"""
================================================================================
Train Script: Geometric Matching Model (v6 - RTX 3090 Optimized + ±180° Curriculum)
================================================================================
[v5 주요 변경사항]
1. RTX 3090 24GB VRAM 최적화: 배치 사이즈 8, ACCUM 4
2. ±60도 회전 강건성을 위한 3단계 커리큘럼 학습
3. 5000장 이미지에 최적화된 에폭 설정 (200 에폭)
4. Mixed Precision (FP16) 활용 극대화
5. 다단계 학습률 스케줄링 (Stage별 LR 조정)

[커리큘럼 전략]
Stage 1 (0-50 에폭):   ±15° → ±30° (기초 학습)
Stage 2 (50-120 에폭): ±30° → ±50° (중급 학습)  
Stage 3 (120-200 에폭): ±50° → ±60° (고급 학습 + 유지)
================================================================================
"""

import sys
import numpy as np

try:
    import numpy._core
except ImportError:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import time
import json
import math
import contextlib


import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# [Hyperparameters] Training Configuration - RTX 3090 24GB Optimized
# =============================================================================
IMG_SIZE = (256, 256)            # [Hyperparameter] 입력 이미지 크기
MAX_SAMPLES_NUM = 5000               # [Hyperparameter] 최대 학습 샘플 수 (5000장)
LIMIT_SAMPLE_NUM = 300              # [Hyperparameter] 데이터셋 샘플 제한 (디버그/빠른 검증용)

# [v6 수정] ±180도 회전 범위 (커리큘럼으로 점진 증가)
ROTATION_MIN = -180.0            # [Hyperparameter] 최종 목표 회전 최소값
ROTATION_MAX = 180.0             # [Hyperparameter] 최종 목표 회전 최대값

# [v5 수정] RTX 3090 24GB 최적화 배치 설정
BATCH_SIZE = 8                   # [Hyperparameter] RTX 3090: 2 → 8 (4배 증가)
ACCUM_STEPS = 4                  # [Hyperparameter] Gradient Accumulation (effective batch = 32)
NUM_EPOCHS = 360                 # [Hyperparameter] 총 학습 에폭 수 (±180° 커리큘럼)
LEARNING_RATE = 3e-4             # [Hyperparameter] 초기 학습률 (더 안정적)
WEIGHT_DECAY = 5e-5              # [Hyperparameter] Weight Decay (약간 감소)

# [모델 차원] - 변경 없음
HIDDEN_DIM = 48                  # [Hyperparameter] Phase 2 임베딩 차원
FEATURE_DIM = 144                # [Hyperparameter] Phase 3 Transformer 차원

# =============================================================================
# [Phase 4 (old 3.5) Training Options]
# =============================================================================
# Phase4(=구 Phase3.5)는 Mini-ConvGRU 기반의 "학습 가능한" 정제기(Refiner)입니다.
# 따라서 학습 스크립트에서 optimizer + checkpoint에 포함하지 않으면,
# 추론 시 Phase4는 랜덤 초기화 가중치로 동작하게 됩니다.
ENABLE_PHASE4_DEFAULT = True
PHASE4_LOSS_WEIGHT_DEFAULT = 1.0
PHASE4_START_EPOCH_DEFAULT = 0
PHASE4_BACKPROP_TO_PHASE23_DEFAULT = False  # True면 Phase4 loss가 Phase2/3에도 역전파됩니다.
PHASE4_VERBOSE_DEFAULT = False              # True면 Phase4 내부 print 로그를 출력합니다.


# [검증 설정]
VAL_SPLIT = 0.1                  # [Hyperparameter] 검증 데이터 비율 (10%)
VAL_INTERVAL = 3                 # [Hyperparameter] 검증 주기 (3 에폭마다)

CHECKPOINT_DIR = "./checkpoints"

# [v6 신규] 6단계 커리큘럼 설정
CURRICULUM_STAGES = [
    (0, 35, 5, 15),    # Stage 1: ±5° → ±15°
    (35, 100, 15, 30),  # Stage 2: ±15° → ±30°
    (100, 210, 30, 45),  # Stage 3: ±30° → ±45°
    (210, 350, 45, 60),  # Stage 4: ±45° → ±60°
]

STAGE_LR_MULTIPLIERS = [1.0, 0.7, 0.5, 0.3] 
# [Warmup 설정]
WARMUP_EPOCHS = 10                # [Hyperparameter] Warmup 에폭 수 (Stage 1의 16%)
WARMUP_START_LR = 1e-7           # [Hyperparameter] Warmup 시작 LR
SCHEDULER_ETA_MIN = 1e-7         # [Hyperparameter] 최소 LR

LOG_INTERVAL = 20                # [Hyperparameter] 로깅 주기

# [v5 신규] 데이터 증강 설정
AUGMENTATION_PROB = 0.3          # [Hyperparameter] 추가 증강 확률
SCALE_JITTER_RANGE = (0.95, 1.05)  # [Hyperparameter] 스케일 변동 범위


# =============================================================================
# [v5 신규] 3단계 커리큘럼 스케줄러
# =============================================================================

class CurriculumScheduler:
    """
    [v5] 3단계 커리큘럼 학습 스케줄러
    
    ±60도 회전 강건성을 위해 점진적으로 난이도를 증가시킵니다.
    
    Stage 1 (기초): ±15° → ±30° (50 에폭)
        - 기본적인 기하학적 관계 학습
        - 빠른 수렴으로 초기 파라미터 안정화
    
    Stage 2 (중급): ±30° → ±50° (70 에폭)
        - 중간 범위 회전에 적응
        - LR을 절반으로 줄여 안정적 학습
    
    Stage 3 (고급): ±50° → ±60° (80 에폭)
        - 극한 회전에서의 미세 조정
        - 가장 낮은 LR로 정밀 학습
    """
    
    def __init__(self, stages=CURRICULUM_STAGES):
        """
        Args:
            stages: [(start_epoch, end_epoch, start_angle, end_angle), ...]
        """
        self.stages = stages
        self.current_stage = 0
        
    def get_rotation_range(self, epoch):
        """
        현재 에폭에 해당하는 회전 범위 반환
        
        Args:
            epoch: 현재 에폭 (0-indexed)
            
        Returns:
            (rot_min, rot_max): 회전 범위 (degree)
        """
        for stage_idx, (start_ep, end_ep, start_ang, end_ang) in enumerate(self.stages):
            if start_ep <= epoch < end_ep:
                self.current_stage = stage_idx
                
                # 스테이지 내 진행률
                progress = (epoch - start_ep) / max(end_ep - start_ep, 1)
                
                # 선형 보간
                current_angle = start_ang + (end_ang - start_ang) * progress
                
                return -current_angle, current_angle
        
        # 마지막 스테이지 이후: 최대 범위 유지
        return -self.stages[-1][3], self.stages[-1][3]
    
    def get_current_stage(self):
        """현재 스테이지 인덱스 반환"""
        return self.current_stage
    
    def get_stage_info(self, epoch):
        """현재 에폭의 스테이지 정보 반환"""
        rot_min, rot_max = self.get_rotation_range(epoch)
        return {
            'stage': self.current_stage + 1,
            'rotation_range': (rot_min, rot_max),
            'lr_multiplier': STAGE_LR_MULTIPLIERS[min(self.current_stage, len(STAGE_LR_MULTIPLIERS)-1)]
        }


# =============================================================================
# [v5 수정] Warmup + Stage-aware Cosine Scheduler
# =============================================================================

class StageAwareWarmupScheduler:
    """
    [v5] Warmup + 스테이지별 LR 조정 스케줄러
    
    각 스테이지 시작 시 LR을 리셋하고 Cosine Annealing 적용
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 curriculum_scheduler, warmup_start_lr=1e-7, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.curriculum = curriculum_scheduler
        
        # 원래 LR 저장
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        self.prev_stage = 0
        
    def step(self):
        self.current_epoch += 1
        stage_info = self.curriculum.get_stage_info(self.current_epoch)
        current_stage = stage_info['stage'] - 1
        lr_mult = stage_info['lr_multiplier']
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i] * lr_mult
            
            if self.current_epoch <= self.warmup_epochs:
                # Phase 1: Linear Warmup
                progress = self.current_epoch / self.warmup_epochs
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
            else:
                # Phase 2: Cosine Annealing (스테이지별)
                if current_stage != self.prev_stage:
                    # 스테이지 변경 시 로그
                    print(f"\n🔄 Stage {current_stage + 1} started! LR multiplier: {lr_mult}")
                
                cosine_epochs = self.total_epochs - self.warmup_epochs
                cosine_progress = (self.current_epoch - self.warmup_epochs) / cosine_epochs
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cosine_progress))
            
            param_group['lr'] = lr
        
        self.prev_stage = current_stage
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# =============================================================================
# [Device Setup]
# =============================================================================

def setup_device():
    """
    [Helper] 학습 장치 설정
    
    TPU > CUDA > CPU 순으로 탐지하여 최적 장치 반환
    """
    device_info = {'type': 'cpu', 'device': torch.device('cpu')}
    
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device_info['type'] = 'tpu'
        device_info['device'] = xm.xla_device()
        device_info['xm'] = xm
        print("✅ TPU Detected")
        return device_info
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        device_info['type'] = 'cuda'
        device_info['device'] = torch.device('cuda')
        
        # [v5] GPU 메모리 정보 출력
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ CUDA Detected - {gpu_name} ({gpu_mem:.1f} GB)")
        
        # RTX 3090 최적화 설정
        if gpu_mem >= 20:  # 20GB 이상
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("   🚀 TF32 & cuDNN Benchmark Enabled")
        
        return device_info
    
    print("⚠️ No GPU/TPU detected - Using CPU")
    return device_info


# =============================================================================
# [v5 수정] Dataset with Advanced Curriculum
# =============================================================================

class GeometricRotationDataset(Dataset):
    """
    [Dataset] 기하학적 회전 데이터셋
    
    v5 변경사항:
    1. 3단계 커리큘럼 지원
    2. 스케일 변동 증강 추가
    3. 피라미드 레벨 최적화 (4 → 5)
    """
    
    def __init__(self, img_dir, is_train=True, max_samples=MAX_SAMPLES_NUM,
                 rot_min=ROTATION_MIN, rot_max=ROTATION_MAX, curriculum_scheduler=None, img_paths=None):
        """
        Args:
            img_dir: 이미지 디렉토리 경로
            is_train: 학습 모드 여부
            max_samples: 최대 샘플 수
            rot_min, rot_max: 회전 범위 (커리큘럼으로 동적 변경)
            curriculum_scheduler: CurriculumScheduler 인스턴스
        """
        # 이미지 경로 리스트를 외부에서 주입할 수 있도록 지원 (train/val split에 사용)
        if img_paths is not None:
            self.img_paths = list(img_paths)
        else:
            self.img_paths = (glob.glob(os.path.join(img_dir, "*.jpg")) +
                              glob.glob(os.path.join(img_dir, "*.png")) +
                              glob.glob(os.path.join(img_dir, "*.jpeg")))
            self.img_paths.sort()

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        if max_samples is not None and len(self.img_paths) > max_samples:
            self.img_paths = self.img_paths[:max_samples]
        self.is_train = is_train
        self.preprocessor = None
        
        # 회전 범위 (커리큘럼으로 동적 업데이트)
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.base_rot_min = rot_min
        self.base_rot_max = rot_max
        
        # [v5] 커리큘럼 스케줄러
        self.curriculum = curriculum_scheduler
        self.current_epoch = 0
        
        print(f"📂 Dataset: {len(self.img_paths)} images loaded")
        print(f"   Target Rotation: {rot_min}° ~ {rot_max}°")
        if curriculum_scheduler:
            print(f"   📈 3-Stage Curriculum: ON")
    
    def set_epoch(self, epoch):
        """
        [v5] 에폭 업데이트 및 커리큘럼 적용
        
        Args:
            epoch: 현재 에폭 (0-indexed)
        """
        self.current_epoch = epoch
        
        if self.curriculum:
            self.rot_min, self.rot_max = self.curriculum.get_rotation_range(epoch)
    
    def _get_preprocessor(self):
        """지연 로딩으로 메모리 효율화"""
        if self.preprocessor is None:
            from pipeline.phase1 import MathGeometricPreprocessor
            self.preprocessor = MathGeometricPreprocessor()
        return self.preprocessor
    
    def __len__(self):
        return len(self.img_paths)
    
    def normalize_affine_matrix(self, matrix_pixel, width, height):
        """픽셀 좌표 → 정규화 좌표 변환"""
        N = np.array([[2.0 / width, 0, -1], 
                      [0, 2.0 / height, -1], 
                      [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_pix_aug = np.vstack([matrix_pixel, [0, 0, 1]])
        M_norm_aug = N @ M_pix_aug @ N_inv
        return M_norm_aug[:2, :]
    
    def __getitem__(self, idx):
        """
        [v5] 데이터 로딩 + 증강
        
        Returns:
            dict: pyramid_a, pyramid_b, w_gt, gt_angle, img_a, img_b
        """
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        
        if img_bgr is None:
            return self.__getitem__((idx + 1) % len(self))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        rows, cols = img_rgb.shape[:2]
        
        # [v5] 회전 각도 샘플링 (커리큘럼 범위 내)
        angle = np.random.uniform(self.rot_min, self.rot_max)
        
        # [v5] 스케일 변동 추가 (학습 모드에서만)
        if self.is_train and np.random.random() < AUGMENTATION_PROB:
            scale = np.random.uniform(*SCALE_JITTER_RANGE)
        else:
            scale = 1.0
        
        # Affine 변환 적용
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        
        # GT 역변환 행렬 계산
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :]
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)
        gt_angle_rad = np.deg2rad(-angle)
        
        # Phase 1 전처리 (피라미드 생성)
        preprocessor = self._get_preprocessor()
        # [v5] 피라미드 레벨 5로 증가 (더 큰 회전에 대응)
        pyramid_a = preprocessor.process_pyramid(img_warped, levels=5)
        pyramid_b = preprocessor.process_pyramid(img_rgb, levels=5)
        
        return {
            'pyramid_a': pyramid_a,
            'pyramid_b': pyramid_b,
            'w_gt': W_gt_mat_norm.astype(np.float32),
            'gt_angle': np.float32(gt_angle_rad),
            'img_a': img_warped,
            'img_b': img_rgb
        }


def collate_fn_geometric(batch):
    """
    [Collate Function] 배치 데이터 조합
    
    피라미드 구조를 유지하면서 배치로 묶음
    """
    batch_size = len(batch)
    levels = len(batch[0]['pyramid_a'])
    
    batched_pyramid_a = [{} for _ in range(levels)]
    batched_pyramid_b = [{} for _ in range(levels)]
    w_gts, gt_angles = [], []
    
    for item in batch:
        w_gts.append(item['w_gt'])
        gt_angles.append(item['gt_angle'])
        
        for l in range(levels):
            for key in item['pyramid_a'][l]:
                if key not in batched_pyramid_a[l]:
                    batched_pyramid_a[l][key] = []
                    batched_pyramid_b[l][key] = []
                batched_pyramid_a[l][key].append(item['pyramid_a'][l][key])
                batched_pyramid_b[l][key].append(item['pyramid_b'][l][key])
    
    for l in range(levels):
        for key in batched_pyramid_a[l]:
            if isinstance(batched_pyramid_a[l][key][0], np.ndarray):
                batched_pyramid_a[l][key] = np.stack(batched_pyramid_a[l][key], axis=0)
                batched_pyramid_b[l][key] = np.stack(batched_pyramid_b[l][key], axis=0)
    
    return {
        'pyramid_a': batched_pyramid_a,
        'pyramid_b': batched_pyramid_b,
        'w_gt': torch.tensor(np.stack(w_gts), dtype=torch.float32),
        'gt_angle': torch.tensor(np.array(gt_angles), dtype=torch.float32)
    }


# =============================================================================
# [Metrics]
# =============================================================================

class MetricTracker:
    """
    [Metrics] 학습 지표 추적
    
    각도 오차, 픽셀 오차, 손실값을 추적하고 통계 제공
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.angle_errors = []
        self.pixel_errors = []
        self.losses = []
        
    def update(self, pred_W, gt_W, pred_angle, gt_angle, loss):
        """지표 업데이트"""
        with torch.no_grad():
            # AMP/autocast 환경에서는 pred_W 등이 fp16으로 나오고,
            # gt_W/gt_angle은 fp32로 유지되는 경우가 많습니다.
            # torch.bmm 같은 연산은 dtype이 반드시 같아야 하므로,
            # 메트릭 계산은 fp32로 통일해 안정적으로 처리합니다.
            pred_W = pred_W.detach().to(dtype=torch.float32)
            gt_W = gt_W.detach().to(dtype=torch.float32)
            pred_angle = pred_angle.detach().to(dtype=torch.float32)
            gt_angle = gt_angle.detach().to(dtype=torch.float32)

            # 각도 오차 (degree)
            # 각도 오차는 주기성이 있으므로 [-pi, pi)로 wrap 후 |diff| 사용
            diff = pred_angle - gt_angle
            diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
            angle_diff = torch.abs(diff) * 180 / np.pi
            self.angle_errors.extend(angle_diff.cpu().numpy().tolist())
            
            # 픽셀 오차 (네 모서리 기준)
            B = pred_W.shape[0]
            corners = torch.tensor([
                [-1., -1., 1.], [1., -1., 1.], 
                [1., 1., 1.], [-1., 1., 1.]
            ], device=pred_W.device, dtype=pred_W.dtype).T.unsqueeze(0).repeat(B, 1, 1)
            
            pts_pred = torch.bmm(pred_W, corners)
            pts_gt = torch.bmm(gt_W, corners)
            
            scale = IMG_SIZE[0] / 2
            pixel_error = torch.norm(pts_pred - pts_gt, dim=1).mean(dim=1) * scale
            self.pixel_errors.extend(pixel_error.cpu().numpy().tolist())
            
            self.losses.append(loss)
            
    def get_metrics(self):
        """통계 반환"""
        return {
            'angle_error_mean': np.mean(self.angle_errors) if self.angle_errors else 0,
            'angle_error_std': np.std(self.angle_errors) if self.angle_errors else 0,
            'pixel_error_mean': np.mean(self.pixel_errors) if self.pixel_errors else 0,
            'pixel_error_std': np.std(self.pixel_errors) if self.pixel_errors else 0,
            'loss_mean': np.mean(self.losses) if self.losses else 0
        }


# =============================================================================
# [Training Loop] (Phase2 + Phase3 + Phase4 optional)
# =============================================================================

def _get_rotor_map_from_result(res: dict):
    """
    Phase3 결과 dict에서 rotor map 텐서를 꺼냅니다.
    - phase3(v5):  res['rotor_map']
    - phase3_archfull: res['delta_rotor_map'] (잔차 ΔW) 또는 res['rotor_map']
    """
    if res is None:
        return None
    rotor = res.get('delta_rotor_map', None)
    if rotor is None:
        rotor = res.get('rotor_map', None)
    return rotor


def sort_results_by_level(results):
    """results(list[dict])를 level(0=Fine) 기준 오름차순 정렬"""
    if results is None:
        return []
    return sorted(results, key=lambda d: d.get('level', 0))


def rotor_map_to_affine(rotor_map):
    """
    Dense Rotor Map (B,H,W,4) = (cos, sin, dx, dy) 를
    Global Affine (B,2,3) 로 변환합니다. (평균 풀링)

    NOTE:
      - dx,dy는 '정규화 좌표계' 기준으로 해석합니다.
      - cos,sin은 normalize_rotor_output()으로 단위화합니다.
    """
    dense_rotor = rotor_map
    avg_rotor = dense_rotor.mean(dim=(1, 2))  # (B,4)

    cos_raw = avg_rotor[:, 0]
    sin_raw = avg_rotor[:, 1]
    dx = avg_rotor[:, 2]
    dy = avg_rotor[:, 3]

    from losses import normalize_rotor_output
    cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)

    row1 = torch.stack([cos_t, -sin_t, dx], dim=1)
    row2 = torch.stack([sin_t,  cos_t, dy], dim=1)
    pred_W = torch.stack([row1, row2], dim=1)
    return pred_W, cos_t, sin_t



def invert_affine_2x3(W: torch.Tensor) -> torch.Tensor:
    """
    W: (B,2,3) -> inverse (B,2,3)
    - grid_sample/affine_grid 컨벤션(out->in)에서 A->B <-> B->A 변환에 사용
    """
    B = W.shape[0]
    device, dtype = W.device, W.dtype
    W_aug = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    W_aug[:, :2, :3] = W
    W_aug[:, 2, 2] = 1.0
    W_inv = torch.inverse(W_aug)
    return W_inv[:, :2, :3]


def get_W_AB_from_phase3_result(res: dict):
    """
    Phase3 result dict에서 'A->B' 방향의 변환행렬을 추출합니다.

    우선순위:
      1) res['W_AB'] (권장: phase3_minpatch_A_dirdebug 계열)
      2) inverse(res['W_global']) (기존 phase3_archfull: W_global=theta_B2A)
      3) rotor_map_to_affine(delta_rotor_map/rotor_map) (구버전 호환)
    """
    if isinstance(res, dict):
        if 'W_AB' in res and res['W_AB'] is not None:
            return res['W_AB']
        if 'W_global' in res and res['W_global'] is not None:
            return invert_affine_2x3(res['W_global'])

        rotor = _get_rotor_map_from_result(res)
        if rotor is None:
            return None
        W_pred, _, _ = rotor_map_to_affine(rotor)
        return W_pred
    return None
def build_W_predictions(results_sorted):
    """Multi-scale consistency를 위한 W 리스트 구축 (fine->coarse).

    주의: phase3_archfull에서는 res['delta_rotor_map']는 '잔차'라서 그대로 쓰면 안 됩니다.
          반드시 누적된 W_global(또는 W_AB)를 사용해야 합니다.
    """
    W_predictions = []
    for res in results_sorted:
        W_ab = get_W_AB_from_phase3_result(res)
        if W_ab is not None:
            W_predictions.append(W_ab)
    return W_predictions


@contextlib.contextmanager
def suppress_stdout(enabled: bool = True):
    """
    Phase4(IterativeRefinementLoop)가 내부에서 print를 많이 수행하므로,
    학습 시 기본적으로 stdout을 억제합니다.
    """
    if not enabled:
        yield
        return

    try:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                yield
    except Exception:
        # fail-open: 억제 실패 시 그냥 출력
        yield


def build_phase4_pyramid_features(transformer, phase2_a, phase2_b, detach: bool = True):
    """
    Phase4(구 Phase3.5) Refiner 입력 텐서 준비

    refiner는 각 피라미드 레벨마다 (B, FEATURE_DIM, H, W) 특징을 필요로 합니다.
    - 채널은 [S | V | B] 3덩어리로 구성(각 FEATURE_DIM//3).
    - 여기서는 Phase3Transformer.prepare_input()을 재사용하여
      Phase2 tuple(S,V,B) -> (B,FEATURE_DIM,H,W)로 변환합니다.

    detach=True:
      - Phase4 loss가 Phase2/3로 역전파되지 않도록 입력 특징을 graph에서 분리합니다.
      - Phase4의 "학습된 리파이너"만 원할 때 기본 권장 옵션입니다.
    """
    feats_a, feats_b = [], []
    for lvl in range(len(phase2_a)):
        if detach:
            with torch.no_grad():
                fa = transformer.prepare_input(phase2_a[lvl]).detach()
                fb = transformer.prepare_input(phase2_b[lvl]).detach()
        else:
            fa = transformer.prepare_input(phase2_a[lvl])
            fb = transformer.prepare_input(phase2_b[lvl])
        feats_a.append(fa)
        feats_b.append(fb)
    return feats_a, feats_b


def train_one_epoch(
    embedder,
    transformer,
    refiner,
    dataloader,
    optimizer,
    criterion,
    device_info,
    scaler=None,
    current_rotation_range=None,
    enable_phase4: bool = True,
    phase4_weight: float = 1.0,
    phase4_backprop_to_phase23: bool = False,
    phase4_verbose: bool = False,
):
    """
    [Training] 한 에폭 학습 (Phase2 + Phase3 + Phase4 옵션)

    - Phase2: CliffordPyramidEmbedder
    - Phase3: Phase3Transformer (Dense Rotor Map)
    - Phase4: Phase35Refiner (Mini-ConvGRU 기반 반복 정제)

    핵심 포인트:
      - Phase4는 학습 가능한 모듈이므로 optimizer/ckpt에 포함되어야 합니다.
      - 다만 Phase4는 '정제기'이므로 Phase3 loss도 반드시 함께 학습시켜야 합니다.
        (Phase4 init_W는 기본적으로 detach되어 Phase3로 gradient가 잘 가지 않습니다.)
    """
    device = device_info['device']
    is_tpu = device_info['type'] == 'tpu'

    embedder.train()
    transformer.train()
    if enable_phase4 and refiner is not None:
        refiner.train()

    # Phase3 / Phase4 별도 지표 + 전체 loss
    metric_p3 = MetricTracker()
    metric_p4 = MetricTracker()
    total_losses = []

    optimizer.zero_grad()

    rot_str = f"[{current_rotation_range[0]:.0f}°~{current_rotation_range[1]:.0f}°]" if current_rotation_range else ""
    pbar = tqdm(dataloader, desc=f"Training {rot_str}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)

        use_amp = (device_info['type'] == 'cuda') and (scaler is not None)

        with torch.amp.autocast('cuda', enabled=use_amp):
            # --------------------------------------------------------------
            # (1) Phase2: Multi-scale Clifford embedding
            # --------------------------------------------------------------
            phase2_a = embedder(pyramid_a_raw, device)
            phase2_b = embedder(pyramid_b_raw, device)

            # --------------------------------------------------------------
            # (2) Phase3: Coarse-to-Fine Transformer -> Dense Rotor Map
            # --------------------------------------------------------------
            results = transformer(phase2_a, phase2_b)
            results_sorted = sort_results_by_level(results)  # level0(Fine) -> ...

            # Phase3 main prediction은 level0(Fine)을 기준으로 사용
            # (기존 v5 fine_tune은 results[0]을 사용했지만, list 순서가 구현에 따라 달라질 수 있어
            #  level을 기준으로 명시적으로 선택합니다.)
            finest_res = results_sorted[0] if len(results_sorted) > 0 else results[0]
            pred_W3 = get_W_AB_from_phase3_result(finest_res)
            if pred_W3 is None:
                raise ValueError('Phase3 result does not contain W_AB/W_global/rotor_map')
            cos3_raw = pred_W3[:, 0, 0]
            sin3_raw = pred_W3[:, 1, 0]
            from losses import normalize_rotor_output
            cos3, sin3 = normalize_rotor_output(cos3_raw, sin3_raw)

            # Multi-scale W 리스트 (IterativeStabilityLoss의 L_multi_scale에 사용)
            W_predictions = build_W_predictions(results_sorted)

            # --------------------------------------------------------------
            # (3) Phase4: Iterative Refinement (학습 가능한 GRU Refiner)
            # --------------------------------------------------------------
            if enable_phase4 and (refiner is not None) and (phase4_weight > 0.0):
                # Phase4 입력 특징 준비 (기본: Phase4 loss는 Phase2/3로 역전파하지 않음)
                feats_a, feats_b = build_phase4_pyramid_features(
                    transformer,
                    phase2_a,
                    phase2_b,
                    detach=(not phase4_backprop_to_phase23),
                )

                # Phase4 내부 print 로그 억제 (학습시 콘솔 폭발 방지)
                with suppress_stdout(enabled=(not phase4_verbose)):
                    pred_W4_B2A, _extra = refiner(
                        feats_a, feats_b,
                        phase3_results=results,
                        device=device,
                    )

                # Phase4에서 얻은 행렬로부터 (cos,sin) 추출
                    pred_W4 = invert_affine_2x3(pred_W4_B2A)  # B->A(theta) -> A->B(W)
                cos4_raw = pred_W4[:, 0, 0]
                sin4_raw = pred_W4[:, 1, 0]
                from losses import normalize_rotor_output
                cos4, sin4 = normalize_rotor_output(cos4_raw, sin4_raw)
            else:
                pred_W4, cos4, sin4 = pred_W3, cos3, sin3

            # --------------------------------------------------------------
            # (4) Loss: UnifiedGeometricLoss(§5) 확장
            #     - L_geo는 pred_W와 무관하므로 1회만 계산
            #     - L_final은 Phase3/Phase4 각각 계산 후 가중합
            #     - L_iter는 (여기서는) multi-scale만 사용
            # --------------------------------------------------------------
            # Phase2 tuple 분해
            S_A, V_A, B_A_tuple = phase2_a[0]
            S_B, V_B, B_B_tuple = phase2_b[0]
            B_A = B_A_tuple[2]  # Rotor Magnitude
            B_B = B_B_tuple[2]

            # §5.1 Geometric Accuracy (W_gt만 사용)
            L_geo, _geo_dict = criterion.geo_loss(S_A, V_A, B_A, S_B, V_B, B_B, w_gt)

            # §5.3 Iterative & Multi-scale (여기서는 L_multi_scale만)
            L_iter, _iter_dict = criterion.iter_loss(delta_W_list=None, W_predictions=W_predictions)

            # §5.2 Final Consistency (Phase3 / Phase4 각각)
            L_final3, _final3 = criterion.final_loss(pred_W3, w_gt, cos3, sin3, gt_angle, S_A, S_B)

            if enable_phase4 and (refiner is not None) and (phase4_weight > 0.0):
                L_final4, _final4 = criterion.final_loss(pred_W4, w_gt, cos4, sin4, gt_angle, S_A, S_B)
            else:
                L_final4 = torch.tensor(0.0, device=device)

            # 통합 Loss (Phase4의 비중을 phase4_weight로 조절)
            total_loss = (
                criterion.alpha * L_geo
                + criterion.beta * (L_final3 + float(phase4_weight) * L_final4)
                + criterion.gamma * L_iter
            )
            loss = total_loss / ACCUM_STEPS

        # --------------------------------------------------------------
        # (5) Backprop + Gradient Accumulation
        # --------------------------------------------------------------
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % ACCUM_STEPS == 0:
            # Gradient clip (Phase4 포함)
            params_for_clip = list(embedder.parameters()) + list(transformer.parameters())
            if enable_phase4 and refiner is not None:
                params_for_clip += list(refiner.parameters())

            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            if is_tpu:
                device_info['xm'].mark_step()

        # --------------------------------------------------------------
        # (6) Metrics
        # --------------------------------------------------------------
        pred_angle3 = torch.atan2(sin3, cos3)
        pred_angle4 = torch.atan2(sin4, cos4)

        # 각 트래커의 loss는 "해당 단계의 FinalConsistency만 반영"한 값으로 기록
        loss_p3 = (criterion.alpha * L_geo + criterion.beta * L_final3 + criterion.gamma * L_iter).item()
        loss_p4 = (criterion.alpha * L_geo + criterion.beta * L_final4 + criterion.gamma * L_iter).item() if enable_phase4 else loss_p3

        metric_p3.update(pred_W3, w_gt, pred_angle3, gt_angle, loss_p3)
        metric_p4.update(pred_W4, w_gt, pred_angle4, gt_angle, loss_p4)
        total_losses.append(total_loss.item())

        if batch_idx % LOG_INTERVAL == 0:
            m4 = metric_p4.get_metrics()
            pbar.set_postfix({
                'tot': f"{np.mean(total_losses):.4f}",
                'p4_loss': f"{m4['loss_mean']:.4f}",
                'p4_ang': f"{m4['angle_error_mean']:.2f}°",
            })

    # 반환: 기본 키는 Phase4(최종 출력) 기준으로 유지
    m3 = metric_p3.get_metrics()
    m4 = metric_p4.get_metrics()
    return {
        'loss_mean': float(np.mean(total_losses)) if total_losses else 0.0,
        'angle_error_mean': m4['angle_error_mean'],
        'angle_error_std': m4['angle_error_std'],
        'pixel_error_mean': m4['pixel_error_mean'],
        'pixel_error_std': m4['pixel_error_std'],

        # 디버그/로깅용 (Phase3)
        'phase3_loss_mean': m3['loss_mean'],
        'phase3_angle_error_mean': m3['angle_error_mean'],
        'phase3_pixel_error_mean': m3['pixel_error_mean'],

        # 디버그/로깅용 (Phase4)
        'phase4_loss_mean': m4['loss_mean'],
    }


@torch.no_grad()
def validate(
    embedder,
    transformer,
    refiner,
    dataloader,
    criterion,
    device_info,
    enable_phase4: bool = True,
    phase4_weight: float = 1.0,
    phase4_backprop_to_phase23: bool = False,
    phase4_verbose: bool = False,
):
    """
    [Validation] 검증 수행 (Phase2 + Phase3 + Phase4 옵션)
    """
    device = device_info['device']

    embedder.eval()
    transformer.eval()
    refiner.eval()
    if enable_phase4 and refiner is not None:
        refiner.eval()

    metric_p3 = MetricTracker()
    metric_p4 = MetricTracker()
    total_losses = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)

        phase2_a = embedder(pyramid_a_raw, device)
        phase2_b = embedder(pyramid_b_raw, device)
        results = transformer(phase2_a, phase2_b)
        results_sorted = sort_results_by_level(results)

        finest_res = results_sorted[0] if len(results_sorted) > 0 else results[0]
        pred_W3 = get_W_AB_from_phase3_result(finest_res)
        if pred_W3 is None:
            raise ValueError('Phase3 result does not contain W_AB/W_global/rotor_map')
        cos3_raw = pred_W3[:, 0, 0]
        sin3_raw = pred_W3[:, 1, 0]
        from losses import normalize_rotor_output
        cos3, sin3 = normalize_rotor_output(cos3_raw, sin3_raw)

        W_predictions = build_W_predictions(results_sorted)

        if enable_phase4 and (refiner is not None) and (phase4_weight > 0.0):
            feats_a, feats_b = build_phase4_pyramid_features(
                transformer,
                phase2_a,
                phase2_b,
                detach=(not phase4_backprop_to_phase23),
            )
            with suppress_stdout(enabled=(not phase4_verbose)):
                pred_W4_B2A, _extra = refiner(
                    feats_a, feats_b,
                    phase3_results=results,
                    device=device,
                )
                pred_W4 = invert_affine_2x3(pred_W4_B2A)  # B->A(theta) -> A->B(W)
            cos4_raw = pred_W4[:, 0, 0]
            sin4_raw = pred_W4[:, 1, 0]
            from losses import normalize_rotor_output
            cos4, sin4 = normalize_rotor_output(cos4_raw, sin4_raw)
        else:
            pred_W4, cos4, sin4 = pred_W3, cos3, sin3

        # Loss (train과 동일한 구성)
        S_A, V_A, B_A_tuple = phase2_a[0]
        S_B, V_B, B_B_tuple = phase2_b[0]
        B_A = B_A_tuple[2]
        B_B = B_B_tuple[2]

        L_geo, _ = criterion.geo_loss(S_A, V_A, B_A, S_B, V_B, B_B, w_gt)
        L_iter, _ = criterion.iter_loss(delta_W_list=None, W_predictions=W_predictions)
        L_final3, _ = criterion.final_loss(pred_W3, w_gt, cos3, sin3, gt_angle, S_A, S_B)
        if enable_phase4 and (refiner is not None) and (phase4_weight > 0.0):
            L_final4, _ = criterion.final_loss(pred_W4, w_gt, cos4, sin4, gt_angle, S_A, S_B)
        else:
            L_final4 = torch.tensor(0.0, device=device)

        total_loss = (
            criterion.alpha * L_geo
            + criterion.beta * (L_final3 + float(phase4_weight) * L_final4)
            + criterion.gamma * L_iter
        )

        pred_angle3 = torch.atan2(sin3, cos3)
        pred_angle4 = torch.atan2(sin4, cos4)

        loss_p3 = (criterion.alpha * L_geo + criterion.beta * L_final3 + criterion.gamma * L_iter).item()
        loss_p4 = (criterion.alpha * L_geo + criterion.beta * L_final4 + criterion.gamma * L_iter).item() if enable_phase4 else loss_p3

        metric_p3.update(pred_W3, w_gt, pred_angle3, gt_angle, loss_p3)
        metric_p4.update(pred_W4, w_gt, pred_angle4, gt_angle, loss_p4)
        total_losses.append(total_loss.item())

    m3 = metric_p3.get_metrics()
    m4 = metric_p4.get_metrics()
    return {
        'loss_mean': float(np.mean(total_losses)) if total_losses else 0.0,
        'angle_error_mean': m4['angle_error_mean'],
        'angle_error_std': m4['angle_error_std'],
        'pixel_error_mean': m4['pixel_error_mean'],
        'pixel_error_std': m4['pixel_error_std'],

        'phase3_loss_mean': m3['loss_mean'],
        'phase3_angle_error_mean': m3['angle_error_mean'],
        'phase3_pixel_error_mean': m3['pixel_error_mean'],
        'phase4_loss_mean': m4['loss_mean'],
    }


# =============================================================================
# [Main Training Function]
# =============================================================================

def train(
    img_dir,
    resume_from=None,
    debug_mode=False,
    enable_phase4=ENABLE_PHASE4_DEFAULT,
    phase4_weight=PHASE4_LOSS_WEIGHT_DEFAULT,
    phase4_start_epoch=PHASE4_START_EPOCH_DEFAULT,
    phase4_backprop_to_phase23=PHASE4_BACKPROP_TO_PHASE23_DEFAULT,
    phase4_verbose=PHASE4_VERBOSE_DEFAULT,
):
    """
    [Main] v5 학습 메인 함수
    
    RTX 3090 24GB에 최적화된 ±60° 커리큘럼 학습
    
    Args:
        img_dir: 이미지 디렉토리 경로
        resume_from: 재개할 체크포인트 경로
        debug_mode: 디버그 모드 (적은 샘플로 테스트)
    """
    
    # -------------------------------------------------------------------------
    # [Mode Settings]
    # -------------------------------------------------------------------------
    if debug_mode:
        print("\n" + "⚡" * 40)
        print("⚡ DEBUG MODE - Limited Samples ⚡")
        print("⚡" * 40 + "\n")
        
        limit_samples = LIMIT_SAMPLE_NUM
        run_epochs = 30  # 디버그용 짧은 에폭
        save_name_prefix = "debug_"
    else:
        limit_samples = MAX_SAMPLES_NUM
        run_epochs = NUM_EPOCHS
        save_name_prefix = "v6_180deg_"
    
    # -------------------------------------------------------------------------
    # [Print Configuration]
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("🚀 Geometric Matching Model Training (v5 - RTX 3090 Optimized)")
    print("=" * 70)
    print(f"   Target Rotation: ±{ROTATION_MAX}° (3-Stage Curriculum)")
    print(f"   Stages:")
    for i, (s, e, sa, ea) in enumerate(CURRICULUM_STAGES):
        print(f"      Stage {i+1}: Epoch {s:3d}-{e:3d} | ±{sa:.0f}° → ±{ea:.0f}°")
    print(f"   Dataset: {limit_samples if limit_samples else 'All'} images")
    print(f"   Epochs: {run_epochs}")
    print(f"   Batch Size: {BATCH_SIZE} × {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS} (effective)")
    print(f"   Learning Rate: {LEARNING_RATE:.2e} (with stage decay)")
    print(f"   Warmup: {WARMUP_EPOCHS} epochs")
    print(f"   Phase4 Refiner: {'ON' if enable_phase4 else 'OFF'} | weight={phase4_weight} | start_epoch={phase4_start_epoch} | backprop_to_phase23={phase4_backprop_to_phase23}")
    print("=" * 70)
    
    # Device Setup
    device_info = setup_device()
    device = device_info['device']
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # ==========================================================================
    # [1] Curriculum Scheduler
    # ==========================================================================
    curriculum = CurriculumScheduler(CURRICULUM_STAGES if not debug_mode else [
        (0, 10, 10.0, 20.0),
        (10, 20, 20.0, 40.0),
        (20, 30, 40.0, 60.0),
    ])
    
    # ==========================================================================
    # [2] Dataset & DataLoader
    # ==========================================================================
    print("\n📊 Loading Dataset...")
    
    # ==========================================================================
    # [2] Datasets & Dataloaders (val: jitter OFF)
    # ==========================================================================
    print("\n📊 Loading Dataset...")

    # ---- 파일 리스트 확보 ----
    all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                          glob.glob(os.path.join(img_dir, "*.jpg")) +
                          glob.glob(os.path.join(img_dir, "*.jpeg")))
    if len(all_img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    # ---- 샘플 수 제한(선택) ----
    if limit_samples is not None and len(all_img_paths) > limit_samples:
        all_img_paths = all_img_paths[:limit_samples]

    # ---- Train/Val split (재현성 확보) ----
    rng = np.random.RandomState(42)
    rng.shuffle(all_img_paths)
    total_size = len(all_img_paths)
    val_size = max(int(total_size * VAL_SPLIT), 1)
    train_size = total_size - val_size
    train_paths = all_img_paths[:train_size]
    val_paths = all_img_paths[train_size:]

    train_dataset = GeometricRotationDataset(
        img_dir=img_dir, is_train=True, max_samples=None,
        rot_min=ROTATION_MIN, rot_max=ROTATION_MAX, curriculum_scheduler=curriculum,
        img_paths=train_paths
    )
    val_dataset = GeometricRotationDataset(
        img_dir=img_dir, is_train=False, max_samples=None,
        rot_min=ROTATION_MIN, rot_max=ROTATION_MAX, curriculum_scheduler=curriculum,
        img_paths=val_paths
    )

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_geometric)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_geometric)

    print(f"   Total: {total_size}, Train: {train_size}, Val: {val_size}")
    print(f"   Batch size: {BATCH_SIZE}")

    # [3] Model
    # ==========================================================================
    print("\n🏗️ Building Model...")
    
    from pipeline.phase2 import CliffordPyramidEmbedder
    from pipeline.phase3 import Phase3Transformer
    from pipeline.phase4 import IterativeRefinementLoop 
    from losses import UnifiedGeometricLoss

    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    # Phase4(구 Phase3.5) - 학습 가능한 Refiner
    refiner = IterativeRefinementLoop(feature_dim=FEATURE_DIM).to(device)

    total_params = (
        sum(p.numel() for p in embedder.parameters())
        + sum(p.numel() for p in transformer.parameters())
        + sum(p.numel() for p in refiner.parameters())
    )
    print(f"   Parameters: {total_params:,}")

    start_epoch = 0
    best_val_loss = float('inf')  # Best 점수 추적을 위한 변수
    
    # ==========================================================================
    # [4] Optimizer & Scheduler
    # ==========================================================================
    optimizer_params = list(embedder.parameters()) + list(transformer.parameters())
    if enable_phase4:
        optimizer_params += list(refiner.parameters())

    optimizer = optim.AdamW(
        optimizer_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # [v5] Stage-aware Warmup Scheduler
    scheduler = StageAwareWarmupScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=run_epochs,
        curriculum_scheduler=curriculum,
        warmup_start_lr=WARMUP_START_LR,
        min_lr=SCHEDULER_ETA_MIN
    )
    
    # [v5] 손실 함수 (큰 회전에 맞게 가중치 조정)
    criterion = UnifiedGeometricLoss(
        alpha=1.0,    # Geometric Accuracy
        beta=1.5,     # Final Consistency (각도 중요)
        gamma=0.1     # Iterative Stability
    ).to(device)
    
    # [v5] Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if device_info['type'] == 'cuda' else None
    
    # ==========================================================================
    # [5] Resume
    # ==========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n📥 Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
        if 'refiner' in checkpoint:
            refiner.load_state_dict(checkpoint['refiner'])
        else:
            print("   ⚠️ Checkpoint has no 'refiner' weights. Phase4 will start from random init.")
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = 0  # 🔥 무조건 0부터 시작하도록 수정
        best_val_loss = float('inf') # 검증 손실 기록도 초기화하는 것이 좋습니다.
        
        # 🔥 [추가] scheduler 상태 동기화
        scheduler.current_epoch = start_epoch
        scheduler.prev_stage = curriculum.get_stage_info(start_epoch)['stage'] - 1
    
    # ==========================================================================
    # [6] Training Loop
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🎯 Starting Training...")
    print("=" * 70)
    
    history = {
        'train_loss': [], 'train_angle': [],
        'val_loss': [], 'val_angle': [],
        'learning_rate': [], 'rotation_range': []
    }
    
    for epoch in range(start_epoch, run_epochs):
        epoch_start = time.time()
        
        # 커리큘럼 업데이트
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)  # val: jitter OFF, rotation range만 동기화
        stage_info = curriculum.get_stage_info(epoch)
        current_rot = stage_info['rotation_range']

        current_stage = stage_info['stage']
        if epoch > 0:
            prev_stage_info = curriculum.get_stage_info(epoch - 1)
            if current_stage != prev_stage_info['stage']:
                # 🔥 스테이지 전환 시 patience & best_val_loss 리셋
                patience_counter = 0
                best_val_loss = float('inf')
                print(f"   🔄 Stage {current_stage} 시작! patience & best_loss 리셋")
        
        # 현재 LR
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📌 Epoch {epoch+1}/{run_epochs} | Stage {stage_info['stage']} | "
              f"Rotation: {current_rot[0]:.0f}°~{current_rot[1]:.0f}° | LR: {current_lr:.2e}")
        print("-" * 50)
        
        # Train
        enable_phase4_now = bool(enable_phase4) and (epoch >= int(phase4_start_epoch)) and (float(phase4_weight) > 0.0)

        train_metrics = train_one_epoch(
            embedder, transformer, refiner,
            train_loader, optimizer, criterion,
            device_info,
            scaler,
            current_rot,
            enable_phase4=enable_phase4_now,
            phase4_weight=float(phase4_weight),
            phase4_backprop_to_phase23=bool(phase4_backprop_to_phase23),
            phase4_verbose=bool(phase4_verbose),
        )
        
        # Scheduler Step
        scheduler.step()
        
        print(f"   [Train] Loss: {train_metrics['loss_mean']:.4f} | "
              f"Angle: {train_metrics['angle_error_mean']:.2f}° ± {train_metrics['angle_error_std']:.2f}° | "
              f"Pixel: {train_metrics['pixel_error_mean']:.2f}px")
        
        history['train_loss'].append(train_metrics['loss_mean'])
        history['train_angle'].append(train_metrics['angle_error_mean'])
        history['learning_rate'].append(current_lr)
        history['rotation_range'].append(current_rot)
        
        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == run_epochs - 1:
            enable_phase4_now = bool(enable_phase4) and (epoch >= int(phase4_start_epoch)) and (float(phase4_weight) > 0.0)
            val_metrics = validate(
                embedder, transformer, refiner,
                val_loader, criterion, device_info,
                enable_phase4=enable_phase4_now,
                phase4_weight=float(phase4_weight),
                phase4_backprop_to_phase23=bool(phase4_backprop_to_phase23),
                phase4_verbose=bool(phase4_verbose),
            )
            
            # 🔥 [수정] 변수 정의: val_metrics에서 loss_mean을 추출하여 정의합니다.
            current_val_loss = val_metrics['loss_mean']
            
            print(f"   [Val]   Loss: {current_val_loss:.4f} | "
                  f"Angle: {val_metrics['angle_error_mean']:.2f}° ± {val_metrics['angle_error_std']:.2f}° | "
                  f"Pixel: {val_metrics['pixel_error_mean']:.2f}px")
            
            # =================================================================
            # [Best Model 저장 로직] Loss 기준 최저점 저장
            # =================================================================
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                rot_tag = int(abs(current_rot[1]))
                mean = float(val_metrics.get('angle_error_mean', 0.0))
                std = float(val_metrics.get('angle_error_std', 0.0))
                best_fname = f"{save_name_prefix}best_rot{rot_tag}_{mean:.2f}+-{std:.2f}.pth"
                best_path = os.path.join(CHECKPOINT_DIR, best_fname)
                stable_best_path = os.path.join(CHECKPOINT_DIR, f"{save_name_prefix}best_model.pth")

                checkpoint = {
                    'epoch': epoch,
                    'embedder': embedder.state_dict(),
                    'transformer': transformer.state_dict(),
                    'refiner': refiner.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'metrics': val_metrics,
                    'training_config': {
                        'version': 'v6_best',
                        'rotation_range': current_rot
                    }
                }
                torch.save(checkpoint, best_path)
                torch.save(checkpoint, stable_best_path)
                best_model_path = best_path
                print(f"   🌟 Best Model Saved! (Loss: {best_val_loss:.4f})")

            # history 업데이트 (current_val_loss 변수 사용)
            history['val_loss'].append(current_val_loss)
            history['val_angle'].append(val_metrics['angle_error_mean'])
                   
        epoch_time = time.time() - epoch_start
        print(f"   ⏱️ Time: {epoch_time:.1f}s")
        
        # Last Model Backup (매 에폭)
        torch.save({
            'epoch': epoch,
            'embedder': embedder.state_dict(),
            'transformer': transformer.state_dict(),
            'refiner': refiner.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'hidden_dim': HIDDEN_DIM,
            'feature_dim': FEATURE_DIM,
            'training_config': {
                'version': 'v5',
                'rotation_range': current_rot
            }
        }, os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}last_model.pth'))
    
    # ==========================================================================
    # [7] Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    
    stable_best_path = os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}best_model.pth')
    print(f"\n💾 Best checkpoint (stable): {stable_best_path}")
    if best_model_path is not None:
        print(f"💾 Best checkpoint (tagged): {best_model_path}")

    load_path = stable_best_path
    if best_model_path is not None and os.path.exists(best_model_path):
        load_path = best_model_path

    if os.path.exists(load_path):
        best_ckpt = torch.load(load_path, weights_only=False)
        final_metrics = best_ckpt.get('metrics', {})
        final_config = best_ckpt.get('training_config', {})
        
        print(f"\n📊 Best Model:")
        print(f"   Val Loss: {best_val_loss:.4f}")
        print(f"   Angle Error: {final_metrics.get('angle_error_mean', 'N/A'):.2f}°")
        print(f"   Pixel Error: {final_metrics.get('pixel_error_mean', 'N/A'):.2f}px")
        print(f"   Trained up to: ±{final_config.get('rotation_range', ['?', '?'])[1]}°")
    
    # History 저장
    with open(os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}history.json'), 'w') as f:
        # rotation_range는 튜플이라 리스트로 변환
        history_serializable = history.copy()
        history_serializable['rotation_range'] = [list(r) for r in history['rotation_range']]
        json.dump(history_serializable, f, indent=2)
    
    return history


# =============================================================================
# [Quick Test]
# =============================================================================

def quick_test(img_dir, checkpoint_path=None, test_angles=[15, 30, 45, 60]):
    """
    [Test] 다양한 각도에서 모델 성능 테스트
    
    Args:
        img_dir: 이미지 디렉토리
        checkpoint_path: 체크포인트 경로
        test_angles: 테스트할 각도 리스트
    """
    print(f"\n🧪 Quick Test - Angles: {test_angles}°")
    
    device_info = setup_device()
    device = device_info['device']
    
    from pipeline.phase2 import CliffordPyramidEmbedder
    from pipeline.phase3 import Phase3Transformer
    from pipeline.phase4 import IterativeRefinementLoop 

    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    refiner = IterativeRefinementLoop (feature_dim=FEATURE_DIM).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        if 'refiner' in ckpt:
            refiner.load_state_dict(ckpt['refiner'])
        
        config = ckpt.get('training_config', {})
        print(f"   Loaded: {checkpoint_path}")
        print(f"   Trained rotation: {config.get('rotation_range', 'Unknown')}")
    
    embedder.eval()
    transformer.eval()
    refiner.eval()
    
    # 각 각도별 테스트
    results = {}
    
    for test_angle in test_angles:
        print(f"\n   Testing ±{test_angle}°...")
        
        test_dataset = GeometricRotationDataset(
            img_dir, max_samples=20,
            rot_min=-test_angle, rot_max=test_angle
        )
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_geometric)
        
        errors = []
        with torch.no_grad():
            for batch in test_loader:
                pyramid_a = batch['pyramid_a']
                pyramid_b = batch['pyramid_b']
                gt_angle_rad = batch['gt_angle'].item()
                
                phase2_a = embedder(pyramid_a, device)
                phase2_b = embedder(pyramid_b, device)
                results_model = transformer(phase2_a, phase2_b)
                
                results_sorted = sort_results_by_level(results_model)
                rotor_map = _get_rotor_map_from_result(results_sorted[0])
                pred_W3, cos3, sin3 = rotor_map_to_affine(rotor_map)
                pred_angle3 = np.degrees(np.arctan2(sin3[0].item(), cos3[0].item()))

                # Phase4 refinement (checkpoint에 refiner가 없으면 랜덤 init이므로 주의)
                feats_a, feats_b = build_phase4_pyramid_features(transformer, phase2_a, phase2_b, detach=True)
                with suppress_stdout(enabled=True):
                    pred_W4, _hist = refiner(feats_a, feats_b, phase3_results=results_model, device=device)
                cos4 = pred_W4[0, 0, 0].item()
                sin4 = pred_W4[0, 1, 0].item()
                pred_angle4 = np.degrees(np.arctan2(sin4, cos4))

                gt_angle_deg = np.degrees(gt_angle_rad)

                # 기본은 Phase4(최종) 기준으로 에러 계산
                error = abs(pred_angle4 - gt_angle_deg)
                errors.append(error)
        
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        results[test_angle] = (mean_err, std_err)
        print(f"      Mean Error: {mean_err:.2f}° ± {std_err:.2f}°")
    
    print(f"\n   📊 Summary:")
    for angle, (mean, std) in results.items():
        status = "✅" if mean < angle * 0.1 else "⚠️" if mean < angle * 0.2 else "❌"
        print(f"      {status} ±{angle}°: {mean:.2f}° ± {std:.2f}°")


# =============================================================================
# [Entry Point]
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Geometric Matching Training v5 (RTX 3090 Optimized)')
    parser.add_argument('--img_dir', type=str, default='./val2017',
                        help='Image directory path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with limited samples')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint for testing')
    

    # Phase4(구 Phase3.5) Refiner 학습 옵션
    parser.add_argument('--disable_phase4', action='store_true',
                        help='Disable Phase4 refiner training (train Phase2+3 only)')
    parser.add_argument('--phase4_weight', type=float, default=PHASE4_LOSS_WEIGHT_DEFAULT,
                        help='Loss weight for Phase4 final consistency term')
    parser.add_argument('--phase4_start_epoch', type=int, default=PHASE4_START_EPOCH_DEFAULT,
                        help='Epoch to start Phase4 training (0-indexed)')
    parser.add_argument('--phase4_backprop_to_phase23', action='store_true',
                        help='Allow Phase4 loss to backprop into Phase2/3 (not recommended by default)')
    parser.add_argument('--phase4_verbose', action='store_true',
                        help='Enable Phase4 internal verbose logs (prints)')

    args = parser.parse_args()
    
    enable_phase4 = (not args.disable_phase4)

    if args.test:
        quick_test(args.img_dir, args.checkpoint)
    else:
        train(
            args.img_dir,
            args.resume,
            args.debug,
            enable_phase4=enable_phase4,
            phase4_weight=args.phase4_weight,
            phase4_start_epoch=args.phase4_start_epoch,
            phase4_backprop_to_phase23=args.phase4_backprop_to_phase23,
            phase4_verbose=args.phase4_verbose,
        )