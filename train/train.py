"""
Training script for the rotation-robust dense matching pipeline.

This version keeps the existing curriculum / optimizer structure, but tightens
the Phase 4 integration points:

- use `Phase3Transformer.prepare_phase4_input()` when available
- keep conventions explicit: Phase 3 returns W_AB for loss/eval and W_B2A for
  warping; Phase 4 keeps B->A internally and is inverted only at the loss edge
- optionally open Phase4 -> Phase2/3 gradients only in the late stage of
  training
- pass the full rotor tuple into the geometric loss so bivector orientation
  consistency is not silently disabled
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
import re
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
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# =============================================================================
# Training defaults
# =============================================================================
IMG_SIZE = (256, 256)            # [Hyperparameter] 입력 이미지 크기
MAX_SAMPLES_NUM = 3000               # [Hyperparameter] 최대 학습 샘플 수 (5000장)
LIMIT_SAMPLE_NUM = 300              # [Hyperparameter] 데이터셋 샘플 제한 (디버그/빠른 검증용)

# [v6 수정] ±180도 회전 범위 (커리큘럼으로 점진 증가)
ROTATION_MIN = -90.0             # [Hyperparameter] 최종 목표 회전 최소값
ROTATION_MAX = 90.0              # [Hyperparameter] 최종 목표 회전 최대값
BATCH_SIZE = 1
ACCUM_STEPS = 16            # effective batch = 12
NUM_EPOCHS = 850
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 1e-4

# [모델 차원] - 변경 없음
HIDDEN_DIM = 48                  # [Hyperparameter] Phase 2 임베딩 차원
FEATURE_DIM = 144                # [Hyperparameter] Phase 3 Transformer 차원

# =============================================================================
# Phase 4 integration options
# =============================================================================
# Phase 4 is a learned recurrent refiner.
# It must be part of the optimizer/checkpoint if you expect it to improve during training.
ENABLE_PHASE4_DEFAULT = True
PHASE4_VERBOSE_DEFAULT = False              # True면 Phase4 내부 print 로그를 출력합니다.
PHASE4_START_EPOCH_DEFAULT = 80
PHASE4_LOSS_WEIGHT_DEFAULT = 0.25
PHASE4_BACKPROP_TO_PHASE23_DEFAULT = False
PHASE4_BACKPROP_START_RATIO_DEFAULT = 0.85

# [검증 설정]
VAL_SPLIT = 0.1                  # [Hyperparameter] 검증 데이터 비율 (10%)
VAL_INTERVAL = 3                 # [Hyperparameter] 검증 주기 (3 에폭마다)

CHECKPOINT_DIR = "./checkpoints"  # legacy fallback only
DEFAULT_MAIN_CHECKPOINT_ROOT = os.path.expanduser("~/work/checkpoints")
DEFAULT_ABLATION_CHECKPOINT_ROOT = os.path.expanduser("~/scratch/ablation")
DEFAULT_GLOBAL_BEST_MIN_ROT = 60.0

CURRICULUM_STAGES = [
    (0,    50,  3,  8),
    (50,  120,  8, 18),
    (120, 200, 18, 30),
    (200, 300, 30, 45),
    (300, 470, 45, 60),    # 5→65°로 살짝 더
    (470, 620, 60, 75),
    (620, 720, 75, 90),
    (720, 850, 10, 90),    # 전체 범위 랜덤 노출
]

STAGE_LR_MULTIPLIERS = [1.0, 0.75, 0.55, 0.38, 0.25, 0.16, 0.12, 0.08]

# [Warmup 설정]
WARMUP_EPOCHS = 10
WARMUP_START_LR = 1e-7
SCHEDULER_ETA_MIN = 1e-8

LOG_INTERVAL = 20                # [Hyperparameter] 로깅 주기

AUGMENTATION_PROB = 0.3          # [Hyperparameter] 추가 증강 확률
SCALE_JITTER_RANGE = (0.95, 1.05)  # [Hyperparameter] 스케일 변동 범위

# =============================================================================
# Curriculum scheduler
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
# Warmup + stage-aware cosine scheduler
# =============================================================================

class StageAwareWarmupScheduler:
    """Warmup + stage-aware cosine decay scheduler."""
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 curriculum_scheduler, warmup_start_lr=1e-7, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.curriculum = curriculum_scheduler

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = -1
        self.prev_stage = 0

        for group in self.optimizer.param_groups:
            group['lr'] = self.warmup_start_lr

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = int(epoch)

        stage_info = self.curriculum.get_stage_info(self.current_epoch)
        current_stage = stage_info['stage'] - 1
        lr_mult = stage_info['lr_multiplier']

        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i] * lr_mult
            if self.current_epoch < self.warmup_epochs:
                progress = self.current_epoch / max(self.warmup_epochs - 1, 1)
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
            else:
                if current_stage != self.prev_stage:
                    print(f"\n🔄 Stage {current_stage + 1} started! LR multiplier: {lr_mult}")
                cosine_epochs = max(self.total_epochs - self.warmup_epochs, 1)
                cosine_progress = (self.current_epoch - self.warmup_epochs) / cosine_epochs
                cosine_progress = min(max(cosine_progress, 0.0), 1.0)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * cosine_progress)
                )
            param_group['lr'] = lr

        self.prev_stage = current_stage

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# =============================================================================
# Device setup
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
        
        # CUDA 학습 공통 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("   🚀 TF32 & cuDNN Benchmark Enabled")
        
        return device_info
    
    print("⚠️ No GPU/TPU detected - Using CPU")
    return device_info


def sanitize_tag(text_value):
    text_value = str(text_value).strip()
    text_value = re.sub(r'[^A-Za-z0-9_.+-]+', '-', text_value)
    text_value = re.sub(r'-{2,}', '-', text_value)
    return text_value.strip('-') or 'exp'


def build_default_experiment_name(debug_mode: bool,
                                  enable_phase4: bool,
                                  phase4_weight: float,
                                  criterion_beta: float,
                                  criterion_gamma: float,
                                  target_rotation: float) -> str:
    mode = 'debug' if debug_mode else 'main'
    p4 = 'p4on' if enable_phase4 else 'p4off'
    return sanitize_tag(
        f"{mode}_gpfull_{p4}_p4w{int(round(phase4_weight * 100)):02d}_"
        f"b{int(round(criterion_beta * 100)):03d}_g{int(round(criterion_gamma * 100)):03d}_"
        f"rot{int(round(target_rotation))}"
    )


def resolve_checkpoint_dir(debug_mode: bool,
                           experiment_name: str | None,
                           checkpoint_root: str | None,
                           checkpoint_dir: str | None,
                           enable_phase4: bool,
                           phase4_weight: float,
                           criterion_beta: float,
                           criterion_gamma: float,
                           target_rotation: float):
    experiment_name = experiment_name or build_default_experiment_name(
        debug_mode=debug_mode,
        enable_phase4=enable_phase4,
        phase4_weight=phase4_weight,
        criterion_beta=criterion_beta,
        criterion_gamma=criterion_gamma,
        target_rotation=target_rotation,
    )
    experiment_name = sanitize_tag(experiment_name)

    if checkpoint_dir:
        run_dir = Path(os.path.expanduser(checkpoint_dir))
    else:
        root = checkpoint_root
        if root is None:
            root = DEFAULT_ABLATION_CHECKPOINT_ROOT if debug_mode else DEFAULT_MAIN_CHECKPOINT_ROOT
        run_dir = Path(os.path.expanduser(root)) / experiment_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir), experiment_name


def resolve_resume_path(resume_from: str | None, experiment_dir: str) -> str | None:
    if not resume_from:
        return None
    resume_path = Path(os.path.expanduser(resume_from))
    if resume_path.is_dir():
        exp_name = Path(experiment_dir).name
        candidates = [
            resume_path / 'best_global.pth',
            resume_path / 'last.pth',
            resume_path / f'{exp_name}__best_global.pth',
            resume_path / f'{exp_name}__last.pth',
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
        return None
    return str(resume_path) if resume_path.exists() else None


def format_metric_tag(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace('-', 'm')


def compute_stage_score(metrics: dict) -> float:
    return float(metrics.get('angle_error_mean', 0.0)) + 0.10 * float(metrics.get('pixel_error_mean', 0.0))


def compute_global_score(metrics: dict) -> float:
    return float(metrics.get('angle_error_mean', 0.0)) + 0.15 * float(metrics.get('pixel_error_mean', 0.0))


def build_checkpoint_payload(epoch, embedder, transformer, refiner, optimizer,
                             metrics, training_config, extra=None):
    payload = {
        'epoch': int(epoch),
        'embedder': embedder.state_dict(),
        'transformer': transformer.state_dict(),
        'refiner': refiner.state_dict() if refiner is not None else None,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'metrics': metrics,
        'training_config': training_config,
    }
    if extra:
        payload.update(extra)
    return payload


def save_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def write_experiment_config(config_path: str, payload: dict):
    serializable = dict(payload)
    if 'rotation_range' in serializable:
        serializable['rotation_range'] = list(serializable['rotation_range'])
    save_json(config_path, serializable)


def build_curriculum_stages(debug_mode: bool, run_epochs: int):
    if debug_mode:
        base = [
            (0, 3, 5, 15),
            (3, 6, 15, 30),
            (6, 9, 30, 45),
            (9, 12, 45, 60),
        ]
    else:
        base = CURRICULUM_STAGES

    base_total = base[-1][1]
    if run_epochs == base_total:
        return list(base)

    scaled = []
    prev_end = 0
    for idx, (start_ep, end_ep, start_ang, end_ang) in enumerate(base):
        new_start = int(round(start_ep / base_total * run_epochs))
        new_end = int(round(end_ep / base_total * run_epochs))
        if idx == 0:
            new_start = 0
        new_start = max(new_start, prev_end)
        new_end = max(new_end, new_start + 1)
        scaled.append((new_start, new_end, start_ang, end_ang))
        prev_end = new_end

    last = list(scaled[-1])
    last[1] = max(run_epochs, last[0] + 1)
    scaled[-1] = tuple(last)
    return scaled

# =============================================================================
# Dataset
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
        """픽셀 좌표 -> 정규화 좌표 변환.

        학습/추론의 내부 warp는 `align_corners=True` 규약을 사용하므로,
        GT affine도 동일한 좌표계로 변환합니다.
        """
        sx = 2.0 / max(width - 1, 1)
        sy = 2.0 / max(height - 1, 1)
        N = np.array([[sx, 0, -1],
                      [0, sy, -1],
                      [0,  0,  1]], dtype=np.float32)
        N_inv = np.linalg.inv(N)
        M_pix_aug = np.vstack([matrix_pixel, [0, 0, 1]]).astype(np.float32)
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
        pyramid_a = preprocessor.process_pyramid(img_warped, levels=6)
        pyramid_b = preprocessor.process_pyramid(img_rgb, levels=6)
        
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
# Metrics
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
# Training helpers
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
    Extract the forward transform used by the losses and evaluation.

    Convention
    ----------
    - Phase 3 keeps `W_B2A` / `W_global` as the backward theta fed to
      affine_grid/grid_sample.
    - The training loss is defined on `W_AB` to match the dataset's `w_gt`.

    Fallback order:
      1) `res["W_AB"]`
      2) `inverse(res["W_global"])`
      3) pooled affine from a rotor map (legacy compatibility)
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
    """Build the per-level transform list used by the multi-scale loss.

    The list is ordered fine -> coarse because `results_sorted` is already sorted
    by level index (0 = finest). Only accumulated transforms are valid here;
    raw residual rotor maps must not be used directly.
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

def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward 값은 유지하면서 gradient만 scale 배율로 통과시킵니다."""
    if scale <= 0.0:
        return x.detach()
    if scale >= 1.0:
        return x
    return x.detach() + float(scale) * (x - x.detach())

def build_phase4_pyramid_features(transformer, phase2_a, phase2_b, detach: bool = True,
                                  backprop_scale: float = 1.0):
    """
    Build the per-level feature volumes consumed by Phase 4.

    The current Phase 3 implementation already exposes
    `prepare_phase4_input()`, which keeps the richer rotor tuple
    `(unit_cos, unit_sin, rotor_mag)` alive for the downstream refiner.
    For backward compatibility we still fall back to `prepare_input()` when an
    older transformer checkpoint/module is loaded.

    `detach=True` keeps Phase 4 as a pure downstream refiner.
    `backprop_scale` lets the Phase 4 loss open a late-stage gradient path back
    into Phase 2/3 without changing the forward values.
    """
    feats_a, feats_b = [], []
    phase4_prep = getattr(transformer, 'prepare_phase4_input', transformer.prepare_input)
    for lvl in range(len(phase2_a)):
        if detach:
            with torch.no_grad():
                fa = phase4_prep(phase2_a[lvl]).detach()
                fb = phase4_prep(phase2_b[lvl]).detach()
        else:
            fa = phase4_prep(phase2_a[lvl])
            fb = phase4_prep(phase2_b[lvl])
            fa = scale_gradient(fa, backprop_scale)
            fb = scale_gradient(fb, backprop_scale)
        feats_a.append(fa)
        feats_b.append(fb)
    return feats_a, feats_b

def resolve_phase4_backprop_scale(epoch: int, total_epochs: int,
                                  phase4_start_epoch: int,
                                  enable_phase4: bool,
                                  phase4_backprop_to_phase23: bool) -> float:
    """Late-stage warm-up for Phase4 -> Phase2/3 gradients.

    Early training keeps Phase 4 detached so Phase 2/3 first learn a stable
    coarse-to-fine transform. Only the last portion of training opens this path.
    """
    if (not enable_phase4) or (not phase4_backprop_to_phase23):
        return 0.0

    late_start_epoch = max(int(phase4_start_epoch), int(total_epochs * PHASE4_BACKPROP_START_RATIO_DEFAULT))
    if epoch < late_start_epoch:
        return 0.0

    late_epochs = max(total_epochs - late_start_epoch, 1)
    progress = (epoch - late_start_epoch + 1) / late_epochs
    return float(np.clip(progress, 0.0, 1.0))

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
    phase4_backprop_scale: float = 0.0,
    accum_steps: int = ACCUM_STEPS,
):
    """
    Train one epoch of Phase 2 + Phase 3, with optional Phase 4 refinement.

    Design notes
    ------------
    - Phase 3 still provides the primary coarse-to-fine estimate.
    - Phase 4 is treated as a learned residual refiner on top of that estimate.
    - By default, Phase 4 does *not* backprop into Phase 2/3 until the late
      stage of training.
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
                    detach=((not phase4_backprop_to_phase23) or (float(phase4_backprop_scale) <= 0.0)),
                    backprop_scale=float(phase4_backprop_scale),
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

            # §5.1 Geometric Accuracy
            # Pass the full rotor tuple so the loss can supervise both magnitude
            # and local rotor orientation consistency.
            L_geo, _geo_dict = criterion.geo_loss(
                S_A, V_A, B_A_tuple,
                S_B, V_B, B_B_tuple,
                w_gt,
            )

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
            loss = total_loss / accum_steps

        # --------------------------------------------------------------
        # (5) Backprop + Gradient Accumulation
        # --------------------------------------------------------------
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
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

    if len(dataloader) % accum_steps != 0:
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
    phase4_backprop_scale: float = 0.0,
    accum_steps: int = ACCUM_STEPS,
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
                detach=((not phase4_backprop_to_phase23) or (float(phase4_backprop_scale) <= 0.0)),
                backprop_scale=float(phase4_backprop_scale),
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

        L_geo, _ = criterion.geo_loss(
            S_A, V_A, B_A_tuple,
            S_B, V_B, B_B_tuple,
            w_gt,
        )
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
    checkpoint_root=None,
    checkpoint_dir=None,
    experiment_name=None,
    criterion_alpha=1.0,
    criterion_beta=1.25,
    criterion_gamma=0.1,
    lambda_angle=None,
    lambda_rotation_inv=None,
    lambda_pixel=None,
    batch_size_override=None,
    accum_steps_override=None,
    run_epochs_override=None,
    limit_samples_override=None,
    val_interval_override=None,
    global_best_min_rot=DEFAULT_GLOBAL_BEST_MIN_ROT,
):
    """Main training entry.

    Key additions
    -------------
    - checkpoint path is configurable and experiment-scoped
    - saves `last`, `best_stage{n}`, and `best_global`
    - experiment metadata is written to `config.json`
    - ablations can override core loss weights from the CLI
    """
    # -------------------------------------------------------------------------
    # [Mode Settings]
    # -------------------------------------------------------------------------
    if debug_mode:
        print("\n" + "⚡" * 40)
        print("⚡ DEBUG / ABLATION MODE ⚡")
        print("⚡" * 40 + "\n")
        default_limit_samples = LIMIT_SAMPLE_NUM
        default_run_epochs = 30
    else:
        default_limit_samples = MAX_SAMPLES_NUM
        default_run_epochs = NUM_EPOCHS

    limit_samples = int(limit_samples_override) if limit_samples_override is not None else default_limit_samples
    run_epochs = int(run_epochs_override) if run_epochs_override is not None else default_run_epochs
    batch_size = int(batch_size_override) if batch_size_override is not None else BATCH_SIZE
    accum_steps = int(accum_steps_override) if accum_steps_override is not None else ACCUM_STEPS
    val_interval = int(val_interval_override) if val_interval_override is not None else VAL_INTERVAL

    curriculum_stages = build_curriculum_stages(debug_mode=debug_mode, run_epochs=run_epochs)
    target_rotation = curriculum_stages[-1][3]

    experiment_dir, experiment_name = resolve_checkpoint_dir(
        debug_mode=debug_mode,
        experiment_name=experiment_name,
        checkpoint_root=checkpoint_root,
        checkpoint_dir=checkpoint_dir,
        enable_phase4=enable_phase4,
        phase4_weight=phase4_weight,
        criterion_beta=criterion_beta,
        criterion_gamma=criterion_gamma,
        target_rotation=target_rotation,
    )

    # -------------------------------------------------------------------------
    # [Print Configuration]
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("🚀 Geometric Matching Model Training")
    print("=" * 80)
    print(f"   Experiment: {experiment_name}")
    print(f"   Checkpoint Dir: {experiment_dir}")
    print(f"   Target Rotation: ±{target_rotation:.0f}° (Curriculum)")
    print("   Stages:")
    for i, (s, e, sa, ea) in enumerate(curriculum_stages):
        print(f"      Stage {i+1}: Epoch {s:3d}-{e:3d} | ±{sa:.0f}° → ±{ea:.0f}°")
    print(f"   Dataset: {limit_samples if limit_samples else 'All'} images")
    print(f"   Epochs: {run_epochs}")
    print(f"   Batch Size: {batch_size} × {accum_steps} = {batch_size * accum_steps} (effective)")
    print(f"   Learning Rate: {LEARNING_RATE:.2e} (with stage decay)")
    print(f"   Warmup: {WARMUP_EPOCHS} epochs")
    print(f"   Validation Interval: every {val_interval} epoch(s)")
    print(f"   Phase4 Refiner: {'ON' if enable_phase4 else 'OFF'} | weight={phase4_weight} | start_epoch={phase4_start_epoch} | backprop_to_phase23={phase4_backprop_to_phase23}")
    print(f"   Loss weights: alpha={criterion_alpha}, beta={criterion_beta}, gamma={criterion_gamma}")
    if lambda_angle is not None or lambda_rotation_inv is not None or lambda_pixel is not None:
        print(f"   Loss overrides: L_angle={lambda_angle}, L_rot_inv={lambda_rotation_inv}, L_pixel={lambda_pixel}")
    if phase4_backprop_to_phase23:
        print(f"   Phase4 -> Phase2/3 backprop schedule: late warm-up from epoch ratio {PHASE4_BACKPROP_START_RATIO_DEFAULT:.2f}")
    print("=" * 80)

    # Device Setup
    device_info = setup_device()
    device = device_info['device']
    os.makedirs(experiment_dir, exist_ok=True)

    # ======================================================================
    # [1] Curriculum Scheduler
    # ======================================================================
    curriculum = CurriculumScheduler(curriculum_stages)

    # ======================================================================
    # [2] Datasets & Dataloaders
    # ======================================================================
    print("\n📊 Loading Dataset...")
    all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                           glob.glob(os.path.join(img_dir, "*.jpg")) +
                           glob.glob(os.path.join(img_dir, "*.jpeg")))
    if len(all_img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    if limit_samples is not None and len(all_img_paths) > limit_samples:
        all_img_paths = all_img_paths[:limit_samples]

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
        img_paths=train_paths,
    )
    val_dataset = GeometricRotationDataset(
        img_dir=img_dir, is_train=False, max_samples=None,
        rot_min=ROTATION_MIN, rot_max=ROTATION_MAX, curriculum_scheduler=curriculum,
        img_paths=val_paths,
    )

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_geometric)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_geometric)

    print(f"   Total: {total_size}, Train: {train_size}, Val: {val_size}")
    print(f"   Batch size: {batch_size}")

    # ======================================================================
    # [3] Model
    # ======================================================================
    print("\n🏗️ Building Model...")

    from pipeline.phase2 import CliffordPyramidEmbedder
    from pipeline.phase3 import Phase3Transformer
    from pipeline.phase4 import IterativeRefinementLoop
    import losses as losses_module
    from losses import UnifiedGeometricLoss

    if lambda_angle is not None:
        losses_module.LAMBDA_ANGLE = float(lambda_angle)
    if lambda_rotation_inv is not None:
        losses_module.LAMBDA_ROTATION_INV = float(lambda_rotation_inv)
    if lambda_pixel is not None:
        losses_module.LAMBDA_PIXEL = float(lambda_pixel)

    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    refiner = IterativeRefinementLoop(feature_dim=FEATURE_DIM).to(device)

    total_params = (
        sum(p.numel() for p in embedder.parameters())
        + sum(p.numel() for p in transformer.parameters())
        + sum(p.numel() for p in refiner.parameters())
    )
    print(f"   Parameters: {total_params:,}")

    # ======================================================================
    # [4] Optimizer & Scheduler & Loss
    # ======================================================================
    optimizer_params = list(embedder.parameters()) + list(transformer.parameters())
    if enable_phase4:
        optimizer_params += list(refiner.parameters())

    optimizer = optim.AdamW(
        optimizer_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = StageAwareWarmupScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=run_epochs,
        curriculum_scheduler=curriculum,
        warmup_start_lr=WARMUP_START_LR,
        min_lr=SCHEDULER_ETA_MIN,
    )
    criterion = UnifiedGeometricLoss(
        alpha=float(criterion_alpha),
        beta=float(criterion_beta),
        gamma=float(criterion_gamma),
    ).to(device)
    scaler = torch.amp.GradScaler('cuda') if device_info['type'] == 'cuda' else None

    config_payload = {
        'experiment_name': experiment_name,
        'experiment_dir': experiment_dir,
        'debug_mode': bool(debug_mode),
        'enable_phase4': bool(enable_phase4),
        'phase4_weight': float(phase4_weight),
        'phase4_start_epoch': int(phase4_start_epoch),
        'phase4_backprop_to_phase23': bool(phase4_backprop_to_phase23),
        'criterion_alpha': float(criterion_alpha),
        'criterion_beta': float(criterion_beta),
        'criterion_gamma': float(criterion_gamma),
        'lambda_angle': float(losses_module.LAMBDA_ANGLE),
        'lambda_rotation_inv': float(losses_module.LAMBDA_ROTATION_INV),
        'lambda_pixel': float(losses_module.LAMBDA_PIXEL),
        'batch_size': int(batch_size),
        'accum_steps': int(accum_steps),
        'run_epochs': int(run_epochs),
        'limit_samples': int(limit_samples),
        'val_interval': int(val_interval),
        'curriculum_stages': [list(s) for s in curriculum_stages],
        'global_best_min_rot': float(global_best_min_rot),
    }
    write_experiment_config(os.path.join(experiment_dir, 'config.json'), config_payload)

    # ======================================================================
    # [5] Resume
    # ======================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    best_stage_scores = {}
    best_global_score = float('inf')
    best_stage_paths = {}
    best_global_path = None
    history_path = os.path.join(experiment_dir, 'history.json')
    history = {
        'experiment_name': experiment_name,
        'train_loss': [], 'train_angle': [], 'train_pixel': [],
        'val_loss': [], 'val_angle': [], 'val_pixel': [],
        'stage': [], 'learning_rate': [], 'rotation_range': [],
        'stage_score': [], 'global_score': [],
    }
    if os.path.exists(history_path):
        try:
            loaded_history = json.load(open(history_path, 'r'))
            if isinstance(loaded_history, dict):
                history.update(loaded_history)
        except Exception:
            pass

    resume_path = resolve_resume_path(resume_from, experiment_dir)
    if resume_path:
        print(f"\n📥 Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
        if checkpoint.get('refiner', None) is not None:
            refiner.load_state_dict(checkpoint['refiner'])
        else:
            print("   ⚠️ Checkpoint has no 'refiner' weights. Phase4 will start from random init.")
        if checkpoint.get('optimizer', None) is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as exc:
                print(f"   ⚠️ Optimizer state was not restored cleanly: {exc}")
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        best_val_loss = float(checkpoint.get('best_val_loss', float('inf')))
        best_stage_scores = checkpoint.get('best_stage_scores', {}) or {}
        best_global_score = float(checkpoint.get('best_global_score', float('inf')))
        best_stage_paths = checkpoint.get('best_stage_paths', {}) or {}
        best_global_path = checkpoint.get('best_global_path', None)
        scheduler.current_epoch = start_epoch
        prev_epoch = max(start_epoch - 1, 0)
        scheduler.prev_stage = curriculum.get_stage_info(prev_epoch)['stage'] - 1

    # ======================================================================
    # [6] Training Loop
    # ======================================================================
    print("\n" + "=" * 70)
    print("🎯 Starting Training...")
    print("=" * 70)

    tagged_last_path = os.path.join(experiment_dir, f'{experiment_name}__last.pth')
    stable_last_path = os.path.join(experiment_dir, 'last.pth')

    for epoch in range(start_epoch, run_epochs):
        epoch_start = time.time()

        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        stage_info = curriculum.get_stage_info(epoch)
        current_rot = stage_info['rotation_range']
        current_stage = int(stage_info['stage'])

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n📌 Epoch {epoch+1}/{run_epochs} | Stage {current_stage} | Rotation: {current_rot[0]:.0f}°~{current_rot[1]:.0f}° | LR: {current_lr:.2e}")
        print("-" * 50)

        enable_phase4_now = bool(enable_phase4) and (epoch >= int(phase4_start_epoch)) and (float(phase4_weight) > 0.0)
        phase4_backprop_scale_now = resolve_phase4_backprop_scale(
            epoch=epoch,
            total_epochs=run_epochs,
            phase4_start_epoch=int(phase4_start_epoch),
            enable_phase4=enable_phase4_now,
            phase4_backprop_to_phase23=bool(phase4_backprop_to_phase23),
        )
        phase4_backprop_to_phase23_now = phase4_backprop_scale_now > 0.0

        train_metrics = train_one_epoch(
            embedder, transformer, refiner,
            train_loader, optimizer, criterion,
            device_info,
            scaler,
            current_rot,
            enable_phase4=enable_phase4_now,
            phase4_weight=float(phase4_weight),
            phase4_backprop_to_phase23=phase4_backprop_to_phase23_now,
            phase4_verbose=bool(phase4_verbose),
            phase4_backprop_scale=float(phase4_backprop_scale_now),
            accum_steps=int(accum_steps),
        )

        print(f"   [Train] Loss: {train_metrics['loss_mean']:.4f} | Angle: {train_metrics['angle_error_mean']:.2f}° ± {train_metrics['angle_error_std']:.2f}° | Pixel: {train_metrics['pixel_error_mean']:.2f}px")
        if phase4_backprop_to_phase23:
            print(f"   [Train] Phase4 -> Phase2/3 grad scale: {phase4_backprop_scale_now:.3f}")

        history['train_loss'].append(float(train_metrics['loss_mean']))
        history['train_angle'].append(float(train_metrics['angle_error_mean']))
        history['train_pixel'].append(float(train_metrics['pixel_error_mean']))
        history['learning_rate'].append(float(current_lr))
        history['rotation_range'].append(list(current_rot))
        history['stage'].append(current_stage)

        current_val_loss = None
        last_metrics_snapshot = {
            'epoch': epoch + 1,
            'stage': current_stage,
            'rotation_range': list(current_rot),
            'train': train_metrics,
            'val': None,
        }

        if (epoch + 1) % val_interval == 0 or epoch == run_epochs - 1:
            enable_phase4_now = bool(enable_phase4) and (epoch >= int(phase4_start_epoch)) and (float(phase4_weight) > 0.0)
            phase4_backprop_scale_now = resolve_phase4_backprop_scale(
                epoch=epoch,
                total_epochs=run_epochs,
                phase4_start_epoch=int(phase4_start_epoch),
                enable_phase4=enable_phase4_now,
                phase4_backprop_to_phase23=bool(phase4_backprop_to_phase23),
            )
            val_metrics = validate(
                embedder, transformer, refiner,
                val_loader, criterion, device_info,
                enable_phase4=enable_phase4_now,
                phase4_weight=float(phase4_weight),
                phase4_backprop_to_phase23=(phase4_backprop_scale_now > 0.0),
                phase4_verbose=bool(phase4_verbose),
                phase4_backprop_scale=float(phase4_backprop_scale_now),
            )
            current_val_loss = float(val_metrics['loss_mean'])
            best_val_loss = min(best_val_loss, current_val_loss)
            val_angle_mean = float(val_metrics['angle_error_mean'])
            val_pixel_mean = float(val_metrics['pixel_error_mean'])
            val_angle_std = float(val_metrics['angle_error_std'])
            stage_score = compute_stage_score(val_metrics)
            global_score = compute_global_score(val_metrics)
            is_global_candidate = abs(float(current_rot[1])) >= float(global_best_min_rot)

            print(f"   [Val]   Loss: {current_val_loss:.4f} | Angle: {val_angle_mean:.2f}° ± {val_angle_std:.2f}° | Pixel: {val_pixel_mean:.2f}px")
            print(f"   [Val]   Stage Score: {stage_score:.4f} | Global Score: {global_score:.4f} | Global Candidate: {is_global_candidate}")

            history['val_loss'].append(current_val_loss)
            history['val_angle'].append(val_angle_mean)
            history['val_pixel'].append(val_pixel_mean)
            history['stage_score'].append(stage_score)
            history['global_score'].append(global_score)
            last_metrics_snapshot['val'] = val_metrics

            rot_tag = int(round(abs(current_rot[1])))
            metric_tag = (
                f"{experiment_name}__stage{current_stage}__rot{rot_tag:02d}__ep{epoch+1:03d}"
                f"__ang{format_metric_tag(val_angle_mean)}__px{format_metric_tag(val_pixel_mean)}"
                f"__vloss{format_metric_tag(current_val_loss, digits=4)}.pth"
            )
            checkpoint_config = {
                **config_payload,
                'rotation_range': list(current_rot),
                'stage': current_stage,
                'epoch': epoch + 1,
                'lr': float(current_lr),
                'phase4_backprop_scale': float(phase4_backprop_scale_now),
            }
            checkpoint_extra = {
                'best_val_loss': current_val_loss,
                'best_stage_scores': best_stage_scores,
                'best_global_score': best_global_score,
                'best_stage_paths': best_stage_paths,
                'best_global_path': best_global_path,
            }
            checkpoint_payload = build_checkpoint_payload(
                epoch=epoch,
                embedder=embedder,
                transformer=transformer,
                refiner=refiner,
                optimizer=optimizer,
                metrics=val_metrics,
                training_config=checkpoint_config,
                extra=checkpoint_extra,
            )

            stage_key = str(current_stage)
            prev_stage_best = float(best_stage_scores.get(stage_key, float('inf')))
            if stage_score < prev_stage_best:
                best_stage_scores[stage_key] = stage_score
                stage_tagged_path = os.path.join(experiment_dir, metric_tag)
                stage_alias_path = os.path.join(experiment_dir, f'{experiment_name}__best_stage{current_stage}.pth')
                best_stage_paths[stage_key] = stage_tagged_path
                checkpoint_payload['best_stage_scores'] = best_stage_scores
                checkpoint_payload['best_stage_paths'] = best_stage_paths
                torch.save(checkpoint_payload, stage_tagged_path)
                torch.save(checkpoint_payload, stage_alias_path)
                print(f"   🌟 Stage-{current_stage} best saved -> {Path(stage_tagged_path).name}")

            if is_global_candidate and global_score < best_global_score:
                best_global_score = global_score
                best_global_tagged = os.path.join(
                    experiment_dir,
                    f"{experiment_name}__best-global__stage{current_stage}__rot{rot_tag:02d}__ep{epoch+1:03d}"
                    f"__ang{format_metric_tag(val_angle_mean)}__px{format_metric_tag(val_pixel_mean)}"
                    f"__vloss{format_metric_tag(current_val_loss, digits=4)}.pth"
                )
                best_global_alias = os.path.join(experiment_dir, f'{experiment_name}__best_global.pth')
                stable_best_global_alias = os.path.join(experiment_dir, 'best_global.pth')
                best_global_path = best_global_tagged
                checkpoint_payload['best_global_score'] = best_global_score
                checkpoint_payload['best_global_path'] = best_global_path
                torch.save(checkpoint_payload, best_global_tagged)
                torch.save(checkpoint_payload, best_global_alias)
                torch.save(checkpoint_payload, stable_best_global_alias)
                print(f"   🏆 Global best saved -> {Path(best_global_tagged).name}")

        epoch_time = time.time() - epoch_start
        print(f"   ⏱️ Time: {epoch_time:.1f}s")

        last_training_config = {
            **config_payload,
            'rotation_range': list(current_rot),
            'stage': current_stage,
            'epoch': epoch + 1,
            'lr': float(current_lr),
            'phase4_backprop_scale': float(phase4_backprop_scale_now),
        }
        last_payload = build_checkpoint_payload(
            epoch=epoch,
            embedder=embedder,
            transformer=transformer,
            refiner=refiner,
            optimizer=optimizer,
            metrics=last_metrics_snapshot,
            training_config=last_training_config,
            extra={
                'best_val_loss': float(current_val_loss) if current_val_loss is not None else best_val_loss,
                'best_stage_scores': best_stage_scores,
                'best_global_score': best_global_score,
                'best_stage_paths': best_stage_paths,
                'best_global_path': best_global_path,
            },
        )
        torch.save(last_payload, tagged_last_path)
        torch.save(last_payload, stable_last_path)
        save_json(history_path, history)

    # ======================================================================
    # [7] Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    print(f"\n💾 Last checkpoint: {stable_last_path}")
    if best_global_path is not None:
        print(f"💾 Best global checkpoint: {best_global_path}")
    if best_stage_paths:
        for stage_key in sorted(best_stage_paths.keys(), key=lambda x: int(x)):
            print(f"💾 Best stage {stage_key}: {best_stage_paths[stage_key]}")

    load_path = os.path.join(experiment_dir, 'best_global.pth')
    if not os.path.exists(load_path):
        load_path = stable_last_path

    if os.path.exists(load_path):
        best_ckpt = torch.load(load_path, map_location=device, weights_only=False)
        final_metrics = best_ckpt.get('metrics', {})
        final_config = best_ckpt.get('training_config', {})
        print(f"\n📊 Reference Model:")
        if isinstance(final_metrics, dict) and 'angle_error_mean' in final_metrics:
            print(f"   Angle Error: {final_metrics.get('angle_error_mean', 0.0):.2f}°")
            print(f"   Pixel Error: {final_metrics.get('pixel_error_mean', 0.0):.2f}px")
            print(f"   Val Loss: {final_metrics.get('loss_mean', 0.0):.4f}")
        print(f"   Trained up to: ±{final_config.get('rotation_range', ['?', '?'])[1]}")

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
                finest_res = results_sorted[0]
                pred_W3 = get_W_AB_from_phase3_result(finest_res)

                from losses import normalize_rotor_output
                cos3_raw = pred_W3[:, 0, 0]
                sin3_raw = pred_W3[:, 1, 0]
                cos3, sin3 = normalize_rotor_output(cos3_raw, sin3_raw)

                pred_angle3 = np.degrees(np.arctan2(sin3[0].item(), cos3[0].item()))

                # Phase4 refinement (checkpoint에 refiner가 없으면 랜덤 init이므로 주의)
                feats_a, feats_b = build_phase4_pyramid_features(transformer, phase2_a, phase2_b, detach=True)
                with suppress_stdout(enabled=True):
                    pred_W4_B2A, _hist = refiner(feats_a, feats_b, phase3_results=results_model, device=device)
                    
                pred_W4 = invert_affine_2x3(pred_W4_B2A)

                from losses import normalize_rotor_output
                cos4_raw = pred_W4[:, 0, 0]
                sin4_raw = pred_W4[:, 1, 0]
                cos4, sin4 = normalize_rotor_output(cos4_raw, sin4_raw)

                pred_angle4 = np.degrees(np.arctan2(sin4[0].item(), cos4[0].item()))

                gt_angle_deg = np.degrees(gt_angle_rad)

                # 기본은 Phase4(최종) 기준으로 에러 계산
                diff = pred_angle4 - gt_angle_deg
                error = abs((diff + 180.0) % 360.0 - 180.0)
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

    parser = argparse.ArgumentParser(description='Geometric Matching Training')
    parser.add_argument('--img_dir', type=str, default='./val2017',
                        help='Image directory path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path or experiment directory')
    parser.add_argument('--debug', action='store_true',
                        help='Debug / ablation mode with reduced default sample count')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint for testing')

    # Experiment / checkpoint routing
    parser.add_argument('--checkpoint_root', type=str, default=None,
                        help='Root directory under which <experiment_name>/ will be created')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Full experiment directory to save into (overrides checkpoint_root)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment/run name used in folder and checkpoint aliases')
    parser.add_argument('--global_best_min_rot', type=float, default=DEFAULT_GLOBAL_BEST_MIN_ROT,
                        help='Only stages whose max rotation is at least this value compete for best_global')

    # Phase4 refiner
    parser.add_argument('--disable_phase4', action='store_true',
                        help='Disable Phase4 refiner training (train Phase2+3 only)')
    parser.add_argument('--phase4_weight', type=float, default=PHASE4_LOSS_WEIGHT_DEFAULT,
                        help='Loss weight for Phase4 final consistency term')
    parser.add_argument('--phase4_start_epoch', type=int, default=PHASE4_START_EPOCH_DEFAULT,
                        help='Epoch to start Phase4 training (0-indexed)')
    parser.add_argument('--phase4_backprop_to_phase23', action='store_true',
                        help='Enable late-stage warm-up backprop from Phase4 loss into Phase2/3')
    parser.add_argument('--phase4_verbose', action='store_true',
                        help='Enable Phase4 internal verbose logs (prints)')

    # Loss / ablation overrides
    parser.add_argument('--criterion_alpha', type=float, default=1.0,
                        help='UnifiedGeometricLoss alpha weight')
    parser.add_argument('--criterion_beta', type=float, default=1.25,
                        help='UnifiedGeometricLoss beta weight')
    parser.add_argument('--criterion_gamma', type=float, default=0.1,
                        help='UnifiedGeometricLoss gamma weight')
    parser.add_argument('--lambda_angle', type=float, default=None,
                        help='Override losses.LAMBDA_ANGLE')
    parser.add_argument('--lambda_rotation_inv', type=float, default=None,
                        help='Override losses.LAMBDA_ROTATION_INV')
    parser.add_argument('--lambda_pixel', type=float, default=None,
                        help='Override losses.LAMBDA_PIXEL')

    # Runtime overrides for short ablations
    parser.add_argument('--batch_size_override', type=int, default=None,
                        help='Override dataloader batch size')
    parser.add_argument('--accum_steps_override', type=int, default=None,
                        help='Override gradient accumulation steps')
    parser.add_argument('--run_epochs_override', type=int, default=None,
                        help='Override total number of epochs')
    parser.add_argument('--limit_samples_override', type=int, default=None,
                        help='Override maximum number of samples used from the dataset')
    parser.add_argument('--val_interval_override', type=int, default=None,
                        help='Override validation interval in epochs')

    args = parser.parse_args()

    enable_phase4 = (not args.disable_phase4)

    if args.test:
        quick_test(args.img_dir, args.checkpoint)
    else:
        train(
            img_dir=args.img_dir,
            resume_from=args.resume,
            debug_mode=args.debug,
            enable_phase4=enable_phase4,
            phase4_weight=args.phase4_weight,
            phase4_start_epoch=args.phase4_start_epoch,
            phase4_backprop_to_phase23=args.phase4_backprop_to_phase23,
            phase4_verbose=args.phase4_verbose,
            checkpoint_root=args.checkpoint_root,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment_name,
            criterion_alpha=args.criterion_alpha,
            criterion_beta=args.criterion_beta,
            criterion_gamma=args.criterion_gamma,
            lambda_angle=args.lambda_angle,
            lambda_rotation_inv=args.lambda_rotation_inv,
            lambda_pixel=args.lambda_pixel,
            batch_size_override=args.batch_size_override,
            accum_steps_override=args.accum_steps_override,
            run_epochs_override=args.run_epochs_override,
            limit_samples_override=args.limit_samples_override,
            val_interval_override=args.val_interval_override,
            global_best_min_rot=args.global_best_min_rot,
        )
