"""
================================================================================
Train Script: Geometric Matching Model (v5 - RTX 3090 Optimized + ±60° Curriculum)
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


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# [Hyperparameters] Training Configuration - RTX 3090 24GB Optimized
# =============================================================================
IMG_SIZE = (256, 256)            # [Hyperparameter] 입력 이미지 크기
MAX_SAMPLES_NUM = 2000               # [Hyperparameter] 최대 학습 샘플 수 (5000장)
LIMIT_SAMPLE_NUM = 300           # [Hyperparameter] 디버그 모드 샘플 수

# [v5 수정] ±60도 회전 범위 (커리큘럼으로 점진 증가)
ROTATION_MIN = -60.0             # [Hyperparameter] 최종 목표 회전 최소값
ROTATION_MAX = 60.0              # [Hyperparameter] 최종 목표 회전 최대값

# [v5 수정] RTX 3090 24GB 최적화 배치 설정
BATCH_SIZE = 8                   # [Hyperparameter] RTX 3090: 2 → 8 (4배 증가)
ACCUM_STEPS = 4                  # [Hyperparameter] Gradient Accumulation (effective batch = 32)
NUM_EPOCHS = 450                 # [Hyperparameter] 총 에폭 수 (±60도 학습에 충분)
LEARNING_RATE = 3e-4             # [Hyperparameter] 초기 학습률 (더 안정적)
WEIGHT_DECAY = 5e-5              # [Hyperparameter] Weight Decay (약간 감소)

# [모델 차원] - 변경 없음
HIDDEN_DIM = 48                  # [Hyperparameter] Phase 2 임베딩 차원
FEATURE_DIM = 144                # [Hyperparameter] Phase 3 Transformer 차원

# [검증 설정]
VAL_SPLIT = 0.1                  # [Hyperparameter] 검증 데이터 비율 (10%)
VAL_INTERVAL = 3                 # [Hyperparameter] 검증 주기 (3 에폭마다)

CHECKPOINT_DIR = "./checkpoints"

# [v5 신규] 3단계 커리큘럼 설정
CURRICULUM_STAGES = [
    # (시작 에폭, 종료 에폭, 시작 각도, 종료 각도)
    (0, 100, 5.0, 20.0),         # Stage 1: 기초 학습 ±15° → ±30°
    (100, 300, 20.0, 45.0),       # Stage 2: 중급 학습 ±30° → ±50°
    (300, 450, 45.0, 60.0),      # Stage 3: 고급 학습 ±50° → ±60°
]

# [v5 신규] Stage별 학습률 조정
STAGE_LR_MULTIPLIERS = [1.0, 0.8, 0.4]  # [Hyperparameter] 각 스테이지별 LR 배율

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
                 rot_min=-60.0, rot_max=60.0, curriculum_scheduler=None):
        """
        Args:
            img_dir: 이미지 디렉토리 경로
            is_train: 학습 모드 여부
            max_samples: 최대 샘플 수
            rot_min, rot_max: 회전 범위 (커리큘럼으로 동적 변경)
            curriculum_scheduler: CurriculumScheduler 인스턴스
        """
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        if not self.img_paths:
            self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        
        if max_samples is not None:
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
            # 각도 오차 (degree)
            angle_diff = torch.abs(pred_angle - gt_angle) * 180 / np.pi
            self.angle_errors.extend(angle_diff.cpu().numpy().tolist())
            
            # 픽셀 오차 (네 모서리 기준)
            B = pred_W.shape[0]
            corners = torch.tensor([
                [-1., -1., 1.], [1., -1., 1.], 
                [1., 1., 1.], [-1., 1., 1.]
            ], device=pred_W.device).T.unsqueeze(0).repeat(B, 1, 1)
            
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
# [Training Loop]
# =============================================================================

def train_one_epoch(embedder, transformer, dataloader, optimizer, criterion, 
                    device_info, scaler=None, current_rotation_range=None):
    """
    [Training] 한 에폭 학습
    
    Args:
        embedder: Phase 2 모델
        transformer: Phase 3 모델
        dataloader: 학습 데이터 로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device_info: 장치 정보
        scaler: AMP GradScaler
        current_rotation_range: 현재 회전 범위 (로깅용)
    """
    device = device_info['device']
    is_tpu = device_info['type'] == 'tpu'
    
    embedder.train()
    transformer.train()
    
    metric_tracker = MetricTracker()
    optimizer.zero_grad()
    
    rot_str = f"[{current_rotation_range[0]:.0f}°~{current_rotation_range[1]:.0f}°]" if current_rotation_range else ""
    pbar = tqdm(dataloader, desc=f"Training {rot_str}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)
        
        # [v5] Mixed Precision 최적화
        use_amp = (device_info['type'] == 'cuda') and (scaler is not None)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward Pass
            phase2_a = embedder(pyramid_a_raw, device)
            phase2_b = embedder(pyramid_b_raw, device)
            results = transformer(phase2_a, phase2_b)
            
            # Dense Rotor → Global Transform
            finest_res = results[0]
            dense_rotor = finest_res['rotor_map']
            avg_rotor = dense_rotor.mean(dim=(1, 2))
            
            cos_raw = avg_rotor[:, 0]
            sin_raw = avg_rotor[:, 1]
            dx = avg_rotor[:, 2]
            dy = avg_rotor[:, 3]
            
            # Rotor 정규화
            from losses import normalize_rotor_output
            cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
            
            # 변환 행렬 구성
            row1 = torch.stack([cos_t, -sin_t, dx], dim=1)
            row2 = torch.stack([sin_t, cos_t, dy], dim=1)
            pred_W = torch.stack([row1, row2], dim=1)
            
            # 손실 계산
            loss, loss_dict = criterion(
                pred_W, w_gt, cos_t, sin_t, gt_angle,
                phase2_a[0], phase2_b[0]
            )
            loss = loss / ACCUM_STEPS
        
        # Backward Pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient Accumulation Step
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(embedder.parameters()) + list(transformer.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(embedder.parameters()) + list(transformer.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
            
            optimizer.zero_grad()
            
            if is_tpu:
                device_info['xm'].mark_step()
        
        # 지표 업데이트
        pred_angle = torch.atan2(sin_t, cos_t)
        metric_tracker.update(pred_W, w_gt, pred_angle, gt_angle, 
                            loss.item() * ACCUM_STEPS)
        
        if batch_idx % LOG_INTERVAL == 0:
            metrics = metric_tracker.get_metrics()
            pbar.set_postfix({
                'loss': f"{metrics['loss_mean']:.4f}",
                'angle': f"{metrics['angle_error_mean']:.2f}°"
            })
    
    return metric_tracker.get_metrics()


@torch.no_grad()
def validate(embedder, transformer, dataloader, criterion, device_info):
    """
    [Validation] 검증 수행
    """
    device = device_info['device']
    
    embedder.eval()
    transformer.eval()
    
    metric_tracker = MetricTracker()
    
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)
        
        phase2_a = embedder(pyramid_a_raw, device)
        phase2_b = embedder(pyramid_b_raw, device)
        results = transformer(phase2_a, phase2_b)
        
        finest_res = results[0]
        dense_rotor = finest_res['rotor_map']
        avg_rotor = dense_rotor.mean(dim=(1, 2))
        
        cos_raw, sin_raw = avg_rotor[:, 0], avg_rotor[:, 1]
        dx, dy = avg_rotor[:, 2], avg_rotor[:, 3]
        
        from losses import normalize_rotor_output
        cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
        
        row1 = torch.stack([cos_t, -sin_t, dx], dim=1)
        row2 = torch.stack([sin_t, cos_t, dy], dim=1)
        pred_W = torch.stack([row1, row2], dim=1)
        
        loss, _ = criterion(pred_W, w_gt, cos_t, sin_t, gt_angle,
                            phase2_a[0], phase2_b[0])
        
        pred_angle = torch.atan2(sin_t, cos_t)
        metric_tracker.update(pred_W, w_gt, pred_angle, gt_angle, loss.item())
    
    return metric_tracker.get_metrics()


# =============================================================================
# [Main Training Function]
# =============================================================================

def train(img_dir, resume_from=None, debug_mode=False):
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
        save_name_prefix = "v5_60deg_"
    
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
    
    full_dataset = GeometricRotationDataset(
        img_dir, is_train=True, max_samples=limit_samples,
        rot_min=ROTATION_MIN, rot_max=ROTATION_MAX,
        curriculum_scheduler=curriculum
    )
    
    total_size = len(full_dataset)
    val_size = max(int(total_size * VAL_SPLIT), 1)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # [v5] num_workers 증가 (RTX 3090에서 CPU 병목 방지)
    num_workers = min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_geometric, num_workers=num_workers, 
        drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn_geometric, num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"   Train: {train_size}, Val: {val_size}")
    print(f"   Workers: {num_workers}")
    
    # ==========================================================================
    # [3] Model
    # ==========================================================================
    print("\n🏗️ Building Model...")
    
    from pipeline.phase2 import CliffordPyramidEmbedder
    from pipeline.phase3 import Phase3Transformer
    from losses import UnifiedGeometricLoss
    
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    total_params = sum(p.numel() for p in embedder.parameters()) + \
                   sum(p.numel() for p in transformer.parameters())
    print(f"   Parameters: {total_params:,}")

    start_epoch = 0
    best_val_loss = float('inf')  # Best 점수 추적을 위한 변수
    
    # ==========================================================================
    # [4] Optimizer & Scheduler
    # ==========================================================================
    optimizer = optim.AdamW(
        list(embedder.parameters()) + list(transformer.parameters()),
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
        full_dataset.set_epoch(epoch)
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
        train_metrics = train_one_epoch(
            embedder, transformer, train_loader, optimizer, criterion,
            device_info, scaler, current_rot
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
            val_metrics = validate(embedder, transformer, val_loader, criterion, device_info)
            
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
                torch.save({
                    'epoch': epoch,
                    'embedder': embedder.state_dict(),
                    'transformer': transformer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'metrics': val_metrics,
                    'training_config': {
                        'version': 'v5_best',
                        'rotation_range': current_rot
                    }
                }, os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}best_model.pth'))
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
    
    best_model_path = os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}best_model.pth')
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, weights_only=False)
        final_metrics = best_ckpt.get('metrics', {})
        final_config = best_ckpt.get('training_config', {})
        
        print(f"\n📊 Best Model:")
        print(f"   Val Loss: {best_val_loss:.4f}")
        print(f"   Angle Error: {final_metrics.get('angle_error_mean', 'N/A'):.2f}°")
        print(f"   Pixel Error: {final_metrics.get('pixel_error_mean', 'N/A'):.2f}px")
        print(f"   Trained up to: ±{final_config.get('current_rotation', ['?', '?'])[1]}°")
    
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
    
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        
        config = ckpt.get('training_config', {})
        print(f"   Loaded: {checkpoint_path}")
        print(f"   Trained rotation: {config.get('rotation_range', 'Unknown')}")
    
    embedder.eval()
    transformer.eval()
    
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
                
                rotor = results_model[0]['rotor_map']
                avg_rotor = rotor.mean(dim=(1, 2))
                
                pred_angle = np.degrees(np.arctan2(avg_rotor[0, 1].item(), avg_rotor[0, 0].item()))
                gt_angle_deg = np.degrees(gt_angle_rad)
                
                error = abs(pred_angle - gt_angle_deg)
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
    
    args = parser.parse_args()
    
    if args.test:
        quick_test(args.img_dir, args.checkpoint)
    else:
        train(args.img_dir, args.resume, args.debug)