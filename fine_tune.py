"""
================================================================================
Train Script: Geometric Matching Model (v4 - Warmup & Advanced Fine-tuning)
================================================================================
Fine-tuning 시 Loss가 안 줄어드는 문제 해결:
1. Linear Warmup: 초반 N 에폭은 LR을 0에서 점진적으로 증가
2. Warmup + Cosine Annealing 결합
3. Layer-wise LR Decay (선택): 앞쪽 레이어는 낮은 LR
4. Patience 자동 조정: Fine-tuning 모드에서는 더 길게 기다림
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

# =============================================================================
# [Hyperparameters] Training Configuration
# =============================================================================
IMG_SIZE = (256, 256)
MAX_SAMPLES = None
LIMIT_SAMPLE_NUM = 100

ROTATION_MIN = -20.0
ROTATION_MAX = 20.0

BATCH_SIZE = 2
ACCUM_STEPS = 16
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 48
FEATURE_DIM = 144

VAL_SPLIT = 0.1
VAL_INTERVAL = 2

# [Early Stopping] - Fine-tuning 모드에서는 자동으로 늘어남
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.001

CHECKPOINT_DIR = "./checkpoints"

# [v4 신규] Warmup 설정
WARMUP_EPOCHS = 5                # [Hyperparameter] Warmup 에폭 수
WARMUP_START_LR = 1e-6           # [Hyperparameter] Warmup 시작 LR (거의 0)
SCHEDULER_ETA_MIN = 1e-6         # [Hyperparameter] 최소 LR

LOG_INTERVAL = 10


# =============================================================================
# [v4 신규] Custom Learning Rate Scheduler with Warmup
# =============================================================================

class WarmupCosineScheduler:
    """
    [v4] Warmup + Cosine Annealing 결합 스케줄러
    
    Phase 1 (Warmup): LR을 warmup_start_lr → target_lr로 선형 증가
    Phase 2 (Cosine): LR을 target_lr → min_lr로 코사인 감소
    
    Fine-tuning 시 모델이 새로운 분포에 적응할 시간을 줌
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 warmup_start_lr=1e-6, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        
        # 원래 LR 저장 (target)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            if self.current_epoch <= self.warmup_epochs:
                # Phase 1: Linear Warmup
                # LR = warmup_start + (target - warmup_start) * (epoch / warmup_epochs)
                progress = self.current_epoch / self.warmup_epochs
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
            else:
                # Phase 2: Cosine Annealing
                # 남은 에폭에서 cosine 감소
                cosine_epochs = self.total_epochs - self.warmup_epochs
                cosine_progress = (self.current_epoch - self.warmup_epochs) / cosine_epochs
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cosine_progress))
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# =============================================================================
# [Device Setup]
# =============================================================================

def setup_device():
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
        print(f"✅ CUDA Detected - {torch.cuda.get_device_name(0)}")
        return device_info
    
    print("⚠️ No GPU/TPU detected - Using CPU")
    return device_info


# =============================================================================
# [Dataset]
# =============================================================================

class GeometricRotationDataset(Dataset):
    def __init__(self, img_dir, is_train=True, max_samples=None,
                 rot_min=-20.0, rot_max=20.0, curriculum_mode=False):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        if not self.img_paths:
            self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
            
        self.is_train = is_train
        self.preprocessor = None
        
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.base_rot_min = rot_min
        self.base_rot_max = rot_max
        
        self.curriculum_mode = curriculum_mode
        self.current_epoch = 0
        self.total_epochs = 1
        
        print(f"📁 Dataset: {len(self.img_paths)} images")
        print(f"   Rotation: {rot_min}° ~ {rot_max}°")
        if curriculum_mode:
            print(f"   📈 Curriculum Mode: ON")
    
    def set_epoch(self, epoch, total_epochs):
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        if self.curriculum_mode:
            progress = min(epoch / max(total_epochs * 0.7, 1), 1.0)
            start_ratio = 0.3
            current_ratio = start_ratio + (1.0 - start_ratio) * progress
            self.rot_min = self.base_rot_min * current_ratio
            self.rot_max = self.base_rot_max * current_ratio
    
    def _get_preprocessor(self):
        if self.preprocessor is None:
            from phase1 import MathGeometricPreprocessor
            self.preprocessor = MathGeometricPreprocessor()
        return self.preprocessor
    
    def __len__(self):
        return len(self.img_paths)
    
    def normalize_affine_matrix(self, matrix_pixel, width, height):
        N = np.array([[2.0 / width, 0, -1], 
                      [0, 2.0 / height, -1], 
                      [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_pix_aug = np.vstack([matrix_pixel, [0, 0, 1]])
        M_norm_aug = N @ M_pix_aug @ N_inv
        return M_norm_aug[:2, :]
    
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        
        if img_bgr is None:
            return self.__getitem__((idx + 1) % len(self))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        rows, cols = img_rgb.shape[:2]
        
        angle = np.random.uniform(self.rot_min, self.rot_max)
        
        scale = 1.0
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :]
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)
        gt_angle_rad = np.deg2rad(-angle)
        
        preprocessor = self._get_preprocessor()
        pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
        pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
        
        return {
            'pyramid_a': pyramid_a,
            'pyramid_b': pyramid_b,
            'w_gt': W_gt_mat_norm.astype(np.float32),
            'gt_angle': np.float32(gt_angle_rad),
            'img_a': img_warped,
            'img_b': img_rgb
        }


def collate_fn_geometric(batch):
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
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.angle_errors = []
        self.pixel_errors = []
        self.losses = []
        
    def update(self, pred_W, gt_W, pred_angle, gt_angle, loss):
        with torch.no_grad():
            angle_diff = torch.abs(pred_angle - gt_angle) * 180 / np.pi
            self.angle_errors.extend(angle_diff.cpu().numpy().tolist())
            
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
                    device_info, scaler=None):
    device = device_info['device']
    is_tpu = device_info['type'] == 'tpu'
    
    embedder.train()
    transformer.train()
    
    metric_tracker = MetricTracker()
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)
        
        use_amp = (device_info['type'] == 'cuda') and (scaler is not None)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            phase2_a = embedder(pyramid_a_raw, device)
            phase2_b = embedder(pyramid_b_raw, device)
            results = transformer(phase2_a, phase2_b)
            
            finest_res = results[0]
            dense_rotor = finest_res['rotor_map']
            avg_rotor = dense_rotor.mean(dim=(1, 2))
            
            cos_raw = avg_rotor[:, 0]
            sin_raw = avg_rotor[:, 1]
            dx = avg_rotor[:, 2]
            dy = avg_rotor[:, 3]
            
            from losses import normalize_rotor_output
            cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
            
            row1 = torch.stack([cos_t, -sin_t, dx], dim=1)
            row2 = torch.stack([sin_t, cos_t, dy], dim=1)
            pred_W = torch.stack([row1, row2], dim=1)
            
            loss, loss_dict = criterion(
                pred_W, w_gt, cos_t, sin_t, gt_angle,
                phase2_a[0], phase2_b[0]
            )
            loss = loss / ACCUM_STEPS
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
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
# [Main Training Function] - v4: Warmup 지원
# =============================================================================

def train(img_dir, resume_from=None, debug_mode=False, finetune_from=None,
          lr_override=None, loss_alpha=None, loss_beta=None, loss_gamma=None,
          rot_min=None, rot_max=None, curriculum_mode=False,
          warmup_epochs=None, no_warmup=False):
    """
    [Main] 학습 메인 함수
    
    Args:
        ... (기존과 동일)
        warmup_epochs: [v4] Warmup 에폭 수 오버라이드
        no_warmup: [v4] True면 Warmup 비활성화
    """
    
    # -------------------------------------------------------------------------
    # [Mode Settings]
    # -------------------------------------------------------------------------
    if debug_mode:
        print("\n" + "⚡" * 40)
        print("⚡ FAST DEBUG MODE ACTIVATED ⚡")
        print("⚡" * 40 + "\n")
        
        limit_samples = LIMIT_SAMPLE_NUM
        run_epochs = NUM_EPOCHS
        run_val_interval = 1
        save_name_prefix = "debug_"
    else:
        limit_samples = MAX_SAMPLES
        run_epochs = NUM_EPOCHS
        run_val_interval = VAL_INTERVAL
        save_name_prefix = ""
    
    # -------------------------------------------------------------------------
    # [v4] Fine-tuning 모드 자동 설정
    # -------------------------------------------------------------------------
    is_finetune = finetune_from is not None
    actual_patience = EARLY_STOP_PATIENCE
    actual_warmup = warmup_epochs if warmup_epochs is not None else WARMUP_EPOCHS
    
    if is_finetune:
        save_name_prefix = "ft_" + save_name_prefix
        
        # [v4] Fine-tuning 시 자동 조정
        if not no_warmup:
            actual_warmup = max(actual_warmup, 5)  # 최소 5 에폭 warmup
        actual_patience = max(actual_patience, 8)  # 더 오래 기다림
        
        print("\n" + "🔧" * 40)
        print("🔧 FINE-TUNING MODE (v4) 🔧")
        print(f"🔧 Loading: {finetune_from}")
        print(f"🔧 Auto-adjusted: Warmup={actual_warmup}, Patience={actual_patience}")
        print("🔧" * 40 + "\n")
    
    if no_warmup:
        actual_warmup = 0
        print("⚠️ Warmup DISABLED")
    
    # -------------------------------------------------------------------------
    # Hyperparameter Settings
    # -------------------------------------------------------------------------
    actual_rot_min = rot_min if rot_min is not None else ROTATION_MIN
    actual_rot_max = rot_max if rot_max is not None else ROTATION_MAX
    actual_lr = lr_override if lr_override is not None else LEARNING_RATE
    actual_alpha = loss_alpha if loss_alpha is not None else 1.0
    actual_beta = loss_beta if loss_beta is not None else 1.0
    actual_gamma = loss_gamma if loss_gamma is not None else 0.1
    
    # [v4] Fine-tuning 시 LR 자동 조정 (지정 안 했을 때)
    if is_finetune and lr_override is None:
        actual_lr = LEARNING_RATE * 0.2  # 기본 LR의 1/5
        print(f"📉 Auto LR for fine-tuning: {actual_lr:.2e} (1/5 of default)")
    
    rot_tag = f"rot{int(abs(actual_rot_max))}_"
    save_name_prefix = rot_tag + save_name_prefix
    
    print("=" * 70)
    print("🚀 Geometric Matching Model Training (v4 - Warmup)")
    print(f"   Rotation Range: {actual_rot_min}° ~ {actual_rot_max}°")
    print(f"   Curriculum: {'ON 📈' if curriculum_mode else 'OFF'}")
    print(f"   Warmup: {actual_warmup} epochs" + (" [Fine-tune Auto]" if is_finetune and warmup_epochs is None else ""))
    print(f"   Epochs: {run_epochs}, Batch: {BATCH_SIZE}")
    print(f"   Learning Rate: {actual_lr:.2e}")
    print(f"   Early Stop Patience: {actual_patience}")
    print(f"   Save Prefix: {save_name_prefix}")
    print("=" * 70)
    
    # Device Setup
    device_info = setup_device()
    device = device_info['device']
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # ==========================================================================
    # [1] Dataset & DataLoader
    # ==========================================================================
    print("\n📊 Loading Dataset...")
    
    full_dataset = GeometricRotationDataset(
        img_dir, is_train=True, max_samples=limit_samples,
        rot_min=actual_rot_min, rot_max=actual_rot_max,
        curriculum_mode=curriculum_mode
    )
    
    total_size = len(full_dataset)
    val_size = max(int(total_size * VAL_SPLIT), 1)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_geometric, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn_geometric, num_workers=0
    )
    
    print(f"   Train: {train_size}, Val: {val_size}")
    
    # ==========================================================================
    # [2] Model
    # ==========================================================================
    print("\n🏗️ Building Model...")
    
    from phase2 import CliffordPyramidEmbedder
    from phase3 import Phase3Transformer
    from losses import UnifiedGeometricLoss
    
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    # Fine-tuning: 모델 가중치만 로드
    if is_finetune and os.path.exists(finetune_from):
        print(f"   📥 Loading weights from: {finetune_from}")
        checkpoint = torch.load(finetune_from, map_location=device)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
        
        # 이전 성능 출력
        if 'metrics' in checkpoint:
            prev = checkpoint['metrics']
            prev_config = checkpoint.get('training_config', {})
            prev_range = prev_config.get('rotation_range', ['?', '?'])
            print(f"   📊 Previous: Loss={checkpoint.get('best_val_loss', 0):.4f}, "
                  f"Angle={prev.get('angle_error_mean', 0):.2f}°, "
                  f"Range={prev_range[0]}°~{prev_range[1]}°")
        print(f"   ✅ Weights loaded! Optimizer will start fresh with warmup.")
    
    total_params = sum(p.numel() for p in embedder.parameters()) + \
                   sum(p.numel() for p in transformer.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # ==========================================================================
    # [3] Optimizer & Scheduler (v4: Warmup)
    # ==========================================================================
    optimizer = optim.AdamW(
        list(embedder.parameters()) + list(transformer.parameters()),
        lr=actual_lr,
        weight_decay=WEIGHT_DECAY
    )
    
    # [v4] Warmup + Cosine Annealing 스케줄러
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=actual_warmup,
        total_epochs=run_epochs,
        warmup_start_lr=WARMUP_START_LR,
        min_lr=SCHEDULER_ETA_MIN
    )
    
    criterion = UnifiedGeometricLoss(
        alpha=actual_alpha, beta=actual_beta, gamma=actual_gamma
    ).to(device)
    
    scaler = torch.amp.GradScaler('cuda') if device_info['type'] == 'cuda' else None
    
    # ==========================================================================
    # [4] Resume (기존 방식)
    # ==========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n📥 Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resumed from epoch {start_epoch}")
    
    # ==========================================================================
    # [5] Training Loop
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🎯 Starting Training...")
    print("=" * 70)
    
    history = {
        'train_loss': [], 'train_angle': [],
        'val_loss': [], 'val_angle': [],
        'learning_rate': []
    }
    
    for epoch in range(start_epoch, run_epochs):
        epoch_start = time.time()
        
        # Curriculum Learning
        if curriculum_mode:
            full_dataset.set_epoch(epoch, run_epochs)
            current_range = f"{full_dataset.rot_min:.1f}°~{full_dataset.rot_max:.1f}°"
        else:
            current_range = f"{actual_rot_min}°~{actual_rot_max}°"
        
        # [v4] 현재 LR 및 Phase 표시
        current_lr = optimizer.param_groups[0]['lr']
        phase_str = "🌡️ WARMUP" if epoch < actual_warmup else "📉 Cosine"
        
        print(f"\n📌 Epoch {epoch+1}/{run_epochs} | {phase_str} | LR: {current_lr:.2e} | Range: {current_range}")
        print("-" * 50)
        
        # Train
        train_metrics = train_one_epoch(
            embedder, transformer, train_loader, optimizer, criterion,
            device_info, scaler
        )
        
        # [v4] Scheduler Step
        scheduler.step()
        
        print(f"   [Train] Loss: {train_metrics['loss_mean']:.4f} | "
              f"Angle: {train_metrics['angle_error_mean']:.2f}° | "
              f"Pixel: {train_metrics['pixel_error_mean']:.2f}px")
        
        history['train_loss'].append(train_metrics['loss_mean'])
        history['train_angle'].append(train_metrics['angle_error_mean'])
        history['learning_rate'].append(current_lr)
        
        # Validation
        if (epoch + 1) % run_val_interval == 0 or epoch == run_epochs - 1:
            val_metrics = validate(embedder, transformer, val_loader, criterion, device_info)
            
            print(f"   [Val]   Loss: {val_metrics['loss_mean']:.4f} | "
                  f"Angle: {val_metrics['angle_error_mean']:.2f}° | "
                  f"Pixel: {val_metrics['pixel_error_mean']:.2f}px")
            
            history['val_loss'].append(val_metrics['loss_mean'])
            history['val_angle'].append(val_metrics['angle_error_mean'])
            
            # [v4] Warmup 기간에는 Best Model 갱신 안 함 (불안정하므로)
            if epoch >= actual_warmup:
                if val_metrics['loss_mean'] < best_val_loss - EARLY_STOP_MIN_DELTA:
                    best_val_loss = val_metrics['loss_mean']
                    patience_counter = 0
                    
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'embedder': embedder.state_dict(),
                        'transformer': transformer.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'hidden_dim': HIDDEN_DIM,
                        'feature_dim': FEATURE_DIM,
                        'metrics': val_metrics,
                        'training_config': {
                            'lr': actual_lr,
                            'warmup_epochs': actual_warmup,
                            'loss_alpha': actual_alpha,
                            'loss_beta': actual_beta,
                            'loss_gamma': actual_gamma,
                            'rotation_range': [actual_rot_min, actual_rot_max],
                            'curriculum_mode': curriculum_mode,
                            'finetune_from': finetune_from
                        }
                    }, checkpoint_path)
                    print(f"   ✅ New Best! Saved to {os.path.basename(checkpoint_path)}")
                else:
                    patience_counter += 1
                    print(f"   ⏳ No improvement ({patience_counter}/{actual_patience})")
                
                if patience_counter >= actual_patience:
                    print(f"\n🛑 Early Stopping at epoch {epoch+1}")
                    break
            else:
                print(f"   🌡️ Warmup phase - skipping best model check")
        
        epoch_time = time.time() - epoch_start
        print(f"   ⏱️ Time: {epoch_time:.1f}s")
        
        # Last model backup
        torch.save({
            'epoch': epoch,
            'embedder': embedder.state_dict(),
            'transformer': transformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'hidden_dim': HIDDEN_DIM,
            'feature_dim': FEATURE_DIM,
            'training_config': {
                'rotation_range': [actual_rot_min, actual_rot_max]
            }
        }, os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}last_model.pth'))
    
    # ==========================================================================
    # [6] Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    
    best_model_path = os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}best_model.pth')
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path)
        final_metrics = best_ckpt.get('metrics', {})
        print(f"\n📊 Best Model:")
        print(f"   Val Loss: {best_val_loss:.4f}")
        print(f"   Angle Error: {final_metrics.get('angle_error_mean', 'N/A'):.2f}°")
        print(f"   Pixel Error: {final_metrics.get('pixel_error_mean', 'N/A'):.2f}px")
    
    with open(os.path.join(CHECKPOINT_DIR, f'{save_name_prefix}history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


# =============================================================================
# [Quick Test]
# =============================================================================

def quick_test(img_dir, checkpoint_path=None, test_rot_max=20.0):
    print(f"\n🧪 Quick Test (±{test_rot_max}°)")
    
    device_info = setup_device()
    device = device_info['device']
    
    from phase2 import CliffordPyramidEmbedder
    from phase3 import Phase3Transformer
    
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        
        config = ckpt.get('training_config', {})
        train_range = config.get('rotation_range', ['?', '?'])
        print(f"   Loaded: {checkpoint_path}")
        print(f"   Trained on: {train_range[0]}° ~ {train_range[1]}°")
    
    embedder.eval()
    transformer.eval()
    
    test_dataset = GeometricRotationDataset(
        img_dir, max_samples=10,
        rot_min=-test_rot_max, rot_max=test_rot_max
    )
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_geometric)
    
    errors = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            pyramid_a = batch['pyramid_a']
            pyramid_b = batch['pyramid_b']
            gt_angle = batch['gt_angle'].item()
            
            phase2_a = embedder(pyramid_a, device)
            phase2_b = embedder(pyramid_b, device)
            results = transformer(phase2_a, phase2_b)
            
            rotor = results[0]['rotor_map']
            avg_rotor = rotor.mean(dim=(1, 2))
            
            pred_angle = np.degrees(np.arctan2(avg_rotor[0, 1].item(), avg_rotor[0, 0].item()))
            gt_angle_deg = np.degrees(gt_angle)
            
            error = abs(pred_angle - gt_angle_deg)
            errors.append(error)
            print(f"   {i+1:2d}: GT={gt_angle_deg:+6.1f}° | Pred={pred_angle:+6.1f}° | Err={error:5.2f}°")
    
    print(f"\n   📊 Mean Error: {np.mean(errors):.2f}° ± {np.std(errors):.2f}°")


# =============================================================================
# [Entry Point]
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Geometric Matching Training v4')
    parser.add_argument('--img_dir', type=str, default='./val2017')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    
    # Fine-tuning
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    # Loss weights
    parser.add_argument('--loss_alpha', type=float, default=None)
    parser.add_argument('--loss_beta', type=float, default=None)
    parser.add_argument('--loss_gamma', type=float, default=None)
    
    # Rotation
    parser.add_argument('--rot_min', type=float, default=None)
    parser.add_argument('--rot_max', type=float, default=None)
    parser.add_argument('--curriculum', action='store_true')
    
    # [v4] Warmup
    parser.add_argument('--warmup', type=int, default=None,
                        help='Warmup epochs (default: 3, auto-increased for fine-tuning)')
    parser.add_argument('--no_warmup', action='store_true',
                        help='Disable warmup completely')
    
    # Test
    parser.add_argument('--test_rot', type=float, default=20.0)
    
    args = parser.parse_args()
    
    if args.test:
        quick_test(args.img_dir, args.checkpoint, args.test_rot)
    else:
        train(
            img_dir=args.img_dir, 
            resume_from=args.resume, 
            debug_mode=args.debug,
            finetune_from=args.finetune,
            lr_override=args.lr,
            loss_alpha=args.loss_alpha,
            loss_beta=args.loss_beta,
            loss_gamma=args.loss_gamma,
            rot_min=args.rot_min,
            rot_max=args.rot_max,
            curriculum_mode=args.curriculum,
            warmup_epochs=args.warmup,
            no_warmup=args.no_warmup
        )