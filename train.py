"""
================================================================================
Train Script: Geometric Matching Model
================================================================================
[Google Colab TPU / GPU 호환]

초기 검증용 학습 스크립트: -20° ~ +20° 회전 범위에서 모델 성능 평가
본격적인 학습 전 모델 구조 및 수렴성 확인용

주요 기능:
1. TPU/GPU/CPU 자동 감지 및 최적화
2. 회전 증강 기반 합성 데이터셋
3. 실시간 검증 지표 (각도 오차, 픽셀 오차)
4. 체크포인트 저장 및 조기 종료
================================================================================
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import json

# =============================================================================
# [Hyperparameters] Training Configuration
# =============================================================================
# [Dataset]
IMG_SIZE = (256, 256)            # [Hyperparameter] 입력 이미지 크기
MAX_SAMPLES = None               # [Hyperparameter] None=전체, 숫자=디버그용

# [Rotation Range for Validation]
ROTATION_MIN = -20.0             # [Hyperparameter] 최소 회전 각도 (도)
ROTATION_MAX = 20.0              # [Hyperparameter] 최대 회전 각도 (도)

# [Training]
BATCH_SIZE = 2                   # [Hyperparameter] 배치 크기 (TPU에서는 8 권장)
ACCUM_STEPS = 16                  # [Hyperparameter] Gradient Accumulation Steps
NUM_EPOCHS = 30                  # [Hyperparameter] 최대 에폭 수
LEARNING_RATE = 5e-4             # [Hyperparameter] 초기 학습률
WEIGHT_DECAY = 1e-4              # [Hyperparameter] AdamW weight decay

# [Model Dimensions] - Phase 2/3과 일치해야 함
HIDDEN_DIM = 48                  # [Hyperparameter] Phase 2 임베딩 차원
FEATURE_DIM = 144                # [Hyperparameter] Phase 3 연산 차원

# [Validation]
VAL_SPLIT = 0.1                  # [Hyperparameter] 검증 데이터 비율
VAL_INTERVAL = 2                 # [Hyperparameter] 검증 수행 주기 (에폭)

# [Early Stopping]
EARLY_STOP_PATIENCE = 5         # [Hyperparameter] 조기 종료 인내심 (에폭)
EARLY_STOP_MIN_DELTA = 0.001     # [Hyperparameter] 최소 개선량

# [Checkpoint]
CHECKPOINT_DIR = "./checkpoints" # [Hyperparameter] 체크포인트 저장 경로
SAVE_BEST_ONLY = True            # [Hyperparameter] 최고 성능만 저장

# [Scheduler]
SCHEDULER_T_MAX = NUM_EPOCHS     # [Hyperparameter] CosineAnnealing 주기
SCHEDULER_ETA_MIN = 1e-5         # [Hyperparameter] 최소 학습률

# [Logging]
LOG_INTERVAL = 10                # [Hyperparameter] 로그 출력 간격 (배치)

# =============================================================================
# [Device Setup] TPU / GPU / CPU 자동 감지
# =============================================================================

def setup_device():
    """
    [Helper] 학습 디바이스 자동 설정
    
    우선순위: TPU > CUDA > CPU
    """
    device_info = {'type': 'cpu', 'device': torch.device('cpu')}
    
    # TPU 확인 (Google Colab)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device_info['type'] = 'tpu'
        device_info['device'] = xm.xla_device()
        device_info['xm'] = xm
        print("✅ TPU Detected - Using TPU for training")
        return device_info
    except ImportError:
        pass
    
    # CUDA 확인
    if torch.cuda.is_available():
        device_info['type'] = 'cuda'
        device_info['device'] = torch.device('cuda')
        print(f"✅ CUDA Detected - {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device_info
    
    print("⚠️ No GPU/TPU detected - Using CPU (training will be slow)")
    return device_info


# =============================================================================
# [Dataset] Geometric Rotation Dataset
# =============================================================================

class GeometricRotationDataset(Dataset):
    """
    [Dataset] 회전 증강 기반 합성 데이터셋
    
    Architecture.md §5와 연계: 학습용 W_GT 생성
    
    원본 이미지를 무작위로 회전시켜 (Source, Target, W_GT) 쌍을 생성합니다.
    """
    
    def __init__(self, img_dir, is_train=True, max_samples=MAX_SAMPLES):
        """
        Args:
            img_dir: 이미지 디렉토리 경로
            is_train: 학습/검증 여부
            max_samples: 최대 샘플 수 (디버그용)
        """
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        if not self.img_paths:
            # PNG도 시도
            self.img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
            
        self.is_train = is_train
        self.preprocessor = None  # Lazy loading
        
        print(f"📁 Dataset: {len(self.img_paths)} images from {img_dir}")
        
    def _get_preprocessor(self):
        """Lazy loading for preprocessor to avoid multiprocessing issues"""
        if self.preprocessor is None:
            from phase1 import MathGeometricPreprocessor
            self.preprocessor = MathGeometricPreprocessor()
        return self.preprocessor
    
    def __len__(self):
        return len(self.img_paths)
    
    def normalize_affine_matrix(self, matrix_pixel, width, height):
        """
        [Helper] 픽셀 좌표계 Affine → 정규화 좌표계 Affine 변환
        
        F.affine_grid는 [-1, 1] 좌표계를 사용하므로 변환 필요
        """
        N = np.array([[2.0 / width, 0, -1], 
                      [0, 2.0 / height, -1], 
                      [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_pix_aug = np.vstack([matrix_pixel, [0, 0, 1]])
        M_norm_aug = N @ M_pix_aug @ N_inv
        return M_norm_aug[:2, :]
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'pyramid_a': Phase 1 피라미드 (Warped),
                'pyramid_b': Phase 1 피라미드 (Original),
                'w_gt': 정답 변환 행렬 (2, 3),
                'gt_angle': 정답 각도 (radian),
                'img_a': 워핑된 이미지 (디버그용),
                'img_b': 원본 이미지 (디버그용)
            }
        """
        # 이미지 로드
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        
        if img_bgr is None:
            # 손상된 이미지 건너뛰기
            return self.__getitem__((idx + 1) % len(self))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        rows, cols = img_rgb.shape[:2]
        
        # 무작위 회전 각도 생성
        angle = np.random.uniform(ROTATION_MIN, ROTATION_MAX)
        
        # Affine 변환 행렬 생성 (회전만, 스케일=1.0)
        scale = 1.0
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        
        # 이미지 워핑 (반사 패딩으로 검은 영역 최소화)
        img_warped = cv2.warpAffine(
            img_rgb, M_warp, (cols, rows), 
            borderMode=cv2.BORDER_REFLECT
        )
        
        # 역변환 행렬 계산 (W_GT: Warped → Original 변환)
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :]
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)
        
        # 정답 각도 (역방향)
        gt_angle_rad = np.deg2rad(-angle)
        
        # Phase 1 피라미드 생성
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
    """
    [Collate Function] 배치 데이터 정리
    
    Phase 1 피라미드는 레벨별로 numpy array를 stack
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
    
    # Stack numpy arrays
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
# [Metrics] 평가 지표
# =============================================================================

class MetricTracker:
    """
    [Metrics] 학습 지표 추적기
    
    - Angle Error (도): 예측 각도와 정답 각도의 절대 차이
    - Pixel Error (px): 코너점 변환 오차의 평균
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.angle_errors = []
        self.pixel_errors = []
        self.losses = []
        
    def update(self, pred_W, gt_W, pred_angle, gt_angle, loss):
        """
        Args:
            pred_W: 예측 변환 (B, 2, 3)
            gt_W: 정답 변환 (B, 2, 3)
            pred_angle: 예측 각도 (B,)
            gt_angle: 정답 각도 (B,)
            loss: 현재 배치 손실
        """
        with torch.no_grad():
            # Angle Error (도)
            angle_diff = torch.abs(pred_angle - gt_angle) * 180 / np.pi
            self.angle_errors.extend(angle_diff.cpu().numpy().tolist())
            
            # Pixel Error (4개 코너점)
            B = pred_W.shape[0]
            corners = torch.tensor([
                [-1., -1., 1.], [1., -1., 1.], 
                [1., 1., 1.], [-1., 1., 1.]
            ], device=pred_W.device).T.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 4)
            
            pts_pred = torch.bmm(pred_W, corners)  # (B, 2, 4)
            pts_gt = torch.bmm(gt_W, corners)
            
            # 픽셀 오차로 변환 (정규화 좌표 → 픽셀)
            # [-1, 1] → [0, IMG_SIZE]
            scale = IMG_SIZE[0] / 2
            pixel_error = torch.norm(pts_pred - pts_gt, dim=1).mean(dim=1) * scale
            self.pixel_errors.extend(pixel_error.cpu().numpy().tolist())
            
            self.losses.append(loss)
            
    def get_metrics(self):
        """평균 지표 반환"""
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
    """
    [Training] 1 에폭 학습
    """
    device = device_info['device']
    is_tpu = device_info['type'] == 'tpu'
    
    embedder.train()
    transformer.train()
    
    metric_tracker = MetricTracker()
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Data to device
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)
        
        # Mixed Precision (CUDA only)
        use_amp = (device_info['type'] == 'cuda') and (scaler is not None)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward Pass
            phase2_a = embedder(pyramid_a_raw, device)
            phase2_b = embedder(pyramid_b_raw, device)
            results = transformer(phase2_a, phase2_b)
            
            # Extract predictions from finest level
            finest_res = results[0]
            dense_rotor = finest_res['rotor_map']
            avg_rotor = dense_rotor.mean(dim=(1, 2))
            
            cos_raw = avg_rotor[:, 0]
            sin_raw = avg_rotor[:, 1]
            dx = avg_rotor[:, 2]
            dy = avg_rotor[:, 3]
            
            # Normalize Rotor
            from losses import normalize_rotor_output
            cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
            
            # Build Affine Matrix
            row1 = torch.stack([cos_t, -sin_t, dx], dim=1)
            row2 = torch.stack([sin_t, cos_t, dy], dim=1)
            pred_W = torch.stack([row1, row2], dim=1)
            
            # Compute Loss
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
        
        # Gradient Accumulation
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
            
            # TPU 동기화
            if is_tpu:
                device_info['xm'].mark_step()
        
        # Metrics Update
        pred_angle = torch.atan2(sin_t, cos_t)
        metric_tracker.update(pred_W, w_gt, pred_angle, gt_angle, 
                            loss.item() * ACCUM_STEPS)
        
        # Progress Bar
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
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    for batch in pbar:
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)
        
        # Forward Pass
        phase2_a = embedder(pyramid_a_raw, device)
        phase2_b = embedder(pyramid_b_raw, device)
        results = transformer(phase2_a, phase2_b)
        
        # Extract predictions
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

def train(img_dir, resume_from=None):
    """
    [Main] 학습 메인 함수
    
    Args:
        img_dir: 학습 이미지 디렉토리
        resume_from: 재개할 체크포인트 경로 (Optional)
    """
    print("=" * 70)
    print("🚀 Geometric Matching Model Training")
    print(f"   Rotation Range: {ROTATION_MIN}° ~ {ROTATION_MAX}°")
    print(f"   Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("=" * 70)
    
    # Device Setup
    device_info = setup_device()
    device = device_info['device']
    
    # Checkpoint Directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # ==========================================================================
    # [1] Dataset & DataLoader
    # ==========================================================================
    print("\n📊 Loading Dataset...")
    
    full_dataset = GeometricRotationDataset(img_dir, is_train=True)
    
    # Train/Val Split
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader (TPU는 DistributedSampler 사용)
    if device_info['type'] == 'tpu':
        import torch_xla.distributed.parallel_loader as pl
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn_geometric, num_workers=2, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn_geometric, num_workers=2
        )
    else:
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
    transformer = Phase3Transformer(
        feature_dim=FEATURE_DIM, 
        embed_dim=HIDDEN_DIM
    ).to(device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in embedder.parameters()) + \
                   sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in embedder.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # ==========================================================================
    # [3] Optimizer, Scheduler, Loss
    # ==========================================================================
    optimizer = optim.AdamW(
        list(embedder.parameters()) + list(transformer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_ETA_MIN
    )
    
    criterion = UnifiedGeometricLoss(alpha=1.0, beta=1.0, gamma=0.1).to(device)
    
    # Mixed Precision Scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if device_info['type'] == 'cuda' else None
    
    # ==========================================================================
    # [4] Resume from Checkpoint
    # ==========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n📥 Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    # ==========================================================================
    # [5] Training Loop
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🎯 Starting Training...")
    print("=" * 70)
    
    history = {
        'train_loss': [], 'train_angle': [],
        'val_loss': [], 'val_angle': []
    }
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"\n📌 Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(
            embedder, transformer, train_loader, optimizer, criterion,
            device_info, scaler
        )
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log Training
        print(f"   [Train] Loss: {train_metrics['loss_mean']:.4f} | "
              f"Angle: {train_metrics['angle_error_mean']:.2f}° ± {train_metrics['angle_error_std']:.2f}° | "
              f"Pixel: {train_metrics['pixel_error_mean']:.2f}px")
        
        history['train_loss'].append(train_metrics['loss_mean'])
        history['train_angle'].append(train_metrics['angle_error_mean'])
        
        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
            val_metrics = validate(embedder, transformer, val_loader, criterion, device_info)
            
            print(f"   [Val]   Loss: {val_metrics['loss_mean']:.4f} | "
                  f"Angle: {val_metrics['angle_error_mean']:.2f}° ± {val_metrics['angle_error_std']:.2f}° | "
                  f"Pixel: {val_metrics['pixel_error_mean']:.2f}px")
            
            history['val_loss'].append(val_metrics['loss_mean'])
            history['val_angle'].append(val_metrics['angle_error_mean'])
            
            # Save Best Model
            if val_metrics['loss_mean'] < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = val_metrics['loss_mean']
                patience_counter = 0
                
                checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'embedder': embedder.state_dict(),
                    'transformer': transformer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'hidden_dim': HIDDEN_DIM,
                    'feature_dim': FEATURE_DIM,
                    'metrics': val_metrics
                }, checkpoint_path)
                print(f"   ✅ New Best Model Saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            
            # Early Stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early Stopping at epoch {epoch+1}")
                break
        
        # Epoch Time
        epoch_time = time.time() - epoch_start
        print(f"   ⏱️ Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        
        # Save Last Model (backup)
        torch.save({
            'epoch': epoch,
            'embedder': embedder.state_dict(),
            'transformer': transformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'hidden_dim': HIDDEN_DIM,
            'feature_dim': FEATURE_DIM
        }, os.path.join(CHECKPOINT_DIR, 'last_model.pth'))
    
    # ==========================================================================
    # [6] Training Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    
    # Load best model for final evaluation
    best_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    final_metrics = best_ckpt.get('metrics', {})
    
    print(f"\n📊 Best Model Performance:")
    print(f"   Validation Loss: {best_val_loss:.4f}")
    print(f"   Angle Error: {final_metrics.get('angle_error_mean', 'N/A')}°")
    print(f"   Pixel Error: {final_metrics.get('pixel_error_mean', 'N/A')}px")
    
    # Save History
    with open(os.path.join(CHECKPOINT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


# =============================================================================
# [Quick Test Function]
# =============================================================================

def quick_test(img_dir, checkpoint_path=None):
    """
    [Test] 빠른 추론 테스트
    
    단일 이미지로 모델 동작 확인
    """
    print("\n🧪 Quick Test Mode")
    
    device_info = setup_device()
    device = device_info['device']
    
    # Load Model
    from phase2 import CliffordPyramidEmbedder
    from phase3 import Phase3Transformer
    
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        print(f"   Loaded: {checkpoint_path}")
    else:
        print("   Using random weights (no checkpoint)")
    
    embedder.eval()
    transformer.eval()
    
    # Test Dataset
    test_dataset = GeometricRotationDataset(img_dir, max_samples=5)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_geometric)
    
    print("\n   Testing on 5 samples...")
    
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
            
            pred_angle = np.degrees(np.arctan2(
                avg_rotor[0, 1].item(), 
                avg_rotor[0, 0].item()
            ))
            gt_angle_deg = np.degrees(gt_angle)
            
            error = abs(pred_angle - gt_angle_deg)
            print(f"   Sample {i+1}: GT={gt_angle_deg:+.1f}° | Pred={pred_angle:+.1f}° | Error={error:.2f}°")
    
    print("\n✅ Quick Test Complete!")


# =============================================================================
# [Entry Point]
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Geometric Matching Training')
    parser.add_argument('--img_dir', type=str, default='./val2017',
                        help='Image directory path')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint for testing')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test(args.img_dir, args.checkpoint)
    else:
        train(args.img_dir, args.resume)
