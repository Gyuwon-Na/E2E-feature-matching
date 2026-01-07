import os
# [Rule 4] 디버깅을 위해 비동기 실행을 차단 (에러 발생 시 정확한 라인 지목)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# [System] OOM 방지를 위한 메모리 단편화 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler # [Rule 4] 최신 Mixed Precision 라이브러리

# 기존 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer

# [Rule 2] Hyperparameters (6GB GPU 최적화 설정)
BATCH_SIZE = 1          # [보완] 메모리 한계로 1로 하향
NUM_EPOCHS = 1000       # 충분한 학습 기회 제공
LEARNING_RATE = 1e-4    
PATIENCE = 15           
IMG_SIZE = (224, 224)   # [보완] 256에서 224로 축소하여 메모리 확보 (OOM 시 128로 더 줄이세요)
FEATURE_DIM = 96        # [보완] 192 -> 96 (S, V, B 각 32채널)
NUM_LAYERS = 1          # [보완] 인코더 깊이 축소
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GeometricDataset(Dataset):
    """
    [Data Pipeline]
    이미지를 로드하고, 임의의 기하학적 변형(Rotation, Scale)을 가해
    Source(A)와 Target(B) 쌍을 생성합니다.
    """
    def __init__(self, img_dir, transform=None, limit=None):
        # [보완] 다양한 확장자 지원
        self.img_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
            
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {os.path.abspath(img_dir)}")
        
        # [Rule 5-2] 랜덤 100장 추출 로직
        if limit is not None and len(self.img_paths) > limit:
            import random
            random.seed(42) 
            self.img_paths = random.sample(self.img_paths, limit)
            print(f"[Dataset] Randomly selected {limit} images from {img_dir}")

        self.preprocessor = MathGeometricPreprocessor()
        self.transform = transform
        print(f"[Dataset] Found {len(self.img_paths)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def normalize_affine_matrix(self, matrix_pixel, width, height):
        """
        [보완] OpenCV의 Pixel 단위 Affine 행렬을 PyTorch의 Normalized 좌표계([-1, 1]) 행렬로 변환
        """
        N = np.array([
            [2.0 / width, 0, -1],
            [0, 2.0 / height, -1],
            [0, 0, 1]
        ])
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
        
        # Random Rotation (-30 ~ 30 deg) & Scale (0.8 ~ 1.2)
        angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.8, 1.2)
        
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        # Calculate W_GT
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]]) 
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :] 

        # [보완] 픽셀 단위 행렬을 정규화된 행렬로 변환
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)

        # Phase 1 Preprocessing
        pyramid_a = self.preprocessor.process_pyramid(img_warped, levels=4)
        pyramid_b = self.preprocessor.process_pyramid(img_rgb, levels=4)

        return {
            'pyramid_a': pyramid_a, 
            'pyramid_b': pyramid_b,
            'w_gt': W_gt_mat_norm.astype(np.float32)
        }

def collate_fn_geometric(batch):
    batch_size = len(batch)
    levels = len(batch[0]['pyramid_a'])
    
    batched_pyramid_a = [{} for _ in range(levels)]
    batched_pyramid_b = [{} for _ in range(levels)]
    w_gts = []

    for item in batch:
        w_gts.append(item['w_gt'])
        for l in range(levels):
            for key in item['pyramid_a'][l]:
                if key not in batched_pyramid_a[l]: batched_pyramid_a[l][key] = []
                if key not in batched_pyramid_b[l]: batched_pyramid_b[l][key] = []
                
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
        'w_gt': torch.tensor(np.stack(w_gts), dtype=torch.float32)
    }

class UnifiedGeometricLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, lambda_c=1.0, lambda_s=0.5):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta   
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0) 

    def get_affine_grid(self, matrix, size):
        B, C, H, W = size
        return F.affine_grid(matrix, [B, C, H, W], align_corners=False)

    def extract_local_rotation(self, w_gt):
        rotation_part = w_gt[:, :2, :2]
        u, s, v = torch.linalg.svd(rotation_part)
        rot_matrix = torch.matmul(u, v.transpose(-2, -1))
        return rot_matrix

    def forward(self, phase2_out_a, phase2_out_b, pred_w_global, w_gt):
        s_a, v_a, b_a = phase2_out_a
        s_b, v_b, b_b = phase2_out_b
        
        size = s_a.shape
        B, C_s, H, W = size
        
        # Warp Features of B
        grid_gt = self.get_affine_grid(w_gt, size)
        
        s_b_warped = F.grid_sample(s_b, grid_gt, align_corners=False)
        
        B, C_v, Comp, H, W = v_b.shape 
        v_b_flat = v_b.view(B, C_v * Comp, H, W) 
        v_b_warped_flat = F.grid_sample(v_b_flat, grid_gt, align_corners=False)
        v_b_warped = v_b_warped_flat.view(B, C_v, Comp, H, W) 
        
        rotor_a = torch.cat([b_a[0], b_a[1]], dim=1) 
        rotor_b = torch.cat([b_b[0], b_b[1]], dim=1)
        rotor_b_warped = F.grid_sample(rotor_b, grid_gt, align_corners=False)

        # Loss Calculations
        l_s = F.mse_loss(s_a, s_b_warped)

        rot_matrix = self.extract_local_rotation(w_gt) 
        v_b_rotated = torch.einsum('bij, bcjhw -> bcihw', rot_matrix, v_b_warped)
        l_v = F.mse_loss(v_a, v_b_rotated)

        # [수정] 하드코딩 제거: 입력 채널에 맞춰 동적으로 Reshape
        # rotor_b_warped shape: (B, C_total, H, W) -> C_total은 hidden_dim * 2
        C_total = rotor_b_warped.shape[1] 
        hidden_dim = C_total // 2
        
        # (B, 64, 2, ...) -> (B, hidden_dim, 2, ...) 로 변경
        rotor_b_warped_reshaped = rotor_b_warped.view(B, hidden_dim, 2, H, W)
        
        rotor_b_rotated = torch.einsum('bij, bcjhw -> bcihw', rot_matrix, rotor_b_warped_reshaped)
        
        # 다시 원래 채널 수로 복구 (B, C_total, H, W)
        rotor_b_rotated_flat = rotor_b_rotated.reshape(B, C_total, H, W)
        
        l_b = F.mse_loss(rotor_a, rotor_b_rotated_flat)

        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.],
            [1., 1., 1.], [-1., 1., 1.]
        ], device=s_a.device).unsqueeze(0).repeat(B, 1, 1) 
        
        corners_gt = torch.bmm(w_gt, corners.transpose(1, 2))      
        corners_pred = torch.bmm(pred_w_global, corners.transpose(1, 2)) 
        l_coord = self.smooth_l1(corners_pred, corners_gt)

        grid_pred = self.get_affine_grid(pred_w_global, size)
        s_b_recon = F.grid_sample(s_b, grid_pred, align_corners=False)
        l_sdf_photo = F.mse_loss(s_a, s_b_recon)

        loss_geometric = l_s + l_v + l_b
        loss_consistency = self.lambda_c * l_coord + self.lambda_s * l_sdf_photo
        
        total_loss = self.alpha * loss_geometric + self.beta * loss_consistency
        
        return total_loss, {
            "loss": total_loss.item(),
            "l_s": l_s.item(),
            "l_v": l_v.item(),
            "l_b": l_b.item(),
            "l_coord": l_coord.item(),
            "l_sdf": l_sdf_photo.item()
        }
    
def train():
    # 1. Setup
    print(f"Training on {DEVICE}")
    dataset = GeometricDataset(img_dir="./val2017", limit=100)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_geometric,
        num_workers=2, # OOM 발생 시 0으로 줄여보세요
        pin_memory=True 
    )

    # 2. Models [보완: 차원 축소]
    embedder = CliffordPyramidEmbedder(hidden_dim=32).to(DEVICE) # 64 -> 32
    transformer = Phase3Transformer(
        feature_dim=FEATURE_DIM, 
        num_layers=NUM_LAYERS, 
        embed_dim=32 
    ).to(DEVICE)

    # 3. Optimizer & Scaler
    optimizer = optim.Adam(
        list(embedder.parameters()) + list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    
    # [Rule 4] Mixed Precision Scaler
    scaler = GradScaler('cuda') 
    criterion = UnifiedGeometricLoss().to(DEVICE)

    # 4. Training Loop
    best_loss = float('inf')
    patience_counter = 0

    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Training Progress", position=0, leave=True, dynamic_ncols=True)

    for epoch in epoch_bar:
        embedder.train()
        transformer.train()
        
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            w_gt = batch['w_gt'].to(DEVICE, non_blocking=True)
            
            # [Rule 4] AMP Context
            with autocast('cuda'):
                pyramid_a_raw = batch['pyramid_a']
                pyramid_b_raw = batch['pyramid_b']

                phase2_a = embedder(pyramid_a_raw, DEVICE)
                phase2_b = embedder(pyramid_b_raw, DEVICE)

                results = transformer(phase2_a, phase2_b)
                
                finest_res = results[0] 
                dense_rotor = finest_res['rotor_map'] 
                
                avg_rotor = dense_rotor.mean(dim=(1, 2)) 
                cos_t, sin_t, dx_t, dy_t = avg_rotor[:, 0], avg_rotor[:, 1], avg_rotor[:, 2], avg_rotor[:, 3]
                
                row1 = torch.stack([cos_t, -sin_t, dx_t], dim=1)
                row2 = torch.stack([sin_t, cos_t, dy_t], dim=1)
                pred_w_global = torch.stack([row1, row2], dim=1) 

                loss, loss_dict = criterion(phase2_a[0], phase2_b[0], pred_w_global, w_gt)

            # [핵심 수정] Scaler Logic
            # 1. Scale Loss & Backward
            scaler.scale(loss).backward()
            
            # 2. Unscale Optimizer (Clipping을 위해 필수)
            scaler.unscale_(optimizer)
            
            # 3. Gradient Clipping (이제 안전함)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            
            # 4. Step & Update
            scaler.step(optimizer)
            scaler.update()
            
            # [보완] 메모리 즉시 정리
            torch.cuda.empty_cache()

            epoch_loss += loss.item()
            batch_count += 1
            
            epoch_bar.set_postfix(
                ep=f"{epoch+1}", 
                loss=f"{loss.item():.4f}", 
                coord=f"{loss_dict['l_coord']:.4f}"
            )

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save({
                    'embedder': embedder.state_dict(),
                    'transformer': transformer.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, "best_model.pth")
                epoch_bar.write(f"  [Epoch {epoch+1}] New Best Loss: {best_loss:.6f}. Model Saved.")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    epoch_bar.write(f"  [Stop] Early Stopping at Epoch {epoch+1}. No improvement.")
                    break

if __name__ == "__main__":
    train()