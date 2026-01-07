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
BATCH_SIZE = 2          # [보완] 메모리 한계로 1로 하향
NUM_EPOCHS = 100       # 충분한 학습 기회 제공
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
        # [수정] M_warp 행렬을 그대로 사용해야 함
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
        W_gt_mat_pixel = M_warp_aug[:2, :] # 역행렬 연산 제거!

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
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, phase2_out_a, phase2_out_b, pred_w_global, w_gt):
        B = w_gt.shape[0]
        
        # 1. Corner Point Error (가장 중요)
        # 이미지 네 모서리 좌표 [-1, 1]
        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.],
            [1., 1., 1.], [-1., 1., 1.]
        ], device=pred_w_global.device).unsqueeze(0).repeat(B, 1, 1)

        # 정답 좌표 변환
        gt_pts = torch.bmm(w_gt, corners.transpose(1, 2))
        # 예측 좌표 변환
        pred_pts = torch.bmm(pred_w_global, corners.transpose(1, 2))
        
        # L1 Loss 사용 (MSE보다 Outlier에 강건함)
        l_coord = self.l1_loss(pred_pts, gt_pts)
        
        return l_coord, {"loss": l_coord.item(), "l_coord": l_coord.item()}
    
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

    # [핵심 추가] 마지막 레이어의 가중치를 조절하여 Identity 변환에서 시작하도록 강제
    # 모델 구조에 따라 다르지만, 보통 마지막 Linear 레이어를 찾아 초기화합니다.
    # 여기서는 Transformer의 출력이 바로 rotor_map이라고 가정하고, 
    # 모델 내부 혹은 학습 루프 진입 전 bias를 설정하는 것이 좋습니다.

    # 간단한 방법: Transformer의 마지막 레이어 가중치를 0으로 만들고, 
    # Bias를 [1, 0, 0, 0] (cos=1, sin=0, dx=0, dy=0)이 되도록 설정
    for m in transformer.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # 마지막 레이어라고 추정되는 부분 (출력 채널이 output dim과 같은 경우)
            if hasattr(m, 'out_channels') and m.out_channels == 4: # rotor dim
                nn.init.constant_(m.weight, 0)
                # cos, sin, dx, dy 순서라면
                nn.init.constant_(m.bias, 0) 
                m.bias.data[0] = 1.0 # cos = 1.0 (Scale=1, Rot=0)
                print("Initialized last layer to Identity transform.")

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