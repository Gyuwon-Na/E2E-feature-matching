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

# 기존 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer

# [Rule 2] Hyperparameters
BATCH_SIZE = 2          # 하이퍼 파라미터 (GPU 메모리에 맞춰 조절)
NUM_EPOCHS = 100        # 하이퍼 파라미터
LEARNING_RATE = 1e-4    # 하이퍼 파라미터
PATIENCE = 10           # 하이퍼 파라미터 (조기 종료 조건: 10 epoch 동안 개선 없으면 종료)
IMG_SIZE = (256, 256)   # 하이퍼 파라미터 (학습용 해상도 고정)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GeometricDataset(Dataset):
    """
    [Data Pipeline]
    이미지를 로드하고, 임의의 기하학적 변형(Rotation, Scale)을 가해
    Source(A)와 Target(B) 쌍을 생성합니다.
    """
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.preprocessor = MathGeometricPreprocessor()
        self.transform = transform
        print(f"[Dataset] Found {len(self.img_paths)} images in {img_dir}")

    def __len__(self):
        return len(self.img_paths)
    
    def normalize_affine_matrix(self,matrix_pixel, width, height):
        """
        OpenCV의 Pixel 단위 Affine 행렬을 PyTorch의 Normalized 좌표계([-1, 1]) 행렬로 변환
        Args:
            matrix_pixel: (2, 3) numpy array (OpenCV format)
            width, height: Image dimensions
        Returns:
            matrix_norm: (2, 3) normalized matrix
        """
        # 1. 정규화 변환 행렬 (Pixel -> Normalized)
        # x_norm = (x_pix / (width/2)) - 1
        # Normalized Space로 가는 변환 행렬 N
        N = np.array([
            [2.0 / width, 0, -1],
            [0, 2.0 / height, -1],
            [0, 0, 1]
        ])
        
        # 2. 역변환 행렬 (Normalized -> Pixel)
        # Pixel Space로 돌아오는 변환 행렬 N_inv
        N_inv = np.linalg.inv(N)
        
        # 3. 3x3 확장 (OpenCV 매트릭스는 2x3이므로)
        M_pix_aug = np.vstack([matrix_pixel, [0, 0, 1]])
        
        # 4. 변환 공식: M_norm = N * M_pix * N_inv
        M_norm_aug = N @ M_pix_aug @ N_inv
        
        return M_norm_aug[:2, :] # 다시 2x3으로 반환

    def __getitem__(self, idx):
        # 1. Load Image
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # 에러 처리를 위해 임의의 데이터 반환 혹은 건너뛰기 (여기선 예외처리)
            return self.__getitem__((idx + 1) % len(self))
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)

        # 2. Generate Random Affine Transform (GT)
        # Target(B) -> Source(A) 로 만드는 변환 행렬 M 생성
        rows, cols = img_rgb.shape[:2]
        
        # Random Rotation (-30 ~ 30 deg) & Scale (0.8 ~ 1.2)
        angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.8, 1.2)
        
        # Center 기준 회전
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        
        # Source Image (A) 생성 (Warped)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        # 3. Calculate W_GT (Alignment Matrix: A -> B)
        # 우리가 학습할 모델은 A를 B로 되돌리는 행렬을 예측해야 함.
        # 따라서 Warp 행렬 M의 역행렬이 정답(W_GT)임.
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]]) 
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :] # 이건 픽셀 단위 (기존)

        # [수정/추가] 픽셀 단위 행렬을 정규화된 행렬로 변환!
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)

        # 4. Phase 1 Preprocessing (CPU bottleneck 주의)
        # levels=4 추천 (L3:32x32 -> L2:64x64 -> L1:128x128 -> L0:256x256)
        # 이렇게 해야 L3, L2에서 Transformer가 2번 돌면서 확실하게 회전을 잡습니다.
        pyramid_a = self.preprocessor.process_pyramid(img_warped, levels=4)
        pyramid_b = self.preprocessor.process_pyramid(img_rgb, levels=4)

        return {
            'pyramid_a': pyramid_a, # List of Dicts (Not Tensor yet)
            'pyramid_b': pyramid_b,
            'w_gt': W_gt_mat_norm.astype(np.float32) # (2, 3)
        }

def collate_fn_geometric(batch):
    """
    [Collate Function]
    Phase 1의 결과물(List of Dicts)을 Batch 단위로 묶어주는 함수
    """
    batch_size = len(batch)
    levels = len(batch[0]['pyramid_a'])
    
    # 배치 단위 피라미드 재구성
    batched_pyramid_a = [{} for _ in range(levels)]
    batched_pyramid_b = [{} for _ in range(levels)]
    w_gts = []

    for item in batch:
        w_gts.append(item['w_gt'])
        for l in range(levels):
            # Phase 1 Dict의 각 요소를 리스트로 모음
            for key in item['pyramid_a'][l]:
                if key not in batched_pyramid_a[l]: batched_pyramid_a[l][key] = []
                if key not in batched_pyramid_b[l]: batched_pyramid_b[l][key] = []
                
                batched_pyramid_a[l][key].append(item['pyramid_a'][l][key])
                batched_pyramid_b[l][key].append(item['pyramid_b'][l][key])

    # Numpy Stacking -> Tensor conversion은 Phase 2 Embedder 내부에서 수행됨.
    # 여기서는 Numpy Array로 Stack만 해둠.
    for l in range(levels):
        for key in batched_pyramid_a[l]:
            if isinstance(batched_pyramid_a[l][key][0], np.ndarray):
                batched_pyramid_a[l][key] = np.stack(batched_pyramid_a[l][key], axis=0)
                batched_pyramid_b[l][key] = np.stack(batched_pyramid_b[l][key], axis=0)
            # resolution, level_index 등은 리스트 유지 혹은 첫번째 값 참조

    return {
        'pyramid_a': batched_pyramid_a,
        'pyramid_b': batched_pyramid_b,
        'w_gt': torch.tensor(np.stack(w_gts), dtype=torch.float32)
    }

class UnifiedGeometricLoss(nn.Module):
    """
    [Phase 5: Unified Geometric Loss]
    제공된 수식에 기반하여 기하학적 정밀도와 뒤틀림 일관성을 계산합니다.
    """
    def __init__(self, alpha=1.0, beta=1.0, lambda_c=1.0, lambda_s=0.5):
        super().__init__()
        self.alpha = alpha # 하이퍼 파라미터 (Geometric Term Weight)
        self.beta = beta   # 하이퍼 파라미터 (Consistency Term Weight)
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0) # coord loss용 beta

    def get_affine_grid(self, matrix, size):
        B, C, H, W = size
        # matrix: (B, 2, 3)
        return F.affine_grid(matrix, [B, C, H, W], align_corners=False)

    def extract_local_rotation(self, w_gt):
        """
        Jacobian Rotation Extraction
        Affine Matrix W_GT의 좌측 2x2 행렬에서 회전 성분만 추출
        """
        # w_gt: (B, 2, 3) -> R: (B, 2, 2)
        rotation_part = w_gt[:, :2, :2]
        # Gram-Schmidt or SVD to orthogonalize (Pure Rotation)
        # 간단히 Polar Decomposition 유사하게 정규화
        u, s, v = torch.svd(rotation_part)
        rot_matrix = torch.matmul(u, v.transpose(1, 2))
        return rot_matrix

    def forward(self, phase2_out_a, phase2_out_b, pred_w_global, w_gt):
        """
        [수정됨] 5차원 벡터 텐서(v_b)의 차원 문제를 해결하기 위해 
        Reshape -> Grid Sample -> Reshape 과정을 거치도록 변경했습니다.
        """
        s_a, v_a, b_a = phase2_out_a
        s_b, v_b, b_b = phase2_out_b
        
        # B, C, H, W 추출 (s_a: B, 64, H, W)
        size = s_a.shape
        B, C_s, H, W = size
        
        # 1. Warp Features of B using W_GT (Back to A's domain)
        grid_gt = self.get_affine_grid(w_gt, size)
        
        # S(Scalar) Warping: (B, 64, H, W) -> OK (4D Input)
        s_b_warped = F.grid_sample(s_b, grid_gt, align_corners=False)
        
        # [Fix] V(Vector) Warping: (B, 64, 2, H, W) -> 5D Input causes Error
        # Solution: (B, 64, 2, H, W) -> (B, 128, H, W) -> Warping -> (B, 64, 2, H, W)
        B, C_v, Comp, H, W = v_b.shape # C_v=64, Comp=2
        v_b_flat = v_b.view(B, C_v * Comp, H, W) # 5D -> 4D Flatten
        v_b_warped_flat = F.grid_sample(v_b_flat, grid_gt, align_corners=False)
        v_b_warped = v_b_warped_flat.view(B, C_v, Comp, H, W) # 4D -> 5D Restore
        
        # Rotor Map (Cos, Sin) Warping
        # b_a elements are (B, 64, H, W) -> Concat to (B, 128, H, W) -> OK
        rotor_a = torch.cat([b_a[0], b_a[1]], dim=1) 
        rotor_b = torch.cat([b_b[0], b_b[1]], dim=1)
        rotor_b_warped = F.grid_sample(rotor_b, grid_gt, align_corners=False)

        # --- [Loss 1] L_s (Scalar Consistency) ---
        l_s = F.mse_loss(s_a, s_b_warped)

        # --- [Loss 2] L_v (Vector Alignment with Jacobian Rotation) ---
        # Jacobian Rotation Matrix 추출
        rot_matrix = self.extract_local_rotation(w_gt) # (B, 2, 2)
        
        # Rotate Warped Vector B
        # v_b_warped is now correctly (B, 64, 2, H, W)
        # einsum: 배차원(b), 채널(c), 높이(h), 너비(w), 벡터성분(i, j)
        # R_ij * V_bj -> V_bi
        v_b_rotated = torch.einsum('bij, bcjhw -> bcihw', rot_matrix, v_b_warped)
        
        l_v = F.mse_loss(v_a, v_b_rotated)

        # --- [Loss 3] L_b (Rotor Consistency) ---
        l_b = F.mse_loss(rotor_a, rotor_b_warped)

        # --- [Loss 4] L_coord (Corner Distance) ---
        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.],
            [1., 1., 1.], [-1., 1., 1.]
        ], device=s_a.device).unsqueeze(0).repeat(B, 1, 1) # (B, 4, 3)
        
        # Transform Corners
        corners_gt = torch.bmm(w_gt, corners.transpose(1, 2))      # (B, 2, 4)
        corners_pred = torch.bmm(pred_w_global, corners.transpose(1, 2)) # (B, 2, 4)
        
        l_coord = self.smooth_l1(corners_pred, corners_gt)

        # --- [Loss 5] L_sdf_photo (SDF Consistency) ---
        # SDF A와 SDF A_hat 비교
        grid_pred = self.get_affine_grid(pred_w_global, size)
        s_b_recon = F.grid_sample(s_b, grid_pred, align_corners=False)
        
        l_sdf_photo = F.mse_loss(s_a, s_b_recon)

        # --- Total Loss ---
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
    dataset = GeometricDataset(img_dir="./img/train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_geometric)

    # 2. Models
    embedder = CliffordPyramidEmbedder(hidden_dim=64).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=192).to(DEVICE) # 64*3 = 192

    # 3. Optimizer
    optimizer = optim.Adam(
        list(embedder.parameters()) + list(transformer.parameters()), 
        lr=LEARNING_RATE
    )
    
    criterion = UnifiedGeometricLoss().to(DEVICE)

    # 4. Training Loop
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        embedder.train()
        transformer.train()
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

        for batch in pbar:
            # Data to GPU (Embedder 내부에서 Tensor 변환 수행)
            pyramid_a_raw = batch['pyramid_a']
            pyramid_b_raw = batch['pyramid_b']
            w_gt = batch['w_gt'].to(DEVICE)

            # A. Phase 2 Embedding
            # embedder takes list of dicts(numpy), returns list of tensors
            # collate_fn에서 묶인 배치를 처리해야 하므로 embedder 로직 일부 수정 필요할 수 있으나
            # 현재 구현상 list of dicts(batched numpy)를 받아서 처리 가능하도록 가정
            # (실제로는 embedder.core 내부 to_tensor에서 numpy->tensor 변환 시 batch dimension 고려되어 있음)
            phase2_a = embedder(pyramid_a_raw, DEVICE)
            phase2_b = embedder(pyramid_b_raw, DEVICE)

            # B. Phase 3 Transformer
            # results: list of dicts per level
            results = transformer(phase2_a, phase2_b)
            
            # C. Loss Calculation (Use Finest Level result)
            finest_res = results[0] # Level 0
            dense_rotor = finest_res['rotor_map'] # (B, H, W, 4) -> (Cos, Sin, dx, dy)
            
            # Global W Prediction from Dense Rotor (Average)
            # Rotor Map의 평균을 내어 Global Transform 추정
            # Cos, Sin -> Rotation, dx, dy -> Translation
            avg_rotor = dense_rotor.mean(dim=(1, 2)) # (B, 4)
            cos_t, sin_t, dx_t, dy_t = avg_rotor[:, 0], avg_rotor[:, 1], avg_rotor[:, 2], avg_rotor[:, 3]
            
            # Construct W_pred (B, 2, 3)
            # R = [[cos, -sin], [sin, cos]]
            row1 = torch.stack([cos_t, -sin_t, dx_t], dim=1)
            row2 = torch.stack([sin_t, cos_t, dy_t], dim=1)
            pred_w_global = torch.stack([row1, row2], dim=1) # (B, 2, 3)

            # Phase 2 Finest Features for Loss
            # phase2_a[0] is (S, V, B) tuple for Level 0
            loss, loss_dict = criterion(phase2_a[0], phase2_b[0], pred_w_global, w_gt)

            # D. Optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (안정성)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), l_coord=loss_dict['l_coord'])

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        # 5. Early Stopping & Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'embedder': embedder.state_dict(),
                'transformer': transformer.state_dict(),
                'optimizer': optimizer.state_dict()
            }, "best_model.pth")
            print("  [Saved] Best Model Updated.")
        else:
            patience_counter += 1
            print(f"  [Info] Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("  [Stop] Early Stopping Triggered.")
            break

if __name__ == "__main__":
    train()