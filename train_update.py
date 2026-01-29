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
from torch.cuda.amp import autocast, GradScaler

# 기존 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer

# --- [Hyperparameters] ---
BATCH_SIZE = 1          # 6GB VRAM 최적화
ACCUM_STEPS = 16        # 안정적인 학습을 위해 누적 단계 상향
NUM_EPOCHS = 100        # Fine-tuning을 위한 충분한 에폭
LEARNING_RATE = 1e-4    # 미세 조정을 위해 소폭 하향
IMG_SIZE = (256, 256)   
PATIENCE = 35           # 사용자 설정 유지
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 48         # 6GB VRAM 대응 차원
FEATURE_DIM = 144       

def normalize_rotor_output(cos_raw, sin_raw):
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude

# --- [Dataset] Hard Case Mining & Scale Unlocking ---
class GeometricDataset(Dataset):
    def __init__(self, img_dir, max_samples=None):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        self.preprocessor = MathGeometricPreprocessor()

    def __len__(self):
        return len(self.img_paths)
    
    def normalize_affine_matrix(self, matrix_pixel, width, height):
        N = np.array([[2.0 / width, 0, -1], [0, 2.0 / height, -1], [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_pix_aug = np.vstack([matrix_pixel, [0, 0, 1]])
        M_norm_aug = N @ M_pix_aug @ N_inv
        return M_norm_aug[:2, :] 

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None: return self.__getitem__((idx + 1) % len(self))
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        rows, cols = img_rgb.shape[:2]

        # 1. Hard Case Mining: 50% 확률로 20도 이상의 어려운 각도 출제
        max_angle = 45.0
        if np.random.rand() < 0.5:
            sign = 1 if np.random.rand() < 0.5 else -1
            angle = sign * np.random.uniform(20.0, max_angle)
        else:
            angle = np.random.uniform(-max_angle, max_angle)

        # 2. Scale Unlock: 0.8 ~ 1.2 범위 학습
        scale = np.random.uniform(0.8, 1.2)
        
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        # GT 변환 행렬 생성
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]]) 
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :] 
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)
        gt_angle_rad = np.deg2rad(-angle) 
        gt_scale = 1.0 / scale # 역변환 기준 스케일

        pyramid_a = self.preprocessor.process_pyramid(img_warped, levels=4)
        pyramid_b = self.preprocessor.process_pyramid(img_rgb, levels=4)

        return {
            'pyramid_a': pyramid_a, 
            'pyramid_b': pyramid_b,
            'w_gt': W_gt_mat_norm.astype(np.float32),
            'gt_angle': np.float32(gt_angle_rad),
            'gt_scale': np.float32(gt_scale)
        }

def collate_fn_geometric(batch):
    levels = len(batch[0]['pyramid_a'])
    batched_pyramid_a = [{} for _ in range(levels)]
    batched_pyramid_b = [{} for _ in range(levels)]
    w_gts, gt_angles, gt_scales = [], [], []

    for item in batch:
        w_gts.append(item['w_gt'])
        gt_angles.append(item['gt_angle'])
        gt_scales.append(item['gt_scale'])
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
        'w_gt': torch.tensor(np.stack(w_gts), dtype=torch.float32),
        'gt_angle': torch.tensor(np.array(gt_angles), dtype=torch.float32),
        'gt_scale': torch.tensor(np.array(gt_scales), dtype=torch.float32)
    }

class UnifiedGeometricLoss(nn.Module):
    def __init__(self, beta=10.0, lambda_angle=25.0, lambda_scale=10.0):
        super().__init__()
        self.beta = beta
        self.lambda_angle = lambda_angle
        self.lambda_scale = lambda_scale
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.5)

    def get_inverse_affine(self, matrix_2x3):
        B = matrix_2x3.shape[0]
        bottom_row = torch.tensor([0., 0., 1.], device=matrix_2x3.device).view(1, 1, 3).repeat(B, 1, 1)
        matrix_3x3 = torch.cat([matrix_2x3, bottom_row], dim=1)
        return torch.linalg.inv(matrix_3x3)[:, :2, :]

    def forward(self, pred_w_global, w_gt, pred_cos, pred_sin, gt_angle_rad, pred_scale, gt_scale, phase2_a_tuple, phase2_b_tuple): 
        B = pred_w_global.shape[0]
        
        # 1. Coordinate Loss
        corners = torch.tensor([[-1.,-1.,1.], [1.,-1.,1.], [1.,1.,1.], [-1.,1.,1.]], device=pred_w_global.device)
        corners = corners.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)
        pts_pred = torch.bmm(pred_w_global, corners)
        pts_gt = torch.bmm(w_gt, corners)
        l_coord = self.smooth_l1(pts_pred, pts_gt)

        # 2. Angle Loss (Penalize large errors more)
        pred_angle = torch.atan2(pred_sin, pred_cos)
        l_angle = 1.0 - torch.cos(pred_angle - gt_angle_rad).mean()

        # 3. Scale Loss (L1 on Log-Scale)
        l_scale = F.l1_loss(torch.log(pred_scale + 1e-6), torch.log(gt_scale + 1e-6))

        # 4. Pixel Consistency Loss
        s_a = phase2_a_tuple[0] 
        pred_w_inv = self.get_inverse_affine(pred_w_global)
        grid_pred = F.affine_grid(pred_w_inv, s_a.size(), align_corners=False)
        s_a_warped = F.grid_sample(s_a, grid_pred, align_corners=False, padding_mode='zeros')
        l_pixel = F.mse_loss(s_a_warped, phase2_b_tuple[0])

        total_loss = self.beta * l_coord + self.lambda_angle * l_angle + self.lambda_scale * l_scale + 0.5 * l_pixel
        return total_loss, {"loss": total_loss.item(), "l_ang": l_angle.item(), "l_scale": l_scale.item()}

def train():
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    dataset = GeometricDataset(img_dir="./val2017", max_samples=2000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_geometric)

    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    # --- [RESUME LOGIC] 기존 모델 불러오기 ---
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        print(f"🔄 Loading pre-trained weights from {ckpt_path} for scale-up training.")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")

    optimizer = optim.AdamW(list(embedder.parameters()) + list(transformer.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = UnifiedGeometricLoss().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    patience_counter = 0 
    embedder.train(); transformer.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i, batch in enumerate(pbar):
            # [수정] 최신 PyTorch API 반영 (FutureWarning 해결)
            with torch.amp.autocast('cuda', enabled=True):
                w_gt = batch['w_gt'].to(DEVICE)
                gt_angle = batch['gt_angle'].to(DEVICE)
                gt_scale = batch['gt_scale'].to(DEVICE)

                p2_a = embedder(batch['pyramid_a'], DEVICE)
                p2_b = embedder(batch['pyramid_b'], DEVICE)
                results = transformer(p2_a, p2_b)
                
                # Phase 3의 4채널 출력을 해석
                avg_rotor = results[0]['rotor_map'].mean(dim=(1, 2)) 
                cos_raw, sin_raw, dx, dy = avg_rotor[:, 0], avg_rotor[:, 1], avg_rotor[:, 2], avg_rotor[:, 3]
                
                cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
                
                # ==============================================================================
                # [에러 해결] p2_a[0][2]는 (cos, sin, mag) 튜플입니다. 
                # 따라서 Magnitude 텐서에 접근하려면 인덱스 [2]를 한 번 더 써야 합니다.
                # p2_a[0] -> Level 0 Tuple (S, V, B)
                # p2_a[0][2] -> B Tuple (unit_cos, unit_sin, rotor_mag)
                # p2_a[0][2][2] -> rotor_mag Tensor ✅
                # ==============================================================================
                mag_a = p2_a[0][2][2].mean()
                mag_b = p2_b[0][2][2].mean()
                
                pred_scale_raw = mag_a / (mag_b + 1e-6) # 단순 비율로 초기 가이드
                pred_scale = torch.clamp(pred_scale_raw, 0.5, 2.0)
                
                # Affine Matrix 조립
                row1 = torch.stack([pred_scale * cos_t, -pred_scale * sin_t, dx], dim=1)
                row2 = torch.stack([pred_scale * sin_t,  pred_scale * cos_t, dy], dim=1)
                pred_w_global = torch.stack([row1, row2], dim=1)

                loss, loss_dict = criterion(pred_w_global, w_gt, cos_t, sin_t, gt_angle, pred_scale, gt_scale, p2_a[0], p2_b[0])
                loss = loss / ACCUM_STEPS

            # [수정] 최신 Scaler 사용 방식
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}", ang=f"{loss_dict['l_ang']:.4f}", sc=f"{loss_dict['l_scale']:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0 
            torch.save({'embedder': embedder.state_dict(), 'transformer': transformer.state_dict(), 'hidden_dim': HIDDEN_DIM, 'feature_dim': FEATURE_DIM}, os.path.join(save_dir, "best_model.pth"))
            print(f" ✅ Saved Best Model at Epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early Stopping...")
                break
            
if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()