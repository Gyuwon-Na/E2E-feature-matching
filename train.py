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
from torch.cuda.amp import autocast, GradScaler # [OOM 방지 1] AMP 모듈

# 기존 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer

# [Hyperparameters for 6GB VRAM]
BATCH_SIZE = 1          # [OOM 방지 2] 배치를 1로 줄임 (무조건!)
ACCUM_STEPS = 8         # [OOM 방지 2] 대신 8번 모아서 업데이트 (실제 배치 8 효과)
NUM_EPOCHS = 200     
CURRICULUM_END_EPOCH = 100   
LEARNING_RATE = 2e-4    
IMG_SIZE = (256, 256)   # 해상도 유지 (256이 한계일 수 있음)
PATIENCE = 15           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [OOM 방지 3] 차원 축소 (phase2, 3 초기화 때 사용)
HIDDEN_DIM = 48         # 기존 64 -> 48로 축소
FEATURE_DIM = 144       # 기존 192 -> 144로 축소 (3의 배수 유지 필수)

# --- [Helper] Rotor Normalization ---
def normalize_rotor_output(cos_raw, sin_raw):
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude

# --- [Dataset] Curriculum Learning & Reflection Padding ---
class GeometricDataset(Dataset):
    def __init__(self, img_dir, max_samples=None):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]
            print(f"⚠️ Debug Mode: Using only {len(self.img_paths)} images!")

        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
        self.preprocessor = MathGeometricPreprocessor()
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

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

        # [수정] Scale 제거 (회전부터 학습하기 위함)
        max_angle = min(5.0 + self.current_epoch * 2.0, 45.0)
        angle = np.random.uniform(-max_angle, max_angle)
        scale = 1.0 # [수정됨] 스케일 1.0 고정
        
        M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        M_warp_aug = np.vstack([M_warp, [0, 0, 1]]) 
        W_gt_mat_pixel = np.linalg.inv(M_warp_aug)[:2, :] 
        W_gt_mat_norm = self.normalize_affine_matrix(W_gt_mat_pixel, cols, rows)
        gt_angle_rad = np.deg2rad(-angle) 

        pyramid_a = self.preprocessor.process_pyramid(img_warped, levels=4)
        pyramid_b = self.preprocessor.process_pyramid(img_rgb, levels=4)

        return {
            'pyramid_a': pyramid_a, 
            'pyramid_b': pyramid_b,
            'w_gt': W_gt_mat_norm.astype(np.float32),
            'gt_angle': np.float32(gt_angle_rad)
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
        'gt_angle': torch.tensor(np.array(gt_angles), dtype=torch.float32)
    }

class UnifiedGeometricLoss(nn.Module):
    def __init__(self, beta=10.0, lambda_angle=5.0):
        super().__init__()
        self.beta = beta
        self.lambda_angle = lambda_angle
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.5)

    def get_inverse_affine(self, matrix_2x3):
        B = matrix_2x3.shape[0]
        bottom_row = torch.tensor([0., 0., 1.], device=matrix_2x3.device).view(1, 1, 3).repeat(B, 1, 1)
        matrix_3x3 = torch.cat([matrix_2x3, bottom_row], dim=1)
        matrix_inv = torch.linalg.inv(matrix_3x3)
        return matrix_inv[:, :2, :]

    def forward(self, pred_w_global, w_gt, pred_cos, pred_sin, gt_angle_rad,
                phase2_a_tuple, phase2_b_tuple): 
        
        B = pred_w_global.shape[0]
        corners = torch.tensor([[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]], device=pred_w_global.device)
        corners = corners.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)
        pts_pred = torch.bmm(pred_w_global, corners)
        pts_gt = torch.bmm(w_gt, corners)
        l_coord = self.smooth_l1(pts_pred, pts_gt)

        pred_angle = torch.atan2(pred_sin, pred_cos)
        l_angle = 1.0 - torch.cos(pred_angle - gt_angle_rad).mean()

        # [Pixel Loss]
        s_a = phase2_a_tuple[0] 
        s_b = phase2_b_tuple[0]
        pred_w_inv = self.get_inverse_affine(pred_w_global)
        grid_pred = F.affine_grid(pred_w_inv, s_b.size(), align_corners=False)
        s_a_warped = F.grid_sample(s_a, grid_pred, align_corners=False, padding_mode='zeros')
        l_pixel = F.mse_loss(s_a_warped, s_b)

        total_loss = self.beta * l_coord + self.lambda_angle * l_angle + 0.5 * l_pixel
        return total_loss, {"loss": total_loss.item(), "l_coord": l_coord.item(), "l_angle": l_angle.item()}

def train():
    print(f"Training on {DEVICE} (6GB VRAM Mode: AMP + GradAccum + SmallDim)")
    
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 경로가 연구실 PC에 맞게 되어 있는지 확인 필요
    dataset = GeometricDataset(img_dir="./val2017", max_samples=100)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_geometric)

    # [OOM 방지 3] 줄어든 차원으로 모델 생성
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    optimizer = optim.AdamW(list(embedder.parameters()) + list(transformer.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = UnifiedGeometricLoss().to(DEVICE)
    
    # [OOM 방지 1] Scaler 초기화
    scaler = GradScaler()

    best_loss = float('inf')
    patience_counter = 0 

    embedder.train(); transformer.train() # 루프 밖으로 이동 (계속 train 모드)

    for epoch in range(NUM_EPOCHS):
        dataset.set_epoch(epoch) 
        
        epoch_loss = 0.0
        optimizer.zero_grad() # Epoch 시작 시 초기화

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i, batch in enumerate(pbar):
            # [OOM 방지 1] Autocast Context
            with autocast():
                pyramid_a_raw = batch['pyramid_a']
                pyramid_b_raw = batch['pyramid_b']
                w_gt = batch['w_gt'].to(DEVICE)
                gt_angle = batch['gt_angle'].to(DEVICE)

                phase2_a = embedder(pyramid_a_raw, DEVICE)
                phase2_b = embedder(pyramid_b_raw, DEVICE)
                results = transformer(phase2_a, phase2_b)
                
                finest_res = results[0] 
                dense_rotor = finest_res['rotor_map'] 
                avg_rotor = dense_rotor.mean(dim=(1, 2)) 
                cos_raw, sin_raw, dx, dy = avg_rotor[:, 0], avg_rotor[:, 1], avg_rotor[:, 2], avg_rotor[:, 3]
                
                cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
                
                # [수정] Scale=1.0 고정 (연구실 PC에서 회전부터 잡기 위해)
                scale_factor = 1.0 
                
                row1 = torch.stack([scale_factor * cos_t, -scale_factor * sin_t, dx], dim=1)
                row2 = torch.stack([scale_factor * sin_t,  scale_factor * cos_t, dy], dim=1)
                pred_w_global = torch.stack([row1, row2], dim=1)

                loss, loss_dict = criterion(pred_w_global, w_gt, cos_t, sin_t, gt_angle, 
                                            phase2_a[0], phase2_b[0])
                
                # [OOM 방지 2] Loss 나누기 (Accumulation)
                loss = loss / ACCUM_STEPS

            # [OOM 방지 1] Scaler로 Backward
            scaler.scale(loss).backward()

            # [OOM 방지 2] 일정 스텝마다 업데이트
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 로깅용 Loss는 다시 곱해서 복원
            current_loss = loss.item() * ACCUM_STEPS
            epoch_loss += current_loss
            pbar.set_postfix(loss=f"{current_loss:.4f}", coord=f"{loss_dict['l_coord']:.4f}")

        # Epoch 종료 후 저장 및 Early Stopping
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Finished | Average Loss: {avg_loss:.6f}") # <--- 이 줄 추가!

        # 모델 저장 (6GB 환경에선 저장도 가끔 실패할 수 있으니 try-except 추천하나 일단 진행)
        if epoch < CURRICULUM_END_EPOCH:
            print(f"  [Warm-up] Curriculum getting harder... (Best Loss reset pending)")
            # 워밍업 중에도 모델 저장은 하고 싶다면 아래 주석 해제
            # if avg_loss < best_loss:
            #     best_loss = avg_loss
            #     torch.save(..., "best_model.pth")
            
            # 워밍업 마지막 에폭일 때, Best Loss를 무한대로 초기화! (과거 세탁)
            if epoch == CURRICULUM_END_EPOCH - 1:
                best_loss = float('inf')
                print("  [Reset] Warm-up Done! Resetting Best Loss for Fair Competition.")

        else:
            # [핵심 수정 2] 이제부터 진짜 승부 (Epoch 31~)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0 
                # Best Model 저장
                torch.save({
                    'embedder': embedder.state_dict(), 
                    'transformer': transformer.state_dict(),
                    'hidden_dim': HIDDEN_DIM,
                    'feature_dim': FEATURE_DIM
                }, os.path.join(save_dir, "best_model.pth"))
                print(f"  [Save] New Best Loss (Hard Mode): {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"  [Info] Patience: {patience_counter}/{PATIENCE} (Best: {best_loss:.6f})")
                
                if patience_counter >= PATIENCE:
                    print("  [Stop] Early Stopping Triggered!")
                    torch.save({'embedder': embedder.state_dict(), 'transformer': transformer.state_dict()}, 
                               os.path.join(save_dir, "final_model_early_stop.pth"))
                    break
            
if __name__ == "__main__":
    # GPU 캐시 비우고 시작
    torch.cuda.empty_cache()
    train()