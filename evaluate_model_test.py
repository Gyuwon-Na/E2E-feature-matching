import os
# [핵심 수정 1] GUI 창을 띄우지 않고 파일로 저장하는 'Agg' 백엔드 사용
import matplotlib
matplotlib.use('Agg') 

import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# 기존 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer
from phase4 import GeometricMPCRefiner 

# [설정] 평가 환경
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Evaluation on: {DEVICE}")

IMG_SIZE = (224, 224)
FEATURE_DIM = 96
NUM_LAYERS = 1
EMBED_DIM = 32
MODEL_PATH = "best_model.pth"
TEST_IMG_DIR = "./val2017"
TEST_SAMPLES = 50

# [결과 저장 폴더 생성]
SAVE_DIR = "./eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)

class GeometricEvaluator:
    def __init__(self):
        print(f"[Init] Loading Model from {MODEL_PATH}...")
        
        self.embedder = CliffordPyramidEmbedder(hidden_dim=EMBED_DIM).to(DEVICE)
        self.transformer = Phase3Transformer(
            feature_dim=FEATURE_DIM, 
            num_layers=NUM_LAYERS, 
            embed_dim=EMBED_DIM
        ).to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            self.embedder.load_state_dict(checkpoint['embedder'])
            self.transformer.load_state_dict(checkpoint['transformer'])
            print("[Init] Model Weights Loaded Successfully.")
        else:
            raise FileNotFoundError(f"No checkpoint found at {MODEL_PATH}")
            
        self.embedder.eval()
        self.transformer.eval()
        self.preprocessor = MathGeometricPreprocessor()
        self.refiner = GeometricMPCRefiner(device=DEVICE)

    def get_affine_grid(self, matrix, size):
        B, C, H, W = size
        return F.affine_grid(matrix, [B, C, H, W], align_corners=False)

    def compute_corner_error(self, pred_w, gt_w, h, w):
        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.],
            [1., 1., 1.], [-1., 1., 1.]
        ], device=DEVICE).unsqueeze(0)
        
        gt_pts = torch.bmm(gt_w, corners.transpose(1, 2))
        pred_pts = torch.bmm(pred_w, corners.transpose(1, 2))
        
        dist_norm = torch.norm(gt_pts - pred_pts, dim=1)
        avg_dist_norm = dist_norm.mean().item()
        
        avg_pixel_scale = (h + w) / 4.0
        return avg_dist_norm * avg_pixel_scale

    def predict_transform(self, img_src, img_tgt):
        with torch.no_grad():
            pyr_src = self.preprocessor.process_pyramid(img_src, levels=4)
            pyr_tgt = self.preprocessor.process_pyramid(img_tgt, levels=4)
            
            p2_src = self.embedder(pyr_src, DEVICE)
            p2_tgt = self.embedder(pyr_tgt, DEVICE)
            
            results = self.transformer(p2_src, p2_tgt)
            
            dense_rotor = results[0]['rotor_map']
            avg_rotor = dense_rotor.mean(dim=(1, 2))
            cos, sin, dx, dy = avg_rotor[:, 0], avg_rotor[:, 1], avg_rotor[:, 2], avg_rotor[:, 3]
            
            row1 = torch.stack([cos, -sin, dx], dim=1)
            row2 = torch.stack([sin, cos, dy], dim=1)
            pred_w = torch.stack([row1, row2], dim=1)
            
            return pred_w, results[0]

    def run_evaluation(self):
        import glob
        import random
        
        img_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
        if not img_paths:
            print("No images found.")
            return

        random.seed(999)
        test_samples = random.sample(img_paths, min(len(img_paths), TEST_SAMPLES))
        
        errors = []
        success_count_5px = 0
        success_count_10px = 0
        times = []

        print(f"\n[Eval] Starting Evaluation on {len(test_samples)} images...")
        
        for i, path in enumerate(tqdm(test_samples)):
            img_raw = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, IMG_SIZE)
            h, w = img_rgb.shape[:2]

            angle = np.random.uniform(-30, 30)
            scale = np.random.uniform(0.8, 1.2)
            M_cv = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            img_warped = cv2.warpAffine(img_rgb, M_cv, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            M_aug = np.vstack([M_cv, [0,0,1]])
            M_inv = np.linalg.inv(M_aug)[:2, :]
            
            N = np.array([[2/w, 0, -1], [0, 2/h, -1], [0, 0, 1]])
            N_inv = np.linalg.inv(N)
            gt_w_np = N @ np.vstack([M_inv, [0,0,1]]) @ N_inv
            gt_w = torch.tensor(gt_w_np[:2]).unsqueeze(0).float().to(DEVICE)

            start_t = time.time()
            pred_w, _ = self.predict_transform(img_warped, img_rgb)
            
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                
            times.append(time.time() - start_t)

            err = self.compute_corner_error(pred_w, gt_w, h, w)
            errors.append(err)
            
            if err < 5.0: success_count_5px += 1
            if err < 10.0: success_count_10px += 1
            
            # [Visualization] 첫 5개 샘플만 저장
            if i < 5: 
                self.visualize(img_warped, img_rgb, pred_w, gt_w, i, err)

        avg_err = np.mean(errors)
        fps = 1.0 / np.mean(times)
        
        print("\n" + "="*40)
        print(f"  [Evaluation Report] (N={len(test_samples)})")
        print("="*40)
        print(f"  * MACE (Avg Error) : {avg_err:.4f} pixels")
        print(f"  * Success Rate (5px): {success_count_5px / len(test_samples) * 100:.2f}%")
        print(f"  * Success Rate (10px): {success_count_10px / len(test_samples) * 100:.2f}%")
        print(f"  * Inference Speed  : {fps:.2f} FPS")
        print("="*40)
        print(f"  * Visualization saved to: {SAVE_DIR}")

    def visualize(self, src, tgt, pred_w, gt_w, idx, err):
        src_tensor = torch.tensor(src).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
        
        grid_pred = self.get_affine_grid(pred_w, src_tensor.shape)
        recon_pred = F.grid_sample(src_tensor, grid_pred, align_corners=False)
        img_recon = recon_pred[0].permute(1,2,0).cpu().numpy()
        
        grid_gt = self.get_affine_grid(gt_w, src_tensor.shape)
        recon_gt = F.grid_sample(src_tensor, grid_gt, align_corners=False)
        img_gt_recon = recon_gt[0].permute(1,2,0).cpu().numpy()

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Evaluation Sample #{idx} (Error: {err:.2f} px)", fontsize=16)
        
        plt.subplot(1, 4, 1)
        plt.title("Input (Warped Source)")
        plt.imshow(src)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("Reference (Target)")
        plt.imshow(tgt)
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title(f"Model Prediction\n(Restored)")
        plt.imshow(np.clip(img_recon, 0, 1))
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("Ground Truth\n(Perfect Align)")
        plt.imshow(np.clip(img_gt_recon, 0, 1))
        plt.axis('off')
        
        plt.tight_layout()
        
        # [핵심 수정 2] show() 대신 savefig() 사용
        save_path = os.path.join(SAVE_DIR, f"result_{idx}.png")
        plt.savefig(save_path)
        plt.close() # 메모리 해제

if __name__ == "__main__":
    evaluator = GeometricEvaluator()
    evaluator.run_evaluation()