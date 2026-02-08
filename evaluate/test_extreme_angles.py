"""
test_extreme_angles.py
30~60도 극한 회전 상황에서 Phase 4 v5 (Angle Booster) 성능 집중 테스트
"""
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM
from phase4.phase4_1 import HierarchicalMPCRefiner  # v5 적용됨

# ==============================================================================
# [Settings]
# ==============================================================================
# 테스트할 이미지 경로 (바꾸셔도 됩니다)
IMG_PATH = "./img/val2017/000000010995.jpg" 
MODEL_PATH = "./checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 테스트할 각도 리스트 (극한 상황 포함)
TEST_ANGLES = [30, 40, 50, 60, -35, -45, -60]

def invert_affine_norm(matrix_2x3):
    """2x3 Normalized Affine의 역행렬 구하기"""
    if isinstance(matrix_2x3, np.ndarray):
        matrix_2x3 = torch.from_numpy(matrix_2x3)
    
    row_bottom = torch.tensor([0., 0., 1.], device=matrix_2x3.device).unsqueeze(0)
    if matrix_2x3.dim() == 2:
        mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=0)
        mat_inv = torch.inverse(mat_3x3)
        return mat_inv[:2, :]
    else:
        B = matrix_2x3.shape[0]
        row_bottom = row_bottom.repeat(B, 1, 1)
        mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=1)
        mat_inv = torch.linalg.inv(mat_3x3)
        return mat_inv[:, :2, :]

def get_grid(w_matrix, width, height, num_points=10):
    x = np.linspace(width * 0.2, width * 0.8, num_points)
    y = np.linspace(height * 0.2, height * 0.8, num_points)
    xv, yv = np.meshgrid(x, y)
    src_pts = np.vstack([xv.flatten(), yv.flatten()]).T
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_aug = np.hstack([src_pts, ones])
    dst_pts = (w_matrix @ src_pts_aug.T).T
    return src_pts, dst_pts

def normalize_rotor_output(cos_raw, sin_raw):
    mag = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw/mag, sin_raw/mag

def run_test():
    print(f"🚀 Extreme Rotation Test on {DEVICE}...")
    
    # 1. Load Model
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    refiner = HierarchicalMPCRefiner(device=DEVICE) # Phase 4 v5
    
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        print("✅ Pretrained Model Loaded.")
    else:
        print("⚠️ Warning: No checkpoint found. Using random weights for Phase 3.")
    
    embedder.eval()
    transformer.eval()

    # 2. Load Image
    if not os.path.exists(IMG_PATH):
        print(f"❌ Image not found: {IMG_PATH}")
        return
    
    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))
    h, w = img_rgb.shape[:2]
    
    save_dir = "./extreme_test_results"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'Angle':^10} | {'P3 Error':^12} | {'P4 Error':^12} | {'Improvement':^15}")
    print("-" * 55)

    # 3. Angle Loop
    for angle in TEST_ANGLES:
        # --- A. Problem Generation ---
        M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # GT Inverse Matrix (Target -> Source)
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
        W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :]

        # --- B. Phase 1~3 (Coarse) ---
        prep = MathGeometricPreprocessor()
        pyr_a = prep.process_pyramid(img_warped, levels=4)
        pyr_b = prep.process_pyramid(img_rgb, levels=4)
        
        with torch.no_grad():
            p2_a = embedder(pyr_a, DEVICE)
            p2_b = embedder(pyr_b, DEVICE)
            results = transformer(p2_a, p2_b)
            
            # Phase 3 Output Extraction
            finest = results[0]
            avg_rotor = finest['rotor_map'].mean(dim=(1, 2))
            cos, sin, dx, dy = avg_rotor[0]
            cos_t, sin_t = normalize_rotor_output(cos, sin)
            
            # Construct Phase 3 Matrix (Normalized Forward)
            row1 = torch.stack([cos_t, -sin_t, dx])
            row2 = torch.stack([sin_t, cos_t, dy])
            W_p3_norm = torch.stack([row1, row2]).unsqueeze(0) # (1, 2, 3)

        # --- C. Phase 4 v5 (Refinement) ---
        # Handover: P3(Forward) -> Invert -> MPC Init(Inverse)
        W_p3_inv = invert_affine_norm(W_p3_norm)
        
        # Run MPC
        W_mpc_inv, _ = refiner.optimize(p2_a, p2_b, W_init=W_p3_inv)
        
        # Result: MPC(Inverse) -> Invert -> Forward (Comparison용)
        W_mpc_norm = invert_affine_norm(W_mpc_inv).detach()

        # --- D. Evaluation & Vis ---
        # Matrix Conversion Helper
        def to_pixel(W_norm):
            N = np.array([[2.0/w, 0, -1], [0, 2.0/h, -1], [0, 0, 1]])
            N_inv = np.linalg.inv(N)
            W_n = W_norm.squeeze().cpu().numpy()
            W_aug = np.vstack([W_n, [0, 0, 1]])
            return (N_inv @ W_aug @ N)[:2, :]

        W_p3_pix = to_pixel(W_p3_norm)
        W_mpc_pix = to_pixel(W_mpc_norm)
        
        # Calculate Errors
        src_pts, _ = get_grid(W_gt_pixel, w, h) # Source points
        _, dst_gt = get_grid(W_gt_pixel, w, h)
        _, dst_p3 = get_grid(W_p3_pix, w, h)
        _, dst_mpc = get_grid(W_mpc_pix, w, h)
        
        err_p3 = np.mean(np.linalg.norm(dst_p3 - dst_gt, axis=1))
        err_mpc = np.mean(np.linalg.norm(dst_mpc - dst_gt, axis=1))
        improv = err_p3 - err_mpc
        
        # Print Stats
        mark = "✅" if err_mpc < 5.0 else "⚠️" if err_mpc < 20.0 else "❌"
        print(f"{angle:^10.1f} | {err_p3:^12.2f} | {err_mpc:^12.2f} | {improv:^15.2f} {mark}")

        # Visualization
        vis_img = np.hstack([img_warped, img_rgb])
        plt.figure(figsize=(12, 6))
        plt.imshow(vis_img)
        
        # Draw Grids
        # Shift target points to the right image
        for k in range(len(src_pts)):
            pt_src = src_pts[k]
            pt_gt = dst_gt[k] + np.array([w, 0])
            pt_p3 = dst_p3[k] + np.array([w, 0])
            pt_mpc = dst_mpc[k] + np.array([w, 0])
            
            if k == 0:
                plt.plot(pt_gt[0], pt_gt[1], 'bx', markersize=8, label='GT')
                plt.plot([pt_src[0], pt_p3[0]], [pt_src[1], pt_p3[1]], 'orange', linestyle='--', alpha=0.5, label='Phase 3')
                plt.plot([pt_src[0], pt_mpc[0]], [pt_src[1], pt_mpc[1]], 'r-', alpha=0.8, label='Phase 4 v5')
            else:
                plt.plot(pt_gt[0], pt_gt[1], 'bx', markersize=8)
                plt.plot([pt_src[0], pt_p3[0]], [pt_src[1], pt_p3[1]], 'orange', linestyle='--', alpha=0.5)
                plt.plot([pt_src[0], pt_mpc[0]], [pt_src[1], pt_mpc[1]], 'r-', alpha=0.8)

        plt.title(f"Angle: {angle} deg | P3 Error: {err_p3:.1f}px -> P4 Error: {err_mpc:.1f}px")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/result_{angle}.png")
        plt.close()

if __name__ == "__main__":
    run_test()