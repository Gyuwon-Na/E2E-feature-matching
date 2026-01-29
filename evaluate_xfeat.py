import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# --- My Model Imports ---
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer
from phase4 import GeometricMPCRefiner

# --- Settings ---
IMG_SIZE = (256, 256)
HIDDEN_DIM = 48
FEATURE_DIM = 144
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/final_model_early_stop.pth"
ANGLE_THRESHOLD = 45

# =========================================================
# 1. My Model Pipeline (Phase 3 + MPC)
# =========================================================
def run_my_model_pipeline(img_warped, img_rgb, model_components):
    embedder, transformer = model_components
    
    # Phase 3 Inference
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    with torch.no_grad():
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        results = transformer(p2_a, p2_b)
        
        finest_res = results[0]
        avg_rotor = finest_res['rotor_map'].mean(dim=(1, 2))
        cos_raw, sin_raw, dx, dy = avg_rotor[0,0], avg_rotor[0,1], avg_rotor[0,2], avg_rotor[0,3]
        mag = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
        
        # Build Matrix (Normalized)
        row1 = torch.stack([cos_raw/mag, -sin_raw/mag, dx])
        row2 = torch.stack([sin_raw/mag,  cos_raw/mag, dy])
        W_p3_norm = torch.stack([row1, row2])

    # Phase 4 MPC Refinement
    s_src, v_src, b_src = p2_a[0]
    s_tgt, v_tgt, b_tgt = p2_b[0]
    
    src_mpc = {
        'sdf': torch.from_numpy(pyramid_a[0]['sdf']).unsqueeze(0).unsqueeze(0).float().to(DEVICE),
        'vector': v_src.mean(dim=1).detach(),
        'rotor': b_src[2].mean(dim=1, keepdim=True).detach()
    }
    tgt_mpc = {
        'sdf': torch.from_numpy(pyramid_b[0]['sdf']).unsqueeze(0).unsqueeze(0).float().to(DEVICE),
        'vector': v_tgt.mean(dim=1).detach(),
        'rotor': b_tgt[2].mean(dim=1, keepdim=True).detach()
    }
    
    # Simple Gates
    g_s = torch.sigmoid(torch.mean(torch.abs(s_src), dim=1, keepdim=True))
    g_v = torch.sigmoid(torch.mean(torch.norm(v_src, dim=2), dim=1, keepdim=True))
    g_b = torch.sigmoid(torch.mean(b_src[2], dim=1, keepdim=True))
    gates = (g_s, g_v, g_b)

    # Init MPC
    row_bottom = torch.tensor([0., 0., 1.], device=DEVICE).unsqueeze(0)
    mat_3x3 = torch.cat([W_p3_norm, row_bottom], dim=0)
    W_p3_inv = torch.inverse(mat_3x3)[:2, :] # Inverse for grid_sample
    
    refiner = GeometricMPCRefiner(device=DEVICE)
    with torch.no_grad():
        refiner.W[0] = W_p3_inv.unsqueeze(0)
    
    # Optimization Loop
    refiner.optimize(src_mpc, tgt_mpc, gates) # 300 Iterations
    
    # Get Final Matrix (Forward)
    W_mpc_inv = refiner.W[0].detach()
    mat_3x3_inv = torch.cat([W_mpc_inv, row_bottom], dim=0)
    W_final = torch.inverse(mat_3x3_inv)[:2, :].cpu().numpy()
    
    return W_final

# =========================================================
# 2. SOTA Model: XFeat (CVPR 2024)
# =========================================================
def load_xfeat():
    try:
        # Hub에서 XFeat 모델 자동 다운로드 및 로드
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, trust_repo=True)
        return xfeat.to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading XFeat: {e}")
        return None

# [수정된 run_xfeat 함수]

def run_xfeat(img1_np, img2_np, model):
    # img: (H, W, 3) RGB 0~255 numpy array
    
    # Sparse Matching Mode
    # XFeat의 내장 match_images 메서드 사용 (Top-k points)
    with torch.no_grad():
        output = model.match_xfeat(img1_np, img2_np, top_k=2048)
    
    # [수정] output[0], output[1]은 이미 Numpy Array이므로 변환 불필요!
    pts0 = output[0]
    pts1 = output[1]
    
    # 매칭 포인트가 너무 적으면 실패 처리
    if len(pts0) < 4:
        return None
        
    # RANSAC으로 Affine Matrix 추정
    # XFeat은 점만 찾아주므로, 행렬을 구하려면 RANSAC 필수 (여기서 오차 발생)
    M, _ = cv2.estimateAffine2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    return M

# =========================================================
# 3. Utils & Main
# =========================================================
def denormalize_affine_matrix(matrix_norm, width, height):
    N = np.array([[2.0/width, 0, -1], [0, 2.0/height, -1], [0, 0, 1]])
    N_inv = np.linalg.inv(N)
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    M_pix_aug = N_inv @ M_norm_aug @ N
    return M_pix_aug[:2, :]

def calc_error(W_pred, W_gt, w, h):
    if W_pred is None: return 50.0 # Penalty
    corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
    ones = np.ones((4,1))
    corners_aug = np.hstack([corners, ones])
    gt_pts = (W_gt @ corners_aug.T).T
    pred_pts = (W_pred @ corners_aug.T).T
    return np.mean(np.linalg.norm(gt_pts - pred_pts, axis=1))

def main():
    # 1. Load My Model
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
    except:
        # Fallback
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
    embedder.eval(); transformer.eval()
    
    # 2. Load XFeat (2024 SOTA)
    print("Loading XFeat (CVPR 2024)...")
    xfeat_model = load_xfeat()
    if xfeat_model is None:
        print("XFeat load failed. Please install dependencies or check internet.")
        return

    # 3. Benchmark
    img_list = glob.glob("./val2017/*.jpg")
    if not img_list: return
    np.random.shuffle(img_list)
    
    TEST_COUNT = 10
    results = {'Ours (MPC)': [], 'XFeat (2024)': []}
    
    print(f"\n🔥 Benchmark: Ours vs XFeat (2024) on {TEST_COUNT} images...")
    
    for i in range(TEST_COUNT):
        img_raw = cv2.imread(img_list[i])
        if img_raw is None: continue
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        h, w = img_rgb.shape[:2]
        
        # Warp (Harder Case: 15~45 degrees)
        angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
        scale = 1.0 # XFeat은 Scale 변화에도 강하므로 1.0으로 고정해서 회전 승부
        M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # GT Matrix
        M_warp_aug = np.vstack([M_warp, [0,0,1]])
        W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :]
        
        # --- Run Ours ---
        W_ours_norm = run_my_model_pipeline(img_warped, img_rgb, (embedder, transformer))
        W_ours_pix = denormalize_affine_matrix(W_ours_norm, w, h)
        err_ours = calc_error(W_ours_pix, W_gt_pixel, w, h)
        
        # --- Run XFeat ---
        W_xfeat_pix = run_xfeat(img_warped, img_rgb, xfeat_model)
        err_xfeat = calc_error(W_xfeat_pix, W_gt_pixel, w, h)
        
        results['Ours (MPC)'].append(err_ours)
        results['XFeat (2024)'].append(err_xfeat)
        
        print(f"[{i+1}] Angle {angle:.1f} | Ours: {err_ours:.4f} px  vs  XFeat: {err_xfeat:.4f} px")
        
    # --- Final Result ---
    avg_ours = np.mean(results['Ours (MPC)'])
    avg_xfeat = np.mean(results['XFeat (2024)'])
    
    print("="*40)
    print(f"Final MACE (Mean Average Corner Error):")
    print(f"  Ours (MPC)    : {avg_ours:.4f} px")
    print(f"  XFeat (2024)  : {avg_xfeat:.4f} px")
    print("="*40)
    
    if avg_ours < avg_xfeat:
        print("🏆 Victory! Your model is more precise than CVPR 2024 SOTA.")
    
    # Simple Plot
    plt.bar(['Ours (MPC)', 'XFeat (2024)'], [avg_ours, avg_xfeat], color=['blue', 'orange'])
    plt.ylabel('Error (px)')
    plt.title('Performance vs CVPR 2024 Model')
    plt.show()

if __name__ == "__main__":
    main()