import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# 프로젝트 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer
from phase4 import GeometricMPCRefiner  # [New] Phase 4 추가

# ==============================================================================
# [Configuration] 
# ==============================================================================
IMG_SIZE = (256, 256)
HIDDEN_DIM = 48          # Train과 동일
FEATURE_DIM = 144        # Train과 동일
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/best_model.pth"
MATCHING_THRESHOLD = 5.0 
ANGLE_THRESHOLD = 45.0

def normalize_rotor_output(cos_raw, sin_raw):
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude

def get_correspondences(w_matrix, width, height, num_points=10):
    """정확도 계산용 격자 생성 (Forward Matrix 기준)"""
    x = np.linspace(width * 0.2, width * 0.8, num_points)
    y = np.linspace(height * 0.2, height * 0.8, num_points)
    xv, yv = np.meshgrid(x, y)
    
    src_pts = np.vstack([xv.flatten(), yv.flatten()]).T 
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_aug = np.hstack([src_pts, ones]) 
    
    dst_pts = (w_matrix @ src_pts_aug.T).T 
    return src_pts, dst_pts

def denormalize_affine_matrix(matrix_norm, width, height):
    """Normalized Matrix -> Pixel Matrix 변환"""
    N = np.array([[2.0/width, 0, -1], [0, 2.0/height, -1], [0, 0, 1]])
    N_inv = np.linalg.inv(N)
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    M_pix_aug = N_inv @ M_norm_aug @ N
    return M_pix_aug[:2, :]

def invert_affine_norm(matrix_2x3):
    """2x3 Normalized Affine의 역행렬 구하기 (Torch)"""
    # 3x3으로 확장 후 역행렬 계산
    if isinstance(matrix_2x3, np.ndarray):
        matrix_2x3 = torch.from_numpy(matrix_2x3)
    
    row_bottom = torch.tensor([0., 0., 1.], device=matrix_2x3.device).unsqueeze(0)
    mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=0)
    mat_inv = torch.inverse(mat_3x3)
    return mat_inv[:2, :]

def evaluate_with_mpc(img_path, model_components):
    embedder, transformer = model_components
    
    # 1. Image Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE)
    rows, cols = img_rgb.shape[:2]

    # 2. Random Warp (문제 출제)
    angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD) # 회전만 적용
    scale = 1.0 
    
    M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # GT Matrix
    M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
    W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :] # 정답 역행렬

    # 3. Phase 1~3 Inference (Coarse Estimation)
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    with torch.no_grad():
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        results = transformer(p2_a, p2_b)
        
        # Phase 3 결과 추출 (Global Rotor)
        finest_res = results[0]
        dense_rotor = finest_res['rotor_map'] 
        avg_rotor = dense_rotor.mean(dim=(1, 2)) 
        
        cos_raw, sin_raw, dx, dy = avg_rotor[0, 0], avg_rotor[0, 1], avg_rotor[0, 2], avg_rotor[0, 3]
        cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
        
        # [Phase 3 Result Matrix] (Normalized Forward)
        row1 = torch.stack([cos_t, -sin_t, dx])
        row2 = torch.stack([sin_t, cos_t, dy])
        W_p3_norm = torch.stack([row1, row2]) # (2, 3)

    # -------------------------------------------------------------------------
    # 4. Phase 4: MPC Refinement (Fine Tuning)
    # -------------------------------------------------------------------------
    print(f"  [MPC] Refining...")
    
    # (1) Data Packing for MPC
    # Phase 2 출력에서 필요한 Feature 추출 (SDF, Vector, Rotor)
    s_src, v_src, b_src = p2_a[0] # Level 0
    s_tgt, v_tgt, b_tgt = p2_b[0] # Level 0
    
    src_mpc = {
        'sdf': torch.from_numpy(pyramid_a[0]['sdf']).unsqueeze(0).unsqueeze(0).float().to(DEVICE),
        'vector': v_src.mean(dim=1).detach(), # 채널 평균으로 대표 벡터 생성
        'rotor': b_src[2].mean(dim=1, keepdim=True).detach()
    }
    tgt_mpc = {
        'sdf': torch.from_numpy(pyramid_b[0]['sdf']).unsqueeze(0).unsqueeze(0).float().to(DEVICE),
        'vector': v_tgt.mean(dim=1).detach(),
        'rotor': b_tgt[2].mean(dim=1, keepdim=True).detach()
    }
    
    # (2) Heuristic Gates (Guidance Net 대신 사용)
    g_s = torch.sigmoid(torch.mean(torch.abs(s_src), dim=1, keepdim=True))
    g_v = torch.sigmoid(torch.mean(torch.norm(v_src, dim=2), dim=1, keepdim=True))
    g_b = torch.sigmoid(torch.mean(b_src[2], dim=1, keepdim=True))
    gates = (g_s, g_v, g_b)

    # (3) MPC Init: Phase 3 결과를 초기값으로 사용
    # 주의: grid_sample은 Inverse Matrix를 사용하므로, Phase 3 Forward 결과를 뒤집어서 초기화
    W_p3_inv = invert_affine_norm(W_p3_norm)
    
    refiner = GeometricMPCRefiner(device=DEVICE)
    with torch.no_grad():
        refiner.W[0] = W_p3_inv.unsqueeze(0) # 초기화 (Handover)

    # (4) Run Optimization
    refiner.optimize(src_mpc, tgt_mpc, gates)
    
    # (5) Final Result Extraction
    # Refiner가 찾은 최적 W는 Inverse Matrix이므로, 다시 Forward로 뒤집어야 정답(W_gt)과 비교 가능
    W_mpc_inv = refiner.W[0].detach()
    W_mpc_norm = invert_affine_norm(W_mpc_inv).cpu().numpy()
    W_p3_norm = W_p3_norm.cpu().numpy()

    # 5. Metric Check
    def calc_error(W_norm):
        W_pixel = denormalize_affine_matrix(W_norm, cols, rows)
        src_pts, dst_pts_pred = get_correspondences(W_pixel, cols, rows)
        _, dst_pts_gt = get_correspondences(W_gt_pixel, cols, rows)
        dist = np.linalg.norm(dst_pts_pred - dst_pts_gt, axis=1)
        return np.mean(dist), W_pixel

    err_p3, W_p3_pixel = calc_error(W_p3_norm)
    err_mpc, W_mpc_pixel = calc_error(W_mpc_norm)
    
    return {
        'img_warped': img_warped,
        'img_rgb': img_rgb,
        'W_p3': W_p3_pixel,
        'W_mpc': W_mpc_pixel,
        'W_gt': W_gt_pixel,
        'err_p3': err_p3,
        'err_mpc': err_mpc,
        'angle': angle
    }

def main():
    print(f"[Final Evaluation] Loading Model from {MODEL_PATH}...")
    
    # 모델 로드
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
    embedder.load_state_dict(checkpoint['embedder'])
    transformer.load_state_dict(checkpoint['transformer'])
    embedder.eval(); transformer.eval()
    
    # 이미지 로드
    TEST_IMG_DIR = "./val2017" # 절대 경로 권장
    img_list = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if not img_list:
        print("Error: No images found.")
        return
    
    np.random.shuffle(img_list)
    test_count = 3
    actual_count = min(len(img_list), test_count)
    
    print(f"Testing on {actual_count} images with MPC Refinement...\n")
    
    for i in range(actual_count):
        res = evaluate_with_mpc(img_list[i], (embedder, transformer))
        if res is None: continue
        
        # --- Visualization ---
        cols = res['img_rgb'].shape[1]
        vis_img = np.hstack([res['img_warped'], res['img_rgb']])
        
        plt.figure(figsize=(14, 7))
        plt.imshow(vis_img)
        
        # Phase 3 vs MPC 성능 비교 출력
        title_str = (f"Test {i+1} | Angle: {res['angle']:.1f}deg\n"
                     f"Phase 3 Error: {res['err_p3']:.2f} px (Coarse)\n"
                     f"Phase 4 Error: {res['err_mpc']:.2f} px (Refined)")
        
        color = 'lime' if res['err_mpc'] < 1.0 else 'orange'
        plt.title(title_str, fontsize=14, fontweight='bold', color=color, loc='left')
        
        # 격자 그리기
        src_pts, _ = get_correspondences(res['W_gt'], cols, cols) # Dummy call for src pts
        _, dst_p3 = get_correspondences(res['W_p3'], cols, cols)
        _, dst_mpc = get_correspondences(res['W_mpc'], cols, cols)
        _, dst_gt = get_correspondences(res['W_gt'], cols, cols)
        
        for k in range(len(src_pts)):
            pt_src = src_pts[k]
            pt_p3 = dst_p3[k] + np.array([cols, 0])
            pt_mpc = dst_mpc[k] + np.array([cols, 0])
            pt_gt = dst_gt[k] + np.array([cols, 0])
            
            # 1. 정답 (파란 X)
            plt.plot(pt_gt[0], pt_gt[1], 'bx', markersize=10, markeredgewidth=2, label='GT' if k==0 else "")
            
            # 2. Phase 3 예측 (노란 점선) - 초기 예측
            plt.plot([pt_src[0], pt_p3[0]], [pt_src[1], pt_p3[1]], 
                     color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Phase 3' if k==0 else "")
            
            # 3. Phase 4 예측 (빨간 실선) - 최종 보정
            plt.plot([pt_src[0], pt_mpc[0]], [pt_src[1], pt_mpc[1]], 
                     color='red', linewidth=1.5, alpha=0.9, marker='o', markersize=3, label='Phase 4 (MPC)' if k==0 else "")

        plt.legend(loc='lower right')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"Image {i+1}: P3 Error {res['err_p3']:.2f} -> MPC Error {res['err_mpc']:.2f}")

if __name__ == "__main__":
    main()