import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# 기존 모듈 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer

# [Hyperparameters]
IMG_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
MATCHING_THRESHOLD = 5.0  # 픽셀 단위 (이 거리 이내면 성공으로 간주)

def denormalize_affine_matrix(matrix_norm, width, height):
    """
    PyTorch Normalized 좌표계([-1, 1]) 행렬을 Pixel 좌표계 행렬로 변환
    """
    # Normalized -> Pixel 변환 행렬
    # x_pix = (x_norm + 1) * (width/2)
    
    # N_inv: Normalized -> Pixel
    N_inv = np.array([
        [width / 2.0, 0, width / 2.0],
        [0, height / 2.0, height / 2.0],
        [0, 0, 1]
    ])
    
    # N: Pixel -> Normalized
    N = np.array([
        [2.0 / width, 0, -1],
        [0, 2.0 / height, -1],
        [0, 0, 1]
    ])
    
    # M_pix = N_inv @ M_norm @ N
    # 하지만 입력 matrix_norm은 좌표를 변환하는 행렬이므로
    # P_norm_new = M_norm @ P_norm_old
    # -> N @ P_pix_new = M_norm @ (N @ P_pix_old)
    # -> P_pix_new = (N_inv @ M_norm @ N) @ P_pix_old
    
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    M_pix_aug = N_inv @ M_norm_aug @ N
    
    return M_pix_aug[:2, :]

def get_correspondences(w_matrix, width, height, num_points=10):
    """
    이미지에 격자 점을 생성하고 w_matrix로 변환된 좌표를 반환
    """
    # 1. 격자 점 생성 (Source Image A 기준)
    x = np.linspace(width * 0.1, width * 0.9, num_points)
    y = np.linspace(height * 0.1, height * 0.9, num_points)
    xv, yv = np.meshgrid(x, y)
    
    src_pts = np.vstack([xv.flatten(), yv.flatten()]).T # (N, 2)
    
    # 2. 변환 적용 (Target Image B에서의 위치 계산)
    # [x', y']^T = W * [x, y, 1]^T
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_aug = np.hstack([src_pts, ones]) # (N, 3)
    
    dst_pts = (w_matrix @ src_pts_aug.T).T # (N, 2)
    
    return src_pts, dst_pts

def evaluate_and_visualize(img_path):
    print(f"\n[Evaluation] Processing {img_path}...")
    
    # 1. Image Load & Preprocess
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Image load failed.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE)
    rows, cols = img_rgb.shape[:2]

    # 2. Generate Warp (Simulation)
    angle = np.random.uniform(-30, 30)
    scale = np.random.uniform(0.8, 1.2)
    M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Ground Truth W 계산 (Pixel 단위)
    # A(Warped) -> B(Original)
    M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
    W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :] # 정답 행렬

    # 4. Model Inference
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    embedder = CliffordPyramidEmbedder(hidden_dim=64).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=192).to(DEVICE)
    
    # Load Checkpoint
    if not os.path.exists(MODEL_PATH):
        print("Checkpoint not found!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    embedder.load_state_dict(checkpoint['embedder'])
    transformer.load_state_dict(checkpoint['transformer'])
    
    embedder.eval()
    transformer.eval()
    
    with torch.no_grad():
        # Phase 2
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        
        # Phase 3
        results = transformer(p2_a, p2_b)
        
        # Extract Global W from Level 0
        finest_res = results[0]
        dense_rotor = finest_res['rotor_map'] # (B, H, W, 4)
        avg_rotor = dense_rotor.mean(dim=(1, 2)) # (B, 4)
        
        cos_t, sin_t, dx_t, dy_t = avg_rotor[0, 0], avg_rotor[0, 1], avg_rotor[0, 2], avg_rotor[0, 3]
        
        # Construct Normalized W (2x3)
        row1 = torch.stack([cos_t, -sin_t, dx_t])
        row2 = torch.stack([sin_t, cos_t, dy_t])
        W_pred_norm = torch.stack([row1, row2]).cpu().numpy()

    # 5. Convert Normalized W -> Pixel W
    W_pred_pixel = denormalize_affine_matrix(W_pred_norm, cols, rows)
    
    # 6. Calculate Accuracy
    # 격자 점(Points)을 생성해서 GT와 Pred로 각각 변환해본 뒤 거리 측정
    src_pts, dst_pts_pred = get_correspondences(W_pred_pixel, cols, rows, num_points=8)
    _, dst_pts_gt = get_correspondences(W_gt_pixel, cols, rows, num_points=8)
    
    distances = np.linalg.norm(dst_pts_pred - dst_pts_gt, axis=1)
    
    inliers = distances < MATCHING_THRESHOLD
    success_rate = (np.sum(inliers) / len(distances)) * 100
    mean_error = np.mean(distances)
    
    print(f"  - GT Angle: {angle:.2f}, Scale: {scale:.2f}")
    print(f"  - Success Rate: {success_rate:.1f}%")
    print(f"  - Mean Pixel Error: {mean_error:.2f} px")

    # 7. Visualization
    vis_img = np.hstack([img_warped, img_rgb]) # 좌: Warped, 우: Original
    
    plt.figure(figsize=(12, 6))
    plt.imshow(vis_img)
    plt.title(f"Geometric Matching Evaluation\nSuccess Rate: {success_rate:.1f}% | Mean Err: {mean_error:.2f}px", 
              fontsize=14, fontweight='bold')
    
    # 선 그리기
    for i in range(len(src_pts)):
        # Source 점 (왼쪽 이미지)
        pt_a = src_pts[i]
        
        # Pred 점 (오른쪽 이미지) -> 좌표 보정 필요 (w 만큼 오른쪽으로 이동)
        pt_b = dst_pts_pred[i]
        pt_b_vis = pt_b + np.array([cols, 0]) 
        
        # GT 점 (오른쪽 이미지) - 비교용으로 작게 표시할 수도 있음 (여기선 생략하고 선 색으로 판별)
        
        color = 'lime' if inliers[i] else 'red'
        alpha = 0.8 if inliers[i] else 0.4
        
        plt.plot([pt_a[0], pt_b_vis[0]], [pt_a[1], pt_b_vis[1]], 
                 color=color, linewidth=1.5, alpha=alpha, marker='o', markersize=4)
        
        # GT 위치를 파란색 X로 살짝 표시 (어디가 정답이었는지)
        if not inliers[i]:
            pt_gt_vis = dst_pts_gt[i] + np.array([cols, 0])
            plt.plot(pt_gt_vis[0], pt_gt_vis[1], 'bx', markersize=6, markeredgewidth=2)
            # 예측 위치에서 정답 위치로 점선 연결 (오차 시각화)
            plt.plot([pt_b_vis[0], pt_gt_vis[0]], [pt_b_vis[1], pt_gt_vis[1]], 
                     color='yellow', linestyle=':', linewidth=1)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 테스트할 이미지 경로 하나를 지정하세요
    # 예: "./img/val/000000000139.jpg"
    # 폴더 내의 첫 번째 이미지를 자동으로 찾습니다.
    import glob
    img_list = glob.glob("./img/val2017/*.jpg")
    
    if len(img_list) > 0:
        # 랜덤으로 3장 뽑아서 테스트
        for _ in range(3):
            target_img = np.random.choice(img_list)
            evaluate_and_visualize(target_img)
    else:
        print("No images found in ./img/val")