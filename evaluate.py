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

# ==============================================================================
# [Configuration] train.py의 6GB VRAM 설정과 100% 동일하게 맞춰야 함
# ==============================================================================
IMG_SIZE = (256, 256)
HIDDEN_DIM = 48          # Train과 동일 (48)
FEATURE_DIM = 144        # Train과 동일 (144)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/best_model.pth" # 저장된 모델 경로
MATCHING_THRESHOLD = 5.0 # 성공 판정 기준 (5픽셀 이내)

def normalize_rotor_output(cos_raw, sin_raw):
    """Rotor를 단위 벡터로 정규화 (train.py와 동일 로직)"""
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude

def get_correspondences(w_matrix, width, height, num_points=10):
    """정확도 계산을 위해 이미지 위에 격자 점을 생성하고 변환"""
    # 이미지 내부에만 점을 찍어서 밖으로 나가는 점 때문에 오차가 튀는 것 방지
    x = np.linspace(width * 0.2, width * 0.8, num_points)
    y = np.linspace(height * 0.2, height * 0.8, num_points)
    xv, yv = np.meshgrid(x, y)
    
    # (N, 2) 좌표 생성
    src_pts = np.vstack([xv.flatten(), yv.flatten()]).T 
    ones = np.ones((src_pts.shape[0], 1))
    src_pts_aug = np.hstack([src_pts, ones]) 
    
    # 행렬 곱으로 변환된 좌표 계산
    dst_pts = (w_matrix @ src_pts_aug.T).T 
    return src_pts, dst_pts

def denormalize_affine_matrix(matrix_norm, width, height):
    """Normalized Matrix -> Pixel Matrix 변환"""
    N = np.array([
        [2.0 / width, 0, -1],
        [0, 2.0 / height, -1],
        [0, 0, 1]
    ])
    N_inv = np.linalg.inv(N)
    
    # matrix_norm은 2x3이므로 3x3으로 확장
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    
    # M_pix = N_inv @ M_norm @ N
    M_pix_aug = N_inv @ M_norm_aug @ N
    return M_pix_aug[:2, :]

def evaluate_single_image(img_path, model_components):
    embedder, transformer = model_components
    filename = os.path.basename(img_path)
    
    # 1. Image Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE)
    rows, cols = img_rgb.shape[:2]

    # 2. Random Warp (문제 출제)
    # [중요] Train과 동일하게 Scale=1.0 고정, 회전만 랜덤 (-45 ~ 45도)
    angle = np.random.uniform(-2, 2)
    scale = 1.0 
    
    M_warp = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    img_warped = cv2.warpAffine(img_rgb, M_warp, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # GT Matrix (Warped -> Original 복원 행렬)
    M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
    W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :] # 정답 역행렬

    # 3. Model Inference
    # (1) Preprocessing
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    # (2) Inference
    with torch.no_grad():
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        results = transformer(p2_a, p2_b)
        
        # Result Extraction
        finest_res = results[0]
        dense_rotor = finest_res['rotor_map'] 
        avg_rotor = dense_rotor.mean(dim=(1, 2)) 
        
        cos_raw, sin_raw, dx, dy = avg_rotor[0, 0], avg_rotor[0, 1], avg_rotor[0, 2], avg_rotor[0, 3]
        
        # [Normalization]
        cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
        
        # [Matrix Build] Train과 동일하게 Scale=1.0 강제 적용
        scale_factor = 1.0
        
        row1 = torch.stack([scale_factor * cos_t, -scale_factor * sin_t, dx])
        row2 = torch.stack([scale_factor * sin_t,  scale_factor * cos_t, dy])
        W_pred_norm = torch.stack([row1, row2]).cpu().numpy()

    # 4. Metric Calculation
    W_pred_pixel = denormalize_affine_matrix(W_pred_norm, cols, rows)
    
    src_pts, dst_pts_pred = get_correspondences(W_pred_pixel, cols, rows)
    _, dst_pts_gt = get_correspondences(W_gt_pixel, cols, rows)
    
    distances = np.linalg.norm(dst_pts_pred - dst_pts_gt, axis=1)
    mean_error = np.mean(distances)
    is_success = mean_error < MATCHING_THRESHOLD
    
    return {
        'img_warped': img_warped,
        'img_rgb': img_rgb,
        'src_pts': src_pts,
        'dst_pts_pred': dst_pts_pred,
        'dst_pts_gt': dst_pts_gt,
        'angle': angle,
        'mean_error': mean_error,
        'distances': distances,
        'is_success': is_success
    }

def main():
    print(f"[Evaluation] Loading Model from {MODEL_PATH}...")
    
    # 모델 초기화 (Train과 동일한 차원 설정 필수!)
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    embedder.load_state_dict(checkpoint['embedder'])
    transformer.load_state_dict(checkpoint['transformer'])
    
    embedder.eval()
    transformer.eval()
    
    # 테스트 이미지 로드
    TEST_IMG_DIR = "./val2017" # 경로 확인
    img_list = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    np.random.shuffle(img_list)
    
    test_count = 5 # 테스트할 이미지 개수
    success_count = 0
    total_error = 0
    
    print(f"[Evaluation] Testing on {test_count} random images...")
    
    for i in range(test_count):
        res = evaluate_single_image(img_list[i], (embedder, transformer))
        if res is None: continue
        
        success_count += int(res['is_success'])
        total_error += res['mean_error']
        
        # --- Visualization ---
        vis_img = np.hstack([res['img_warped'], res['img_rgb']])
        plt.figure(figsize=(12, 6))
        plt.imshow(vis_img)
        
        title_color = 'green' if res['is_success'] else 'red'
        plt.title(f"Test {i+1} | Angle: {res['angle']:.1f} deg | Err: {res['mean_error']:.2f} px", 
                  fontsize=14, fontweight='bold', color=title_color)
        
        cols = res['img_rgb'].shape[1]
        
        # 매칭 라인 그리기
        for k in range(len(res['src_pts'])):
            # 왼쪽(Source) -> 오른쪽(Target 예측)
            pt_src = res['src_pts'][k]
            pt_pred = res['dst_pts_pred'][k] + np.array([cols, 0]) # 오른쪽 이미지로 좌표 이동
            pt_gt = res['dst_pts_gt'][k] + np.array([cols, 0])
            
            # 예측 선 (빨강)
            plt.plot([pt_src[0], pt_pred[0]], [pt_src[1], pt_pred[1]], 
                     color='red', linewidth=1, alpha=0.6, marker='o', markersize=2)
            
            # 정답 위치 (파란 X)
            plt.plot(pt_gt[0], pt_gt[1], 'bx', markersize=8, markeredgewidth=2)
            
            # 오차 라인 (노란 점선: 예측 -> 정답)
            plt.plot([pt_pred[0], pt_gt[0]], [pt_pred[1], pt_gt[1]], 
                     color='yellow', linestyle=':', linewidth=1.5)

        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    print("="*50)
    print(f"Final Result: Success Rate {(success_count/test_count)*100:.1f}%")
    print(f"Average Error: {total_error/test_count:.2f} px")
    print("="*50)

if __name__ == "__main__":
    main()