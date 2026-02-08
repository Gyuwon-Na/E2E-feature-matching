import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt

# 프로젝트 모듈
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer

# ==============================================================================
# [Configuration]
# ==============================================================================
IMG_SIZE = (256, 256)
HIDDEN_DIM = 48
FEATURE_DIM = 144
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/best_model.pth"
ANGLE_THRESHOLD = 60.0

# [Hyperparameter] 시각화 오차 임계값 (픽셀 단위)
ERROR_THRESHOLD_EXCELLENT = 10.0    # [Hyperparameter] 우수 상한 (≤5px)
ERROR_THRESHOLD_ACCEPTABLE = 15.0  # [Hyperparameter] 허용 상한 (5~15px)

# ==============================================================================
# [수정됨] 변환 행렬 생성 함수
# ==============================================================================
def normalize_rotor_output(cos_raw, sin_raw):
    """[Phase 5] Rotor 정규화"""
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude

def denormalize_affine_matrix(matrix_norm, width, height):
    """
    [수정됨] 정규화된 Affine Matrix를 픽셀 좌표계로 복원
    
    fine_tune.py의 normalize_affine_matrix의 역연산
    N^-1 @ M_norm @ N 형태로 복원
    """
    # Normalization Matrix
    N = np.array([
        [2.0 / width, 0, -1],
        [0, 2.0 / height, -1],
        [0, 0, 1]
    ])
    N_inv = np.linalg.inv(N)
    
    # 3x3으로 확장
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    
    # 역정규화
    M_pixel_aug = N_inv @ M_norm_aug @ N
    
    return M_pixel_aug[:2, :]

def rotor_to_normalized_affine(cos, sin, dx, dy):
    """
    [수정됨] 모델 출력을 정규화된 Affine Matrix (2x3)로 변환
    
    모델이 학습한 그대로의 형태 (중심 보정 없음)
    """
    M = np.zeros((2, 3))
    M[0, 0] = cos
    M[0, 1] = -sin
    M[0, 2] = dx
    M[1, 0] = sin
    M[1, 1] = cos
    M[1, 2] = dy
    return M

# ==============================================================================
# [수정됨] 점 변환 함수
# ==============================================================================
def transform_points(M, pts):
    """
    [수정됨] Affine Matrix로 점들 변환
    
    OpenCV의 cv2.transform 대신 직접 행렬곱 사용
    (방향 명확성을 위해)
    """
    # pts: (N, 2) -> (N, 3) 동차좌표
    pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
    
    # 변환: (2, 3) @ (3, N)^T = (2, N)^T
    transformed = (M @ pts_homo.T).T
    
    return transformed

def get_grid_points(width, height, num_points=10):
    """격자 점 생성"""
    x = np.linspace(width * 0.2, width * 0.8, num_points)
    y = np.linspace(height * 0.2, height * 0.8, num_points)
    xv, yv = np.meshgrid(x, y)
    pts = np.stack([xv.flatten(), yv.flatten()], axis=1)
    return pts

# ==============================================================================
# [Main Evaluation Logic]
# ==============================================================================
def evaluate_image(img_path, model_components):
    embedder, transformer = model_components
    
    # 1. Image Load
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: 
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE)
    h, w = img_rgb.shape[:2]

    # 2. Random Warp (문제 출제)
    angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
    
    # [수정됨] GT Matrix 생성
    # Original(B) -> Warped(A) 방향 (이미지를 회전시키는 행렬)
    M_forward_pixel = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_warped = cv2.warpAffine(img_rgb, M_forward_pixel, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # [수정됨] 모델이 학습한 GT는 Warped(A) -> Original(B) 방향
    # 즉, 위 변환의 역행렬
    M_gt_pixel = cv2.invertAffineTransform(M_forward_pixel)

    # 3. Inference
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    with torch.no_grad():
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        results = transformer(p2_a, p2_b)
        
        # 결과 추출
        avg_rotor = results[0]['rotor_map'].mean(dim=(1, 2))
        cos, sin = normalize_rotor_output(avg_rotor[0,0], avg_rotor[0,1])
        dx, dy = avg_rotor[0,2], avg_rotor[0,3]
        
        # [수정됨] 예측 행렬 생성 (정규화 좌표계)
        cos_np = cos.item()
        sin_np = sin.item()
        dx_np = dx.item()
        dy_np = dy.item()
        
        M_pred_norm = rotor_to_normalized_affine(cos_np, sin_np, dx_np, dy_np)
        M_pred_pixel = denormalize_affine_matrix(M_pred_norm, w, h)

    # 4. [수정됨] 점 변환 및 시각화
    # Source Points: Warped 이미지 위의 격자점
    src_pts = get_grid_points(w, h)
    
    # GT Destination: M_gt로 변환된 위치 (Original 이미지 위)
    dst_gt = transform_points(M_gt_pixel, src_pts)
    
    # Predicted Destination: M_pred로 변환된 위치
    dst_pred = transform_points(M_pred_pixel, src_pts)
    
    # Error 계산 (거리 평균)
    error = np.linalg.norm(dst_pred - dst_gt, axis=1).mean()
    
    return {
        'img_vis': np.hstack([img_warped, img_rgb]),
        'src_pts': src_pts,
        'dst_gt': dst_gt,
        'dst_pred': dst_pred,
        'error': error,
        'angle': angle,
        'w': w
    }

def main():
    # 모델 로드
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return
        
    embedder.eval()
    transformer.eval()
    
    # 테스트
    img_list = glob.glob("./img/val2017/*.jpg")
    if not img_list:
        print("❌ No images found in ./img/val2017/")
        return
        
    np.random.shuffle(img_list)
    
    for i in range(min(5, len(img_list))):
        res = evaluate_image(img_list[i], (embedder, transformer))
        if res is None: 
            continue
        
        # ==================================================================
        # [개선] 오차 통계 계산
        # ==================================================================
        errors_per_point = np.linalg.norm(res['dst_pred'] - res['dst_gt'], axis=1)
        max_error = errors_per_point.max()
        min_error = errors_per_point.min()
        median_error = np.median(errors_per_point)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # [Left] 전체 시각화 (기존)
        axes[0].imshow(res['img_vis'])
        axes[0].set_title(f"Angle: {res['angle']:.1f}° | Avg: {res['error']:.1f}px | "
                         f"Min: {min_error:.1f}px | Max: {max_error:.1f}px")
        
        offset = np.array([res['w'], 0])
        
        # 왼쪽 이미지 점들
        axes[0].scatter(res['src_pts'][:,0], res['src_pts'][:,1], 
                       c='lime', s=15, label='Start', zorder=3, edgecolors='black', linewidths=0.5)
        
        gt_pts = res['dst_gt'] + offset
        pred_pts = res['dst_pred'] + offset
        
        # ==================================================================
        # [수정됨] 절대 픽셀 기준으로 Good/Bad 분류
        # ==================================================================
        excellent_mask = errors_per_point <= ERROR_THRESHOLD_EXCELLENT
        acceptable_mask = (errors_per_point > ERROR_THRESHOLD_EXCELLENT) & \
                         (errors_per_point <= ERROR_THRESHOLD_ACCEPTABLE)
        bad_mask = errors_per_point > ERROR_THRESHOLD_ACCEPTABLE
        
        # 통계 계산
        excellent_count = np.sum(excellent_mask)
        acceptable_count = np.sum(acceptable_mask)
        bad_count = np.sum(bad_mask)
        
        # ==================================================================
        # 정답 표시 (파란 X)
        # ==================================================================
        axes[0].scatter(gt_pts[:,0], gt_pts[:,1], 
                       c='blue', marker='x', s=60, label='GT', zorder=4, linewidths=2)
        
        # ==================================================================
        # [수정됨] 3단계 색상으로 예측 표시
        # ==================================================================
        # 우수 (≤5px: 초록)
        if excellent_count > 0:
            axes[0].scatter(pred_pts[excellent_mask, 0], pred_pts[excellent_mask, 1], 
                           c='lime', s=30, label=f'Excellent (≤5px): {excellent_count}', 
                           zorder=5, edgecolors='darkgreen', linewidths=1.5, marker='o')
        
        # 허용 (5~15px: 노랑)
        if acceptable_count > 0:
            axes[0].scatter(pred_pts[acceptable_mask, 0], pred_pts[acceptable_mask, 1], 
                           c='yellow', s=35, label=f'Acceptable (5-15px): {acceptable_count}', 
                           zorder=4, edgecolors='orange', linewidths=1.5, marker='o')
        
        # 불량 (>15px: 빨강)
        if bad_count > 0:
            axes[0].scatter(pred_pts[bad_mask, 0], pred_pts[bad_mask, 1], 
                           c='red', s=50, label=f'Bad (>15px): {bad_count}', 
                           zorder=3, edgecolors='darkred', linewidths=2, marker='o')
        
        # ==================================================================
        # 오차 벡터 (GT -> Pred, 색상별로)
        # ==================================================================
        for j in range(len(gt_pts)):
            if excellent_mask[j]:
                color, alpha, linewidth = 'lime', 0.3, 0.5
            elif acceptable_mask[j]:
                color, alpha, linewidth = 'yellow', 0.5, 1.0
            else:  # bad_mask
                color, alpha, linewidth = 'red', 0.8, 2.0
            
            axes[0].plot([gt_pts[j, 0], pred_pts[j, 0]], 
                        [gt_pts[j, 1], pred_pts[j, 1]], 
                        c=color, alpha=alpha, linewidth=linewidth, zorder=2)
        
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].axis('off')
        
        # ==================================================================
        # [Right] 오차 히트맵
        # ==================================================================
        axes[1].set_title("Error Heatmap (px)")
        
        # 격자를 이미지로 변환
        grid_size = int(np.sqrt(len(errors_per_point)))
        error_map = errors_per_point.reshape(grid_size, grid_size)
        
        im = axes[1].imshow(error_map, cmap='hot', interpolation='bilinear')
        plt.colorbar(im, ax=axes[1], label='Error (px)')
        
        # ==================================================================
        # [수정됨] 통계 텍스트 (정확도 비율 추가)
        # ==================================================================
        total_points = len(errors_per_point)
        excellent_ratio = (excellent_count / total_points) * 100
        acceptable_ratio = (acceptable_count / total_points) * 100
        bad_ratio = (bad_count / total_points) * 100
        
        stats_text = f"【Statistics】\n"
        stats_text += f"Mean: {res['error']:.1f}px\n"
        stats_text += f"Median: {median_error:.1f}px\n"
        stats_text += f"Min: {min_error:.1f}px\n"
        stats_text += f"Max: {max_error:.1f}px\n\n"
        stats_text += f"【Accuracy】\n"
        stats_text += f"✓ Excellent: {excellent_ratio:.1f}%\n"
        stats_text += f"△ Acceptable: {acceptable_ratio:.1f}%\n"
        stats_text += f"✗ Bad: {bad_ratio:.1f}%"
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()