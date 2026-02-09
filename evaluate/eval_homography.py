"""
================================================================================
Phase 4까지 포함된 통합 평가 스크립트
================================================================================
Phase 1 (Preprocessing) -> Phase 2 (Embedding) -> Phase 3 (Transformer) -> Phase 4 (MPC Refiner)
================================================================================
"""

import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 프로젝트 모듈
from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder
from pipeline.phase3 import Phase3Transformer

# Phase 4 import (동일 디렉토리에 있다고 가정)
sys.path.append(current_dir)
from phase4.phase4_2 import GeometricMPCRefiner

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
ERROR_THRESHOLD_EXCELLENT = 5.0
ERROR_THRESHOLD_ACCEPTABLE = 10.0

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

def affine_matrix_to_pytorch_W(M_pixel, width, height):
    """
    픽셀 좌표계의 Affine Matrix를 PyTorch grid_sample용 W로 변환
    
    Args:
        M_pixel: (2, 3) affine matrix in pixel coordinates
        width, height: image dimensions
    
    Returns:
        W: (1, 2, 3) tensor for grid_sample
    """
    # Normalization matrix
    N = np.array([
        [2.0 / width, 0, -1],
        [0, 2.0 / height, -1],
        [0, 0, 1]
    ])
    
    # 3x3으로 확장
    M_aug = np.vstack([M_pixel, [0, 0, 1]])
    
    # 정규화
    M_norm = N @ M_aug @ np.linalg.inv(N)
    
    # (1, 2, 3) 텐서로 변환
    W = torch.from_numpy(M_norm[:2, :]).float().unsqueeze(0)
    
    return W

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
    """격자점 생성"""
    x = np.linspace(width * 0.2, width * 0.8, num_points)
    y = np.linspace(height * 0.2, height * 0.8, num_points)
    xv, yv = np.meshgrid(x, y)
    pts = np.stack([xv.flatten(), yv.flatten()], axis=1)
    return pts

# ==============================================================================
# [NEW] Phase 4 적용 함수
# ==============================================================================
def apply_phase4_refinement(p2_a, p2_b, phase3_results, refiner, device):
    """
    Phase 3 결과를 Phase 4로 정제
    
    Args:
        p2_a, p2_b: Phase 2 출력 (피라미드 딕셔너리)
        phase3_results: Phase 3 출력
        refiner: GeometricMPCRefiner 인스턴스
        device: torch device
    
    Returns:
        W_refined: (1, 2, 3) refined affine matrix (정규화 좌표계)
        loss_history: Phase 4 최적화 loss 히스토리
    """
    # Phase 3 예측 추출
    avg_rotor = phase3_results[0]['rotor_map'].mean(dim=(1, 2))
    cos_raw, sin_raw = avg_rotor[0, 0], avg_rotor[0, 1]
    cos, sin = normalize_rotor_output(cos_raw, sin_raw)
    
    # 각도와 스케일 계산
    cos_np = cos.item()
    sin_np = sin.item()
    angle_rad = np.arctan2(sin_np, cos_np)
    scale = np.sqrt(cos_np**2 + sin_np**2)
    
    print(f"\n[Phase 3 Output] Angle: {np.degrees(angle_rad):.2f}°, Scale: {scale:.3f}")
    
    # Phase 4 초기화
    refiner.global_filtering_init(angle_rad, scale)
    
    # Source와 Target 딕셔너리 준비
    # Phase 2 출력: [(sdf, vector, (unit_cos, unit_sin, magnitude)), ...]
    
    # Finest level (마지막 = 최고 해상도)
    src_sdf, src_vector, src_rotor_tuple = p2_a[-1]
    tgt_sdf, tgt_vector, tgt_rotor_tuple = p2_b[-1]
    
    # Rotor 튜플을 텐서로 변환
    src_unit_cos, src_unit_sin, src_magnitude = src_rotor_tuple
    tgt_unit_cos, tgt_unit_sin, tgt_magnitude = tgt_rotor_tuple
    
    src_rotor = torch.cat([src_unit_cos, src_unit_sin, src_magnitude], dim=1)
    tgt_rotor = torch.cat([tgt_unit_cos, tgt_unit_sin, tgt_magnitude], dim=1)
    
    src_dict = {'sdf': src_sdf, 'vector': src_vector, 'rotor': src_rotor}
    tgt_dict = {'sdf': tgt_sdf, 'vector': tgt_vector, 'rotor': tgt_rotor}
    
    print(f"[DEBUG] Shapes - sdf: {src_sdf.shape}, vector: {src_vector.shape}, rotor: {src_rotor.shape}")
    
    # Gates (균등 가중치로 시작)
    gates = (
        torch.ones(1, 1, 1, 1).to(device),  # g_s
        torch.ones(1, 1, 1, 1).to(device),  # g_v
        torch.ones(1, 1, 1, 1).to(device)   # g_b
    )
    
    # Phase 4 최적화 실행
    W_refined, loss_history = refiner.optimize(src_dict, tgt_dict, gates)
    
    return W_refined, loss_history

# ==============================================================================
# [Main Evaluation Logic with Phase 4]
# ==============================================================================
def evaluate_image_with_phase4(img_path, model_components, refiner, use_phase4=True):
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

    # 3. Inference Phase 1-3
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    with torch.no_grad():
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        phase3_results = transformer(p2_a, p2_b)

        # Phase 3 결과 추출
        avg_rotor = phase3_results[0]['rotor_map'].mean(dim=(1, 2))
        cos, sin = normalize_rotor_output(avg_rotor[0,0], avg_rotor[0,1])
        dx, dy = avg_rotor[0,2], avg_rotor[0,3]

        cos_np = cos.item()
        sin_np = sin.item()
        dx_np = dx.item()
        dy_np = dy.item()

        M_phase3_norm = rotor_to_normalized_affine(cos_np, sin_np, dx_np, dy_np)
        M_phase3_pixel = denormalize_affine_matrix(M_phase3_norm, w, h)

    if use_phase4:
        W_refined, loss_history = apply_phase4_refinement(
            p2_a, p2_b, phase3_results, refiner, DEVICE
        )

        W_np = W_refined.detach().cpu().numpy()[0]
        M_phase4_pixel = denormalize_affine_matrix(W_np, w, h)
    else:
        loss_history = []
        M_phase4_pixel = M_phase3_pixel


    # 5. 점 변환 및 시각화 준비
    src_pts = get_grid_points(w, h)
    
    # GT Destination
    dst_gt = transform_points(M_gt_pixel, src_pts)
    
    # Phase 3 Prediction
    dst_phase3 = transform_points(M_phase3_pixel, src_pts)
    error_phase3 = np.linalg.norm(dst_phase3 - dst_gt, axis=1).mean()
    
    # Phase 4 Prediction
    if use_phase4:
        dst_phase4 = transform_points(M_phase4_pixel, src_pts)
        error_phase4 = np.linalg.norm(dst_phase4 - dst_gt, axis=1).mean()
    else:
        dst_phase4 = dst_phase3
        error_phase4 = error_phase3
    
    return {
        'img_vis': np.hstack([img_warped, img_rgb]),
        'src_pts': src_pts,
        'dst_gt': dst_gt,
        'dst_phase3': dst_phase3,
        'dst_phase4': dst_phase4,
        'error_phase3': error_phase3,
        'error_phase4': error_phase4,
        'loss_history': loss_history,
        'angle': angle,
        'w': w,
        'use_phase4': use_phase4
    }

def visualize_results(res):
    """결과 시각화"""
    errors_phase3 = np.linalg.norm(res['dst_phase3'] - res['dst_gt'], axis=1)
    errors_phase4 = np.linalg.norm(res['dst_phase4'] - res['dst_gt'], axis=1)
    
    # Phase 3 통계
    max_error_p3 = errors_phase3.max()
    min_error_p3 = errors_phase3.min()
    median_error_p3 = np.median(errors_phase3)
    
    # Phase 4 통계
    max_error_p4 = errors_phase4.max()
    min_error_p4 = errors_phase4.min()
    median_error_p4 = np.median(errors_phase4)
    
    # Figure 구성
    if res['use_phase4'] and len(res['loss_history']) > 0:
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 0.8])
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = list(axes)
    
    # ==================================================================
    # [Left] 전체 시각화
    # ==================================================================
    axes[0].imshow(res['img_vis'])
    title_text = f"GT Angle: {res['angle']:.1f}°\n"
    title_text += f"Phase 3: {res['error_phase3']:.1f}px"
    if res['use_phase4']:
        improvement = res['error_phase3'] - res['error_phase4']
        title_text += f" → Phase 4: {res['error_phase4']:.1f}px (Δ {improvement:+.1f}px)"
    axes[0].set_title(title_text, fontsize=10)
    
    offset = np.array([res['w'], 0])
    
    # Source points (왼쪽 이미지)
    axes[0].scatter(res['src_pts'][:,0], res['src_pts'][:,1], 
                   c='lime', s=15, label='Start', zorder=3, edgecolors='black', linewidths=0.5)
    
    gt_pts = res['dst_gt'] + offset
    phase3_pts = res['dst_phase3'] + offset
    phase4_pts = res['dst_phase4'] + offset
    
    # GT (파란 X)
    axes[0].scatter(gt_pts[:,0], gt_pts[:,1], 
                   c='blue', marker='x', s=60, label='GT', zorder=6, linewidths=2)
    
    # Phase 3 (빨간 O)
    axes[0].scatter(phase3_pts[:,0], phase3_pts[:,1], 
                   c='red', marker='o', s=40, label=f'Phase 3 ({res["error_phase3"]:.1f}px)', 
                   zorder=4, edgecolors='darkred', linewidths=1.5, alpha=0.6)
    
    # Phase 4 (초록 O) - if applied
    if res['use_phase4']:
        axes[0].scatter(phase4_pts[:,0], phase4_pts[:,1], 
                       c='lime', marker='o', s=50, label=f'Phase 4 ({res["error_phase4"]:.1f}px)', 
                       zorder=5, edgecolors='darkgreen', linewidths=2)
    
    # 오차 벡터
    for j in range(len(gt_pts)):
        # Phase 3 오차 (빨간 선)
        axes[0].plot([gt_pts[j, 0], phase3_pts[j, 0]], 
                    [gt_pts[j, 1], phase3_pts[j, 1]], 
                    c='red', alpha=0.3, linewidth=1.0, zorder=2)
        
        # Phase 4 오차 (초록 선)
        if res['use_phase4']:
            axes[0].plot([gt_pts[j, 0], phase4_pts[j, 0]], 
                        [gt_pts[j, 1], phase4_pts[j, 1]], 
                        c='lime', alpha=0.5, linewidth=1.5, zorder=3)
    
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].axis('off')
    
    # ==================================================================
    # [Middle Left] Phase 3 Error Heatmap
    # ==================================================================
    axes[1].set_title("Phase 3 Error Heatmap (px)")
    grid_size = int(np.sqrt(len(errors_phase3)))
    error_map_p3 = errors_phase3.reshape(grid_size, grid_size)
    im1 = axes[1].imshow(error_map_p3, cmap='hot', interpolation='bilinear')
    plt.colorbar(im1, ax=axes[1], label='Error (px)')
    
    stats_text_p3 = f"【Phase 3 Statistics】\n"
    stats_text_p3 += f"Mean: {res['error_phase3']:.1f}px\n"
    stats_text_p3 += f"Median: {median_error_p3:.1f}px\n"
    stats_text_p3 += f"Min: {min_error_p3:.1f}px\n"
    stats_text_p3 += f"Max: {max_error_p3:.1f}px"
    
    axes[1].text(0.02, 0.98, stats_text_p3, transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    axes[1].axis('off')
    
    # ==================================================================
    # [Middle Right] Phase 4 Error Heatmap
    # ==================================================================
    if res['use_phase4']:
        axes[2].set_title("Phase 4 Error Heatmap (px)")
        error_map_p4 = errors_phase4.reshape(grid_size, grid_size)
        im2 = axes[2].imshow(error_map_p4, cmap='hot', interpolation='bilinear')
        plt.colorbar(im2, ax=axes[2], label='Error (px)')
        
        stats_text_p4 = f"[Phase 4 Statistics]\n"
        stats_text_p4 += f"Mean: {res['error_phase4']:.1f}px\n"
        stats_text_p4 += f"Median: {median_error_p4:.1f}px\n"
        stats_text_p4 += f"Min: {min_error_p4:.1f}px\n"
        stats_text_p4 += f"Max: {max_error_p4:.1f}px\n\n"
        improvement = res['error_phase3'] - res['error_phase4']
        improvement_pct = (improvement / res['error_phase3']) * 100
        stats_text_p4 += f"【Improvement】\n"
        stats_text_p4 += f"Δ: {improvement:+.1f}px\n"
        stats_text_p4 += f"Rate: {improvement_pct:+.1f}%"
        
        axes[2].text(0.02, 0.98, stats_text_p4, transform=axes[2].transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        axes[2].axis('off')
        
        # ==================================================================
        # [Right] Loss History
        # ==================================================================
        if len(res['loss_history']) > 0:
            axes[3].set_title("Phase 4 Optimization")
            axes[3].plot(res['loss_history'], c='blue', linewidth=2)
            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('Loss')
            axes[3].grid(True, alpha=0.3)
            
            # Best loss 표시
            min_loss_idx = np.argmin(res['loss_history'])
            min_loss = res['loss_history'][min_loss_idx]
            axes[3].scatter([min_loss_idx], [min_loss], c='red', s=100, 
                           marker='*', zorder=5, label=f'Best: {min_loss:.4f}')
            axes[3].legend()
    else:
        axes[2].text(0.5, 0.5, 'Phase 4 Disabled', transform=axes[2].transAxes,
                    ha='center', va='center', fontsize=14, color='gray')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

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
    
    # Phase 4 Refiner 초기화
    refiner = GeometricMPCRefiner(device=DEVICE)
    
    # 테스트
    img_list = glob.glob("./img/val2017/*.jpg")
    if not img_list:
        print("❌ No images found in ./img/val2017/")
        return
        
    np.random.shuffle(img_list)
    
    # Phase 4 사용 여부 선택
    use_phase4 = True  # False로 설정하면 Phase 3만 평가
    
    for i in range(min(5, len(img_list))):
        print(f"\n{'='*80}")
        print(f"Evaluating image {i+1}/5: {os.path.basename(img_list[i])}")
        print(f"{'='*80}")
        
        res = evaluate_image_with_phase4(
            img_list[i], 
            (embedder, transformer), 
            refiner,
            use_phase4=use_phase4
        )
        
        if res is None: 
            continue
        
        visualize_results(res)

if __name__ == "__main__":
    main()