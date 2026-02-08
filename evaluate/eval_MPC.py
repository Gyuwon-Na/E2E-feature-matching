import os
import cv2
import numpy as np
import torch
import glob
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ==============================================================================
# [Import Area] 비교할 Phase 4 모듈들을 여기서 가져오세요
# ==============================================================================
from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder
from pipeline.phase3 import Phase3Transformer

# Import all refiner variants
from phase4.phase4_1 import HierarchicalMPCRefiner as RefinerBase
from phase4.phase4_2 import HierarchicalMPCRefiner as RefinerV2
from phase4.phase4_3 import HierarchicalMPCRefiner as RefinerV3

# NEW: Import proposed solutions
import sys
sys.path.insert(0, '/home/claude')
from phase4.phase4_ensemble import EnsembleMPCRefiner
from phase4.phase4_dynamic import DynamicScoutRefiner

# 비교할 후보군 등록 (이름, 클래스)
CANDIDATES = [
    ("V1 Baseline", RefinerBase), 
    ("V2 Advanced", RefinerV2),
    ("V3 Scout", RefinerV3),
    ("Ensemble", EnsembleMPCRefiner),      # Solution 2
    ("Dynamic", DynamicScoutRefiner),       # Solution 3
]

# ==============================================================================
# [Configuration]
# ==============================================================================
IMG_SIZE = (256, 256)
HIDDEN_DIM = 48
FEATURE_DIM = 144
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/last_model.pth"
TEST_IMG_DIR = "./img/val2017"
ANGLE_THRESHOLD = 60.0

# ==============================================================================
# [Helper Functions] 기존 코드 재사용 (정규화/비정규화/역행렬)
# ==============================================================================
def normalize_rotor_output(cos_raw, sin_raw):
    mag = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / mag, sin_raw / mag

def invert_affine_norm(matrix_2x3):
    if isinstance(matrix_2x3, np.ndarray):
        matrix_2x3 = torch.from_numpy(matrix_2x3)
    row_bottom = torch.tensor([0., 0., 1.], device=matrix_2x3.device).unsqueeze(0)
    if matrix_2x3.dim() == 2:
        mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=0)
        return torch.inverse(mat_3x3)[:2, :]
    else:
        B = matrix_2x3.shape[0]
        row_bottom = row_bottom.repeat(B, 1, 1)
        mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=1)
        return torch.linalg.inv(mat_3x3)[:, :2, :]

def denormalize_affine_matrix(matrix_norm, width, height):
    N = np.array([[2.0/width, 0, -1], [0, 2.0/height, -1], [0, 0, 1]])
    N_inv = np.linalg.inv(N)
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    return (N_inv @ M_norm_aug @ N)[:2, :]

def get_correspondences(w_matrix, width, height, num_points=10):
    x = np.linspace(width * 0.2, width * 0.8, num_points)
    y = np.linspace(height * 0.2, height * 0.8, num_points)
    xv, yv = np.meshgrid(x, y)
    src_pts = np.vstack([xv.flatten(), yv.flatten()]).T 
    ones = np.ones((src_pts.shape[0], 1))
    dst_pts = (w_matrix @ np.hstack([src_pts, ones]).T).T 
    return src_pts, dst_pts

# ==============================================================================
# [Main Logic]
# ==============================================================================
def run_comparison():
    # 1. Load Core Models (Phase 2 & 3)
    print(f"Loading Core Models from {MODEL_PATH}...")
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    embedder.eval()
    transformer.eval()
    preprocessor = MathGeometricPreprocessor()

    # 2. Instantiate Refiners
    print(f"Initializing {len(CANDIDATES)} Refiner Candidates...")
    refiners = {}
    for name, cls in CANDIDATES:
        refiners[name] = cls(device=DEVICE) # 각 클래스 인스턴스화

    # 3. Load Images
    img_list = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if not img_list:
        print("No images found.")
        return
    np.random.shuffle(img_list)
    test_imgs = img_list[:5] # 5장만 테스트

    # 4. Evaluation Loop
    results_summary = {name: {'total_error': 0, 'total_time': 0} for name, _ in CANDIDATES}
    results_summary['Phase3'] = {'total_error': 0, 'total_time': 0}

    for idx, img_path in enumerate(test_imgs):
        print(f"\n[{idx+1}/{len(test_imgs)}] Processing {os.path.basename(img_path)}...")
        
        # --- Pre-processing & Inference (Common) ---
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        h, w = img_rgb.shape[:2]

        # Random Warp
        angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
        M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # GT Matrix
        M_warp_aug = np.vstack([M_warp, [0, 0, 1]])
        W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :] # Inverse Warp

        # Phase 1~3 Inference
        pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
        pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
        
        with torch.no_grad():
            p2_a = embedder(pyramid_a, DEVICE)
            p2_b = embedder(pyramid_b, DEVICE)
            p3_res = transformer(p2_a, p2_b)
            
            # Phase 3 Matrix Extraction
            dense_rotor = p3_res[0]['rotor_map']
            avg_rotor = dense_rotor.mean(dim=(1, 2))
            cos, sin = normalize_rotor_output(avg_rotor[0,0], avg_rotor[0,1])
            dx, dy = avg_rotor[0,2], avg_rotor[0,3]
            
            row1 = torch.stack([cos, -sin, dx])
            row2 = torch.stack([sin, cos, dy])
            W_p3_norm = torch.stack([row1, row2]).unsqueeze(0)
            
            # Handover Matrix (Forward -> Inverse)
            W_p3_inv = invert_affine_norm(W_p3_norm)

        # --- Calculate Phase 3 Error ---
        W_p3_pixel = denormalize_affine_matrix(W_p3_norm.cpu().numpy()[0], w, h)
        _, dst_gt = get_correspondences(W_gt_pixel, w, h)
        _, dst_p3 = get_correspondences(W_p3_pixel, w, h)
        err_p3 = np.mean(np.linalg.norm(dst_p3 - dst_gt, axis=1))
        results_summary['Phase3']['total_error'] += err_p3
        
        print(f"  > Angle: {angle:.1f}° | Phase 3 Error: {err_p3:.2f} px")

        # --- Run Each Candidate ---
        candidate_results = {} # Visualization용 데이터 저장
        
        for name, _ in CANDIDATES:
            refiner = refiners[name]
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # [Core Refinement]
            # p2_a, p2_b, 초기값(Inverse)을 전달
            W_refined_inv, _ = refiner.optimize(p2_a, p2_b, W_init=W_p3_inv)
            
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 결과 처리 (Inverse -> Forward 변환 후 픽셀 좌표)
            W_refined_norm = invert_affine_norm(W_refined_inv).detach().cpu().numpy()[0]
            W_refined_pixel = denormalize_affine_matrix(W_refined_norm, w, h)
            
            # 에러 계산
            _, dst_refined = get_correspondences(W_refined_pixel, w, h)
            err_refined = np.mean(np.linalg.norm(dst_refined - dst_gt, axis=1))
            
            # 통계 저장
            results_summary[name]['total_error'] += err_refined
            results_summary[name]['total_time'] += elapsed_ms
            
            candidate_results[name] = {
                'error': err_refined,
                'time': elapsed_ms,
                'pts': dst_refined,
                'matrix': W_refined_pixel
            }
            
            # 개선율 계산 (양수면 개선, 음수면 악화)
            improv = err_p3 - err_refined
            print(f"    - [{name}] Error: {err_refined:.2f} px ({improv:+.2f}) | Time: {elapsed_ms:.1f} ms")

        # --- Visualization ---
        visualize_comparison(img_warped, img_rgb, dst_gt, dst_p3, candidate_results, angle)

    # 5. Final Report
    print_final_report(results_summary, len(test_imgs))

# ==============================================================================
# [Visualization & Reporting Helper]
# ==============================================================================
def visualize_comparison(img_warped, img_rgb, gt_pts, p3_pts, candidates_res, angle):
    vis_img = np.hstack([img_warped, img_rgb])
    h, w = img_rgb.shape[:2]
    offset = np.array([w, 0])
    
    plt.figure(figsize=(15, 8))
    plt.imshow(vis_img)
    plt.title(f"Refiner Comparison (Angle: {angle:.1f}°)", fontsize=15, fontweight='bold')
    
    # 1. Start Points (Left Image)
    # 격자 원본 위치 (대략적인 시각화)
    grid_w = np.linspace(w*0.2, w*0.8, 10)
    xv, yv = np.meshgrid(grid_w, grid_w)
    plt.scatter(xv, yv, c='lime', s=10, alpha=0.5)

    # 2. GT & Phase 3 (Right Image)
    plt.plot(gt_pts[:,0]+w, gt_pts[:,1], 'bx', markersize=8, markeredgewidth=2, label='Ground Truth')
    
    # Phase 3는 선으로만 표현 (복잡도 줄이기 위함)
    # 대표적으로 첫 10개 점만 선으로 연결해서 보여줌
    for i in range(len(gt_pts)):
        lbl = 'Phase 3 (Base)' if i == 0 else ""
        plt.plot([gt_pts[i,0]+w, p3_pts[i,0]+w], [gt_pts[i,1], p3_pts[i,1]], 
                 color='orange', linestyle='--', alpha=0.4, label=lbl)

    # 3. Candidates
    colors = ['red', 'cyan', 'magenta', 'yellow']
    for idx, (name, res) in enumerate(candidates_res.items()):
        color = colors[idx % len(colors)]
        pts = res['pts']
        err = res['error']
        t = res['time']
        
        # 예측점 찍기
        plt.scatter(pts[:,0]+w, pts[:,1], c=color, s=20, edgecolors='black', linewidths=0.5, zorder=5+idx,
                   label=f"{name}: {err:.1f}px ({t:.0f}ms)")
        
        # 에러 벡터 그리기 (GT -> Pred)
        for i in range(len(pts)):
            plt.plot([gt_pts[i,0]+w, pts[i,0]+w], [gt_pts[i,1], pts[i,1]], 
                     color=color, linewidth=1, alpha=0.6)

    plt.legend(loc='upper right', framealpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def print_final_report(summary, count):
    print("\n" + "="*60)
    print(f" FINAL REPORT (Average over {count} images)")
    print("="*60)
    print(f"{'Method':<20} | {'Avg Error (px)':<15} | {'Avg Time (ms)':<15}")
    print("-" * 60)
    
    # Phase 3 먼저 출력
    p3_err = summary['Phase3']['total_error'] / count
    print(f"{'Phase 3 (Base)':<20} | {p3_err:<15.2f} | {'-':<15}")
    print("-" * 60)
    
    for name, data in summary.items():
        if name == 'Phase3': continue
        avg_err = data['total_error'] / count
        avg_time = data['total_time'] / count
        
        # Phase 3 대비 개선율 표시
        improvement = p3_err - avg_err
        sign = "+" if improvement > 0 else ""
        
        err_str = f"{avg_err:.2f} ({sign}{improvement:.2f})"
        print(f"{name:<20} | {err_str:<15} | {avg_time:<15.1f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_comparison()