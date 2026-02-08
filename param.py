"""
tune_and_log_mpc_nopandas.py
Pandas 없이 순수 Python만으로 MPC 파라미터 튜닝 및 CSV 저장을 수행하는 스크립트
"""
import os
import cv2
import numpy as np
import torch
import time
import csv
from collections import defaultdict

# 프로젝트 모듈
import phase4  # 설정을 덮어쓰기 위해 임포트
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from phase3 import Phase3Transformer, FEATURE_DIM
from phase4 import HierarchicalMPCRefiner
# ==============================================================================
# [Experiment Setup] 4차 튜닝: Champion(Exp1) 미세 조정 및 스케줄링
# ==============================================================================
EXPERIMENTS = [
    {
        "name": "Exp1_Baseline", # 기준점 (부동의 1위)
        "config": {
            'levels': [2, 1, 0],
            'iters': [100, 50, 30],
            'base_lrs': [0.005, 0.002, 0.001],
            'angle_boost': [20.0, 5.0, 1.0],
            'weights': [[0.0, 2.0, 0.5], [0.5, 1.0, 0.5], [1.0, 0.2, 0.1]]
        }
    },
    {
        "name": "Exp10_SDF_Assist",
        # [전략] Level 2에서 픽셀 정보(SDF)를 아주 조금(0.1) 추가
        # 목표: 방향은 맞는데 위치가 붕 뜨는 현상(Drift) 방지
        "config": {
            'levels': [2, 1, 0],
            'iters': [100, 50, 30],
            'base_lrs': [0.005, 0.002, 0.001],
            'angle_boost': [20.0, 5.0, 1.0],
            'weights': [
                [0.1, 2.0, 0.5], # [변경] SDF 0.0 -> 0.1 (위치 고정용)
                [0.5, 1.0, 0.5], 
                [1.0, 0.2, 0.1]
            ]
        }
    },
    {
        "name": "Exp11_Scale_Lock",
        # [전략] Level 2에서 Rotor 가중치 2배 증가
        # 목표: 회전하면서 이미지가 커지거나 작아지는(Scale Drift) 현상 억제
        "config": {
            'levels': [2, 1, 0],
            'iters': [100, 50, 30],
            'base_lrs': [0.005, 0.002, 0.001],
            'angle_boost': [20.0, 5.0, 1.0],
            'weights': [
                [0.0, 2.0, 1.0], # [변경] Rotor 0.5 -> 1.0 (스케일 고정)
                [0.5, 1.0, 0.5], 
                [1.0, 0.2, 0.1]
            ]
        }
    },
    {
        "name": "Exp12_TwoStage_Coarse",
        # [전략] "Level 2를 두 번 쪼개서 실행" (Coarse -> Coarse_Fine -> Fine)
        # 1차(Level 2): 조금 높은 LR(0.008)로 과감하게 접근
        # 2차(Level 2): 낮은 LR(0.003)로 Level 2 안에서 정밀 안착
        # 3차(Level 1), 4차(Level 0): 기존대로 진행
        "config": {
            'levels': [2, 2, 1, 0], # [핵심] Level 2가 두 번 들어감!
            'iters': [60, 60, 50, 30], # 횟수 분배 (60+60 = 120회)
            'base_lrs': [0.008, 0.003, 0.002, 0.001], # LR 스케줄링 효과 (Fast -> Slow)
            'angle_boost': [20.0, 20.0, 5.0, 1.0],
            'weights': [
                [0.0, 2.0, 0.5], # 1차: 방향만 봄
                [0.1, 2.0, 0.5], # 2차: 픽셀(SDF) 살짝 추가해서 고정
                [0.5, 1.0, 0.5], 
                [1.0, 0.2, 0.1]
            ]
        }
    }
]

# 공통 설정
IMG_PATH = "./img/val2017/000000000632.jpg"
MODEL_PATH = "./checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_ANGLES = [30, 45, 60, -30, -45, -60] 
OUTPUT_CSV = "mpc_tuning_results.csv"

# ==============================================================================
# Helper Functions
# ==============================================================================
def invert_affine_norm(matrix_2x3):
    if isinstance(matrix_2x3, np.ndarray): matrix_2x3 = torch.from_numpy(matrix_2x3)
    row_bottom = torch.tensor([0., 0., 1.], device=matrix_2x3.device).unsqueeze(0)
    if matrix_2x3.dim() == 2:
        mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=0)
        return torch.inverse(mat_3x3)[:2, :]
    else:
        B = matrix_2x3.shape[0]
        row_bottom = row_bottom.repeat(B, 1, 1)
        mat_3x3 = torch.cat([matrix_2x3, row_bottom], dim=1)
        return torch.linalg.inv(mat_3x3)[:, :2, :]

def normalize_rotor_output(cos, sin):
    mag = torch.sqrt(cos**2 + sin**2 + 1e-6)
    return cos/mag, sin/mag

def to_pixel(W_norm, w, h):
    N = np.array([[2.0/w, 0, -1], [0, 2.0/h, -1], [0, 0, 1]])
    N_inv = np.linalg.inv(N)
    W_n = W_norm.squeeze().cpu().numpy()
    W_aug = np.vstack([W_n, [0, 0, 1]])
    return (N_inv @ W_aug @ N)[:2, :]

def get_correspondence_error(W_pred, W_gt, w, h):
    x = np.linspace(w*0.2, w*0.8, 10)
    y = np.linspace(h*0.2, h*0.8, 10)
    xv, yv = np.meshgrid(x, y)
    pts = np.vstack([xv.flatten(), yv.flatten(), np.ones_like(xv.flatten())])
    
    pt_pred = W_pred @ pts
    pt_gt = W_gt @ pts
    
    dist = np.linalg.norm(pt_pred - pt_gt, axis=0)
    return np.mean(dist)

# ==============================================================================
# Main Tuning Loop
# ==============================================================================
def run_tuning():
    print(f"🧪 Starting MPC Parameter Tuning on {DEVICE} (No Pandas Mode)...")
    
    # 1. Load Fixed Models
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        except:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        print("✅ Pretrained models loaded.")
    else:
        print("⚠️ No checkpoint found!")
        return

    embedder.eval()
    transformer.eval()
    
    # 2. Load Image
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print("❌ Image not found")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))
    h, w = img_rgb.shape[:2]
    
    results = []

    # 3. Experiment Loop
    for exp in EXPERIMENTS:
        exp_name = exp['name']
        config = exp['config']
        
        print(f"\n🔄 Running Experiment: {exp_name}")
        
        # [Monkey Patching] 설정 교체
        phase4.MPC_CONFIG = config
        refiner = HierarchicalMPCRefiner(device=DEVICE)
        
        # Angle Loop
        for angle in TEST_ANGLES:
            # A. Prepare Data
            M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
            M_gt_aug = np.vstack([M_warp, [0, 0, 1]])
            W_gt_pixel = np.linalg.inv(M_gt_aug)[:2, :] 
            
            # B. Phase 1~3
            prep = MathGeometricPreprocessor()
            pyr_a = prep.process_pyramid(img_warped, levels=5) 
            pyr_b = prep.process_pyramid(img_rgb, levels=5)
            
            with torch.no_grad():
                p2_a = embedder(pyr_a, DEVICE)
                p2_b = embedder(pyr_b, DEVICE)
                res_p3 = transformer(p2_a, p2_b)
                
                finest = res_p3[0]
                avg_rotor = finest['rotor_map'].mean(dim=(1, 2))
                cos, sin, dx, dy = avg_rotor[0]
                cos_t, sin_t = normalize_rotor_output(cos, sin)
                row1 = torch.stack([cos_t, -sin_t, dx])
                row2 = torch.stack([sin_t, cos_t, dy])
                W_p3_norm = torch.stack([row1, row2]).unsqueeze(0)
            
            # C. Phase 4
            start_time = time.time()
            W_p3_inv = invert_affine_norm(W_p3_norm)
            
            try:
                W_mpc_inv, _ = refiner.optimize(p2_a, p2_b, W_init=W_p3_inv)
                W_mpc_norm = invert_affine_norm(W_mpc_inv).detach()
                mpc_time = time.time() - start_time
                
                # D. Evaluate
                W_p3_pix = to_pixel(W_p3_norm, w, h)
                W_mpc_pix = to_pixel(W_mpc_norm, w, h)
                
                err_p3 = get_correspondence_error(W_p3_pix, W_gt_pixel, w, h)
                err_mpc = get_correspondence_error(W_mpc_pix, W_gt_pixel, w, h)
                improvement = err_p3 - err_mpc
                
                # Log Data (Dict)
                log_entry = {
                    "Experiment": exp_name,
                    "Angle": angle,
                    "P3_Error": round(float(err_p3), 2),
                    "P4_Error": round(float(err_mpc), 2),
                    "Improvement": round(float(improvement), 2),
                    "Time_sec": round(mpc_time, 3),
                    "Iters": str(config['iters']),
                    "Boost": str(config['angle_boost'])
                }
                results.append(log_entry)
                
                print(f"   Angle {angle:3d}° | P3: {err_p3:6.2f} -> P4: {err_mpc:6.2f} | Imp: {improvement:6.2f}")
                
            except Exception as e:
                print(f"   Angle {angle:3d}° | Failed: {str(e)}")

    if not results:
        print("No results generated.")
        return

    # 4. Save to CSV (Using built-in csv module)
    print(f"\n💾 Saving results to {OUTPUT_CSV}...")
    headers = results[0].keys()
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    # 5. Summary Analysis (Manual Aggregation without Pandas)
    print("\n📊 Summary (Average Improvement by Experiment):")
    
    # 데이터 집계
    summary = defaultdict(lambda: {'P4_Error': 0.0, 'Improvement': 0.0, 'Time_sec': 0.0, 'count': 0})
    
    for row in results:
        exp = row['Experiment']
        summary[exp]['P4_Error'] += row['P4_Error']
        summary[exp]['Improvement'] += row['Improvement']
        summary[exp]['Time_sec'] += row['Time_sec']
        summary[exp]['count'] += 1
    
    # 표 출력
    print("-" * 85)
    print(f"{'Experiment Name':<30} | {'Avg P4 Err':<12} | {'Avg Improv':<12} | {'Avg Time':<10}")
    print("-" * 85)
    
    for exp_name, stats in summary.items():
        count = stats['count']
        avg_err = stats['P4_Error'] / count
        avg_imp = stats['Improvement'] / count
        avg_time = stats['Time_sec'] / count
        
        print(f"{exp_name:<30} | {avg_err:<12.2f} | {avg_imp:<12.2f} | {avg_time:<10.3f}s")
    print("-" * 85)

if __name__ == "__main__":
    run_tuning()