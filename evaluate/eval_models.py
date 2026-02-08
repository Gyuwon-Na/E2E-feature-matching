import os
import cv2
import numpy as np
import torch
import glob
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM

# ==============================================================================
# [Configuration]
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "./checkpoints/best_model.pth"
LAST_MODEL_PATH = "./checkpoints/last_model.pth" # 또는 v5_60deg_last_model.pth 등 실제 파일명 확인
IMG_DIR = "./img/val2017"
NUM_TEST_SAMPLES = 10
ANGLE_MIN = 45
ANGLE_MAX = 60

def load_model(model_path, device):
    """모델 로드 함수"""
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    if os.path.exists(model_path):
        # map_location을 사용하여 CPU/CUDA 호환성 확보
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        embedder.eval()
        transformer.eval()
        print(f"✅ Loaded: {model_path}")
        return embedder, transformer
    else:
        print(f"❌ Model not found: {model_path}")
        return None, None

def evaluate_single_image(embedder, transformer, img_path, angle, device):
    """단일 이미지에 대한 Phase 3 각도 오차 계산"""
    # 이미지 로드
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))
    h, w = img_rgb.shape[:2]
    
    # 이미지 회전 (Input 생성)
    M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 전처리
    preprocessor = MathGeometricPreprocessor()
    pyr_c = preprocessor.process_pyramid(img_warped, levels=5)
    pyr_t = preprocessor.process_pyramid(img_rgb, levels=5)
    
    with torch.no_grad():
        f_c = embedder(pyr_c, device)
        f_t = embedder(pyr_t, device)
        
        # Phase 3 추론
        results = transformer(f_c, f_t)
        rotor = results[0]['rotor_map'].mean(dim=(1,2))
        cos, sin = rotor[0,0].item(), rotor[0,1].item()
        
        # 예측 각도 (Phase 3는 복원 각도를 예측함 -> 부호가 반대여야 정상)
        pred_rad = np.arctan2(sin, cos)
        pred_deg = np.degrees(pred_rad)
        
        # 오차 계산: |입력각도 + 예측각도| (예: 입력 30도, 예측 -28도 -> 오차 2도)
        error = abs(angle + pred_deg)
        
    return error

def main():
    print(f"🚀 Model Comparison (Best vs Last) on {DEVICE}")
    print(f"   Samples: {NUM_TEST_SAMPLES}")
    
    # 1. 테스트 데이터셋 준비
    img_list = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if not img_list:
        print("❌ No images found.")
        return
        
    # 랜덤 50장 선택
    if len(img_list) > NUM_TEST_SAMPLES:
        test_images = np.random.choice(img_list, NUM_TEST_SAMPLES, replace=False)
    else:
        test_images = img_list
    
    # 테스트 케이스 고정 (동일한 조건 비교를 위해 미리 생성)
    test_cases = []
    for img_path in test_images:
        angle = np.random.uniform(*[(-ANGLE_MAX, -ANGLE_MIN), (ANGLE_MIN, ANGLE_MAX)][np.random.randint(2)])
        test_cases.append((img_path, angle))
        
    print(f"   Prepared {len(test_cases)} test cases.")

    # 2. Best Model 평가
    print("\n--- Evaluating Best Model ---")
    emb_best, trans_best = load_model(BEST_MODEL_PATH, DEVICE)
    errors_best = []
    
    if emb_best:
        for i, (path, ang) in enumerate(test_cases):
            err = evaluate_single_image(emb_best, trans_best, path, ang, DEVICE)
            if err is not None:
                errors_best.append(err)
            if (i+1) % 10 == 0: print(f"   Processing {i+1}/{len(test_cases)}...")
            
        # 메모리 정리 (VRAM 확보)
        del emb_best, trans_best
        torch.cuda.empty_cache()

    # 3. Last Model 평가
    print("\n--- Evaluating Last Model ---")
    emb_last, trans_last = load_model(LAST_MODEL_PATH, DEVICE)
    errors_last = []
    
    if emb_last:
        for i, (path, ang) in enumerate(test_cases):
            err = evaluate_single_image(emb_last, trans_last, path, ang, DEVICE)
            if err is not None:
                errors_last.append(err)
            if (i+1) % 10 == 0: print(f"   Processing {i+1}/{len(test_cases)}...")

    # 4. 결과 비교 리포트
    print("\n" + "="*60)
    print("📢 Final Comparison Report")
    print("="*60)
    
    mean_best = np.mean(errors_best) if errors_best else float('inf')
    std_best = np.std(errors_best) if errors_best else 0.0
    
    mean_last = np.mean(errors_last) if errors_last else float('inf')
    std_last = np.std(errors_last) if errors_last else 0.0

    print(f"🏆 Best Model ({os.path.basename(BEST_MODEL_PATH)})")
    print(f"   Mean Error: {mean_best:.4f}°")
    print(f"   Std Dev:    {std_best:.4f}°")
    print("-" * 60)
    print(f"🕒 Last Model ({os.path.basename(LAST_MODEL_PATH)})")
    print(f"   Mean Error: {mean_last:.4f}°")
    print(f"   Std Dev:    {std_last:.4f}°")
    print("="*60)
    
    diff = mean_last - mean_best
    if diff > 0:
        print(f"🚀 Best Model is better by {diff:.4f}°")
    elif diff < 0:
        print(f"🚀 Last Model is better by {-diff:.4f}°")
    else:
        print("Both models perform identically.")

if __name__ == "__main__":
    main()