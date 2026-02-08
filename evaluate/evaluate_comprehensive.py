import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 라이브러리 체크
try:
    import kornia
    from kornia.feature import LoFTR
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

try:
    from pipeline.phase1 import MathGeometricPreprocessor
    from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
    from pipeline.phase3 import Phase3Transformer, FEATURE_DIM
except ImportError:
    print("❌ Critical: Phase modules not found.")
    sys.exit(1)

# ==============================================================================
# [Configuration]
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
IMG_DIR = "./img/val2017"
NUM_TEST_IMAGES = 100      # 100장이면 통계적으로 충분함
ROTATION_RANGE = (-180, 180) # 전방위 테스트

# ==============================================================================
# [Helper Functions]
# ==============================================================================
def load_user_model(path, device):
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            embedder.load_state_dict(ckpt['embedder'])
            transformer.load_state_dict(ckpt['transformer'])
            embedder.eval()
            transformer.eval()
            return embedder, transformer
        except:
            return None, None
    return None, None

def get_error_user(embedder, transformer, img_src, img_tgt, true_angle, device):
    preprocessor = MathGeometricPreprocessor()
    # User Model Resize
    src_s = cv2.resize(img_src, (256, 256))
    tgt_s = cv2.resize(img_tgt, (256, 256))
    
    pyr_c = preprocessor.process_pyramid(src_s, levels=5)
    pyr_t = preprocessor.process_pyramid(tgt_s, levels=5)
    
    with torch.no_grad():
        f_c = embedder(pyr_c, device)
        f_t = embedder(pyr_t, device)
        res = transformer(f_c, f_t)
        rotor = res[0]['rotor_map'].mean(dim=(1,2))
        cos, sin = rotor[0,0].item(), rotor[0,1].item()
        
        # 예측: 복원 각도 -> 입력 각도 변환 (-pred)
        pred_deg = -np.degrees(np.arctan2(sin, cos))
        
        # 오차 (주기성 고려)
        diff = abs(true_angle - pred_deg)
        diff = min(diff, 360 - diff)
        return diff

def get_error_loftr(matcher, img_src, img_tgt, true_angle, device):
    # Resize for LoFTR
    src_g = cv2.cvtColor(cv2.resize(img_src, (640, 480)), cv2.COLOR_RGB2GRAY)
    tgt_g = cv2.cvtColor(cv2.resize(img_tgt, (640, 480)), cv2.COLOR_RGB2GRAY)
    
    t_src = torch.from_numpy(src_g).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    t_tgt = torch.from_numpy(tgt_g).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        res = matcher({"image0": t_src, "image1": t_tgt})
        mkpts0 = res['keypoints0'].cpu().numpy()
        mkpts1 = res['keypoints1'].cpu().numpy()
        
    if len(mkpts0) < 10: return 180.0 # Fail
    
    M, _ = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    if M is None: return 180.0
    
    pred_deg = -np.degrees(np.arctan2(-M[0,1], M[0,0]))
    
    diff = abs(true_angle - pred_deg)
    diff = min(diff, 360 - diff)
    return diff

# ==============================================================================
# [Main Logic]
# ==============================================================================
def main():
    print(f"📊 Comprehensive Evaluation (Mean is not enough!) on {DEVICE}")
    print(f"   Test Set: {NUM_TEST_IMAGES} images, Range: {ROTATION_RANGE}°")
    
    # 1. 데이터 준비
    all_images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if len(all_images) < NUM_TEST_IMAGES:
        print("❌ Not enough images.")
        return
    
    np.random.seed(42)
    test_images_paths = np.random.choice(all_images, NUM_TEST_IMAGES, replace=False)
    
    # 각 이미지마다 랜덤 각도 하나씩 배정 (고정된 테스트셋 생성)
    test_set = []
    for path in test_images_paths:
        angle = np.random.uniform(*ROTATION_RANGE)
        test_set.append((path, angle))

    # 2. 모델 리스트업
    models_info = []
    if HAS_KORNIA:
        models_info.append(('LoFTR', 'loftr', None))
    
    ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    for ckpt in ckpt_files:
        name = os.path.basename(ckpt)
        models_info.append((name, 'user', ckpt))
        
    # 3. 평가 실행
    results = {} # {name: [error_list]}
    
    for name, mtype, path in models_info:
        print(f"\nEvaluating {name}...")
        
        # 모델 로드
        if mtype == 'loftr':
            model_obj = LoFTR(pretrained='outdoor').to(DEVICE).eval()
        else:
            emb, trans = load_user_model(path, DEVICE)
            if emb is None: continue
            model_obj = (emb, trans)
            
        errors = []
        for path, angle in tqdm(test_set, leave=False):
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # 회전
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # 추론
            try:
                if mtype == 'loftr':
                    err = get_error_loftr(model_obj, img_rot, img, angle, DEVICE)
                else:
                    err = get_error_user(model_obj[0], model_obj[1], img_rot, img, angle, DEVICE)
            except:
                err = 180.0
            
            errors.append(err)
            
        results[name] = np.array(errors)
        
        # 메모리 해제
        del model_obj
        torch.cuda.empty_cache()

    # 4. 종합 리포트 출력
    print("\n" + "="*80)
    print(f"{'Model Name':<30} | {'Mean':<6} | {'Median':<6} | {'Max':<6} | {'<1°(%)':<8} | {'<3°(%)':<8}")
    print("-" * 80)
    
    for name, errs in results.items():
        mean_val = np.mean(errs)
        median_val = np.median(errs)
        max_val = np.max(errs)
        
        # 성공률 (Success Rate)
        acc_1deg = np.mean(errs < 1.0) * 100
        acc_3deg = np.mean(errs < 3.0) * 100
        
        # 하이라이트: Median이 낮고 Max가 낮아야 진짜 좋은 모델
        print(f"{name:<30} | {mean_val:6.2f} | {median_val:6.2f} | {max_val:6.1f} | {acc_1deg:8.1f} | {acc_3deg:8.1f}")
        
    print("="*80)
    
    # 5. CDF 그래프 (성공률 곡선)
    # X축: 허용 오차(Threshold), Y축: 성공률(Accuracy)
    plt.figure(figsize=(10, 6))
    thresholds = np.linspace(0, 30, 100) # 0도 ~ 30도까지 확인
    
    for name, errs in results.items():
        success_rates = [np.mean(errs < t) * 100 for t in thresholds]
        
        # 스타일
        linewidth = 3 if 'best' in name else 1.5
        linestyle = '--' if 'LoFTR' in name else '-'
        
        plt.plot(thresholds, success_rates, label=f"{name} (AUC)", linewidth=linewidth, linestyle=linestyle)
        
    plt.title("Cumulative Success Rate (CDF)")
    plt.xlabel("Error Threshold (deg)")
    plt.ylabel("Success Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.xlim(0, 15) # 15도까지만 확대해서 보여줌 (중요 구간)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('comprehensive_eval.png')
    print("✅ Saved detailed graph to 'comprehensive_eval.png'")

if __name__ == "__main__":
    main()