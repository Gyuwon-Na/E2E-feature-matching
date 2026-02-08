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
NUM_EVAL_IMAGES = 5        # 공정한 평가를 위해 5장 평균
ANGLES = list(range(-60, 60, 10)) # 0 ~ 180도, 15도 간격

# ==============================================================================
# [Inference Helper]
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
    # Resize to standard size for consistent metric
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
    
    # LoFTR 예측 각도 추출
    pred_deg = -np.degrees(np.arctan2(-M[0,1], M[0,0]))
    
    diff = abs(true_angle - pred_deg)
    diff = min(diff, 360 - diff)
    return diff

# ==============================================================================
# [Main Logic]
# ==============================================================================
def main():
    print(f"🏆 Finding the BEST Model on {DEVICE}")
    print(f"   Test Config: {NUM_EVAL_IMAGES} images × {len(ANGLES)} angles (0~180°)")
    
    # 1. 이미지 데이터셋 준비
    all_images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if len(all_images) < NUM_EVAL_IMAGES:
        print("❌ Not enough images.")
        return
    
    # 랜덤 샘플링 (고정 시드)
    np.random.seed(42)
    test_images = np.random.choice(all_images, NUM_EVAL_IMAGES, replace=False)
    
    # 메모리에 로드
    loaded_images = []
    for path in test_images:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        loaded_images.append(img)

    # 2. 평가할 모델 리스트업
    models_info = [] # (name, type, path/object)
    
    # (A) LoFTR
    if HAS_KORNIA:
        models_info.append(('LoFTR (SOTA)', 'loftr', None))
    
    # (B) My Checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    for ckpt in ckpt_files:
        name = os.path.basename(ckpt)
        models_info.append((name, 'user', ckpt))
        
    # 3. 대규모 채점 시작
    leaderboard = {} # {name: mean_error}
    model_curves = {} # {name: [error_at_0, error_at_15, ...]}
    
    for name, mtype, path in models_info:
        print(f"\nEvaluating {name}...")
        
        # 모델 로드
        model_obj = None
        if mtype == 'loftr':
            model_obj = LoFTR(pretrained='outdoor').to(DEVICE).eval()
        else:
            emb, trans = load_user_model(path, DEVICE)
            if emb is None: 
                print(f"   Skipping {name} (Load failed)")
                continue
            model_obj = (emb, trans)
            
        # 모든 이미지 & 모든 각도에 대해 테스트
        total_errors = []
        angle_errors = {a: [] for a in ANGLES} # 각도별 에러 평균용
        
        for img in tqdm(loaded_images, leave=False):
            h, w = img.shape[:2]
            for angle in ANGLES:
                # 회전
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
                # 추론
                err = 180.0
                try:
                    if mtype == 'loftr':
                        err = get_error_loftr(model_obj, img_rot, img, angle, DEVICE)
                    else:
                        err = get_error_user(model_obj[0], model_obj[1], img_rot, img, angle, DEVICE)
                except:
                    err = 180.0 # Error Handling
                
                total_errors.append(err)
                angle_errors[angle].append(err)
        
        # 점수 집계
        mean_mae = np.mean(total_errors)
        leaderboard[name] = mean_mae
        
        # 곡선 데이터 저장 (각도별 평균)
        curve = [np.mean(angle_errors[a]) for a in ANGLES]
        model_curves[name] = curve
        
        print(f"   👉 Mean Error: {mean_mae:.2f}°")
        
        # 메모리 해제
        del model_obj
        torch.cuda.empty_cache()

    # 4. 순위 발표
    print("\n" + "="*50)
    print("🥇 FINAL LEADERBOARD (Lower is Better)")
    print("="*50)
    
    # 오름차순 정렬 (에러 낮은 순) - Pure Python Sort
    sorted_lb = sorted(leaderboard.items(), key=lambda item: item[1])
    
    best_model_name = sorted_lb[0][0]
    
    for rank, (name, score) in enumerate(sorted_lb):
        prefix = "🥇" if rank == 0 else "🥈" if rank == 1 else "🥉" if rank == 2 else f"{rank+1}."
        print(f"{prefix} {name:<30} : {score:.4f}° (Avg Error)")
        
    print("="*50)
    print(f"🎉 The Winner is [{best_model_name}]!")

    # 5. 그래프 그리기 (상위 3개 + LoFTR만)
    top_models = [n for n, s in sorted_lb[:3]] # Top 3
    if 'LoFTR (SOTA)' in leaderboard and 'LoFTR (SOTA)' not in top_models:
        top_models.append('LoFTR (SOTA)') # 비교를 위해 LoFTR 추가
        
    plt.figure(figsize=(12, 7))
    for name in top_models:
        curve = model_curves[name]
        
        linewidth = 4 if name == best_model_name else 2
        linestyle = '--' if 'LoFTR' in name else '-'
        marker = '*' if name == best_model_name else 'o'
        
        plt.plot(ANGLES, curve, label=f"{name} (Avg: {leaderboard[name]:.1f}°)", 
                 linewidth=linewidth, linestyle=linestyle, marker=marker)
        
    plt.title(f"Performance Curve: Top Models vs Rotation\n(Winner: {best_model_name})")
    plt.xlabel("Rotation Angle (deg)")
    plt.ylabel("Mean Angle Error (deg)")
    plt.ylim(0, 100) # 가독성을 위해 Y축 제한
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('best_model_ranking.png')
    print("✅ Performance graph saved to 'best_model_ranking.png'")

if __name__ == "__main__":
    main()