import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import sys

# 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 라이브러리 체크
try:
    import kornia
    from kornia.feature import LoFTR
    HAS_KORNIA = True
except ImportError:
    print("❌ Kornia not found. LoFTR will be skipped.")
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
IMG_SIZE_LOFTR = (640, 480) 
# 테스트할 각도 범위 (0 ~ 180도, 10도 간격)
ANGLES = list(range(-60, 60, 10))

# ==============================================================================
# [Inference Functions]
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

def run_user_model_angle(embedder, transformer, img_src, img_tgt, device):
    """User Model: Return predicted angle (deg)"""
    preprocessor = MathGeometricPreprocessor()
    # 내부 리사이즈 (256x256)
    src_s = cv2.resize(img_src, (256, 256))
    tgt_s = cv2.resize(img_tgt, (256, 256))
    
    pyr_c = preprocessor.process_pyramid(src_s, levels=5)
    pyr_t = preprocessor.process_pyramid(tgt_s, levels=5)
    
    with torch.no_grad():
        f_c = embedder(pyr_c, device)
        f_t = embedder(pyr_t, device)
        results = transformer(f_c, f_t)
        rotor = results[0]['rotor_map'].mean(dim=(1,2))
        cos, sin = rotor[0,0].item(), rotor[0,1].item()
        return -np.degrees(np.arctan2(sin, cos))

def run_loftr_angle(matcher, img_src, img_tgt, device):
    """LoFTR: Return predicted angle (deg) or None if failed"""
    # Grayscale + Resize
    src_g = cv2.cvtColor(cv2.resize(img_src, IMG_SIZE_LOFTR), cv2.COLOR_RGB2GRAY)
    tgt_g = cv2.cvtColor(cv2.resize(img_tgt, IMG_SIZE_LOFTR), cv2.COLOR_RGB2GRAY)
    
    t_src = torch.from_numpy(src_g).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    t_tgt = torch.from_numpy(tgt_g).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        res = matcher({"image0": t_src, "image1": t_tgt})
        mkpts0 = res['keypoints0'].cpu().numpy()
        mkpts1 = res['keypoints1'].cpu().numpy()
        
    if len(mkpts0) < 10: return None
    
    M, _ = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    if M is None: return None
    
    return -np.degrees(np.arctan2(-M[0,1], M[0,0]))

# ==============================================================================
# [Main]
# ==============================================================================
def main():
    print(f"🚀 Robustness Curve Comparison on {DEVICE}")
    
    # 1. 이미지 로드
    img_list = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if not img_list: return
    img_path = np.random.choice(img_list)
    print(f"   Target Image: {os.path.basename(img_path)}")
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 결과 저장소: {모델이름: [에러리스트]}
    curves = {}
    
    # 2. 모델 리스트업
    models_to_test = []
    
    # (1) LoFTR
    if HAS_KORNIA:
        models_to_test.append(('LoFTR', 'loftr'))
        
    # (2) Checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    for ckpt in ckpt_files:
        name = os.path.basename(ckpt).replace('.pth', '')
        models_to_test.append((name, ckpt))
        
    # 3. 평가 루프
    for name, path_or_type in models_to_test:
        print(f"   Processing {name}...")
        errors = []
        
        # 모델 로드 (메모리 절약을 위해 루프 안에서 로드/해제)
        model = None
        is_loftr = (path_or_type == 'loftr')
        
        if is_loftr:
            model = LoFTR(pretrained='outdoor').to(DEVICE).eval()
        else:
            embedder, transformer = load_user_model(path_or_type, DEVICE)
            model = (embedder, transformer)
            if embedder is None:
                print(f"     ⚠️ Failed to load {name}")
                continue

        # 각도 루프
        for angle in ANGLES:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_rot = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            try:
                if is_loftr:
                    pred = run_loftr_angle(model, img_rot, img_rgb, DEVICE)
                else:
                    pred = run_user_model_angle(model[0], model[1], img_rot, img_rgb, DEVICE)
                
                if pred is not None:
                    # 오차 계산 (0~180도, 주기성 고려 안함 - 절대 오차)
                    # 입력이 angle이면 예측은 angle이어야 함 (부호 보정됨)
                    diff = abs(angle - pred)
                    # 360도 주기성 고려 (예: 350도 vs 10도 -> 20도 차이)
                    diff = min(diff, 360 - diff)
                    errors.append(diff)
                else:
                    errors.append(180.0) # Fail
            except Exception:
                errors.append(180.0)
        
        curves[name] = errors
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()

    # 4. 시각화
    plt.figure(figsize=(12, 8))
    
    # 스타일 지정
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    
    for i, (name, errs) in enumerate(curves.items()):
        marker = markers[i % len(markers)]
        color = colors[i]
        
        # LoFTR나 Baseline은 점선으로, 내 모델은 실선으로
        linestyle = '--' if 'LoFTR' in name or 'ViT' in name else '-'
        linewidth = 3 if 'best' in name or 'last' in name else 1.5
        alpha = 1.0 if 'best' in name or 'last' in name else 0.7
        
        plt.plot(ANGLES, errs, label=name, marker=marker, 
                 linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha)

    plt.title(f"Rotation Robustness Curves (0° - 180°)\nImage: {os.path.basename(img_path)}")
    plt.xlabel("Input Rotation Angle (deg)")
    plt.ylabel("Angle Error (deg)")
    plt.ylim(-5, 185)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 범례 밖으로
    plt.tight_layout()
    
    save_path = "robustness_curves_all.png"
    plt.savefig(save_path)
    print(f"✅ Comparison plot saved to {save_path}")
    # plt.show() # 서버 환경이면 주석 유지

if __name__ == "__main__":
    main()