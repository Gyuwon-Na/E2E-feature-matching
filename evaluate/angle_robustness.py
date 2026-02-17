"""
================================================================================
Rotation Robustness Comparison - 실제 설치 가능 버전
================================================================================
사용자 체크포인트와 실제로 설치 가능한 SOTA 모델들을 비교합니다.

비교 모델 (7개 + 사용자 체크포인트):
✅ LoFTR (Kornia)
✅ RoMa
✅ DKM
✅ Aspanformer
✅ TopicFM
✅ RoRD
✅ PMatch
✅ User Checkpoints

설치 방법은 REAL_INSTALLATION_GUIDE.md 참조
================================================================================
"""

import glob
import os
import sys
import gc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# ==============================================================================
# [Configuration]
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
IMG_DIR = "./img/val2017"
ANGLES = list(range(-180, 181, 15))  # -180° ~ 180°, 15° 간격

# 모델별 활성화 토글
MODEL_TOGGLES: Dict[str, bool] = {
    "LoFTR": False,          # Kornia (pip install kornia)
    "RoMa": True,           # PyPI (pip install romatch)
    "DKM": False,            # Git (pip install git+...)
    "Aspanformer": False,   # 수동 설치 필요
    "TopicFM": False,       # 수동 설치 필요
    "RoRD": False,          # 수동 설치 필요
    "PMatch": False,        # 수동 설치 필요
    "Checkpoints": False,    # 사용자 체크포인트
}

# 모델별 최적 입력 크기 (메모리 효율)
MODEL_INPUT_SIZES: Dict[str, Tuple[int, int]] = {
    "LoFTR": (640, 480),
    "RoMa": (560, 560),
    "DKM": (512, 384),
    "Aspanformer": (640, 480),
    "TopicFM": (640, 480),
    "RoRD": (512, 512),
    "PMatch": (640, 480),
    "Checkpoints": (256, 256),
}


# ==============================================================================
# [Phase Modules Import]
# ==============================================================================
try:
    from phase1 import MathGeometricPreprocessor
    from phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
    from phase3 import Phase3Transformer, FEATURE_DIM
    HAS_PHASE_MODULES = True
except ImportError:
    print("⚠️  Phase modules not found. User checkpoints will be skipped.")
    HAS_PHASE_MODULES = False


@dataclass
class ModelEntry:
    """모델 정보 저장"""
    name: str
    model_type: str  # loftr | roma | dkm | aspanformer | topicfm | rord | pmatch | checkpoint
    source: str      # checkpoint path or identifier


# ==============================================================================
# [Common Helpers]
# ==============================================================================
def circular_angle_error(true_deg: float, pred_deg: float) -> float:
    """회전각 주기성을 고려한 최소 오차"""
    return abs((true_deg - pred_deg + 180.0) % 360.0 - 180.0)


def estimate_angle_from_keypoints(mkpts0: np.ndarray, mkpts1: np.ndarray) -> Optional[float]:
    """매칭점 쌍으로부터 회전각 추정"""
    # 입력 검증
    if mkpts0 is None or mkpts1 is None:
        return None
    
    # numpy array로 변환
    if not isinstance(mkpts0, np.ndarray):
        mkpts0 = np.array(mkpts0)
    if not isinstance(mkpts1, np.ndarray):
        mkpts1 = np.array(mkpts1)
    
    # 차원 확인
    if mkpts0.ndim != 2 or mkpts1.ndim != 2:
        return None
    
    # 점 개수 확인 (최소 10개 필요)
    if len(mkpts0) < 10 or len(mkpts1) < 10:
        return None
    
    # 같은 개수인지 확인
    if len(mkpts0) != len(mkpts1):
        min_len = min(len(mkpts0), len(mkpts1))
        mkpts0 = mkpts0[:min_len]
        mkpts1 = mkpts1[:min_len]
    
    # shape 확인 (N, 2)
    if mkpts0.shape[1] != 2 or mkpts1.shape[1] != 2:
        return None
    
    # dtype 확인
    mkpts0 = mkpts0.astype(np.float32)
    mkpts1 = mkpts1.astype(np.float32)
    
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            mkpts0, mkpts1, 
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99,
            maxIters=2000
        )
        
        if M is None:
            return None
        
        # 회전 행렬에서 각도 추출
        return -np.degrees(np.arctan2(-M[0, 1], M[0, 0]))
        
    except cv2.error as e:
        # OpenCV 에러 발생 시 None 반환
        return None


def safe_resize(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Aspect ratio 유지하며 리사이즈"""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if new_w != target_w or new_h != target_h:
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        resized = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )
    
    return resized


def clear_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==============================================================================
# [User Checkpoint Model]
# ==============================================================================
def load_user_model(path: str, device: str):
    """사용자 체크포인트 모델 로드"""
    if not HAS_PHASE_MODULES:
        print(f"     ⚠️  Phase modules not available")
        return None, None
    
    if not os.path.exists(path):
        print(f"     ❌ Checkpoint not found: {path}")
        return None, None
    
    try:
        print(f"     📂 Loading checkpoint: {os.path.basename(path)}")
        
        embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
        transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
        
        ckpt = torch.load(path, map_location=device, weights_only=False)
        
        # 키 확인
        if "embedder" not in ckpt or "transformer" not in ckpt:
            print(f"     ❌ Invalid checkpoint format")
            print(f"        Expected keys: ['embedder', 'transformer']")
            print(f"        Found keys: {list(ckpt.keys())}")
            return None, None
        
        embedder.load_state_dict(ckpt["embedder"])
        transformer.load_state_dict(ckpt["transformer"])
        embedder.eval()
        transformer.eval()
        
        print(f"     ✅ Checkpoint loaded successfully")
        return embedder, transformer
        
    except Exception as e:
        print(f"     ❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_user_model_angle(embedder, transformer, img_src, img_tgt, device: str) -> Optional[float]:
    """사용자 모델로 회전각 추정"""
    if not HAS_PHASE_MODULES:
        return None
    
    preprocessor = MathGeometricPreprocessor()
    
    src_s = cv2.resize(img_src, (256, 256))
    tgt_s = cv2.resize(img_tgt, (256, 256))

    pyr_src = preprocessor.process_pyramid(src_s, levels=5)
    pyr_tgt = preprocessor.process_pyramid(tgt_s, levels=5)

    try:
        with torch.no_grad():
            feat_src = embedder(pyr_src, device)
            feat_tgt = embedder(pyr_tgt, device)
            results = transformer(feat_src, feat_tgt)
            rotor = results[0]["rotor_map"].mean(dim=(1, 2))
            
            cos_v, sin_v = rotor[0, 0].item(), rotor[0, 1].item()
            return -np.degrees(np.arctan2(sin_v, cos_v))
    except Exception as e:
        print(f"     ⚠️  User model error: {e}")
        return None


# ==============================================================================
# [1. LoFTR - Kornia]
# ==============================================================================
def load_loftr(device: str):
    """LoFTR 모델 로드"""
    try:
        from kornia.feature import LoFTR
        matcher = LoFTR(pretrained="outdoor").to(device).eval()
        return matcher
    except Exception as e:
        print(f"     ⚠️  LoFTR load failed: {e}")
        print(f"     → Install: pip install kornia")
        return None


def run_loftr_angle(matcher, img_src, img_tgt, device: str) -> Optional[float]:
    """LoFTR로 회전각 추정"""
    target_size = MODEL_INPUT_SIZES["LoFTR"]
    
    src_g = cv2.cvtColor(safe_resize(img_src, target_size), cv2.COLOR_RGB2GRAY)
    tgt_g = cv2.cvtColor(safe_resize(img_tgt, target_size), cv2.COLOR_RGB2GRAY)

    t_src = torch.from_numpy(src_g).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    t_tgt = torch.from_numpy(tgt_g).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            res = matcher({"image0": t_src, "image1": t_tgt})
            mkpts0 = res["keypoints0"].cpu().numpy()
            mkpts1 = res["keypoints1"].cpu().numpy()
        return estimate_angle_from_keypoints(mkpts0, mkpts1)
    except Exception as e:
        print(f"     ⚠️  LoFTR inference error: {e}")
        return None


# ==============================================================================
# [2. RoMa - Robust Matcher]
# ==============================================================================
def load_roma(device: str):
    """RoMa 모델 로드"""
    try:
        from romatch import roma_outdoor
        matcher = roma_outdoor(device=device)
        return matcher
    except Exception as e:
        print(f"     ⚠️  RoMa load failed: {e}")
        print(f"     → Install: pip install romatch")
        return None


def run_roma_angle(matcher, img_src, img_tgt, device: str) -> Optional[float]:
    """RoMa로 회전각 추정 (Dense Correspondence)"""
    import tempfile
    
    target_size = MODEL_INPUT_SIZES["RoMa"]
    
    src_resized = safe_resize(img_src, target_size)
    tgt_resized = safe_resize(img_tgt, target_size)
    
    H, W = src_resized.shape[:2]
    
    try:
        # RoMa는 파일 경로를 입력으로 받음 - 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_src:
            cv2.imwrite(tmp_src.name, cv2.cvtColor(src_resized, cv2.COLOR_RGB2BGR))
            src_path = tmp_src.name
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_tgt:
            cv2.imwrite(tmp_tgt.name, cv2.cvtColor(tgt_resized, cv2.COLOR_RGB2BGR))
            tgt_path = tmp_tgt.name
        
        try:
            with torch.no_grad():
                # RoMa API: match() -> sample() -> to_pixel_coordinates()
                warp, certainty = matcher.match(src_path, tgt_path, device=device)
                
                # sample() 메서드 사용 (공식 API)
                matches, match_certainty = matcher.sample(warp, certainty)
                
                # to_pixel_coordinates() 사용
                kptsA, kptsB = matcher.to_pixel_coordinates(matches, H, W, H, W)
                
                # numpy로 변환
                mkpts0 = kptsA.cpu().numpy()
                mkpts1 = kptsB.cpu().numpy()
                
                # 높은 certainty만 사용
                certainty_vals = match_certainty.cpu().numpy()
                threshold = np.percentile(certainty_vals, 70)  # 상위 30%
                
                mask = certainty_vals > threshold
                mkpts0 = mkpts0[mask]
                mkpts1 = mkpts1[mask]
                
                if len(mkpts0) < 10:
                    # 임계값 낮춤
                    threshold = np.percentile(certainty_vals, 50)
                    mask = certainty_vals > threshold
                    mkpts0 = kptsA.cpu().numpy()[mask]
                    mkpts1 = kptsB.cpu().numpy()[mask]
                
                if len(mkpts0) < 10:
                    return None
                
                return estimate_angle_from_keypoints(mkpts0, mkpts1)
                
        finally:
            # 임시 파일 삭제
            import os
            if os.path.exists(src_path):
                os.unlink(src_path)
            if os.path.exists(tgt_path):
                os.unlink(tgt_path)
                
    except Exception as e:
        print(f"     ⚠️  RoMa inference error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# [3. DKM - Dense Kernel Matching]
# ==============================================================================
def load_dkm(device: str):
    """DKM 모델 로드"""
    try:
        # DKMv3 사용 (DKM의 최신 버전)
        from dkm import DKMv3_outdoor
        matcher = DKMv3_outdoor(device=device)
        return matcher
    except Exception as e:
        print(f"     ⚠️  DKM load failed: {e}")
        print(f"     → Install: cd ~/DKM && pip install --user -e .")
        return None


def run_dkm_angle(matcher, img_src, img_tgt, device: str) -> Optional[float]:
    """DKM으로 회전각 추정 (Dense Correspondence)"""
    import tempfile
    
    target_size = MODEL_INPUT_SIZES["DKM"]
    
    src_resized = safe_resize(img_src, target_size)
    tgt_resized = safe_resize(img_tgt, target_size)
    
    try:
        # DKM도 파일 경로를 입력으로 받을 수 있음 - 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_src:
            cv2.imwrite(tmp_src.name, cv2.cvtColor(src_resized, cv2.COLOR_RGB2BGR))
            src_path = tmp_src.name
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_tgt:
            cv2.imwrite(tmp_tgt.name, cv2.cvtColor(tgt_resized, cv2.COLOR_RGB2BGR))
            tgt_path = tmp_tgt.name
        
        try:
            with torch.no_grad():
                # DKM API: match(im_A_path, im_B_path, device)
                dense_matches, certainty = matcher.match(src_path, tgt_path, device=device)
                
                B, H, W, _ = dense_matches.shape
                certainty_map = certainty[0].cpu().numpy()
                
                # 임계값 조정 (RoMa와 동일한 로직)
                certainty_max = certainty_map.max()
                certainty_mean = certainty_map.mean()
                
                if certainty_max < 0.1:
                    threshold = certainty_mean
                else:
                    threshold = np.percentile(certainty_map, 70)  # 90 -> 70
                
                mask = certainty_map > threshold
                
                y_coords, x_coords = np.where(mask)
                num_matches = len(x_coords)
                
                if num_matches < 10:
                    threshold = np.percentile(certainty_map, 50)
                    mask = certainty_map > threshold
                    y_coords, x_coords = np.where(mask)
                    num_matches = len(x_coords)
                
                if num_matches < 10:
                    return None
                
                num_samples = min(100, num_matches)
                indices = np.random.choice(num_matches, num_samples, replace=False)
                mkpts0 = np.stack([x_coords[indices], y_coords[indices]], axis=1).astype(np.float32)
                
                matches_np = dense_matches[0].cpu().numpy()
                mkpts1 = matches_np[y_coords[indices], x_coords[indices], :]
                
                return estimate_angle_from_keypoints(mkpts0, mkpts1)
        finally:
            # 임시 파일 삭제
            import os
            if os.path.exists(src_path):
                os.unlink(src_path)
            if os.path.exists(tgt_path):
                os.unlink(tgt_path)
                
    except Exception as e:
        print(f"     ⚠️  DKM inference error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# [4-7. 수동 설치 모델들 - Placeholder]
# ==============================================================================
def load_manual_install_model(model_name: str, device: str):
    """
    Aspanformer / TopicFM / RoRD / PMatch는 수동 설치 필요
    
    설치 방법:
    1. REAL_INSTALLATION_GUIDE.md 참조
    2. 각 모델의 GitHub 레포지토리 클론
    3. Conda 환경 생성 및 가중치 다운로드
    4. 아래 함수에 로드 코드 추가
    """
    print(f"     ⚠️  {model_name} requires manual installation")
    print(f"     → See REAL_INSTALLATION_GUIDE.md for instructions")
    return None


# ==============================================================================
# [Model List Builder]
# ==============================================================================
def build_models_to_test() -> List[ModelEntry]:
    """테스트할 모델 리스트 생성"""
    models_to_test: List[ModelEntry] = []

    if MODEL_TOGGLES.get("LoFTR", False):
        models_to_test.append(ModelEntry("LoFTR", "loftr", "loftr"))

    if MODEL_TOGGLES.get("RoMa", False):
        models_to_test.append(ModelEntry("RoMa", "roma", "roma"))

    if MODEL_TOGGLES.get("DKM", False):
        models_to_test.append(ModelEntry("DKM", "dkm", "dkm"))

    # 수동 설치 모델들
    for model_name in ["Aspanformer", "TopicFM", "RoRD", "PMatch"]:
        if MODEL_TOGGLES.get(model_name, False):
            models_to_test.append(ModelEntry(model_name, "manual", model_name.lower()))

    # 사용자 체크포인트
    if MODEL_TOGGLES.get("Checkpoints", False) and HAS_PHASE_MODULES:
        ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
        for ckpt in ckpt_files:
            ckpt_name = os.path.basename(ckpt).replace(".pth", "")
            models_to_test.append(ModelEntry(ckpt_name, "checkpoint", ckpt))

    return models_to_test


# ==============================================================================
# [Main]
# ==============================================================================
def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🚀 Rotation Robustness Comparison")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # 이미지 로드
    img_list = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if not img_list:
        print(f"❌ No images found in {IMG_DIR}")
        return

    img_path = np.random.choice(img_list)
    print(f"📷 Test Image: {os.path.basename(img_path)}\n")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Failed to load image")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 모델 리스트 구성
    curves: Dict[str, List[float]] = {}
    models_to_test = build_models_to_test()
    
    if not models_to_test:
        print("❌ No models enabled. Check MODEL_TOGGLES in the script.")
        return
    
    print(f"📋 Models to test: {len(models_to_test)}")
    for entry in models_to_test:
        print(f"   - {entry.name} ({entry.model_type})")
    print()

    # 각 모델별로 테스트
    for entry in models_to_test:
        print(f"🔄 Processing {entry.name}...")
        errors: List[float] = []
        model_obj = None
        infer_fn: Optional[Callable] = None

        try:
            # 모델 로드
            if entry.model_type == "loftr":
                model_obj = load_loftr(DEVICE)
                if model_obj is None:
                    continue
                infer_fn = lambda src, tgt, dev: run_loftr_angle(model_obj, src, tgt, dev)

            elif entry.model_type == "roma":
                model_obj = load_roma(DEVICE)
                if model_obj is None:
                    continue
                infer_fn = lambda src, tgt, dev: run_roma_angle(model_obj, src, tgt, dev)

            elif entry.model_type == "dkm":
                model_obj = load_dkm(DEVICE)
                if model_obj is None:
                    continue
                infer_fn = lambda src, tgt, dev: run_dkm_angle(model_obj, src, tgt, dev)

            elif entry.model_type == "manual":
                model_obj = load_manual_install_model(entry.name, DEVICE)
                if model_obj is None:
                    continue

            elif entry.model_type == "checkpoint":
                embedder, transformer = load_user_model(entry.source, DEVICE)
                if embedder is None:
                    continue
                model_obj = (embedder, transformer)
                infer_fn = lambda src, tgt, dev: run_user_model_angle(model_obj[0], model_obj[1], src, tgt, dev)

            else:
                print(f"     ⚠️  Unknown type: {entry.model_type}")
                continue

            # 각도별 테스트
            for i, angle in enumerate(ANGLES):
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_rot = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)

                pred = None
                try:
                    pred = infer_fn(img_rot, img_rgb, DEVICE)
                except Exception as e:
                    pred = None

                if pred is None:
                    errors.append(180.0)
                else:
                    errors.append(circular_angle_error(angle, pred))
                
                if (i + 1) % 5 == 0:
                    avg_error = np.mean(errors)
                    print(f"     Progress: {i+1}/{len(ANGLES)}, avg error: {avg_error:.2f}°")

            curves[entry.name] = errors
            avg_error = np.mean(errors)
            print(f"     ✅ Completed: avg = {avg_error:.2f}° (±{np.std(errors):.2f}°)\n")

        except Exception as e:
            print(f"     ❌ Fatal error: {e}\n")
            continue

        finally:
            del model_obj
            clear_memory()

    # 결과 확인
    if not curves:
        print("❌ No model was successfully evaluated.")
        return

    # 시각화
    print(f"📊 Generating comparison plot...")
    plt.figure(figsize=(16, 9))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "x", "+"]
    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))

    for i, (name, errs) in enumerate(curves.items()):
        marker = markers[i % len(markers)]
        color = colors[i]

        is_baseline = name in {"LoFTR", "RoMa", "DKM", "Aspanformer", "TopicFM", "RoRD", "PMatch"}
        linestyle = "--" if is_baseline else "-"
        linewidth = 2.8 if ("best" in name.lower() or "last" in name.lower()) else 1.8

        plt.plot(
            ANGLES,
            errs,
            label=f"{name} (avg={np.mean(errs):.2f}°)",
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            alpha=0.9,
            markersize=5,
        )

    plt.title(f"Rotation Robustness Curves\nImage: {os.path.basename(img_path)}", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Input Rotation Angle (degrees)", fontsize=14)
    plt.ylabel("Angle Error (degrees)", fontsize=14)
    plt.ylim(-5, 185)
    plt.xlim(-185, 185)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    save_path = "robustness_curves_all.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Comparison plot saved to {save_path}\n")
    
    # 결과 요약
    print("=" * 80)
    print("📈 Summary (Sorted by Average Error)")
    print("=" * 80)
    sorted_results = sorted(curves.items(), key=lambda x: np.mean(x[1]))
    for rank, (name, errs) in enumerate(sorted_results, 1):
        avg_err = np.mean(errs)
        std_err = np.std(errs)
        print(f"   {rank}. {name:20s}: {avg_err:6.2f}° (±{std_err:5.2f}°)")
    print("=" * 80)


if __name__ == "__main__":
    main()