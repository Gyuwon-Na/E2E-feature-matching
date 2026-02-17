"""
python3 evaluate/angle_robustness_LoFTR.py  --project_dir /home/gyuwon/E2E-feature-matching  --ckpt checkpoints/rot_90_1.32.pth   --img_dir img/val2017/   --num_samples 30   --topk 800  --conf_thresh 0.2   --trim 0.2
"""

import os, sys, glob, random, argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Utils
# -------------------------
def circular_angle_error(true_deg: float, pred_deg: float) -> float:
    return abs((true_deg - pred_deg + 180.0) % 360.0 - 180.0)

def safe_resize_pad(img: np.ndarray, target_wh):
    tw, th = target_wh
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if nw != tw or nh != th:
        resized = cv2.copyMakeBorder(resized, 0, th-nh, 0, tw-nw, cv2.BORDER_CONSTANT, value=0)
    return resized

def rotate_image(img: np.ndarray, angle_deg: float):
    h, w = img.shape[:2]
    A = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    rot = cv2.warpAffine(img, A, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rot

def weighted_kabsch_2d(p0, p1, w=None):
    """
    Weighted rigid transform (R,t) minimizing ||R p0 + t - p1||^2
    No RANSAC.
    Returns (R(2,2), t(2,), angle_deg) or None
    """
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    n = min(len(p0), len(p1))
    if n < 10:
        return None
    p0 = p0[:n]
    p1 = p1[:n]

    if w is None:
        w = np.ones((n,), dtype=np.float64)
    else:
        w = np.asarray(w, np.float64)[:n]
        w = np.clip(w, 1e-8, None)
    w = w / (w.sum() + 1e-12)

    c0 = (p0 * w[:, None]).sum(axis=0)
    c1 = (p1 * w[:, None]).sum(axis=0)
    X = p0 - c0
    Y = p1 - c1

    H = X.T @ (Y * w[:, None])
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = c1 - R @ c0
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return R, t, angle

def trimmed_kabsch_angle(p0, p1, w=None, trim_ratio=0.2):
    """
    1) weighted Kabsch
    2) compute residuals ||R p0 + t - p1||
    3) keep (1-trim_ratio) best residuals, re-fit
    Still NO RANSAC (deterministic trimming).
    """
    fit1 = weighted_kabsch_2d(p0, p1, w=w)
    if fit1 is None:
        return None
    R, t, ang1 = fit1

    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    n = min(len(p0), len(p1))
    p0 = p0[:n]; p1 = p1[:n]
    if w is None:
        w = np.ones((n,), dtype=np.float64)
    else:
        w = np.asarray(w, np.float64)[:n]

    pred = (p0 @ R.T) + t[None, :]
    res = np.linalg.norm(pred - p1, axis=1)

    keep = max(10, int(n * (1.0 - trim_ratio)))
    idx = np.argsort(res)[:keep]
    fit2 = weighted_kabsch_2d(p0[idx], p1[idx], w=w[idx] if w is not None else None)
    if fit2 is None:
        return ang1
    return fit2[2]

# -------------------------
# Main
# -------------------------
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", DEVICE)

    project_dir = Path(args.project_dir).resolve()
    sys.path.insert(0, str(project_dir))

    # ours imports
    from pipeline.phase1 import MathGeometricPreprocessor
    from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
    from pipeline.phase3 import Phase3Transformer, FEATURE_DIM

    # LoFTR
    from kornia.feature import LoFTR
    loftr = LoFTR(pretrained="outdoor").to(DEVICE).eval()

    # Ours
    ckpt_path = (project_dir / args.ckpt).resolve()
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    embedder.load_state_dict(ckpt["embedder"])
    transformer.load_state_dict(ckpt["transformer"])
    embedder.eval(); transformer.eval()
    preprocessor = MathGeometricPreprocessor()

    # Config
    ANGLES = list(range(-90, 91, args.step))
    COMMON_SIZE = (args.width, args.height)  # SAME for both models (fair)
    OURS_SIZE = (256, 256)                   # 내부 입력은 고정(모델 설계)
    LOFTR_SIZE = COMMON_SIZE                 # LoFTR도 동일 크기에서 평가

    # images
    img_dir = (project_dir / args.img_dir).resolve()
    imgs = glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png"))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images in {img_dir}")

    random.seed(args.seed)
    sample_paths = random.sample(imgs, k=min(args.num_samples, len(imgs)))
    print("N_SAMPLES:", len(sample_paths), "IMG_DIR:", img_dir)

    loftr_angerr_all = []
    ours_angerr_all = []

    for i, pth in enumerate(sample_paths):
        img = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2RGB)

        # FAIR: same resize for both before rotation generation
        img_common = safe_resize_pad(img, COMMON_SIZE)

        loftr_errs = []
        ours_errs  = []

        for ang in ANGLES:
            # rotate in SAME coordinate system
            rot_common = rotate_image(img_common, ang)

            # -------------------------
            # LoFTR angle via (TopK + Trimmed Kabsch)
            # -------------------------
            src_g = cv2.cvtColor(rot_common, cv2.COLOR_RGB2GRAY)
            tgt_g = cv2.cvtColor(img_common, cv2.COLOR_RGB2GRAY)

            t0 = torch.from_numpy(src_g).float().div(255.0)[None, None].to(DEVICE)
            t1 = torch.from_numpy(tgt_g).float().div(255.0)[None, None].to(DEVICE)

            with torch.no_grad():
                out = loftr({"image0": t0, "image1": t1})

            mk0 = out["keypoints0"].detach().cpu().numpy()
            mk1 = out["keypoints1"].detach().cpu().numpy()
            conf = out.get("confidence", None)
            conf = conf.detach().cpu().numpy() if conf is not None else None

            pred_l = None
            if mk0 is not None and len(mk0) >= args.min_matches:
                if conf is not None:
                    # confidence threshold + TopK
                    keep = conf >= args.conf_thresh
                    mk0_f = mk0[keep]
                    mk1_f = mk1[keep]
                    conf_f = conf[keep]
                    if len(mk0_f) >= args.min_matches:
                        order = np.argsort(-conf_f)  # desc
                        order = order[:min(args.topk, len(order))]
                        mk0_f = mk0_f[order]; mk1_f = mk1_f[order]; conf_f = conf_f[order]
                        pred_l = trimmed_kabsch_angle(mk0_f, mk1_f, w=conf_f, trim_ratio=args.trim)
                else:
                    # no confidence -> TopK by count only
                    mk0_f = mk0[:min(args.topk, len(mk0))]
                    mk1_f = mk1[:min(args.topk, len(mk1))]
                    pred_l = trimmed_kabsch_angle(mk0_f, mk1_f, w=None, trim_ratio=args.trim)

            loftr_errs.append(180.0 if pred_l is None else circular_angle_error(ang, pred_l))

            # -------------------------
            # Ours angle (rotor)
            # -------------------------
            # Use SAME rotated image, but feed into Ours pipeline resolution
            rot_o = safe_resize_pad(rot_common, OURS_SIZE)
            ref_o = safe_resize_pad(img_common, OURS_SIZE)

            # ours pipeline expects 256x256
            src_s = cv2.resize(rot_o, OURS_SIZE)
            tgt_s = cv2.resize(ref_o, OURS_SIZE)

            pyr_src = preprocessor.process_pyramid(src_s, levels=5)
            pyr_tgt = preprocessor.process_pyramid(tgt_s, levels=5)

            with torch.no_grad():
                feat_src = embedder(pyr_src, DEVICE)
                feat_tgt = embedder(pyr_tgt, DEVICE)
                results = transformer(feat_src, feat_tgt)
                rotor = results[0]["rotor_map"].mean(dim=(1, 2))
                cos_v, sin_v = rotor[0, 0].item(), rotor[0, 1].item()
                pred_o = -np.degrees(np.arctan2(sin_v, cos_v))

            ours_errs.append(circular_angle_error(ang, pred_o))

        loftr_angerr_all.append(loftr_errs)
        ours_angerr_all.append(ours_errs)

        if (i + 1) % 5 == 0:
            print(f"  processed {i+1}/{len(sample_paths)}")

    loftr_angerr_all = np.array(loftr_angerr_all, dtype=np.float32)
    ours_angerr_all  = np.array(ours_angerr_all,  dtype=np.float32)

    loftr_m = np.mean(loftr_angerr_all, axis=0)
    loftr_s = np.std(loftr_angerr_all, axis=0)
    ours_m  = np.mean(ours_angerr_all,  axis=0)
    ours_s  = np.std(ours_angerr_all,   axis=0)

    # ✅ ONE FIGURE: both curves + bands
    plt.figure(figsize=(14, 7))
    plt.plot(ANGLES, loftr_m, marker="o", markersize=3, label=f"LoFTR (TopK+TrimKabsch) avg={loftr_m.mean():.1f}°")
    plt.fill_between(ANGLES, loftr_m - loftr_s, loftr_m + loftr_s, alpha=0.2)

    plt.plot(ANGLES, ours_m, marker="o", markersize=3, label=f"Ours (rotor) avg={ours_m.mean():.1f}°")
    plt.fill_between(ANGLES, ours_m - ours_s, ours_m + ours_s, alpha=0.2)

    plt.title(f"Rotation Robustness (NO RANSAC): Angle Error (mean±std)\n"
              f"N={len(sample_paths)} | Common={COMMON_SIZE} | topk={args.topk} | conf>={args.conf_thresh} | trim={args.trim}")
    plt.xlabel("Rotation Angle (deg)")
    plt.ylabel("Angle Error (deg)")
    plt.ylim(-5, 185)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--img_dir", type=str, default="val2017")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=10)

    # fair + robust options (NO RANSAC)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--topk", type=int, default=800)
    ap.add_argument("--min_matches", type=int, default=50)
    ap.add_argument("--conf_thresh", type=float, default=0.2)
    ap.add_argument("--trim", type=float, default=0.2)  # 0.2 = drop worst 20% residuals

    args = ap.parse_args()
    main(args)
