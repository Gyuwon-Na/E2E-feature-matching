"""
================================================================================
Evaluation Script: LoFTR + RANSAC (Full Dataset)
================================================================================
기능:
1. 전체 데이터셋에 대해 LoFTR 매칭 수행
2. RANSAC으로 Affine Matrix 추정
3. Transformer 모델과 동일한 기준으로 Angle Error & MACE 통계 산출

실행 방법:
python3 eval_transformer_models.py --img_dir ./img/val2017 --samples 10
================================================================================
"""

import os
import argparse
import cv2
import numpy as np
import torch
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# LoFTR 라이브러리 (kornia)
try:
    import kornia
    from kornia.feature import LoFTR
except ImportError:
    print("❌ kornia가 설치되지 않았습니다. 'pip install kornia'를 실행하세요.")
    exit()

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# =============================================================================
# [Configuration] 
# =============================================================================
IMG_SIZE = (640, 480)  # LoFTR는 해상도가 높을수록 좋음 (8의 배수 권장)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANGLE_THRESHOLD = 60.0 # 학습 때와 동일한 각도 범위

# =============================================================================
# [Metrics & Utils]
# =============================================================================

def compute_mace(pred_M, gt_M, width, height):
    """MACE (Mean Average Corner Error) 계산"""
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T
    
    pred_pts = pred_M @ corners
    gt_pts = gt_M @ corners
    return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=0))

def get_angle_from_matrix(M):
    """OpenCV Matrix에서 각도 추출 (-sin 보정 포함)"""
    # M = [[cos, sin, tx], [-sin, cos, ty]] (OpenCV Y-down)
    # 따라서 각도는 arctan2(-M[1,0], M[0,0])
    return np.degrees(np.arctan2(-M[1, 0], M[0, 0]))

def get_ang_diff(a, b):
    """Cyclic Angle Difference"""
    d = abs(a - b) % 360
    return min(d, 360 - d)

# =============================================================================
# [Dataset Loader] 즉석에서 데이터 생성
# =============================================================================
class SimpleEvaluationDataset:
    def __init__(self, img_dir, max_samples=500):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                         glob.glob(os.path.join(img_dir, "*.png"))
        
        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")
            
        # 셔플 후 개수 제한
        np.random.shuffle(self.img_paths)
        self.img_paths = self.img_paths[:max_samples]
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # 1. 이미지 로드
        path = self.img_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return None
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE) # (W, H)
        h, w = img_rgb.shape[:2]
        
        # 2. Random Warp (문제 생성)
        angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
        
        # M_dist: Orig -> Warped
        M_dist = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_warped = cv2.warpAffine(img_rgb, M_dist, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # GT Matrix: Warped -> Orig (정답)
        M_dist_aug = np.vstack([M_dist, [0, 0, 1]])
        M_gt = np.linalg.inv(M_dist_aug)[:2, :]
        
        return {
            'img_target': img_rgb,   # Original
            'img_source': img_warped, # Warped
            'M_gt': M_gt,
            'gt_angle': -angle, # 복원 각도
            'path': path
        }

# =============================================================================
# [Evaluator]
# =============================================================================
class LoFTREvaluator:
    def __init__(self):
        print(f"✅ Loading LoFTR on {DEVICE}...")
        self.matcher = LoFTR(pretrained='outdoor').to(DEVICE)
        self.matcher.eval()
        
    def process_single(self, sample):
        img_source = sample['img_source'] # Warped
        img_target = sample['img_target'] # Original
        
        # 1. Preprocess (Grayscale -> Tensor)
        def preprocess(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tensor = torch.from_numpy(gray / 255.0).float()[None, None].to(DEVICE)
            return tensor

        batch = {
            'image0': preprocess(img_source),
            'image1': preprocess(img_target)
        }
        
        # 2. Inference
        with torch.no_grad():
            output = self.matcher(batch)
            
        mkpts0 = output['keypoints0'].cpu().numpy()
        mkpts1 = output['keypoints1'].cpu().numpy()
        
        # 3. RANSAC Estimation
        if len(mkpts0) < 4:
            return None # 매칭 실패
            
        # Affine 변환 추정
        M_pred, inliers = cv2.estimateAffine2D(mkpts0, mkpts1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if M_pred is None:
            return None
            
        return M_pred, len(mkpts0)

    def evaluate_dataset(self, dataset, save_dir):
        results = []
        fail_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🚀 Evaluating {len(dataset)} samples...")
        
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            if sample is None: continue
            
            # Inference
            res = self.process_single(sample)
            
            if res is None:
                fail_count += 1
                # 실패 시 페널티 부여 (통계 왜곡 방지 위해 제외하거나 Max값 부여)
                # 여기서는 '실패'로 카운트하고 통계에서 제외합니다.
                continue
                
            M_pred, num_matches = res
            M_gt = sample['M_gt']
            h, w = sample['img_target'].shape[:2]
            
            # Metrics
            mace = compute_mace(M_pred, M_gt, w, h)
            
            pred_deg = get_angle_from_matrix(M_pred)
            gt_deg = sample['gt_angle']
            ang_err = get_ang_diff(gt_deg, pred_deg)
            
            results.append({
                'idx': i,
                'mace': mace,
                'ang_err': ang_err,
                'num_matches': num_matches,
                'sample': sample,
                'M_pred': M_pred,
                'pred_deg': pred_deg
            })
            
        # Statistics
        if len(results) == 0:
            print("❌ All samples failed.")
            return

        maces = [r['mace'] for r in results]
        ang_errs = [r['ang_err'] for r in results]
        
        print("\n" + "="*50)
        print("📊 LoFTR Evaluation Summary")
        print("="*50)
        print(f"Samples      : {len(dataset)}")
        print(f"Success      : {len(results)} ({len(results)/len(dataset)*100:.1f}%)")
        print(f"Failed       : {fail_count} ({fail_count/len(dataset)*100:.1f}%)")
        print("-" * 50)
        print(f"🎯 Angle Error: {np.mean(ang_errs):.4f}° ± {np.std(ang_errs):.4f}°")
        print(f"📏 MACE       : {np.mean(maces):.4f} px ± {np.std(maces):.4f} px")
        print("="*50)
        
        # Visualization (Best 3 / Worst 3)
        sorted_res = sorted(results, key=lambda x: x['mace'])
        
        self.save_visuals(sorted_res[:3], save_dir, prefix="BEST")
        self.save_visuals(sorted_res[-3:], save_dir, prefix="WORST")
        print(f"🖼️ Visualizations saved to {save_dir}")

    def save_visuals(self, res_list, save_dir, prefix):
        for rank, item in enumerate(res_list):
            sample = item['sample']
            img_source = sample['img_source']
            img_target = sample['img_target']
            M_pred = item['M_pred']
            h, w = img_target.shape[:2]
            
            # 복원
            img_corr = cv2.warpAffine(img_source, M_pred, (w, h))
            
            # 오버레이 (Target=Green, Pred=Red)
            gray_tgt = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
            gray_cor = cv2.cvtColor(img_corr, cv2.COLOR_RGB2GRAY)
            
            overlay = np.zeros_like(img_target)
            overlay[..., 1] = gray_tgt
            overlay[..., 0] = gray_cor
            overlay[..., 2] = gray_cor
            
            vis = np.hstack([img_source, overlay, img_target])
            
            plt.figure(figsize=(15, 5))
            plt.imshow(vis)
            plt.title(f"[{prefix} #{rank+1}] MACE: {item['mace']:.2f}px | AngErr: {item['ang_err']:.2f}° | Matches: {item['num_matches']}")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"loftr_{prefix}_{rank+1}_mace{item['mace']:.0f}.png"))
            plt.close()

# =============================================================================
# [Main]
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./img/val2017', help='Image Directory')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--save_dir', type=str, default='./loftr_results', help='Output Directory')
    args = parser.parse_args()
    
    # Run
    dataset = SimpleEvaluationDataset(args.img_dir, max_samples=args.samples)
    evaluator = LoFTREvaluator()
    evaluator.evaluate_dataset(dataset, args.save_dir)