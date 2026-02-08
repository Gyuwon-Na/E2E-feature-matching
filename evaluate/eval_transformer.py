"""
================================================================================
Evaluation Script: Geometric Matching Transformer
================================================================================
기능:
1. Angle Error (각도 오차) 측정
2. MACE (Mean Average Corner Error) 측정 (Pixel 단위)
3. 체스판(Chessboard) 시각화 및 워핑 결과 저장

실행 방법:
python3 eval_transformer.py --checkpoint ./checkpoints/best_model.pth --img_dir ./img/val2017
================================================================================
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# 프로젝트 모듈 임포트
from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM
from train.fine_tune import GeometricRotationDataset, collate_fn_geometric, IMG_SIZE
from train.losses import normalize_rotor_output

# =============================================================================
# [Helper] 유틸리티 함수
# =============================================================================

def create_chessboard(height, width, block_size=32):
    """시각화를 위한 체스판 이미지 생성"""
    check = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if (x // block_size + y // block_size) % 2 == 0:
                check[y:y+block_size, x:x+block_size] = [255, 255, 255]
            else:
                check[y:y+block_size, x:x+block_size] = [0, 0, 0]
    return check

def rotor_to_affine_matrix(cos, sin, dx, dy, width, height):
    """
    모델 출력(Normalized Rotor) -> Pixel 단위 Affine Matrix (2x3) 변환
    
    모델의 dx, dy는 정규화된 좌표계([-1, 1]) 기준이므로 
    픽셀 좌표계([0, W], [0, H])로 변환해야 OpenCV에서 사용 가능합니다.
    """
    # 1. Rotation Matrix (2x2)
    R = np.array([[cos, -sin], [sin, cos]])
    
    # 2. Translation (Normalized -> Pixel)
    # 정규 좌표계에서 (dx, dy) 이동은 픽셀 좌표계에서 (dx*W/2, dy*H/2)에 해당
    tx = dx * (width / 2.0)
    ty = dy * (height / 2.0)
    
    # 3. Center Correction
    # 회전이 이미지 중심(W/2, H/2) 기준이 되도록 보정
    # T_final = T_move + Center - R @ Center
    center = np.array([width / 2.0, height / 2.0])
    offset = center - R @ center
    
    final_tx = tx + offset[0]
    final_ty = ty + offset[1]
    
    M = np.zeros((2, 3))
    M[:2, :2] = R
    M[0, 2] = final_tx
    M[1, 2] = final_ty
    
    return M

def compute_mace(pred_M, gt_M, width, height):
    """
    MACE (Mean Average Corner Error) 계산
    이미지의 네 모서리가 예측된 변환과 실제 변환에 의해 이동된 위치의 차이(유클리드 거리) 평균
    """
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T  # (3, 4)
    
    pred_pts = pred_M @ corners  # (2, 4)
    gt_pts = gt_M @ corners      # (2, 4)
    
    distances = np.linalg.norm(pred_pts - gt_pts, axis=0)
    return np.mean(distances)

# =============================================================================
# [Evaluator] 평가 클래스
# =============================================================================

# ... (앞부분 임포트 및 create_chessboard, rotor_to_affine_matrix, compute_mace 함수는 그대로 유지) ...

# =============================================================================
# [Evaluator] 평가 클래스
# =============================================================================

class GeometricEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {self.device}")
        
        # 모델 초기화
        self.embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(self.device)
        self.transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(self.device)
        
        self.load_checkpoint(checkpoint_path)
        
        self.embedder.eval()
        self.transformer.eval()

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        print(f"📥 Loading checkpoint: {path}")
        # [수정됨] weights_only=False 추가 (PyTorch 2.6+ 호환)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.embedder.load_state_dict(ckpt['embedder'])
        self.transformer.load_state_dict(ckpt['transformer'])
        
        conf = ckpt.get('training_config', {})
        print(f"   Training Config: {conf}")

    def evaluate(self, dataloader, dataset, vis_dir=None):
        all_results = []  # (mace, angle_err, sample_idx, ...) 저장용 리스트
        
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            print(f"📂 Visualization will be saved to: {vis_dir}")

        # [메모리 정리]
        torch.cuda.empty_cache()

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                # 1. 데이터 로드
                pyramid_a = batch['pyramid_a']
                pyramid_b = batch['pyramid_b']
                gt_angle_rad = batch['gt_angle'].numpy()
                
                with torch.amp.autocast('cuda'):
                    # 2. 모델 추론
                    phase2_a = self.embedder(pyramid_a, self.device)
                    phase2_b = self.embedder(pyramid_b, self.device)
                    results = self.transformer(phase2_a, phase2_b)
                
                # 3. 결과 파싱
                finest_res = results[0] 
                dense_rotor = finest_res['rotor_map']
                avg_rotor = dense_rotor.mean(dim=(1, 2))
                
                cos_raw, sin_raw = avg_rotor[:, 0], avg_rotor[:, 1]
                dx_raw, dy_raw = avg_rotor[:, 2], avg_rotor[:, 3]
                
                cos_pred, sin_pred = normalize_rotor_output(cos_raw, sin_raw)
                
                # CPU 변환
                cos_pred = cos_pred.float().cpu().numpy()
                sin_pred = sin_pred.float().cpu().numpy()
                dx_pred = dx_raw.float().cpu().numpy()
                dy_pred = dy_raw.float().cpu().numpy()
                
                # 배치 내 샘플 루프
                for i in range(len(gt_angle_rad)):
                    # A. Angle Error
                    pred_angle_rad = np.arctan2(sin_pred[i], cos_pred[i])
                    gt_deg = np.degrees(gt_angle_rad[i])
                    pred_deg = np.degrees(pred_angle_rad)
                    
                    angle_diff = np.abs(pred_deg - gt_deg)
                    angle_diff = np.minimum(angle_diff, 360 - angle_diff)
                    
                    # B. MACE
                    M_gt_pixel = cv2.getRotationMatrix2D(
                        (IMG_SIZE[1]/2, IMG_SIZE[0]/2), -gt_deg, 1.0
                    ) 
                    M_pred_pixel = rotor_to_affine_matrix(
                        cos_pred[i], sin_pred[i], 
                        dx_pred[i], dy_pred[i], 
                        IMG_SIZE[1], IMG_SIZE[0]
                    )
                    
                    mace = compute_mace(M_pred_pixel, M_gt_pixel, IMG_SIZE[1], IMG_SIZE[0])
                    
                    # 결과 저장
                    global_idx = idx * dataloader.batch_size + i
                    all_results.append({
                        'idx': global_idx,
                        'mace': mace,
                        'angle_err': angle_diff,
                        'gt_deg': gt_deg,
                        'pred_deg': pred_deg,
                        'M_pred': M_pred_pixel
                    })

        # 4. 통계 계산
        maces = [r['mace'] for r in all_results]
        angles = [r['angle_err'] for r in all_results]
        
        mean_ang = np.mean(angles)
        std_ang = np.std(angles)
        mean_mace = np.mean(maces)
        std_mace = np.std(maces)

        # 5. 시각화 저장
        if vis_dir:
            # MACE 기준 내림차순 정렬
            sorted_results = sorted(all_results, key=lambda x: x['mace'], reverse=True)
            
            # (1) Worst 10개 저장
            print("\n📸 Saving Worst 10 cases (Real Images)...")
            for rank, item in enumerate(sorted_results[:10]):
                data_item = dataset[item['idx']]
                real_img_b = data_item['img_b']
                
                save_name = f"rank{rank+1}_WORST_idx{item['idx']}_mace{item['mace']:.1f}.png"
                self.visualize_alignment(
                    real_img_b, item['M_pred'], item['gt_deg'], item['pred_deg'], item['mace'], 
                    vis_dir, filename=save_name
                )
            
            # (2) Best 5개 저장 (수정된 부분)
            print("📸 Saving Best 5 cases...")
            for rank, item in enumerate(sorted_results[-5:]):
                data_item = dataset[item['idx']]
                real_img_b = data_item['img_b'] # 여기서도 이미지를 가져와야 함!
                
                save_name = f"rank{rank+1}_BEST_idx{item['idx']}_mace{item['mace']:.1f}.png"
                self.visualize_alignment(
                    real_img_b, item['M_pred'], item['gt_deg'], item['pred_deg'], item['mace'], 
                    vis_dir, filename=save_name
                )

        return mean_ang, std_ang, mean_mace, std_mace
    
    def visualize_alignment(self, target_img_rgb, M_pred, gt_angle, pred_angle, mace, save_dir, filename=None):
        """실제 이미지를 사용하여 시각화"""
        h, w = target_img_rgb.shape[:2]
        
        # 1. GT 각도로 회전된 Source 이미지 생성 (입력 상황 재현)
        M_gt_inv = cv2.getRotationMatrix2D((w/2, h/2), -gt_angle, 1.0)
        source_img = cv2.warpAffine(target_img_rgb, M_gt_inv, (w, h))
        
        # 2. 모델 예측으로 복원 (Source -> Pred Matrix -> Corrected)
        corrected_img = cv2.warpAffine(source_img, M_pred, (w, h))
        
        # 3. 오버레이 (Target=Green, Corrected=Magenta)
        gray_target = cv2.cvtColor(target_img_rgb, cv2.COLOR_RGB2GRAY)
        gray_corrected = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2GRAY)
        
        overlay = np.zeros_like(target_img_rgb)
        overlay[:, :, 1] = gray_target      # Green
        overlay[:, :, 0] = gray_corrected   # Red
        overlay[:, :, 2] = gray_corrected   # Blue
        
        # 원본 보기 좋게 합치기
        vis_img = np.hstack([source_img, overlay, target_img_rgb])
        
        # 저장
        plt.figure(figsize=(15, 5))
        plt.imshow(vis_img)
        plt.title(f"MACE: {mace:.2f}px | GT: {gt_angle:.1f}° | Pred: {pred_angle:.1f}°")
        plt.axis('off')
        
        if filename is None:
            filename = "result.png"
        
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
        
# =============================================================================
# [Main] 실행
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Geometric Transformer')
    parser.add_argument('--img_dir', type=str, default='./val2017', help='Image directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--vis_dir', type=str, default='./eval_results', help='Output directory for visuals')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to evaluate')
    args = parser.parse_args()

    # 1. Dataset 준비
    dataset = GeometricRotationDataset(
        args.img_dir, 
        is_train=False, 
        max_samples=args.samples,
        rot_min=-60.0, rot_max=60.0
    )
    
    # [수정됨] 8GB VRAM을 위해 batch_size를 8 -> 1로 변경
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn_geometric,
        num_workers=4
    )

    # 2. Evaluator 실행
    evaluator = GeometricEvaluator(args.checkpoint)
    
    print(f"\n🚀 Starting Evaluation on {len(dataset)} samples (Batch Size=1, FP16)...")
    mean_ang, std_ang, mean_mace, std_mace = evaluator.evaluate(dataloader, dataset, args.vis_dir)    
    
    # 3. 결과 출력
    print("\n" + "="*50)
    print("📊 Evaluation Results")
    print("="*50)
    print(f"🎯 Angle Error: {mean_ang:.4f}° ± {std_ang:.4f}°")
    print(f"📏 MACE (px) : {mean_mace:.4f} px ± {std_mace:.4f} px")
    print("="*50)
    print(f"🖼️ Visualizations saved to: {args.vis_dir}")

if __name__ == "__main__":
    main()