"""
================================================================================
Checkpoint Fusion & Self-Evaluation Visualization
================================================================================
기능:
1. 여러 체크포인트 융합 (Angle 특화 + Pixel 특화 = 균형)
2. Self-Evaluation: Transformer 출력 vs MPC 정제 후 비교 시각화
3. 단계별 에러 분석

사용법:
    # 체크포인트 융합
    python eval_self.py --mode fuse --ckpt1 angle_best.pth --ckpt2 pixel_best.pth
    
    # Self-Evaluation 시각화
    python eval_self.py --mode visualize --checkpoint best_model.pth --data_dir ./val2017
================================================================================
"""


import sys
import numpy as np

# [긴급 패치] Colab(NumPy 2.x)에서 만든 모델을 로컬(NumPy 1.x)에서 억지로 열기
# 로컬엔 'numpy._core'가 없으므로, 기존 'numpy.core'를 가리키도록 사기(?)를 칩니다.
try:
    import numpy._core
except ImportError:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# =============================================================================
# [Hyperparameters]
# =============================================================================
DEFAULT_IMG_SIZE = (256, 256)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MPC_ITERATIONS = 100  # 시각화용으로 줄임


# =============================================================================
# [Part 1] Checkpoint Fusion
# =============================================================================

class CheckpointFuser:
    """
    [체크포인트 융합기]
    
    여러 모델의 가중치를 섞어 장점을 결합합니다.
    
    전략:
    1. Simple Average: 단순 평균
    2. Weighted Average: 가중 평균 (성능 기반)
    3. Layer-wise Fusion: 레이어별 다른 비율
    """
    
    def __init__(self):
        self.checkpoints = []
        self.weights = []
        
    def add_checkpoint(self, path: str, weight: float = 1.0, 
                       metrics: Optional[Dict] = None):
        """
        체크포인트 추가
        
        Args:
            path: 체크포인트 경로
            weight: 융합 가중치
            metrics: 성능 지표 (angle_error, pixel_error 등)
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.checkpoints.append({
            'path': path,
            'data': ckpt,
            'metrics': metrics or {}
        })
        self.weights.append(weight)
        print(f"Added: {path} (weight={weight})")
        
    def fuse_simple_average(self) -> Dict:
        """단순 평균 융합"""
        if len(self.checkpoints) < 2:
            raise ValueError("Need at least 2 checkpoints")
        
        merged = {
            'embedder': {},
            'transformer': {},
            'hidden_dim': self.checkpoints[0]['data']['hidden_dim'],
            'feature_dim': self.checkpoints[0]['data']['feature_dim']
        }
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        norm_weights = [w / total_weight for w in self.weights]
        
        # Embedder 융합
        for key in self.checkpoints[0]['data']['embedder']:
            merged['embedder'][key] = sum(
                w * ckpt['data']['embedder'][key] 
                for w, ckpt in zip(norm_weights, self.checkpoints)
            )
        
        # Transformer 융합
        for key in self.checkpoints[0]['data']['transformer']:
            merged['transformer'][key] = sum(
                w * ckpt['data']['transformer'][key]
                for w, ckpt in zip(norm_weights, self.checkpoints)
            )
        
        return merged
    
    def fuse_performance_weighted(self, 
                                   angle_importance: float = 0.5,
                                   pixel_importance: float = 0.5) -> Dict:
        """
        성능 기반 가중 융합
        
        각 체크포인트의 angle/pixel 성능을 기반으로 자동 가중치 계산
        """
        if len(self.checkpoints) < 2:
            raise ValueError("Need at least 2 checkpoints")
        
        # 성능 기반 가중치 계산
        auto_weights = []
        for ckpt in self.checkpoints:
            metrics = ckpt['metrics']
            angle_err = metrics.get('angle_error', 10.0)
            pixel_err = metrics.get('pixel_error', 40.0)
            
            # 에러가 작을수록 높은 가중치
            angle_score = 1.0 / (angle_err + 1e-6)
            pixel_score = 1.0 / (pixel_err + 1e-6)
            
            combined_score = (angle_importance * angle_score + 
                            pixel_importance * pixel_score)
            auto_weights.append(combined_score)
        
        # 정규화
        total = sum(auto_weights)
        norm_weights = [w / total for w in auto_weights]
        
        print("\n[Performance-based Weights]")
        for ckpt, w in zip(self.checkpoints, norm_weights):
            print(f"  {ckpt['path']}: {w:.3f}")
        
        # 기존 weights를 auto_weights로 대체
        original_weights = self.weights
        self.weights = norm_weights
        
        result = self.fuse_simple_average()
        
        # 복원
        self.weights = original_weights
        
        return result
    
    def fuse_layerwise(self, 
                       angle_ckpt_idx: int = 0,
                       pixel_ckpt_idx: int = 1,
                       encoder_from: str = 'angle',
                       decoder_from: str = 'pixel') -> Dict:
        """
        레이어별 선택적 융합
        
        Encoder는 angle 특화 모델에서, Decoder는 pixel 특화 모델에서 가져오기
        """
        if len(self.checkpoints) < 2:
            raise ValueError("Need at least 2 checkpoints")
        
        angle_ckpt = self.checkpoints[angle_ckpt_idx]['data']
        pixel_ckpt = self.checkpoints[pixel_ckpt_idx]['data']
        
        merged = {
            'embedder': {},
            'transformer': {},
            'hidden_dim': angle_ckpt['hidden_dim'],
            'feature_dim': angle_ckpt['feature_dim']
        }
        
        # Embedder: angle 모델 사용 (기하학적 특징 추출)
        merged['embedder'] = angle_ckpt['embedder'].copy()
        
        # Transformer 레이어별 선택
        for key in angle_ckpt['transformer']:
            if 'encoder' in key.lower():
                # Encoder: 지정된 소스에서
                source = angle_ckpt if encoder_from == 'angle' else pixel_ckpt
            elif 'decoder' in key.lower() or 'cross' in key.lower():
                # Decoder/Cross-Attention: 지정된 소스에서
                source = pixel_ckpt if decoder_from == 'pixel' else angle_ckpt
            else:
                # 나머지: 평균
                merged['transformer'][key] = (
                    angle_ckpt['transformer'][key] * 0.5 +
                    pixel_ckpt['transformer'][key] * 0.5
                )
                continue
            
            merged['transformer'][key] = source['transformer'][key].clone()
        
        return merged
    
    def save_fused(self, merged: Dict, save_path: str):
        """융합된 체크포인트 저장"""
        torch.save(merged, save_path)
        print(f"\n✅ Fused checkpoint saved: {save_path}")
        
    def clear(self):
        """초기화"""
        self.checkpoints = []
        self.weights = []


def fuse_checkpoints_interactive():
    """대화형 체크포인트 융합"""
    fuser = CheckpointFuser()
    
    print("=" * 60)
    print("Checkpoint Fusion Tool")
    print("=" * 60)
    
    # 체크포인트 추가
    while True:
        path = input("\nCheckpoint path (or 'done'): ").strip()
        if path.lower() == 'done':
            break
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        angle_err = float(input("  Angle error (deg): ") or "10")
        pixel_err = float(input("  Pixel error (px): ") or "40")
        
        fuser.add_checkpoint(path, metrics={
            'angle_error': angle_err,
            'pixel_error': pixel_err
        })
    
    if len(fuser.checkpoints) < 2:
        print("Need at least 2 checkpoints!")
        return
    
    # 융합 방법 선택
    print("\nFusion methods:")
    print("  1. Simple Average")
    print("  2. Performance-weighted")
    print("  3. Layer-wise")
    
    method = input("Select (1/2/3): ").strip()
    
    if method == '1':
        merged = fuser.fuse_simple_average()
    elif method == '2':
        angle_imp = float(input("Angle importance (0-1): ") or "0.5")
        pixel_imp = float(input("Pixel importance (0-1): ") or "0.5")
        merged = fuser.fuse_performance_weighted(angle_imp, pixel_imp)
    elif method == '3':
        merged = fuser.fuse_layerwise()
    else:
        print("Invalid selection")
        return
    
    # 저장
    save_path = input("\nSave path: ") or "fused_model.pth"
    fuser.save_fused(merged, save_path)


# =============================================================================
# [Part 2] Self-Evaluation Visualizer
# =============================================================================

class SelfEvaluationVisualizer:
    """
    [Self-Evaluation 시각화]
    
    Transformer 출력과 MPC 정제 후를 비교하여
    각 단계의 기여도를 시각화합니다.
    """
    
    def __init__(self, checkpoint_path: str, device: str = DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # 모델 로드
        self._load_models()
        
    def _load_models(self):
        """모델 로드"""
        from phase1 import MathGeometricPreprocessor
        from phase2 import CliffordPyramidEmbedder
        from phase3 import Phase3Transformer
        from phase4 import GeometricMPCRefiner
        
        self.preprocessor = MathGeometricPreprocessor()
        self.embedder = CliffordPyramidEmbedder(hidden_dim=48).to(self.device)
        self.transformer = Phase3Transformer(feature_dim=144, embed_dim=48).to(self.device)
        
        # 체크포인트 로드
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.embedder.load_state_dict(ckpt['embedder'])
            self.transformer.load_state_dict(ckpt['transformer'])
            print(f"Loaded: {self.checkpoint_path}")
        
        self.embedder.eval()
        self.transformer.eval()
        
        # MPC는 매번 새로 초기화
        self.mpc_class = GeometricMPCRefiner
        
    def _add_batch_dim(self, pyramid):
        """배치 차원 추가"""
        batched = []
        for level_data in pyramid:
            batched_level = {}
            for key, value in level_data.items():
                if isinstance(value, np.ndarray):
                    batched_level[key] = value[np.newaxis, ...]
                else:
                    batched_level[key] = value
            batched.append(batched_level)
        return batched
    
    def _normalize_affine(self, M, width, height):
        """픽셀 → 정규화 좌표계 변환"""
        N = np.array([[2.0/width, 0, -1], [0, 2.0/height, -1], [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_aug = np.vstack([M, [0, 0, 1]])
        M_norm = N @ M_aug @ N_inv
        return M_norm[:2]
    
    def _compute_errors(self, pred_H, gt_H, img_size):
        """에러 계산"""
        H, W = img_size
        
        # Corner points
        corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float32).T
        
        if pred_H.shape[0] == 2:
            pred_H = np.vstack([pred_H, [0, 0, 1]])
        if gt_H.shape[0] == 2:
            gt_H = np.vstack([gt_H, [0, 0, 1]])
        
        pred_corners = pred_H @ corners
        gt_corners = gt_H @ corners
        
        pred_corners = pred_corners[:2] / (pred_corners[2:] + 1e-8)
        gt_corners = gt_corners[:2] / (gt_corners[2:] + 1e-8)
        
        # MACE (Mean Average Corner Error)
        mace = np.linalg.norm(pred_corners - gt_corners, axis=0).mean()
        
        # Angle error
        pred_angle = np.arctan2(pred_H[1, 0], pred_H[0, 0])
        gt_angle = np.arctan2(gt_H[1, 0], gt_H[0, 0])
        angle_error = np.abs(np.degrees(pred_angle - gt_angle))
        
        return {
            'mace': mace,
            'angle_error': angle_error,
            'corner_errors': np.linalg.norm(pred_corners - gt_corners, axis=0)
        }
    
    @torch.no_grad()
    def evaluate_single(self, img_rgb: np.ndarray, 
                        angle: float, 
                        use_mpc: bool = True,
                        mpc_iterations: int = MPC_ITERATIONS) -> Dict:
        """
        단일 이미지 평가
        
        Returns:
            결과 딕셔너리 (transformer/mpc 별 예측 및 에러)
        """
        H, W = img_rgb.shape[:2]
        
        # 변환 적용
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        img_warped = cv2.warpAffine(img_rgb, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        
        # GT 역변환
        M_aug = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_aug)[:2]
        gt_H_norm = self._normalize_affine(M_inv, W, H)
        
        # Phase 1
        pyramid_a = self._add_batch_dim(self.preprocessor.process_pyramid(img_warped, levels=4))
        pyramid_b = self._add_batch_dim(self.preprocessor.process_pyramid(img_rgb, levels=4))
        
        # Phase 2
        phase2_a = self.embedder(pyramid_a, self.device)
        phase2_b = self.embedder(pyramid_b, self.device)
        
        # Phase 3 (Transformer)
        results = self.transformer(phase2_a, phase2_b)
        finest = results[0]
        rotor = finest['rotor_map']
        avg_rotor = rotor.mean(dim=(1, 2))
        
        cos_t, sin_t = avg_rotor[0, 0].item(), avg_rotor[0, 1].item()
        dx, dy = avg_rotor[0, 2].item(), avg_rotor[0, 3].item()
        
        # 정규화
        mag = np.sqrt(cos_t**2 + sin_t**2 + 1e-6)
        cos_t, sin_t = cos_t / mag, sin_t / mag
        
        transformer_H = np.array([
            [cos_t, -sin_t, dx],
            [sin_t, cos_t, dy]
        ], dtype=np.float32)
        
        transformer_errors = self._compute_errors(transformer_H, gt_H_norm, (H, W))
        
        result = {
            'img_orig': img_rgb,
            'img_warped': img_warped,
            'gt_angle': angle,
            'gt_H': gt_H_norm,
            'transformer_H': transformer_H,
            'transformer_errors': transformer_errors,
            'phase2_a': phase2_a,
            'phase2_b': phase2_b,
            'pyramid_a': pyramid_a,
            'pyramid_b': pyramid_b,
        }
        
        # Phase 4 (MPC) - no_grad 밖에서 실행
        if use_mpc:
            mpc_H, mpc_errors = self._run_mpc(
                result, transformer_H, gt_H_norm, (H, W), mpc_iterations
            )
            result['mpc_H'] = mpc_H
            result['mpc_errors'] = mpc_errors
        
        return result
    
    def _run_mpc(self, result_dict, initial_H, gt_H, img_size, iterations):
        """MPC 실행 (gradient 필요하므로 별도 메서드)"""
        # MPC는 gradient가 필요
        mpc = self.mpc_class(device=self.device)
        mpc.iterations = iterations
        
        # 초기화
        cos_t, sin_t = initial_H[0, 0], initial_H[1, 0]
        angle = np.arctan2(sin_t, cos_t)
        scale = np.sqrt(cos_t**2 + sin_t**2)
        mpc.global_filtering_init(mean_rotor=angle, mean_scale=scale)
        
        # 데이터 준비 (detach + clone으로 gradient 연결 끊기)
        phase2_a = result_dict['phase2_a']
        phase2_b = result_dict['phase2_b']
        pyramid_a = result_dict['pyramid_a']
        pyramid_b = result_dict['pyramid_b']
        
        s_a, v_a, b_a = phase2_a[0]
        s_b, v_b, b_b = phase2_b[0]
        
        # [수정된 부분] .unsqueeze(1)을 사용하여 (B, C, H, W) 4차원으로 맞춤
        src_data = {
            'sdf': torch.tensor(pyramid_a[0]['sdf']).unsqueeze(1).float().to(self.device),
            'vector': v_a.mean(dim=1).detach().clone(),
            'rotor': b_a[2].mean(dim=1, keepdim=True).detach().clone()
        }
        tgt_data = {
            'sdf': torch.tensor(pyramid_b[0]['sdf']).unsqueeze(1).float().to(self.device),
            'vector': v_b.mean(dim=1).detach().clone(),
            'rotor': b_b[2].mean(dim=1, keepdim=True).detach().clone()
        }
        
        # Gates (detach 필수!)
        g_s = torch.sigmoid(torch.mean(torch.abs(s_a), dim=1, keepdim=True)).detach().clone()
        g_v = torch.sigmoid(torch.mean(torch.norm(v_a, dim=2), dim=1, keepdim=True)).detach().clone()
        g_b = torch.sigmoid(torch.mean(b_a[2], dim=1, keepdim=True)).detach().clone()
        gates = (g_s, g_v, g_b)
        
        with torch.enable_grad():
            try:
                _ = mpc.optimize(src_data, tgt_data, gates)
                mpc_H = mpc.W.detach().cpu().numpy()[0]
            except Exception as e:
                print(f"MPC failed: {e}")
                import traceback
                traceback.print_exc() # 에러 자세히 보기
                mpc_H = initial_H
        
        mpc_errors = self._compute_errors(mpc_H, gt_H, img_size)
        return mpc_H, mpc_errors
    
    def visualize_comparison(self, result: Dict, save_path: Optional[str] = None):
        """
        Transformer vs MPC 비교 시각화
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        img_orig = result['img_orig']
        img_warped = result['img_warped']
        gt_angle = result['gt_angle']
        H, W = img_orig.shape[:2]
        
        # Row 1: 입력 이미지들
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_orig)
        ax1.set_title('Original (Target)', fontsize=11)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_warped)
        ax2.set_title(f'Warped (Source)\nGT: {gt_angle:.1f}°', fontsize=11)
        ax2.axis('off')
        
        # Transformer 결과
        t_H = result['transformer_H']
        if t_H.shape[0] == 2:
            t_H_full = np.vstack([t_H, [0, 0, 1]])
        else:
            t_H_full = t_H
        
        # 픽셀 좌표계로 변환
        N_inv = np.array([[W/2, 0, W/2], [0, H/2, H/2], [0, 0, 1]])
        N = np.linalg.inv(N_inv)
        t_H_pixel = N_inv @ t_H_full @ N
        
        t_aligned = cv2.warpPerspective(img_warped, t_H_pixel, (W, H))
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(t_aligned)
        t_err = result['transformer_errors']
        ax3.set_title(f'Transformer Output\nAngle Err: {t_err["angle_error"]:.2f}°\nMACE: {t_err["mace"]:.2f}px', fontsize=11)
        ax3.axis('off')
        
        # MPC 결과 (있으면)
        if 'mpc_H' in result:
            m_H = result['mpc_H']
            if m_H.shape[0] == 2:
                m_H_full = np.vstack([m_H, [0, 0, 1]])
            else:
                m_H_full = m_H
            
            m_H_pixel = N_inv @ m_H_full @ N
            m_aligned = cv2.warpPerspective(img_warped, m_H_pixel, (W, H))
            
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(m_aligned)
            m_err = result['mpc_errors']
            ax4.set_title(f'After MPC\nAngle Err: {m_err["angle_error"]:.2f}°\nMACE: {m_err["mace"]:.2f}px', fontsize=11)
            ax4.axis('off')
        
        # Row 2: 차이 맵
        ax5 = fig.add_subplot(gs[1, 0:2])
        diff_transformer = np.abs(t_aligned.astype(float) - img_orig.astype(float)).mean(axis=2)
        im5 = ax5.imshow(diff_transformer, cmap='hot', vmin=0, vmax=50)
        ax5.set_title('Difference: Transformer vs Original', fontsize=11)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        if 'mpc_H' in result:
            ax6 = fig.add_subplot(gs[1, 2:4])
            diff_mpc = np.abs(m_aligned.astype(float) - img_orig.astype(float)).mean(axis=2)
            im6 = ax6.imshow(diff_mpc, cmap='hot', vmin=0, vmax=50)
            ax6.set_title('Difference: MPC vs Original', fontsize=11)
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        # Row 3: 코너 에러 비교
        ax7 = fig.add_subplot(gs[2, :])
        
        corners_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        x = np.arange(4)
        width = 0.35
        
        t_corner_errs = result['transformer_errors']['corner_errors']
        bars1 = ax7.bar(x - width/2, t_corner_errs, width, label='Transformer', color='steelblue', alpha=0.8)
        
        if 'mpc_errors' in result:
            m_corner_errs = result['mpc_errors']['corner_errors']
            bars2 = ax7.bar(x + width/2, m_corner_errs, width, label='MPC', color='coral', alpha=0.8)
        
        ax7.set_ylabel('Corner Error (normalized units)')
        ax7.set_title('Per-Corner Error Comparison', fontsize=12)
        ax7.set_xticks(x)
        ax7.set_xticklabels(corners_names)
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)
        
        # 개선율 텍스트
        if 'mpc_errors' in result:
            t_mace = result['transformer_errors']['mace']
            m_mace = result['mpc_errors']['mace']
            improvement = (t_mace - m_mace) / t_mace * 100 if t_mace > 0 else 0
            
            fig.text(0.5, 0.02, 
                    f'MPC Improvement: MACE {t_mace:.2f} → {m_mace:.2f} ({improvement:+.1f}%)',
                    ha='center', fontsize=12, fontweight='bold',
                    color='green' if improvement > 0 else 'red')
        
        plt.suptitle('Self-Evaluation: Transformer vs MPC Refinement', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
    def run_batch_evaluation(self, img_dir: str, 
                              angles: List[float] = None,
                              num_samples: int = 20,
                              save_dir: Optional[str] = None):
        """
        배치 평가 및 통계 시각화
        """
        if angles is None:
            angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        
        img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
        if not img_paths:
            print(f"No images in {img_dir}")
            return
        
        all_results = []
        
        for i in tqdm(range(num_samples), desc="Evaluating"):
            img_path = img_paths[i % len(img_paths)]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, DEFAULT_IMG_SIZE)
            
            angle = angles[i % len(angles)]
            
            result = self.evaluate_single(img, angle, use_mpc=True)
            all_results.append(result)
        
        # 통계 시각화
        self._plot_statistics(all_results, save_dir)
        
        return all_results
    
    def _plot_statistics(self, results: List[Dict], save_dir: Optional[str] = None):
        """통계 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        gt_angles = [r['gt_angle'] for r in results]
        t_angle_errs = [r['transformer_errors']['angle_error'] for r in results]
        t_maces = [r['transformer_errors']['mace'] for r in results]
        
        has_mpc = 'mpc_errors' in results[0]
        if has_mpc:
            m_angle_errs = [r['mpc_errors']['angle_error'] for r in results]
            m_maces = [r['mpc_errors']['mace'] for r in results]
        
        # 1. Angle Error vs GT Angle
        ax1 = axes[0, 0]
        ax1.scatter(gt_angles, t_angle_errs, alpha=0.6, label='Transformer', c='steelblue')
        if has_mpc:
            ax1.scatter(gt_angles, m_angle_errs, alpha=0.6, label='MPC', c='coral')
        ax1.set_xlabel('Ground Truth Angle (°)')
        ax1.set_ylabel('Angle Error (°)')
        ax1.set_title('Angle Error vs GT Angle')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. MACE vs GT Angle
        ax2 = axes[0, 1]
        ax2.scatter(gt_angles, t_maces, alpha=0.6, label='Transformer', c='steelblue')
        if has_mpc:
            ax2.scatter(gt_angles, m_maces, alpha=0.6, label='MPC', c='coral')
        ax2.set_xlabel('Ground Truth Angle (°)')
        ax2.set_ylabel('MACE (normalized)')
        ax2.set_title('MACE vs GT Angle')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Error Distribution (Histogram)
        ax3 = axes[1, 0]
        ax3.hist(t_angle_errs, bins=20, alpha=0.6, label='Transformer', color='steelblue')
        if has_mpc:
            ax3.hist(m_angle_errs, bins=20, alpha=0.6, label='MPC', color='coral')
        ax3.set_xlabel('Angle Error (°)')
        ax3.set_ylabel('Count')
        ax3.set_title('Angle Error Distribution')
        ax3.legend()
        
        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Summary Statistics (n={len(results)})
        {'='*40}
        
        Transformer:
          Angle Error: {np.mean(t_angle_errs):.2f}° ± {np.std(t_angle_errs):.2f}°
          MACE: {np.mean(t_maces):.2f} ± {np.std(t_maces):.2f}
        """
        
        if has_mpc:
            improvement = (np.mean(t_maces) - np.mean(m_maces)) / np.mean(t_maces) * 100
            stats_text += f"""
        After MPC:
          Angle Error: {np.mean(m_angle_errs):.2f}° ± {np.std(m_angle_errs):.2f}°
          MACE: {np.mean(m_maces):.2f} ± {np.std(m_maces):.2f}
          
        Improvement: {improvement:+.1f}%
            """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Self-Evaluation Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'statistics.png'), dpi=150, bbox_inches='tight')
        
        plt.show()


# =============================================================================
# [Main Entry Points]
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Checkpoint Fusion & Self-Evaluation')
    parser.add_argument('--mode', type=str, choices=['fuse', 'visualize', 'batch'],
                        default='visualize', help='Operation mode')
    
    # Fusion arguments
    parser.add_argument('--ckpt1', type=str, help='First checkpoint (angle-optimized)')
    parser.add_argument('--ckpt2', type=str, help='Second checkpoint (pixel-optimized)')
    parser.add_argument('--ckpt3', type=str, help='Third checkpoint (optional)')
    parser.add_argument('--output', type=str, default='fused_model.pth',
                        help='Output path for fused model')
    parser.add_argument('--fusion_method', type=str, 
                        choices=['average', 'performance', 'layerwise'],
                        default='performance', help='Fusion method')
    
    # Visualization arguments
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Model checkpoint for visualization')
    parser.add_argument('--data_dir', type=str, default='./val2017',
                        help='Image directory')
    parser.add_argument('--angle', type=float, default=15.0,
                        help='Test angle for single visualization')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples for batch evaluation')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.mode == 'fuse':
        # 체크포인트 융합
        if not args.ckpt1 or not args.ckpt2:
            print("Need --ckpt1 and --ckpt2 for fusion")
            fuse_checkpoints_interactive()
            return
        
        fuser = CheckpointFuser()
        
        # 체크포인트 추가 (메트릭은 수동 입력 필요)
        fuser.add_checkpoint(args.ckpt1, metrics={'angle_error': 5.0, 'pixel_error': 46.0})
        fuser.add_checkpoint(args.ckpt2, metrics={'angle_error': 10.0, 'pixel_error': 31.0})
        
        if args.ckpt3:
            fuser.add_checkpoint(args.ckpt3, metrics={'angle_error': 7.0, 'pixel_error': 35.0})
        
        if args.fusion_method == 'average':
            merged = fuser.fuse_simple_average()
        elif args.fusion_method == 'performance':
            merged = fuser.fuse_performance_weighted(angle_importance=0.5, pixel_importance=0.5)
        else:
            merged = fuser.fuse_layerwise()
        
        fuser.save_fused(merged, args.output)
        
    elif args.mode == 'visualize':
        # 단일 이미지 시각화
        visualizer = SelfEvaluationVisualizer(args.checkpoint)
        
        # 테스트 이미지 로드
        img_paths = list(Path(args.data_dir).glob('*.jpg'))
        if not img_paths:
            print(f"No images in {args.data_dir}")
            return
        
        img = cv2.imread(str(img_paths[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, DEFAULT_IMG_SIZE)
        
        result = visualizer.evaluate_single(img, args.angle, use_mpc=True)
        
        os.makedirs(args.save_dir, exist_ok=True)
        visualizer.visualize_comparison(
            result, 
            save_path=os.path.join(args.save_dir, f'comparison_angle{args.angle}.png')
        )
        
    elif args.mode == 'batch':
        # 배치 평가
        visualizer = SelfEvaluationVisualizer(args.checkpoint)
        visualizer.run_batch_evaluation(
            args.data_dir,
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )


if __name__ == "__main__":
    main()