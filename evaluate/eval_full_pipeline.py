"""
================================================================================
Full Pipeline Evaluation: Phase 1 → 2 → 3 → 4 (MPC Refiner)
================================================================================
디버깅 버전 - 에러 메시지 상세 출력
================================================================================
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import glob
import time
import argparse
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Project modules
from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM

# Phase 4
from phase4.phase4_2 import GeometricMPCRefiner


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class EvalConfig:
    """Evaluation configuration"""
    img_size: Tuple[int, int] = (256, 256)
    hidden_dim: int = HIDDEN_DIM
    feature_dim: int = FEATURE_DIM
    device: str = 'cuda'
    model_path: str = './checkpoints/rot_90_1.32.pth'
    img_dir: str = './img/val2017'
    output_dir: str = './eval_results'
    
    # Test parameters
    angle_range: Tuple[float, float] = (-60, 60)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    
    # Thresholds
    angle_threshold_excellent: float = 2.0
    angle_threshold_good: float = 5.0
    angle_threshold_acceptable: float = 10.0
    
    pixel_threshold_excellent: float = 3.0
    pixel_threshold_good: float = 5.0
    pixel_threshold_acceptable: float = 10.0


# ==============================================================================
# Utility Functions
# ==============================================================================
def normalize_rotor_output(cos_raw: torch.Tensor, sin_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize rotor output to unit vector"""
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude


def rotor_to_affine_matrix(cos: float, sin: float, dx: float = 0, dy: float = 0, scale: float = 1.0) -> np.ndarray:
    """Convert rotor parameters to 2x3 affine matrix (normalized coordinates)"""
    M = np.zeros((2, 3))
    M[0, 0] = scale * cos
    M[0, 1] = -scale * sin
    M[0, 2] = dx
    M[1, 0] = scale * sin
    M[1, 1] = scale * cos
    M[1, 2] = dy
    return M


def denormalize_affine_matrix(matrix_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized affine matrix to pixel coordinates"""
    N = np.array([
        [2.0 / width, 0, -1],
        [0, 2.0 / height, -1],
        [0, 0, 1]
    ])
    N_inv = np.linalg.inv(N)
    
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    M_pixel_aug = N_inv @ M_norm_aug @ N
    
    return M_pixel_aug[:2, :]


def get_grid_points(width: int, height: int, num_points: int = 10) -> np.ndarray:
    """Generate grid points for evaluation"""
    x = np.linspace(width * 0.15, width * 0.85, num_points)
    y = np.linspace(height * 0.15, height * 0.85, num_points)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=1)


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply affine transformation to points"""
    pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ pts_homo.T).T


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images (simplified version)"""
    try:
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Convert to grayscale if needed
        if img1.ndim == 3:
            img1 = np.mean(img1, axis=2)
        if img2.ndim == 3:
            img2 = np.mean(img2, axis=2)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    except Exception as e:
        print(f"SSIM computation error: {e}")
        return 0.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images"""
    try:
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    except Exception as e:
        print(f"PSNR computation error: {e}")
        return 0.0


# ==============================================================================
# Pipeline Evaluator
# ==============================================================================
class FullPipelineEvaluator:
    """Full pipeline evaluation class"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        print(f"[DEBUG] Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Preprocessor
        self.preprocessor = MathGeometricPreprocessor()
        
        # Phase 4 Refiner
        self.refiner = GeometricMPCRefiner(
            device=self.device,
            config={
                'verbose': False,
                'iterations': 100,
                'patience': 25
            }
        )
    
    def _load_models(self):
        """Load Phase 2 and Phase 3 models"""
        self.embedder = CliffordPyramidEmbedder(hidden_dim=self.config.hidden_dim).to(self.device)
        self.transformer = Phase3Transformer(
            feature_dim=self.config.feature_dim,
            embed_dim=self.config.hidden_dim
        ).to(self.device)
        
        if os.path.exists(self.config.model_path):
            print(f"Loading checkpoint: {self.config.model_path}")
            ckpt = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
            self.embedder.load_state_dict(ckpt['embedder'])
            self.transformer.load_state_dict(ckpt['transformer'])
            print("✅ Models loaded successfully")
        else:
            print(f"⚠️ Checkpoint not found: {self.config.model_path}")
            print("   Using untrained models")
        
        self.embedder.eval()
        self.transformer.eval()
    
    def create_test_case(self, img_rgb: np.ndarray) -> Dict:
        """Create a test case with random transformation"""
        h, w = img_rgb.shape[:2]
        
        # Random angle
        angle = np.random.uniform(*self.config.angle_range)
        
        # Random scale (optional)
        scale = np.random.uniform(*self.config.scale_range)
        
        # Create warped image (source)
        M_forward = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        img_warped = cv2.warpAffine(img_rgb, M_forward, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # GT inverse matrix (what the model should predict)
        M_forward_aug = np.vstack([M_forward, [0, 0, 1]])
        M_gt_pixel = np.linalg.inv(M_forward_aug)[:2, :]
        
        return {
            'img_source': img_warped,
            'img_target': img_rgb,
            'gt_angle': -angle,
            'gt_scale': 1.0 / scale,
            'M_gt_pixel': M_gt_pixel,
            'applied_angle': angle,
            'applied_scale': scale
        }
    
    def run_phase3(self, pyramid_a, pyramid_b) -> Dict:
        """Run Phase 2 + Phase 3"""
        with torch.no_grad():
            # Phase 2
            features_a = self.embedder(pyramid_a, self.device)
            features_b = self.embedder(pyramid_b, self.device)
            
            # Phase 3
            results = self.transformer(features_a, features_b)
            
            # Extract prediction
            avg_rotor = results[0]['rotor_map'].mean(dim=(1, 2))
            cos_raw, sin_raw = avg_rotor[0, 0], avg_rotor[0, 1]
            cos, sin = normalize_rotor_output(cos_raw, sin_raw)
            dx, dy = avg_rotor[0, 2].item(), avg_rotor[0, 3].item()
            
            cos_val = cos.item()
            sin_val = sin.item()
            angle_rad = np.arctan2(sin_val, cos_val)
            scale = np.sqrt(cos_val**2 + sin_val**2)
        
        return {
            'angle_rad': angle_rad,
            'angle_deg': np.degrees(angle_rad),
            'scale': scale,
            'dx': dx,
            'dy': dy,
            'cos': cos_val,
            'sin': sin_val,
            'features_a': features_a,
            'features_b': features_b,
            'results': results
        }
    
    def run_phase4(self, phase3_output: Dict, features_a, features_b) -> Dict:
        """Run Phase 4 refinement"""
        # Initialize from Phase 3
        self.refiner.global_filtering_init(
            mean_rotor=phase3_output['angle_rad'],
            mean_scale=phase3_output['scale']
        )
        
        # Prepare feature dictionaries
        # Get finest level features
        src_sdf, src_vector, src_rotor_tuple = features_a[-1]
        tgt_sdf, tgt_vector, tgt_rotor_tuple = features_b[-1]
        
        # Debug shapes
        # print(f"[DEBUG] src_sdf shape: {src_sdf.shape}")
        # print(f"[DEBUG] src_vector shape: {src_vector.shape}")
        # print(f"[DEBUG] src_rotor_tuple: {type(src_rotor_tuple)}, len={len(src_rotor_tuple) if isinstance(src_rotor_tuple, tuple) else 'N/A'}")
        
        # Combine rotor tuple into tensor
        if isinstance(src_rotor_tuple, tuple):
            src_unit_cos, src_unit_sin, src_magnitude = src_rotor_tuple
            tgt_unit_cos, tgt_unit_sin, tgt_magnitude = tgt_rotor_tuple
            src_rotor = torch.cat([src_unit_cos, src_unit_sin, src_magnitude], dim=1)
            tgt_rotor = torch.cat([tgt_unit_cos, tgt_unit_sin, tgt_magnitude], dim=1)
        else:
            # Already a tensor
            src_rotor = src_rotor_tuple
            tgt_rotor = tgt_rotor_tuple
        
        src_dict = {'sdf': src_sdf, 'vector': src_vector, 'rotor': src_rotor}
        tgt_dict = {'sdf': tgt_sdf, 'vector': tgt_vector, 'rotor': tgt_rotor}
        
        # Gates (uniform for now)
        gates = (
            torch.ones(1, 1, 1, 1, device=self.device),
            torch.ones(1, 1, 1, 1, device=self.device),
            torch.ones(1, 1, 1, 1, device=self.device)
        )
        
        # Optimize
        W_refined, loss_history = self.refiner.optimize(src_dict, tgt_dict, gates)
        
        # Extract parameters
        angle_deg, scale, tx, ty = self.refiner.get_transform_params(W_refined)
        
        return {
            'W': W_refined,
            'angle_deg': angle_deg,
            'angle_rad': np.radians(angle_deg),
            'scale': scale,
            'tx': tx,
            'ty': ty,
            'loss_history': loss_history
        }
    
    def evaluate_single(self, img_rgb: np.ndarray, use_phase4: bool = True) -> Optional[Dict]:
        """Evaluate on a single image"""
        h, w = img_rgb.shape[:2]
        
        # Create test case
        test_case = self.create_test_case(img_rgb)
        
        # Phase 1: Preprocessing
        t_start = time.time()
        pyramid_a = self.preprocessor.process_pyramid(test_case['img_source'], levels=4)
        pyramid_b = self.preprocessor.process_pyramid(test_case['img_target'], levels=4)
        t_phase1 = time.time() - t_start
        
        # Phase 2 + 3
        t_start = time.time()
        phase3_output = self.run_phase3(pyramid_a, pyramid_b)
        t_phase3 = time.time() - t_start
        
        # Phase 3 matrix (normalized -> pixel)
        M_p3_norm = rotor_to_affine_matrix(
            phase3_output['cos'], phase3_output['sin'],
            phase3_output['dx'], phase3_output['dy'],
            phase3_output['scale']
        )
        M_p3_pixel = denormalize_affine_matrix(M_p3_norm, w, h)
        
        # Phase 4 (optional)
        t_phase4 = 0.0
        if use_phase4:
            t_start = time.time()
            phase4_output = self.run_phase4(
                phase3_output,
                phase3_output['features_a'],
                phase3_output['features_b']
            )
            t_phase4 = time.time() - t_start
            
            W_np = phase4_output['W'].detach().cpu().numpy()[0]
            M_p4_pixel = denormalize_affine_matrix(W_np, w, h)
            p4_angle = phase4_output['angle_deg']
            p4_scale = phase4_output['scale']
            loss_history = phase4_output['loss_history']
        else:
            M_p4_pixel = M_p3_pixel
            p4_angle = phase3_output['angle_deg']
            p4_scale = phase3_output['scale']
            loss_history = []
        
        # Compute metrics
        grid_pts = get_grid_points(w, h)
        
        # GT transformation
        dst_gt = transform_points(test_case['M_gt_pixel'], grid_pts)
        
        # Phase 3 transformation
        dst_p3 = transform_points(M_p3_pixel, grid_pts)
        pixel_err_p3 = np.linalg.norm(dst_p3 - dst_gt, axis=1).mean()
        
        # Phase 4 transformation
        dst_p4 = transform_points(M_p4_pixel, grid_pts)
        pixel_err_p4 = np.linalg.norm(dst_p4 - dst_gt, axis=1).mean()
        
        # Angle errors
        angle_err_p3 = abs(phase3_output['angle_deg'] - test_case['gt_angle'])
        angle_err_p4 = abs(p4_angle - test_case['gt_angle'])
        
        # Aligned images
        img_p3_aligned = cv2.warpAffine(
            test_case['img_source'], M_p3_pixel, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )
        img_p4_aligned = cv2.warpAffine(
            test_case['img_source'], M_p4_pixel, (w, h),
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Image quality metrics
        ssim_p3 = compute_ssim(test_case['img_target'], img_p3_aligned)
        ssim_p4 = compute_ssim(test_case['img_target'], img_p4_aligned)
        psnr_p3 = compute_psnr(test_case['img_target'], img_p3_aligned)
        psnr_p4 = compute_psnr(test_case['img_target'], img_p4_aligned)
        
        return {
            # Ground truth
            'gt_angle': test_case['gt_angle'],
            'gt_scale': test_case['gt_scale'],
            'applied_angle': test_case['applied_angle'],
            
            # Phase 3 results
            'p3_angle': phase3_output['angle_deg'],
            'p3_scale': phase3_output['scale'],
            'p3_angle_err': angle_err_p3,
            'p3_pixel_err': pixel_err_p3,
            'p3_ssim': ssim_p3,
            'p3_psnr': psnr_p3,
            
            # Phase 4 results
            'p4_angle': p4_angle,
            'p4_scale': p4_scale,
            'p4_angle_err': angle_err_p4,
            'p4_pixel_err': pixel_err_p4,
            'p4_ssim': ssim_p4,
            'p4_psnr': psnr_p4,
            
            # Improvement
            'angle_improvement': angle_err_p3 - angle_err_p4,
            'pixel_improvement': pixel_err_p3 - pixel_err_p4,
            
            # Runtime
            't_phase1': t_phase1,
            't_phase3': t_phase3,
            't_phase4': t_phase4,
            't_total': t_phase1 + t_phase3 + t_phase4,
            
            # For visualization
            'img_source': test_case['img_source'],
            'img_target': test_case['img_target'],
            'img_p3_aligned': img_p3_aligned,
            'img_p4_aligned': img_p4_aligned,
            'grid_pts': grid_pts,
            'dst_gt': dst_gt,
            'dst_p3': dst_p3,
            'dst_p4': dst_p4,
            'loss_history': loss_history,
            'use_phase4': use_phase4
        }
    
    def evaluate_dataset(self, image_paths: List[str], num_samples: int = None,
                        use_phase4: bool = True) -> List[Dict]:
        """Evaluate on multiple images"""
        if num_samples and num_samples < len(image_paths):
            image_paths = list(np.random.choice(image_paths, num_samples, replace=False))
        
        results = []
        errors = []
        
        for img_path in tqdm(image_paths, desc="Evaluating"):
            # Load image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                errors.append(f"Failed to load: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, self.config.img_size)
            
            # Evaluate
            try:
                result = self.evaluate_single(img_rgb, use_phase4=use_phase4)
                if result is not None:
                    result['image'] = os.path.basename(img_path)
                    results.append(result)
            except Exception as e:
                error_msg = f"Error on {os.path.basename(img_path)}: {str(e)}"
                errors.append(error_msg)
                # Print first few errors in detail
                if len(errors) <= 3:
                    print(f"\n[ERROR] {error_msg}")
                    traceback.print_exc()
                continue
        
        # Print error summary
        if errors:
            print(f"\n⚠️ {len(errors)} errors occurred during evaluation")
            if len(errors) > 3:
                print(f"   (Showing first 3 errors, {len(errors) - 3} more hidden)")
        
        print(f"\n✅ Successfully evaluated {len(results)}/{len(image_paths)} images")
        
        return results


# ==============================================================================
# Analysis and Visualization
# ==============================================================================
def compute_statistics(results: List[Dict], config: EvalConfig) -> Dict:
    """Compute evaluation statistics"""
    n = len(results)
    if n == 0:
        return {}
    
    # Extract arrays
    p3_angle_errs = np.array([r['p3_angle_err'] for r in results])
    p4_angle_errs = np.array([r['p4_angle_err'] for r in results])
    p3_pixel_errs = np.array([r['p3_pixel_err'] for r in results])
    p4_pixel_errs = np.array([r['p4_pixel_err'] for r in results])
    
    improvements_angle = np.array([r['angle_improvement'] for r in results])
    improvements_pixel = np.array([r['pixel_improvement'] for r in results])
    
    ssim_p3 = np.array([r['p3_ssim'] for r in results])
    ssim_p4 = np.array([r['p4_ssim'] for r in results])
    
    runtimes = np.array([r['t_total'] for r in results])
    
    # Success rates
    p3_excellent = (p3_angle_errs < config.angle_threshold_excellent).sum() / n * 100
    p3_good = (p3_angle_errs < config.angle_threshold_good).sum() / n * 100
    p3_acceptable = (p3_angle_errs < config.angle_threshold_acceptable).sum() / n * 100
    
    p4_excellent = (p4_angle_errs < config.angle_threshold_excellent).sum() / n * 100
    p4_good = (p4_angle_errs < config.angle_threshold_good).sum() / n * 100
    p4_acceptable = (p4_angle_errs < config.angle_threshold_acceptable).sum() / n * 100
    
    improved_count = (improvements_angle > 0).sum()
    
    return {
        'num_samples': n,
        
        # Phase 3
        'p3_angle_err_mean': float(p3_angle_errs.mean()),
        'p3_angle_err_std': float(p3_angle_errs.std()),
        'p3_angle_err_median': float(np.median(p3_angle_errs)),
        'p3_pixel_err_mean': float(p3_pixel_errs.mean()),
        'p3_pixel_err_std': float(p3_pixel_errs.std()),
        'p3_ssim_mean': float(ssim_p3.mean()),
        'p3_success_excellent': p3_excellent,
        'p3_success_good': p3_good,
        'p3_success_acceptable': p3_acceptable,
        
        # Phase 4
        'p4_angle_err_mean': float(p4_angle_errs.mean()),
        'p4_angle_err_std': float(p4_angle_errs.std()),
        'p4_angle_err_median': float(np.median(p4_angle_errs)),
        'p4_pixel_err_mean': float(p4_pixel_errs.mean()),
        'p4_pixel_err_std': float(p4_pixel_errs.std()),
        'p4_ssim_mean': float(ssim_p4.mean()),
        'p4_success_excellent': p4_excellent,
        'p4_success_good': p4_good,
        'p4_success_acceptable': p4_acceptable,
        
        # Improvement
        'improvement_rate': improved_count / n * 100,
        'angle_improvement_mean': float(improvements_angle.mean()),
        'pixel_improvement_mean': float(improvements_pixel.mean()),
        
        # Runtime
        'runtime_mean': float(runtimes.mean()),
        'runtime_std': float(runtimes.std())
    }


def print_summary(stats: Dict):
    """Print evaluation summary"""
    if not stats:
        print("No statistics to display")
        return
        
    print("\n" + "=" * 80)
    print("FULL PIPELINE EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nSamples evaluated: {stats['num_samples']}")
    
    print("\n" + "-" * 40)
    print("PHASE 3 (Transformer)")
    print("-" * 40)
    print(f"  Angle Error:  {stats['p3_angle_err_mean']:.2f} +/- {stats['p3_angle_err_std']:.2f} deg (median: {stats['p3_angle_err_median']:.2f} deg)")
    print(f"  Pixel Error:  {stats['p3_pixel_err_mean']:.2f} +/- {stats['p3_pixel_err_std']:.2f} px")
    print(f"  SSIM:         {stats['p3_ssim_mean']:.4f}")
    print(f"  Success Rate: <2 deg: {stats['p3_success_excellent']:.1f}% | <5 deg: {stats['p3_success_good']:.1f}% | <10 deg: {stats['p3_success_acceptable']:.1f}%")
    
    print("\n" + "-" * 40)
    print("PHASE 4 (MPC Refinement)")
    print("-" * 40)
    print(f"  Angle Error:  {stats['p4_angle_err_mean']:.2f} +/- {stats['p4_angle_err_std']:.2f} deg (median: {stats['p4_angle_err_median']:.2f} deg)")
    print(f"  Pixel Error:  {stats['p4_pixel_err_mean']:.2f} +/- {stats['p4_pixel_err_std']:.2f} px")
    print(f"  SSIM:         {stats['p4_ssim_mean']:.4f}")
    print(f"  Success Rate: <2 deg: {stats['p4_success_excellent']:.1f}% | <5 deg: {stats['p4_success_good']:.1f}% | <10 deg: {stats['p4_success_acceptable']:.1f}%")
    
    print("\n" + "-" * 40)
    print("IMPROVEMENT (Phase 3 -> Phase 4)")
    print("-" * 40)
    print(f"  Improved:     {stats['improvement_rate']:.1f}% of samples")
    print(f"  Angle Delta:  {stats['angle_improvement_mean']:+.2f} deg")
    print(f"  Pixel Delta:  {stats['pixel_improvement_mean']:+.2f} px")
    
    print("\n" + "-" * 40)
    print("RUNTIME")
    print("-" * 40)
    print(f"  Total:        {stats['runtime_mean']:.3f} +/- {stats['runtime_std']:.3f} s/image")
    
    print("\n" + "=" * 80)


def plot_results(results: List[Dict], stats: Dict, save_path: str = None):
    """Create visualization plots"""
    if not results:
        print("No results to plot")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    p3_angle_errs = [r['p3_angle_err'] for r in results]
    p4_angle_errs = [r['p4_angle_err'] for r in results]
    p3_pixel_errs = [r['p3_pixel_err'] for r in results]
    p4_pixel_errs = [r['p4_pixel_err'] for r in results]
    gt_angles = [abs(r['gt_angle']) for r in results]
    improvements = [r['angle_improvement'] for r in results]
    
    # 1. Angle Error Comparison
    ax = axes[0, 0]
    ax.boxplot([p3_angle_errs, p4_angle_errs], labels=['Phase 3', 'Phase 4'])
    ax.set_ylabel('Angle Error (deg)')
    ax.set_title('Angle Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. Pixel Error Comparison
    ax = axes[0, 1]
    ax.boxplot([p3_pixel_errs, p4_pixel_errs], labels=['Phase 3', 'Phase 4'])
    ax.set_ylabel('Pixel Error (px)')
    ax.set_title('Pixel Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. Improvement Distribution
    ax = axes[0, 2]
    ax.hist(improvements, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.axvline(np.mean(improvements), color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(improvements):.2f} deg')
    ax.set_xlabel('Angle Improvement (deg)')
    ax.set_ylabel('Count')
    ax.set_title('Phase 4 Improvement Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error vs GT Angle
    ax = axes[1, 0]
    ax.scatter(gt_angles, p3_angle_errs, alpha=0.5, label='Phase 3', c='red')
    ax.scatter(gt_angles, p4_angle_errs, alpha=0.5, label='Phase 4', c='green')
    ax.set_xlabel('Ground Truth Angle (deg)')
    ax.set_ylabel('Angle Error (deg)')
    ax.set_title('Error vs Input Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Phase 3 vs Phase 4 Scatter
    ax = axes[1, 1]
    ax.scatter(p3_angle_errs, p4_angle_errs, alpha=0.5, c='steelblue')
    max_val = max(max(p3_angle_errs), max(p4_angle_errs)) if p3_angle_errs and p4_angle_errs else 10
    ax.plot([0, max_val], [0, max_val], 'r--', label='No change')
    ax.set_xlabel('Phase 3 Error (deg)')
    ax.set_ylabel('Phase 4 Error (deg)')
    ax.set_title('Phase 3 vs Phase 4 Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Success Rate Bar Chart
    ax = axes[1, 2]
    categories = ['<2 deg\n(Excellent)', '<5 deg\n(Good)', '<10 deg\n(Acceptable)']
    p3_rates = [stats['p3_success_excellent'], stats['p3_success_good'], stats['p3_success_acceptable']]
    p4_rates = [stats['p4_success_excellent'], stats['p4_success_good'], stats['p4_success_acceptable']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, p3_rates, width, label='Phase 3', color='salmon')
    ax.bar(x + width/2, p4_rates, width, label='Phase 4', color='lightgreen')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Full Pipeline Evaluation (n={stats['num_samples']})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def visualize_sample(result: Dict, save_path: str = None):
    """Visualize a single sample result"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    h, w = result['img_target'].shape[:2]
    
    # Row 1: Images
    axes[0, 0].imshow(result['img_source'])
    axes[0, 0].set_title(f"Source (Rotated {result['applied_angle']:.1f} deg)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(result['img_p3_aligned'])
    axes[0, 1].set_title(f"Phase 3 Aligned\nError: {result['p3_angle_err']:.2f} deg")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result['img_p4_aligned'])
    color = 'green' if result['p4_angle_err'] < result['p3_angle_err'] else 'red'
    axes[0, 2].set_title(f"Phase 4 Aligned\nError: {result['p4_angle_err']:.2f} deg", color=color)
    axes[0, 2].axis('off')
    
    # Row 2: Analysis
    # Checkerboard
    def create_checkerboard(img1, img2, block_size=32):
        mask = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    mask[y:y+block_size, x:x+block_size] = 1.0
        mask = np.dstack([mask]*3)
        return (img1 * mask + img2 * (1 - mask)).astype(np.uint8)
    
    check_p3 = create_checkerboard(result['img_target'], result['img_p3_aligned'])
    check_p4 = create_checkerboard(result['img_target'], result['img_p4_aligned'])
    
    axes[1, 0].imshow(check_p3)
    axes[1, 0].set_title("Phase 3 Checkerboard")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(check_p4)
    axes[1, 1].set_title("Phase 4 Checkerboard")
    axes[1, 1].axis('off')
    
    # Loss history
    if result['loss_history']:
        axes[1, 2].plot(result['loss_history'], 'b-', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Phase 4 Optimization')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Phase 4 Disabled', ha='center', va='center', fontsize=14)
        axes[1, 2].axis('off')
    
    plt.suptitle(
        f"GT: {result['gt_angle']:.1f} deg | P3: {result['p3_angle']:.1f} deg (err: {result['p3_angle_err']:.2f} deg) | "
        f"P4: {result['p4_angle']:.1f} deg (err: {result['p4_angle_err']:.2f} deg) | "
        f"Delta: {result['angle_improvement']:+.2f} deg",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Full Pipeline Evaluation')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--use_phase4', type=lambda x: x.lower() == 'true', default=True, help='Use Phase 4 refinement')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--img_dir', type=str, default='./img/val2017')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--visualize', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Config
    config = EvalConfig(
        model_path=args.model_path,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    print("=" * 80)
    print("FULL PIPELINE EVALUATION")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Phase 4: {'Enabled' if args.use_phase4 else 'Disabled'}")
    
    # Load images
    image_paths = glob.glob(os.path.join(config.img_dir, "*.jpg"))
    if not image_paths:
        print(f"No images found in {config.img_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = FullPipelineEvaluator(config)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate_dataset(
        image_paths,
        num_samples=args.num_samples,
        use_phase4=args.use_phase4
    )
    
    if not results:
        print("No results generated - check error messages above")
        return
    
    # Compute statistics
    stats = compute_statistics(results, config)
    
    # Print summary
    print_summary(stats)
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save statistics
    stats_path = os.path.join(config.output_dir, 'statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    
    # Plot results
    plot_path = os.path.join(config.output_dir, 'evaluation_plots.png')
    plot_results(results, stats, save_path=plot_path)
    
    # Visualize samples
    if args.visualize > 0 and len(results) >= 3:
        print(f"\nVisualizing {min(args.visualize, len(results))} samples...")
        sorted_by_improvement = sorted(results, key=lambda x: x['angle_improvement'], reverse=True)
        
        samples_to_show = []
        samples_to_show.append(sorted_by_improvement[0])  # Best
        samples_to_show.append(sorted_by_improvement[-1])  # Worst
        samples_to_show.append(sorted_by_improvement[len(sorted_by_improvement)//2])  # Median
        
        for i, sample in enumerate(samples_to_show[:args.visualize]):
            vis_path = os.path.join(config.output_dir, f'sample_{i+1}.png')
            visualize_sample(sample, save_path=vis_path)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()