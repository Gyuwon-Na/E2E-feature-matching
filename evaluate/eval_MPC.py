"""
================================================================================
Phase 4 Model Comparison & Evaluation Script
================================================================================
다양한 Phase 4 구현을 체계적으로 평가하고 비교합니다.

평가 지표:
1. Runtime (초)
2. Improvement (Phase 3 대비 각도 오차 개선)
3. Average Angle Error (도)
4. Average Pixel Error (L1 distance)

사용법:
    python3 eval_MPC.py --model all --num_samples 50
    python3 eval_MPC.py --model 1 --num_samples 20
================================================================================
"""

import os
import sys
import cv2
import csv
import numpy as np
import torch
import torch.nn.functional as F
import glob
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM


# ==============================================================================
# Phase 4 Model Registry
# ==============================================================================

PHASE4_MODELS = {
    '1': {
        'name': 'HierarchicalMPC (Original)',
        'module': 'phase4.phase4_1',
        'class': 'HierarchicalMPCRefiner',
        'description': 'Coarse-to-fine hierarchical optimization'
    },
    '2': {
        'name': 'GeometricMPC (Improved)',
        'module': 'phase4.phase4_2',  # 변경
        'class': 'GeometricMPCRefiner',
        'description': 'Robust MPC with decomposed params'
    }
}

def import_phase4_model(model_key):
    """
    동적으로 Phase 4 모델 import
    
    Args:
        model_key: PHASE4_MODELS의 key
    
    Returns:
        Model class
    """
    if model_key not in PHASE4_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(PHASE4_MODELS.keys())}")
    
    config = PHASE4_MODELS[model_key]
    module_name = config['module']
    class_name = config['class']
    
    try:
        # Import module
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        print(f"✅ Loaded: {config['name']}")
        print(f"   Module: {module_name}")
        print(f"   Description: {config['description']}")
        
        return model_class
    
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return None
    except AttributeError as e:
        print(f"❌ Class {class_name} not found in {module_name}: {e}")
        return None

# ==============================================================================
# Evaluation Utilities
# ==============================================================================

class Phase4Evaluator:
    """Phase 4 모델 평가기"""
    
    def __init__(self, device='cuda', img_size=(256, 256)):
        self.device = device
        self.img_size = img_size
        
        # Load Phase 1-3 models
        self.embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
        self.transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
        
        self.preprocessor = MathGeometricPreprocessor()
        
        # Try to load checkpoint
        model_path = "./checkpoints/best_model.pth"
        if os.path.exists(model_path):
            print(f"Loading checkpoint: {model_path}")
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            self.embedder.load_state_dict(ckpt['embedder'])
            self.transformer.load_state_dict(ckpt['transformer'])
        else:
            print("⚠️  No checkpoint found. Using untrained models.")
        
        self.embedder.eval()
        self.transformer.eval()
    
    def create_test_case(self, img_rgb):
        """
        테스트 케이스 생성
        
        Returns:
            img_source, img_target, true_angle
        """
        h, w = img_rgb.shape[:2]
        
        # Random rotation
        true_angle = np.random.uniform(20, 60) * (1 if np.random.rand() > 0.5 else -1)
        
        M = cv2.getRotationMatrix2D((w/2, h/2), true_angle, 1.0)
        img_source = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        img_target = img_rgb
        
        return img_source, img_target, true_angle
    
    def run_phase3(self, img_source, img_target):
        """
        Phase 3 실행
        
        Returns:
            pred_angle_p3, pyr_src, pyr_tgt
        """
        pyr_src = self.preprocessor.process_pyramid(img_source, levels=5)
        pyr_tgt = self.preprocessor.process_pyramid(img_target, levels=5)
        
        with torch.no_grad():
            f_src = self.embedder(pyr_src, self.device)
            f_tgt = self.embedder(pyr_tgt, self.device)
            res_p3 = self.transformer(f_src, f_tgt)
            rotor_p3 = res_p3[0]['rotor_map'].mean(dim=(1, 2))
        
        # Extract angle
        cos_val = rotor_p3[0, 0].item()
        sin_val = rotor_p3[0, 1].item()
        pred_angle_p3 = np.degrees(np.arctan2(sin_val, cos_val))
        
        return pred_angle_p3, pyr_src, pyr_tgt
    
    def extract_mpc_inputs(self, pyr_dict):
        """MPC 입력 데이터 추출"""
        input_dict = pyr_dict[0]
        s, v, b = self.embedder.core(input_dict, self.device)
        
        mpc_data = {
            'sdf': torch.tensor(input_dict['sdf'][np.newaxis, np.newaxis, ...]).float().to(self.device),
            'vector': v.mean(dim=1).detach(),
            'rotor': b[2].mean(dim=1, keepdim=True).detach()
        }
        
        # Gates
        inv_s = torch.mean(torch.abs(s), dim=1, keepdim=True)
        inv_v = torch.mean(torch.norm(v, dim=2), dim=1, keepdim=True)
        inv_b = torch.mean(b[2], dim=1, keepdim=True)
        
        g_s = torch.sigmoid(inv_s).detach()
        g_v = torch.sigmoid(inv_v).detach()
        g_b = torch.sigmoid(inv_b).detach()
        gates = (g_s, g_v, g_b)
        
        feature_mag = inv_v.detach()
        return mpc_data, gates, feature_mag
    
    def compute_pixel_error(self, img1, img2):
        """픽셀 L1 에러 계산"""
        return np.abs(img1.astype(float) - img2.astype(float)).mean()
    
    def evaluate_single(self, model_class, img_rgb, verbose=False):
        """
        단일 이미지에 대해 평가
        
        Returns:
            results dict
        """
        # Create test case
        img_source, img_target, true_angle = self.create_test_case(img_rgb)
        h, w = img_target.shape[:2]
        
        # Phase 3
        pred_angle_p3, pyr_src, pyr_tgt = self.run_phase3(img_source, img_target)
        err_p3 = abs(pred_angle_p3 + true_angle)
        
        # Phase 3 alignment
        M_p3 = cv2.getRotationMatrix2D((w/2, h/2), pred_angle_p3, 1.0)
        img_p3_aligned = cv2.warpAffine(img_source, M_p3, (w, h), borderMode=cv2.BORDER_REFLECT)
        pixel_err_p3 = self.compute_pixel_error(img_target, img_p3_aligned)
        
        # Phase 4
        try:
            # Prepare inputs
            src_mpc_data, src_gates, src_feat = self.extract_mpc_inputs(pyr_src)
            tgt_mpc_data, _, _ = self.extract_mpc_inputs(pyr_tgt)
            
            # Priority map
            local_avg = F.avg_pool2d(src_feat, kernel_size=3, stride=1, padding=1)
            rotor_variance = torch.abs(src_feat - local_avg)
            
            # Initialize model
            refiner = model_class(device=self.device)
            
            # Check if model has compute_priority_map method
            if hasattr(refiner, 'compute_priority_map'):
                priority_map = refiner.compute_priority_map(rotor_variance, src_feat)
            else:
                priority_map = None
            
            # Initialize
            init_angle_rad = np.radians(pred_angle_p3)
            
            if hasattr(refiner, 'global_filtering_init'):
                refiner.global_filtering_init(mean_rotor=init_angle_rad, mean_scale=1.0)
            
            # Run optimization
            start_time = time.time()
            
            # Check optimization signature
            if model_class.__name__ == 'HierarchicalMPCRefiner':
                # Hierarchical needs pyramid features
                W_init = torch.eye(2, 3).unsqueeze(0).to(self.device)
                
                # Construct initial W from Phase 3
                cos_a = np.cos(init_angle_rad)
                sin_a = np.sin(init_angle_rad)
                W_init[0, 0, 0] = cos_a
                W_init[0, 0, 1] = -sin_a
                W_init[0, 1, 0] = sin_a
                W_init[0, 1, 1] = cos_a
                
                with torch.no_grad():
                    f_src = self.embedder(pyr_src, self.device)
                    f_tgt = self.embedder(pyr_tgt, self.device)
                
                W_final, loss_history = refiner.optimize(f_src, f_tgt, W_init, priority_map)
                
                # Extract angle from W
                W_np = W_final.detach().cpu().numpy()[0]
                pred_angle_p4 = np.degrees(np.arctan2(W_np[1, 0], W_np[0, 0]))
                scale_p4 = np.sqrt(W_np[0, 0]**2 + W_np[1, 0]**2)
                tx_p4, ty_p4 = W_np[0, 2], W_np[1, 2]
                
            else:
                # GeometricMPC
                W_final, loss_history = refiner.optimize(
                    src_mpc_data,
                    tgt_mpc_data,
                    src_gates,
                    priority_map=priority_map
                )
                
                pred_angle_p4, scale_p4, tx_p4, ty_p4 = refiner.get_transform_params()
            
            runtime = time.time() - start_time
            
            # Apply Phase 4
            M_p4 = cv2.getRotationMatrix2D((w/2, h/2), pred_angle_p4, scale_p4)
            M_p4[0, 2] += tx_p4 * w / 2
            M_p4[1, 2] += ty_p4 * h / 2
            
            img_p4_aligned = cv2.warpAffine(img_source, M_p4, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            err_p4 = abs(pred_angle_p4 + true_angle)
            pixel_err_p4 = self.compute_pixel_error(img_target, img_p4_aligned)
            
            improvement = err_p3 - err_p4
            success = improvement > 0
            
        except Exception as e:
            if verbose:
                print(f"  ❌ Phase 4 failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback
            runtime = 0.0
            pred_angle_p4 = pred_angle_p3
            err_p4 = err_p3
            pixel_err_p4 = pixel_err_p3
            improvement = 0.0
            success = False
        
        return {
            'true_angle': true_angle,
            'pred_angle_p3': pred_angle_p3,
            'pred_angle_p4': pred_angle_p4,
            'err_p3': err_p3,
            'err_p4': err_p4,
            'pixel_err_p3': pixel_err_p3,
            'pixel_err_p4': pixel_err_p4,
            'improvement': improvement,
            'runtime': runtime,
            'success': success
        }
    
    def evaluate_model(self, model_key, image_paths, num_samples=None):
        """
        모델 전체 평가
        
        Args:
            model_key: PHASE4_MODELS key
            image_paths: List of image paths
            num_samples: Number of samples to evaluate (None = all)
        
        Returns:
            results DataFrame
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {PHASE4_MODELS[model_key]['name']}")
        print(f"{'='*80}")
        
        # Import model
        model_class = import_phase4_model(model_key)
        if model_class is None:
            return None
        
        # Sample images
        if num_samples and num_samples < len(image_paths):
            image_paths = np.random.choice(image_paths, num_samples, replace=False)
        
        results = []
        
        for img_path in tqdm(image_paths, desc="Evaluating"):
            # Load image
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, self.img_size)
            
            # Evaluate
            result = self.evaluate_single(model_class, img_rgb, verbose=False)
            result['image'] = os.path.basename(img_path)
            result['model'] = model_key
            results.append(result)
        
        return results

# ==============================================================================
# Analysis & Visualization
# ==============================================================================

def mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0

def success_rate(results):
    return 100.0 * sum(r['success'] for r in results) / len(results)

def get_column(results, key):
    return np.array([r[key] for r in results], dtype=np.float32)

def save_csv(path, results):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def print_summary(results_dict):
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    summaries = []

    for model_key, results in results_dict.items():
        if not results:
            continue

        model_name = PHASE4_MODELS[model_key]['name']

        summary = {
            'Model': model_name,
            'Avg Runtime (s)': mean(get_column(results, 'runtime')),
            'Avg Improvement (°)': mean(get_column(results, 'improvement')),
            'Avg Angle Error (°)': mean(get_column(results, 'err_p4')),
            'Avg Pixel Error': mean(get_column(results, 'pixel_err_p4')),
            'Success Rate (%)': success_rate(results),
            'P3 Baseline Error (°)': mean(get_column(results, 'err_p3'))
        }
        summaries.append(summary)

    # 정렬 (개선량 기준)
    summaries.sort(key=lambda x: x['Avg Improvement (°)'], reverse=True)

    # 출력
    for s in summaries:
        print("-"*80)
        for k, v in s.items():
            print(f"{k:22s}: {v:.4f}" if isinstance(v, float) else f"{k:22s}: {v}")

    # Best model
    if summaries:
        best = summaries[0]
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {best['Model']}")
        print("="*80)
        for k, v in best.items():
            print(f"{k:22s}: {v:.4f}" if isinstance(v, float) else f"{k:22s}: {v}")

    return summaries

def plot_comparison(results_dict, save_path=None):
    """결과 비교 그래프"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(results_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # 1. Runtime comparison
    ax = axes[0, 0]
    runtimes = [results_dict[m]['runtime'].mean() for m in models]
    model_names = [PHASE4_MODELS[m]['name'] for m in models]
    
    ax.bar(range(len(models)), runtimes, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Average Runtime Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Improvement distribution
    ax = axes[0, 1]
    for idx, model in enumerate(models):
        df = results_dict[model]
        ax.hist(df['improvement'], bins=20, alpha=0.5, 
                label=PHASE4_MODELS[model]['name'], color=colors[idx])
    
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.set_xlabel('Improvement (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Improvement Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Angle error comparison
    ax = axes[1, 0]
    data_to_plot = []
    labels = []
    
    for model in models:
        df = results_dict[model]
        data_to_plot.append(df['err_p4'])
        labels.append(PHASE4_MODELS[model]['name'])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Angle Error (degrees)')
    ax.set_title('Angle Error Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Success rate
    ax = axes[1, 1]
    success_rates = [(results_dict[m]['success'].sum() / len(results_dict[m])) * 100 
                     for m in models]
    
    ax.bar(range(len(models)), success_rates, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (Improvement > 0)')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4 Model Evaluation')
    parser.add_argument('--model', type=str, default='all',
                        help='Model to evaluate (1, 2, 1_final, 2_final, or all)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of images to evaluate')
    parser.add_argument('--img_dir', type=str, default='./img/val2017',
                        help='Image directory')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load images
    image_paths = glob.glob(os.path.join(args.img_dir, "*.jpg"))
    if not image_paths:
        print(f"❌ No images found in {args.img_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Create evaluator
    evaluator = Phase4Evaluator(device=device)
    
    # Determine models to evaluate
    if args.model == 'all':
        models_to_eval = list(PHASE4_MODELS.keys())
    else:
        models_to_eval = [args.model]
    
    print(f"\nEvaluating models: {models_to_eval}")
    
    # Evaluate
    results_dict = {}
    
    for model_key in models_to_eval:
        df = evaluator.evaluate_model(model_key, image_paths, args.num_samples)
        if df is not None:
            results_dict[model_key] = df
    
    # Summary
    summary_df = print_summary(results_dict)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    # Save individual results
    
    for model_key, results in results_dict.items():
        save_csv(
            os.path.join(args.output, f"{model_key}_results.csv"),
            results
        )

    # Save summary
    summary_path = os.path.join(args.output, 'summary.csv')
    summaries = print_summary(results_dict)

    save_csv(
        summary_path,
        summaries
    )
    print(f"📁 Saved: {summary_path}")
    
    # Plot
    plot_path = os.path.join(args.output, 'comparison.png')
    plot_comparison(results_dict, save_path=plot_path)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()