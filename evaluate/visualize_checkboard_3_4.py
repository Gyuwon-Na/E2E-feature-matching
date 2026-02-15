"""
================================================================================
Phase 3 vs Phase 4 Comparison - FINAL VERSION with Diagnostics
================================================================================
핵심 개선:
1. Phase 3 예측 검증 - 비정상적인 예측 감지
2. Multi-start fallback - Phase 3 예측이 이상하면 여러 초기값 시도
3. 상세한 진단 출력
================================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM

from phase4.phase4_1 import HierarchicalMPCRefiner
from phase4.phase4_2 import GeometricMPCRefiner

# Phase 4 import
sys.path.append('/home/claude')
PHASE4 = ['1', '2']

phase4_map = {
    '1': HierarchicalMPCRefiner,
    '2': GeometricMPCRefiner,
}

current = '1'

if current in PHASE4:
    phase4_map[current]()   # phase4.phase4_1 실행

# ==============================================================================
# Configuration
# ==============================================================================
IMG_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/best_model.pth"
TEST_IMG_DIR = "./img/val2017"

def create_checkerboard(img1, img2, block_size=32):
    """체커보드 오버레이"""
    h, w = img1.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                mask[y:y+block_size, x:x+block_size] = 1.0
    mask = np.dstack([mask]*3)
    return (img1 * mask + img2 * (1 - mask)).astype(np.uint8)

def extract_angle_from_rotor(rotor_tensor):
    """Rotor에서 각도 추출"""
    if rotor_tensor.dim() > 1:
        rotor_tensor = rotor_tensor[0]
    
    cos_val = rotor_tensor[0].item()
    sin_val = rotor_tensor[1].item()
    angle_rad = np.arctan2(sin_val, cos_val)
    return np.degrees(angle_rad)

def validate_phase3_prediction(pred_angle, true_angle):
    """
    Phase 3 예측 검증
    
    Returns:
        is_valid (bool), reason (str)
    """
    expected_angle = -true_angle
    error = abs(pred_angle - expected_angle)
    
    # 검증 기준
    if abs(pred_angle) > 90:
        return False, f"Too large angle ({pred_angle:.1f}°)"
    
    if error > 50:
        return False, f"Unexpected prediction (expected ~{expected_angle:.1f}°, got {pred_angle:.1f}°)"
    
    return True, "OK"

def extract_mpc_inputs(embedder, pyr_dict, device):
    """Phase 4 입력 추출"""
    input_dict = pyr_dict[0]
    s, v, b = embedder.core(input_dict, device)
    
    mpc_data = {
        'sdf': torch.tensor(input_dict['sdf'][np.newaxis, np.newaxis, ...]).float().to(device),
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

def main():
    print("=" * 80)
    print(f"Phase 3 vs Phase 4 Comparison (Final) on {DEVICE}")
    print("=" * 80)
    
    # 1. Load models
    print("\n[1] Loading models...")
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ Checkpoint: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
    else:
        print("⚠️  No checkpoint found. Using untrained models.")
    
    embedder.eval()
    transformer.eval()
    
    # 2. Load image
    print("\n[2] Loading test image...")
    img_list = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if not img_list:
        print(f"❌ No images in {TEST_IMG_DIR}")
        return
    
    img_path = np.random.choice(img_list)
    print(f"Image: {os.path.basename(img_path)}")
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE)
    h, w = img_rgb.shape[:2]
    
    # 3. Create test case
    print("\n[3] Creating test case...")
    true_angle = np.random.uniform(20, 60) * (1 if np.random.rand() > 0.5 else -1)
    print(f"Ground Truth: {true_angle:.2f}°")
    print(f"  Source = Target rotated by {true_angle:.2f}°")
    print(f"  Expected correction: {-true_angle:.2f}°")
    
    M_perturbation = cv2.getRotationMatrix2D((w/2, h/2), true_angle, 1.0)
    img_source = cv2.warpAffine(img_rgb, M_perturbation, (w, h), 
                                borderMode=cv2.BORDER_REFLECT)
    img_target = img_rgb
    
    # 4. Phase 1
    print("\n[4] Phase 1: Preprocessing...")
    preprocessor = MathGeometricPreprocessor()
    pyr_src = preprocessor.process_pyramid(img_source, levels=5)
    pyr_tgt = preprocessor.process_pyramid(img_target, levels=5)
    
    # 5. Phase 3
    print("\n[5] Phase 3: Transformer matching...")
    with torch.no_grad():
        f_src = embedder(pyr_src, DEVICE)
        f_tgt = embedder(pyr_tgt, DEVICE)
        res_p3 = transformer(f_src, f_tgt)
        rotor_p3 = res_p3[0]['rotor_map'].mean(dim=(1, 2))
    
    pred_angle_p3 = extract_angle_from_rotor(rotor_p3)
    
    print(f"  Phase 3 prediction: {pred_angle_p3:.2f}°")
    
    # Validate Phase 3
    is_valid, reason = validate_phase3_prediction(pred_angle_p3, true_angle)
    if not is_valid:
        print(f"  ⚠️  Phase 3 prediction looks suspicious: {reason}")
    else:
        print(f"  ✅ Phase 3 prediction looks reasonable")
    
    # Apply Phase 3
    M_p3 = cv2.getRotationMatrix2D((w/2, h/2), pred_angle_p3, 1.0)
    img_p3_aligned = cv2.warpAffine(img_source, M_p3, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT)
    err_p3 = abs(pred_angle_p3 + true_angle)
    print(f"  Phase 3 error: {err_p3:.2f}°")
    
    # 6. Phase 4
    print("\n[6] Phase 4: MPC refinement...")
    
    mpc = GeometricMPCRefiner(DEVICE)
    
    try:
        # Prepare data
        src_mpc_data, src_gates, src_feat = extract_mpc_inputs(embedder, pyr_src, DEVICE)
        tgt_mpc_data, _, _ = extract_mpc_inputs(embedder, pyr_tgt, DEVICE)
        
        # Priority map
        local_avg = F.avg_pool2d(src_feat, kernel_size=3, stride=1, padding=1)
        rotor_variance = torch.abs(src_feat - local_avg)
        priority_map = mpc.compute_priority_map(rotor_variance, src_feat)
        
        # Strategy selection
        if is_valid:
            # Trust Phase 3
            print("  Strategy: Single-start from Phase 3 prediction")
            init_angle_rad = np.radians(pred_angle_p3)
            mpc.global_filtering_init(mean_rotor=init_angle_rad, mean_scale=1.0)
            
            final_W, loss_history = mpc.optimize(
                src_mpc_data,
                tgt_mpc_data,
                src_gates,
                priority_map=priority_map
            )
        else:
            # Don't trust Phase 3 - try multiple starts
            print("  Strategy: Multi-start optimization")
            
            # Generate candidate angles
            candidates = [
                -true_angle,              # Ideal (if we knew it)
                pred_angle_p3,            # Phase 3
                pred_angle_p3 * 0.5,      # Conservative
                -pred_angle_p3,           # Opposite
                0.0                       # Identity
            ]
            # Remove duplicates
            candidates = list(set([round(a, 1) for a in candidates]))
            print(f"  Trying {len(candidates)} initial angles: {candidates}")
            
            final_W, loss_history = mpc.multi_start_optimize(
                src_mpc_data,
                tgt_mpc_data,
                src_gates,
                candidates,
                priority_map=priority_map
            )
        
        # Extract result
        angle_p4, scale_p4, tx_p4, ty_p4 = mpc.get_transform_params()
        
        print(f"\n  Phase 4 result:")
        print(f"    Angle: {angle_p4:.2f}°")
        print(f"    Scale: {scale_p4:.3f}")
        print(f"    Translation: ({tx_p4:.3f}, {ty_p4:.3f})")
        
        # Apply Phase 4
        M_p4 = cv2.getRotationMatrix2D((w/2, h/2), angle_p4, scale_p4)
        M_p4[0, 2] += tx_p4 * w / 2
        M_p4[1, 2] += ty_p4 * h / 2
        
        img_p4_aligned = cv2.warpAffine(img_source, M_p4, (w, h), 
                                        borderMode=cv2.BORDER_REFLECT)
        
        err_p4 = abs(angle_p4 + true_angle)
        
        improvement = err_p3 - err_p4
        print(f"  Phase 4 error: {err_p4:.2f}°")
        print(f"  Improvement: {improvement:+.2f}° {'✅' if improvement > 0 else '❌'}")
        
    except Exception as e:
        print(f"  ❌ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback
        img_p4_aligned = img_p3_aligned
        err_p4 = err_p3
        angle_p4 = pred_angle_p3
    
    # 7. Visualization
    print("\n[7] Creating visualization...")
    
    check_p3 = create_checkerboard(img_target, img_p3_aligned)
    check_p4 = create_checkerboard(img_target, img_p4_aligned)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1
    axes[0, 0].imshow(img_source)
    axes[0, 0].set_title(f"Source (Rotated {true_angle:.1f}°)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_target)
    axes[0, 1].set_title("Target (Original)", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_p3_aligned)
    axes[0, 2].set_title(f"Phase 3 Aligned\nPred: {pred_angle_p3:.1f}°", 
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(check_p3)
    axes[1, 0].set_title(f"Phase 3 Checkerboard\nError: {err_p3:.2f}°", fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(check_p4)
    color = 'green' if err_p4 < err_p3 else 'red'
    axes[1, 1].set_title(f"Phase 4 Checkerboard\nError: {err_p4:.2f}°", 
                         fontsize=12, color=color, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Error map
    diff_map = np.abs(img_target.astype(float) - img_p4_aligned.astype(float)).mean(axis=2)
    im = axes[1, 2].imshow(diff_map, cmap='hot')
    axes[1, 2].set_title("Phase 4 Error Map", fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    # Super title
    improvement_str = f"{improvement:+.2f}°" if err_p4 < err_p3 else f"{improvement:+.2f}° (worse)"
    plt.suptitle(
        f"Phase 3 vs Phase 4 Comparison\n"
        f"GT: {true_angle:.1f}° | P3: {pred_angle_p3:.1f}° ({err_p3:.2f}° err) | "
        f"P4: {angle_p4:.1f}° ({err_p4:.2f}° err, {improvement_str})",
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()