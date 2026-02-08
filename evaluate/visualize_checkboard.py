import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder
from pipeline.phase3 import Phase3Transformer
from phase4.phase4_1 import HierarchicalMPCRefiner

# ==============================================================================
# [Configuration]
# ==============================================================================
IMG_SIZE = (256, 256)
HIDDEN_DIM = 48
FEATURE_DIM = 144
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/best_model.pth"
TEST_IMG_DIR = "./img/val2017" 

# ==============================================================================
# [Helper Functions]
# ==============================================================================
def create_checkerboard(img1, img2, block_size=32):
    """
    두 이미지를 체커보드 패턴으로 섞습니다.
    img1: Target (Ground Truth)
    img2: Aligned Source
    """
    h, w = img1.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 격자 마스크 생성
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if ((y // block_size) + (x // block_size)) % 2 == 0:
                mask[y:y+block_size, x:x+block_size] = 1.0
                
    mask = mask[:, :, np.newaxis] # (H, W, 1)
    
    # 블렌딩
    checker = img1 * mask + img2 * (1.0 - mask)
    return checker.astype(np.uint8)

def invert_affine(W_2x3):
    """2x3 행렬 역변환"""
    if isinstance(W_2x3, torch.Tensor):
        W_2x3 = W_2x3.detach().cpu().numpy()
    if W_2x3.ndim == 3: W_2x3 = W_2x3[0]
    
    W_3x3 = np.vstack([W_2x3, [0, 0, 1]])
    try:
        W_inv = np.linalg.inv(W_3x3)
    except:
        return np.eye(2, 3)
    return W_inv[:2, :]

def denormalize_affine(W_norm, w, h):
    """Normalized -> Pixel"""
    N = np.array([[2.0/w, 0, -1], [0, 2.0/h, -1], [0, 0, 1]])
    N_inv = np.linalg.inv(N)
    W_aug = np.vstack([W_norm, [0, 0, 1]])
    return (N_inv @ W_aug @ N)[:2, :]

# ==============================================================================
# [Main Visualization Logic]
# ==============================================================================
def visualize_results():
    print(f"🚀 Starting Visualization on {DEVICE}...")
    
    # 1. Load Models
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    refiner = HierarchicalMPCRefiner(device=DEVICE) # Phase 4
    preprocessor = MathGeometricPreprocessor()
    
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        print("✅ Models loaded successfully.")
    else:
        print("⚠️ Checkpoint not found.")
        return

    embedder.eval()
    transformer.eval()
    
    # 2. Load Image
    img_list = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    np.random.shuffle(img_list)
    
    for img_path in img_list[:5]: # 5장 테스트
        img_name = os.path.basename(img_path)
        print(f"\nProcessing {img_name}...")
        
        # Load & Resize
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        h, w = img_rgb.shape[:2]
        
        # Random Warp (Simulation)
        angle = np.random.uniform(10,20) * (1 if np.random.rand() > 0.5 else -1)
        M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # ----------------------------------------------------------------------
        # Phase 1~3: Coarse Alignment
        # ----------------------------------------------------------------------
        pyr_src = preprocessor.process_pyramid(img_warped, levels=4)
        pyr_tgt = preprocessor.process_pyramid(img_rgb, levels=4)
        
        with torch.no_grad():
            f_src = embedder(pyr_src, DEVICE)
            f_tgt = embedder(pyr_tgt, DEVICE)
            p3_out = transformer(f_src, f_tgt)
            
            # Extract Phase 3 Matrix
            rotor = p3_out[0]['rotor_map'].mean(dim=(1,2))
            cos, sin = rotor[0,0], rotor[0,1]
            mag = torch.sqrt(cos**2 + sin**2 + 1e-8)
            dx, dy = rotor[0,2], rotor[0,3]
            
            row1 = torch.stack([cos/mag, -sin/mag, dx])
            row2 = torch.stack([sin/mag,  cos/mag, dy])
            W_p3_norm = torch.stack([row1, row2]).unsqueeze(0) # Forward (Source->Target)
            
            # Prepare for Phase 4 (Target->Source, Inverted)
            W_p3_inv_norm = torch.from_numpy(invert_affine(W_p3_norm[0])).float().unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------------------------
        # Phase 4: Fine Refinement (Optimization)
        # ----------------------------------------------------------------------
        # Gradient ON
        W_p4_inv_norm, _ = refiner.optimize(f_src, f_tgt, W_p3_inv_norm)
        
        # ----------------------------------------------------------------------
        # Visualization Preparation
        # ----------------------------------------------------------------------
        # 1. Phase 3 Result Image
        W_p3_pixel = denormalize_affine(W_p3_norm[0].cpu().numpy(), w, h)
        img_p3 = cv2.warpAffine(img_warped, W_p3_pixel, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Phase 4 Result Image
        # MPC output (Target->Source) ==> Invert ==> (Source->Target)
        W_p4_norm = invert_affine(W_p4_inv_norm.detach())
        W_p4_pixel = denormalize_affine(W_p4_norm, w, h)
        img_p4 = cv2.warpAffine(img_warped, W_p4_pixel, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # 3. Create Checkerboards
        check_p3 = create_checkerboard(img_rgb, img_p3, block_size=32)
        check_p4 = create_checkerboard(img_rgb, img_p4, block_size=32)
        
        # ----------------------------------------------------------------------
        # Plotting
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Left: Input
        axes[0].imshow(img_warped)
        axes[0].set_title(f"Input (Warped: {angle:.1f}°)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Center: Phase 3
        axes[1].imshow(check_p3)
        axes[1].set_title("Phase 3 (Coarse)\nTarget + Source Mix", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Right: Phase 4
        axes[2].imshow(check_p4)
        axes[2].set_title("Phase 4 (Fine + MPC)\nTarget + Source Mix", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 저장 옵션 (필요시 주석 해제)
        # plt.savefig(f"./vis_result_{img_name}", dpi=150)
        # plt.close()

if __name__ == "__main__":
    visualize_results()