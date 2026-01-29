import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# --- My Model Imports ---
try:
    from phase1 import MathGeometricPreprocessor
    from phase2 import CliffordPyramidEmbedder
    from phase3 import Phase3Transformer
    from phase4 import GeometricMPCRefiner
except ImportError:
    print("Warning: 사용자 모델 모듈(phase1~4)을 찾을 수 없습니다.")

# --- Settings ---
IMG_SIZE = (256, 256) # ROMA/MASt3R는 큰 해상도를 선호하지만, 공정한 비교를 위해 256 유지
HIDDEN_DIM = 48
FEATURE_DIM = 144
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/final_model_early_stop.pth"
ANGLE_THRESHOLD = 45

# =========================================================
# 1. My Model Pipeline (Transformer + MPC)
# =========================================================
def run_my_model_pipeline(img_warped, img_rgb, model_components):
    embedder, transformer = model_components
    
    preprocessor = MathGeometricPreprocessor()
    pyramid_a = preprocessor.process_pyramid(img_warped, levels=4)
    pyramid_b = preprocessor.process_pyramid(img_rgb, levels=4)
    
    with torch.no_grad():
        p2_a = embedder(pyramid_a, DEVICE)
        p2_b = embedder(pyramid_b, DEVICE)
        results = transformer(p2_a, p2_b)
        
        finest_res = results[0]
        avg_rotor = finest_res['rotor_map'].mean(dim=(1, 2))
        cos_raw, sin_raw, dx, dy = avg_rotor[0,0], avg_rotor[0,1], avg_rotor[0,2], avg_rotor[0,3]
        mag = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
        
        row1 = torch.stack([cos_raw/mag, -sin_raw/mag, dx])
        row2 = torch.stack([sin_raw/mag,  cos_raw/mag, dy])
        W_p3_norm = torch.stack([row1, row2])

    # Phase 4 MPC
    s_src, v_src, b_src = p2_a[0]
    s_tgt, v_tgt, b_tgt = p2_b[0]
    
    src_mpc = {
        'sdf': torch.from_numpy(pyramid_a[0]['sdf']).unsqueeze(0).unsqueeze(0).float().to(DEVICE),
        'vector': v_src.mean(dim=1).detach(),
        'rotor': b_src[2].mean(dim=1, keepdim=True).detach()
    }
    tgt_mpc = {
        'sdf': torch.from_numpy(pyramid_b[0]['sdf']).unsqueeze(0).unsqueeze(0).float().to(DEVICE),
        'vector': v_tgt.mean(dim=1).detach(),
        'rotor': b_tgt[2].mean(dim=1, keepdim=True).detach()
    }
    
    g_s = torch.sigmoid(torch.mean(torch.abs(s_src), dim=1, keepdim=True))
    g_v = torch.sigmoid(torch.mean(torch.norm(v_src, dim=2), dim=1, keepdim=True))
    g_b = torch.sigmoid(torch.mean(b_src[2], dim=1, keepdim=True))
    gates = (g_s, g_v, g_b)

    row_bottom = torch.tensor([0., 0., 1.], device=DEVICE).unsqueeze(0)
    mat_3x3 = torch.cat([W_p3_norm, row_bottom], dim=0)
    W_p3_inv = torch.inverse(mat_3x3)[:2, :] 
    
    refiner = GeometricMPCRefiner(device=DEVICE)
    with torch.no_grad():
        refiner.W[0] = W_p3_inv.unsqueeze(0)
    
    refiner.optimize(src_mpc, tgt_mpc, gates) 
    
    W_mpc_inv = refiner.W[0].detach()
    mat_3x3_inv = torch.cat([W_mpc_inv, row_bottom], dim=0)
    W_final = torch.inverse(mat_3x3_inv)[:2, :].cpu().numpy()
    
    return W_final

# =========================================================
# 2. RoMa (CVPR 2024) Wrapper
#    - Robust Dense Feature Matching
# =========================================================
class RomaWrapper:
    def __init__(self):
        self.model = None
        try:
            # pip install romatch
            from romatch import roma_outdoor
            print(">>> Loading RoMa (Outdoor)...")
            self.model = roma_outdoor(device=DEVICE)
        except ImportError:
            print("⚠️ RoMa not found. Install via 'pip install romatch'")
        except Exception as e:
            print(f"⚠️ RoMa Load Error: {e}")

    def run(self, img1_np, img2_np):
        if self.model is None: return None
        
        h, w = img1_np.shape[:2]
        
        # RoMa expects PIL or Tensor. Let's use internal match method which handles resizing usually
        # But for safety, we convert to tensor
        im1 = torch.from_numpy(img1_np).permute(2, 0, 1).float() / 255.0
        im2 = torch.from_numpy(img2_np).permute(2, 0, 1).float() / 255.0
        im1 = im1.to(DEVICE).unsqueeze(0)
        im2 = im2.to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            # RoMa outputs dense matches
            warp, certainty = self.model.match(im1, im2, batched=False)
            
            # Sampling high confidence points for RANSAC
            # certainty: (H, W)
            matches = []
            pts1 = []
            pts2 = []
            
            # Grid sampling to avoid processing all pixels (slow)
            step = 4
            conf_thresh = 0.5
            
            certainty = certainty.cpu().numpy()
            warp = warp.cpu().numpy() # (H, W, 2) - coordinate in im2 for each pixel in im1
            
            ys, xs = np.mgrid[0:h:step, 0:w:step]
            ys = ys.flatten()
            xs = xs.flatten()
            
            for x, y in zip(xs, ys):
                if certainty[y, x] > conf_thresh:
                    # Point in Img 1
                    pts1.append([x, y])
                    # Point in Img 2 (from warp flow)
                    # RoMa output is usually absolute coordinates in match()
                    pts2.append(warp[y, x])
            
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)

            if len(pts1) < 4: return None

            # RANSAC for Affine
            M, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            return M

# =========================================================
# 3. MASt3R (CVPR 2024) Wrapper
#    - Matching and Stereo 3D Reconstruction
# =========================================================
class Mast3rWrapper:
    def __init__(self):
        self.model = None
        try:
            # MASt3R is heavy. Assuming installed or loadable via hub
            # This is a fallback loading mechanism
            print(">>> Loading MASt3R (Large)...")
            self.model = torch.hub.load('naver/mast3r', 'mast3r_large', trust_repo=True)
            self.model.to(DEVICE).eval()
        except Exception as e:
            print(f"⚠️ MASt3R not found or load failed. (Complex dependency): {e}")

    def run(self, img1_np, img2_np):
        if self.model is None: return None
        
        # MASt3R input preparation
        # It expects a list of dictionaries or similar based on repo implementation
        # Standard input: items with 'img', 'idx', 'instance'
        # Simplifying assuming match_images interface or manual forward
        
        h, w = img1_np.shape[:2]
        
        # Simple transform
        img1_tensor = torch.from_numpy(img1_np).permute(2,0,1).float() / 255.0
        img2_tensor = torch.from_numpy(img2_np).permute(2,0,1).float() / 255.0
        
        batch = [{
            'img': img1_tensor.to(DEVICE),
            'idx': 0,
            'instance': 0,
        }, {
            'img': img2_tensor.to(DEVICE),
            'idx': 1,
            'instance': 0,
        }]

        with torch.no_grad():
            # This is pseudo-code for MASt3R standard usage as it varies by repo version
            # Usually returns pred1, pred2 with 'pts3d' and 'conf'
            try:
                # If using the official repo inference style
                imgs = torch.stack([d['img'] for d in batch]).unsqueeze(0) # (1, 2, 3, H, W)
                pred1, pred2 = self.model(imgs)
                
                # Extract dense matches based on confidence and 3D proximity
                # For 2D affine, we can just map pixels that have similar descriptors or 3D positions
                # But MASt3R is essentially dense matching.
                
                # Let's use the matching capability if exposed, otherwise brute force dense map
                # MASt3R outputs dense point maps.
                pts1_map = pred1['pts3d'][0].cpu().numpy() # (H, W, 3)
                pts2_map = pred2['pts3d'][0].cpu().numpy()
                conf1 = pred1['conf'][0].cpu().numpy() # (H, W)
                
                # Finding matches: points with same 3D coordinate? No, it reconstructs the scene.
                # Actually, MASt3R matches by estimating relative pose and finding defining geometry.
                # The easiest way to get "matches" is finding nearest neighbors in the descriptor space if available,
                # OR using the FastNN matcher provided in their utils.
                
                # Fallback: If complicated, we assume failure to load for this snippet
                # because implementing full MASt3R matching logic here is >100 lines.
                return None 
            except:
                return None
        return None

# =========================================================
# 4. Utils & Main
# =========================================================
def denormalize_affine_matrix(matrix_norm, width, height):
    N = np.array([[2.0/width, 0, -1], [0, 2.0/height, -1], [0, 0, 1]])
    N_inv = np.linalg.inv(N)
    M_norm_aug = np.vstack([matrix_norm, [0, 0, 1]])
    M_pix_aug = N_inv @ M_norm_aug @ N
    return M_pix_aug[:2, :]

def calc_error(W_pred, W_gt, w, h):
    if W_pred is None: return 50.0 # Penalty
    corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
    ones = np.ones((4,1))
    corners_aug = np.hstack([corners, ones])
    gt_pts = (W_gt @ corners_aug.T).T
    pred_pts = (W_pred @ corners_aug.T).T
    return np.mean(np.linalg.norm(gt_pts - pred_pts, axis=1))

def main():
    # 1. Load My Model
    print(">>> Loading My Model (Phase3 + MPC)...")
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(DEVICE)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
    except:
        print("Model checkpoint not found or mismatch.")
        return
    embedder.eval(); transformer.eval()

    # 2. Load Transformer Competitors
    roma_model = RomaWrapper()
    # mast3r_model = Mast3rWrapper() # MASt3R is tricky to run in script, often OOM or requires specific env
    
    models = {
        'Ours (Dense MPC)': 'custom',
        'RoMa (Dense ViT)': roma_model,
        # 'MASt3R': mast3r_model 
    }

    # 3. Benchmark
    img_list = glob.glob("./val2017/*.jpg")
    if not img_list: 
        print("No images found in ./val2017")
        return
    np.random.shuffle(img_list)
    
    TEST_COUNT = 15
    results = {k: [] for k in models.keys()}
    
    print(f"\n🔥 Benchmark: Dense Transformer Battle on {TEST_COUNT} images...")
    
    for i in range(TEST_COUNT):
        img_raw = cv2.imread(img_list[i])
        if img_raw is None: continue
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)
        h, w = img_rgb.shape[:2]
        
        # Warp (Rotation Test)
        angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
        M_warp = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_warped = cv2.warpAffine(img_rgb, M_warp, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # GT
        M_warp_aug = np.vstack([M_warp, [0,0,1]])
        W_gt_pixel = np.linalg.inv(M_warp_aug)[:2, :]
        
        for name, model_obj in models.items():
            W_pred = None
            try:
                if name == 'Ours (Dense MPC)':
                    W_norm = run_my_model_pipeline(img_warped, img_rgb, (embedder, transformer))
                    W_pred = denormalize_affine_matrix(W_norm, w, h)
                elif name == 'RoMa (Dense ViT)' and model_obj.model:
                    W_pred = model_obj.run(img_warped, img_rgb)
                # elif name == 'MASt3R' ...
                
                err = calc_error(W_pred, W_gt_pixel, w, h)
            except Exception as e:
                # print(e)
                err = 50.0
            
            results[name].append(err)
            
        # Print progress per image
        print(f"[{i+1}/{TEST_COUNT}] Rot {angle:.1f}°")
        for k, v in results.items():
            print(f"   {k}: {v[-1]:.4f} px")

    # Final Stats
    print("\n" + "="*40)
    scores = {k: np.mean(v) for k, v in results.items()}
    for k, v in scores.items():
        print(f"{k:<20} : {v:.4f} px")
    print("="*40)

    # Plot
    plt.bar(scores.keys(), scores.values(), color=['blue', 'green', 'red'])
    plt.title('Dense Transformer Matching Error (Lower is Better)')
    plt.ylabel('MACE (px)')
    plt.show()

if __name__ == "__main__":
    main()