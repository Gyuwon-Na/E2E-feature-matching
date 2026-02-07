"""
================================================================================
Phase 4 v5: Parameter-Specific Optimization (Angle Booster)
================================================================================
[Architecture.md §4 참조]

문제점: 큰 각도 오차 시, 학습률이 낮아 제자리 걸음을 하거나 지역 최솟값에 빠짐.
해결책: 파라미터를 그룹별로 분리하여 '회전(Angle)'에 더 높은 학습률(Momentum)을 부여.

[v5 핵심 변경사항]
1. Parameter Split: d_angle, d_scale, d_trans를 각각 별도의 Parameter로 분리
2. LR Scheduling: Angle에는 10~20배 높은 LR을 적용하여 회전부터 맞추도록 유도
3. Coarse Strategy: Level 2에서는 SDF(픽셀)를 거의 무시하고 Vector(방향)만 보고 회전
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# [Hyperparameters] Angle-Boosted Tuning
# =============================================================================
MPC_CONFIG = {
    'levels': [2, 1, 0],
    'iters': [100, 50, 30],  # Level 2 반복 횟수 증가 (충분히 돌릴 시간 확보)
    
    # 기본 학습률 (Scale, Translation용)
    'base_lrs': [0.005, 0.002, 0.001],
    
    # [핵심] 각도(Angle) 전용 부스트 배율 (예: 10배 더 과감하게 회전)
    'angle_boost': [20.0, 5.0, 1.0], 

    'weights': [
        # w_sdf(픽셀), w_vector(방향), w_rotor(회전량)
        [0.0, 2.0, 0.5],   # Level 2: 픽셀 무시(0.0), 방향(2.0)에 올인 -> 큰 회전 잡기
        [0.5, 1.0, 0.5],   # Level 1: 균형 잡기
        [1.0, 0.2, 0.1]    # Level 0: 미세 픽셀 조정
    ]
}

class HierarchicalMPCRefiner(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.epsilon = 1e-6
        
        # [v5] 파라미터 개별 분리 (Optimizer에서 LR을 다르게 주기 위함)
        # 초기값은 모두 0 (Identity)
        self.p_angle = nn.Parameter(torch.zeros(1).to(device))
        self.p_scale = nn.Parameter(torch.zeros(1).to(device))
        self.p_trans = nn.Parameter(torch.zeros(1, 2).to(device))

    def reset_params(self):
        with torch.no_grad():
            self.p_angle.data.zero_()
            self.p_scale.data.zero_()
            self.p_trans.data.zero_()

    def decompose_affine(self, W):
        """행렬에서 기하학적 파라미터 추출"""
        a = W[:, 0, 0]
        b = W[:, 0, 1]
        tx = W[:, 0, 2]
        ty = W[:, 1, 2]
        
        scale = torch.sqrt(a**2 + b**2 + 1e-8)
        angle = torch.atan2(W[:, 1, 0], W[:, 0, 0])
        
        return angle, scale, tx, ty

    def construct_affine(self, angle, scale, tx, ty):
        """파라미터 -> 행렬 재조립 (Center Pivot)"""
        B = angle.shape[0]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        # Rotation + Scale
        row1_rot = torch.stack([scale * cos, -scale * sin], dim=1)
        row2_rot = torch.stack([scale * sin,  scale * cos], dim=1)
        
        # Translation
        t_x = tx.unsqueeze(1)
        t_y = ty.unsqueeze(1)
        
        row1 = torch.cat([row1_rot, t_x], dim=1)
        row2 = torch.cat([row2_rot, t_y], dim=1)
        
        return torch.stack([row1, row2], dim=1)

    def get_current_transform(self, base_params):
        """Base + Delta(Learned) -> Current W"""
        base_angle, base_scale, base_tx, base_ty = base_params
        
        # 1. Angle Update (Additive)
        curr_angle = base_angle + self.p_angle
        
        # 2. Scale Update (Safe Multiplicative)
        # Tanh로 ±20% 제한 -> 발산 방지
        scale_factor = 1.0 + torch.tanh(self.p_scale) * 0.2
        curr_scale = base_scale * scale_factor
        
        # 3. Translation Update (Additive)
        curr_tx = base_tx + self.p_trans[:, 0]
        curr_ty = base_ty + self.p_trans[:, 1]
        
        return self.construct_affine(curr_angle, curr_scale, curr_tx, curr_ty)

    def compute_energy(self, src_feats, tgt_feats, W_current, weights):
        w_sdf, w_vec, w_rot = weights
        B, _, H, W = src_feats['sdf'].shape
        
        # Warping (Padding='border'로 경계 안정화)
        grid = F.affine_grid(W_current, [B, 1, H, W], align_corners=False)
        
        warped_sdf = F.grid_sample(src_feats['sdf'], grid, align_corners=False, padding_mode='border')
        warped_vec = F.grid_sample(src_feats['vector'], grid, align_corners=False, padding_mode='zeros')
        warped_rot = F.grid_sample(src_feats['rotor'], grid, align_corners=False, padding_mode='zeros')
        
        # Valid Mask
        mask = (F.grid_sample(torch.ones_like(warped_sdf), grid, align_corners=False) > 0.9).float()
        
        # Vector Rotation Correction (중요!)
        rot_mat = W_current[:, :2, :2] 
        vec_perm = warped_vec.permute(0, 2, 3, 1)
        warped_vec_corr = torch.einsum('bij,bhwj->bhwi', rot_mat, vec_perm).permute(0, 3, 1, 2)

        # Losses
        diff_sdf = torch.abs(warped_sdf - tgt_feats['sdf'])
        
        # Cosine Distance (1 - cos)
        sim_vec = F.cosine_similarity(warped_vec_corr, tgt_feats['vector'], dim=1)
        diff_vec = (1.0 - sim_vec).unsqueeze(1)
        
        diff_rot = torch.abs(warped_rot - tgt_feats['rotor'])
        
        loss_map = w_sdf * diff_sdf + w_vec * diff_vec + w_rot * diff_rot
        return (loss_map * mask).sum() / (mask.sum() + 1e-6)

    def optimize(self, pyramid_a_feats, pyramid_b_feats, W_init):
        # Initial Decomposition
        with torch.no_grad():
            base_angle, base_scale, base_tx, base_ty = self.decompose_affine(W_init)
            base_params = (base_angle, base_scale, base_tx, base_ty)

        loss_history = []
        print("\n[Phase 4 v5] Angle-Boosted MPC Refinement...")

        for stage_idx, level in enumerate(MPC_CONFIG['levels']):
            
            # Data Packing
            def pack_feats(p_tuple):
                s, v, b = p_tuple
                return {
                    'sdf': s[:, :1, :, :].detach(),
                    'vector': v.mean(dim=1).detach(),        
                    'rotor': b[2].mean(dim=1, keepdim=True).detach()
                }

            safe_level = min(level, len(pyramid_a_feats)-1)
            feat_a = pack_feats(pyramid_a_feats[safe_level])
            feat_b = pack_feats(pyramid_b_feats[safe_level])
            
            self.reset_params()
            
            # [핵심] Parameter Groups로 Learning Rate 차등 적용
            base_lr = MPC_CONFIG['base_lrs'][stage_idx]
            angle_boost = MPC_CONFIG['angle_boost'][stage_idx]
            
            optimizer = optim.Adam([
                {'params': [self.p_angle], 'lr': base_lr * angle_boost}, # Angle: 고속 학습
                {'params': [self.p_scale, self.p_trans], 'lr': base_lr}  # Scale/Trans: 저속 학습
            ])
            
            curr_weights = MPC_CONFIG['weights'][stage_idx]
            
            # Optimization Loop
            for i in range(MPC_CONFIG['iters'][stage_idx]):
                optimizer.zero_grad()
                
                W_pred = self.get_current_transform(base_params)
                loss = self.compute_energy(feat_a, feat_b, W_pred, curr_weights)
                
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

            # Update Base Params
            with torch.no_grad():
                base_angle += self.p_angle
                base_scale *= (1.0 + torch.tanh(self.p_scale) * 0.2)
                base_tx += self.p_trans[:, 0]
                base_ty += self.p_trans[:, 1]
                base_params = (base_angle, base_scale, base_tx, base_ty)
        
        W_final = self.construct_affine(*base_params)
        return W_final, loss_history
    
# ==============================================================================
# [Integration Logic]
# ==============================================================================

def run_integrated_pipeline():
    """
    [Phase 4 통합 테스트]
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pipeline running on: {device}")

    # 1. 이미지 로드
    IMG_PATH = "./img/val2017/000000000632.jpg" # 경로 확인 필요
    if not os.path.exists(IMG_PATH):
        # Fallback for testing without image
        img_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    else:
        img_raw = cv2.imread(IMG_PATH)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    
    rows, cols = img_rgb.shape[:2]
    
    # 2. 30도 회전 시뮬레이션
    M_gt = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1.0)
    img_warped = cv2.warpAffine(img_rgb, M_gt, (cols, rows))

    # Phase 1, 2
    preprocessor = MathGeometricPreprocessor()
    pyramid_src = preprocessor.process_pyramid(img_warped, levels=4) 
    pyramid_tgt = preprocessor.process_pyramid(img_rgb, levels=4)
    
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        p2_src = embedder(pyramid_src, device)
        p2_tgt = embedder(pyramid_tgt, device)

    # Phase 3 Mockup (Initial Guess: GT + Noise)
    M_gt_aug = np.vstack([M_gt, [0, 0, 1]])
    W_gt_inv = np.linalg.inv(M_gt_aug)[:2, :]
    W_init_np = W_gt_inv + np.random.normal(0, 0.05, size=(2, 3)) # Noise
    W_init_tensor = torch.from_numpy(W_init_np).float().unsqueeze(0).to(device)

    # Phase 4 Run
    refiner = HierarchicalMPCRefiner(device=device)
    W_refined, loss_curve = refiner.optimize(p2_src, p2_tgt, W_init_tensor)
    
    print("\n[Optimization Complete]")
    print(f"Final Loss: {loss_curve[-1]:.6f}")
    
    plt.plot(loss_curve)
    plt.title("Geometric MPC Loss")
    plt.show()

if __name__ == "__main__":
    run_integrated_pipeline()