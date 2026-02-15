"""
================================================================================
Phase 4 v6: Unified Hierarchical MPC with Priority Map
================================================================================
핵심 수정:
1. compute_priority_map 메서드 추가 (visualize 스크립트와 호환)
2. 초기화 로직 개선 - Phase 3 예측을 더 보수적으로 사용
3. Learning rate 조정 - 너무 큰 각도 변화 방지
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# =============================================================================
# [Hyperparameters] Conservative Tuning
# =============================================================================
MPC_CONFIG = {
    'levels': [2, 1, 0],
    'iters': [80, 50, 30],
    'base_lrs': [0.002, 0.001, 0.0005],  # 더 보수적인 LR
    'angle_boost': [10.0, 3.0, 1.0],     # Angle boost 감소
    'weights': [
        [0.0, 2.0, 0.5],   # Level 2: Vector 중심
        [0.5, 1.0, 0.5],   # Level 1: 균형
        [1.0, 0.3, 0.2]    # Level 0: SDF 중심
    ]
}

class HierarchicalMPCRefiner(nn.Module):
    """
    계층적 MPC 정제 모듈
    
    변환 방향: Source -> Target
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.epsilon = 1e-6
        
        # 파라미터 개별 분리
        self.p_angle = nn.Parameter(torch.zeros(1).to(device))
        self.p_scale = nn.Parameter(torch.zeros(1).to(device))
        self.p_trans = nn.Parameter(torch.zeros(1, 2).to(device))
        
        # Tracking
        self.best_loss = float('inf')
        self.best_params = None

    def reset_params(self):
        """파라미터를 Identity로 리셋"""
        with torch.no_grad():
            self.p_angle.data.zero_()
            self.p_scale.data.zero_()
            self.p_trans.data.zero_()

    def decompose_affine(self, W):
        """Affine 행렬 분해"""
        a = W[:, 0, 0]
        b = W[:, 1, 0]
        
        scale = torch.sqrt(a**2 + b**2 + self.epsilon)
        angle = torch.atan2(b, a)
        
        tx = W[:, 0, 2]
        ty = W[:, 1, 2]
        
        return angle, scale, tx, ty

    def construct_affine(self, angle, scale, tx, ty):
        """파라미터로부터 Affine 행렬 구성"""
        B = angle.shape[0]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        row1 = torch.stack([scale * cos_a, -scale * sin_a, tx], dim=1)
        row2 = torch.stack([scale * sin_a,  scale * cos_a, ty], dim=1)
        
        return torch.stack([row1, row2], dim=1)

    def get_current_transform(self, base_params):
        """Base + Delta -> Current W"""
        base_angle, base_scale, base_tx, base_ty = base_params
        
        # Angle update (additive)
        curr_angle = base_angle + self.p_angle
        
        # Scale update (multiplicative, bounded)
        scale_delta = torch.tanh(self.p_scale) * 0.1  # ±10% only
        curr_scale = base_scale * (1.0 + scale_delta)
        
        # Translation update
        curr_tx = base_tx + self.p_trans[:, 0]
        curr_ty = base_ty + self.p_trans[:, 1]
        
        return self.construct_affine(curr_angle, curr_scale, curr_tx, curr_ty)

    def compute_priority_map(self, rotor_variance, feature_magnitude):
        """
        Priority Map 계산 (Phase 4_2와 호환성 확보)
        
        Args:
            rotor_variance: Local variance map
            feature_magnitude: Feature strength map
        
        Returns:
            Priority map (0~1 normalized)
        """
        # 분산 하한 제한
        safe_variance = torch.clamp(rotor_variance, min=1e-3)
        stability = 1.0 / safe_variance
        
        # Raw priority
        raw_priority = stability * feature_magnitude
        
        # Robust normalization
        B = raw_priority.shape[0]
        priority_map = torch.zeros_like(raw_priority)
        
        for b in range(B):
            flat = raw_priority[b].view(-1)
            v_min = flat.min()
            v_max = torch.quantile(flat, 0.99)  # 99th percentile
            
            clipped = torch.clamp(raw_priority[b], min=v_min, max=v_max)
            priority_map[b] = (clipped - v_min) / (v_max - v_min + self.epsilon)
        
        return priority_map

    def compute_energy(self, src_feats, tgt_feats, W_current, weights, priority_map=None):
        """
        기하학적 에너지 계산
        
        Args:
            src_feats: Source feature dict
            tgt_feats: Target feature dict
            W_current: Current transformation (B, 2, 3)
            weights: (w_sdf, w_vec, w_rot)
            priority_map: Optional spatial weighting
        """
        w_sdf, w_vec, w_rot = weights
        B, _, H, W_size = src_feats['sdf'].shape
        
        # Affine grid
        grid = F.affine_grid(W_current, [B, 1, H, W_size], align_corners=False)
        
        # Warping
        warped_sdf = F.grid_sample(src_feats['sdf'], grid, 
                                    align_corners=False, padding_mode='border')
        warped_vec = F.grid_sample(src_feats['vector'], grid, 
                                    align_corners=False, padding_mode='zeros')
        warped_rot = F.grid_sample(src_feats['rotor'], grid, 
                                    align_corners=False, padding_mode='zeros')
        
        # Valid mask
        ones_mask = torch.ones_like(src_feats['sdf'])
        warped_mask = F.grid_sample(ones_mask, grid, align_corners=False)
        valid_mask = (warped_mask > 0.9).float()
        
        # Vector rotation correction
        rot_mat = W_current[:, :2, :2]
        vec_permuted = warped_vec.permute(0, 2, 3, 1)
        vec_rotated = torch.einsum('bij,bhwj->bhwi', rot_mat, vec_permuted)
        vec_corrected = vec_rotated.permute(0, 3, 1, 2)
        
        # Energy terms
        diff_sdf = torch.abs(warped_sdf - tgt_feats['sdf'])
        
        sim_vec = F.cosine_similarity(vec_corrected, tgt_feats['vector'], dim=1)
        diff_vec = (1.0 - sim_vec).unsqueeze(1)
        
        diff_rot = torch.abs(warped_rot - tgt_feats['rotor'])
        
        # Weighted energy
        energy_map = w_sdf * diff_sdf + w_vec * diff_vec + w_rot * diff_rot
        
        # Priority weighting (optional)
        if priority_map is not None:
            energy_map = energy_map * priority_map
        
        # Masked average
        masked_energy = energy_map * valid_mask
        valid_count = valid_mask.sum() + self.epsilon
        
        return masked_energy.sum() / valid_count

    def optimize(self, pyramid_a_feats, pyramid_b_feats, W_init, priority_map=None):
        """
        계층적 최적화
        
        Args:
            pyramid_a_feats: Source pyramid
            pyramid_b_feats: Target pyramid
            W_init: Initial transform (B, 2, 3)
            priority_map: Optional priority (현재 미사용, API 호환성용)
        
        Returns:
            W_final, loss_history
        """
        # Initial decomposition
        with torch.no_grad():
            base_angle, base_scale, base_tx, base_ty = self.decompose_affine(W_init)
            base_params = (base_angle, base_scale, base_tx, base_ty)
            
            # 초기 각도 출력
            init_angle_deg = np.degrees(base_angle.item())
            print(f"\n[Phase 4] Initial angle: {init_angle_deg:.2f}°")

        loss_history = []
        self.best_loss = float('inf')

        # Coarse-to-fine
        for stage_idx, level in enumerate(MPC_CONFIG['levels']):
            
            # Feature packing
            def pack_feats(p_tuple):
                s, v, b = p_tuple
                return {
                    'sdf': s[:, :1, :, :].detach(),
                    'vector': v.mean(dim=1).detach(),
                    'rotor': b[2].mean(dim=1, keepdim=True).detach()
                }
            
            safe_level = min(level, len(pyramid_a_feats) - 1)
            feat_a = pack_feats(pyramid_a_feats[safe_level])
            feat_b = pack_feats(pyramid_b_feats[safe_level])
            
            # Reset deltas
            self.reset_params()
            
            # Optimizer setup
            base_lr = MPC_CONFIG['base_lrs'][stage_idx]
            angle_boost = MPC_CONFIG['angle_boost'][stage_idx]
            
            optimizer = optim.Adam([
                {'params': [self.p_angle], 'lr': base_lr * angle_boost},
                {'params': [self.p_scale, self.p_trans], 'lr': base_lr}
            ])
            
            # Learning rate scheduler (cosine decay)
            n_iters = MPC_CONFIG['iters'][stage_idx]
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_iters, eta_min=base_lr * 0.1
            )
            
            curr_weights = MPC_CONFIG['weights'][stage_idx]
            
            # Optimization loop
            stage_best_loss = float('inf')
            stage_best_state = None
            
            for i in range(n_iters):
                optimizer.zero_grad()
                
                W_pred = self.get_current_transform(base_params)
                loss = self.compute_energy(feat_a, feat_b, W_pred, curr_weights)
                
                # Track best
                if loss.item() < stage_best_loss:
                    stage_best_loss = loss.item()
                    stage_best_state = {
                        'angle': self.p_angle.detach().clone(),
                        'scale': self.p_scale.detach().clone(),
                        'trans': self.p_trans.detach().clone()
                    }
                
                loss.backward()
                
                # Gradient clipping (prevent explosion)
                torch.nn.utils.clip_grad_norm_([self.p_angle, self.p_scale, self.p_trans], 1.0)
                
                optimizer.step()
                scheduler.step()
                
                loss_history.append(loss.item())
                
                if i % 20 == 0:
                    curr_angle_deg = np.degrees((base_angle + self.p_angle).item())
                    curr_scale = (base_scale * (1.0 + torch.tanh(self.p_scale) * 0.1)).item()
                    print(f"  L{level} [{i:3d}/{n_iters}] Loss={loss.item():.6f} | "
                          f"Angle={curr_angle_deg:.2f}° | Scale={curr_scale:.3f}")
            
            # Restore best state of this stage
            if stage_best_state is not None:
                with torch.no_grad():
                    self.p_angle.copy_(stage_best_state['angle'])
                    self.p_scale.copy_(stage_best_state['scale'])
                    self.p_trans.copy_(stage_best_state['trans'])
            
            # Update base params
            with torch.no_grad():
                base_angle = base_angle + self.p_angle
                base_scale = base_scale * (1.0 + torch.tanh(self.p_scale) * 0.1)
                base_tx = base_tx + self.p_trans[:, 0]
                base_ty = base_ty + self.p_trans[:, 1]
                base_params = (base_angle, base_scale, base_tx, base_ty)
                
            final_angle_deg = np.degrees(base_angle.item())
            print(f"  Stage {stage_idx} done. Angle: {final_angle_deg:.2f}°\n")
        
        W_final = self.construct_affine(*base_params)
        return W_final, loss_history

    def get_transform_params(self, W=None):
        """
        변환 파라미터 추출 (Phase 4_2 API 호환)
        
        Args:
            W: Optional transformation matrix. If None, uses self.W
        
        Returns:
            angle (degrees), scale, tx, ty
        """
        if W is None:
            # For compatibility with phase4_2
            if not hasattr(self, 'W'):
                raise ValueError("No transformation matrix available")
            W_np = self.W.detach().cpu().numpy()[0]
        else:
            W_np = W.detach().cpu().numpy()[0]
        
        # Extract parameters
        angle_rad = np.arctan2(W_np[1, 0], W_np[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        scale = np.sqrt(W_np[0, 0]**2 + W_np[1, 0]**2)
        
        tx = W_np[0, 2]
        ty = W_np[1, 2]
        
        return angle_deg, scale, tx, ty