"""
================================================================================
Phase 4 Dynamic v3: Angle-Error-Aware Strategy
================================================================================
[Root Cause Analysis]
Dynamic V2 실패 이유:
- Energy/Coherence는 "각도 오차"를 직접 측정하지 못함
- 59° 오차도 "coherence 높음" → Scouting Skip → 실패

[Solution]
1. Phase 3 행렬에서 회전 성분 추출
2. GT와의 각도 차이 직접 계산 (불가능)
3. 대신: Rotor Field의 "분산"으로 간접 측정
   - 회전이 크면 → Rotor 일관성 낮음 → 분산 큼
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

MPC_CONFIG = {
    'levels': [2, 1, 0],
    'iters': [100, 50, 30],
    'base_lrs': [0.005, 0.002, 0.001],
    'angle_boost': [20.0, 5.0, 1.0],
    'weights': [[0.0, 2.0, 0.5], [0.5, 1.0, 0.5], [1.0, 0.2, 0.1]]
}

class DynamicScoutRefiner(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        self.p_angle = nn.Parameter(torch.zeros(1).to(device))
        self.p_scale = nn.Parameter(torch.zeros(1).to(device))
        self.p_trans = nn.Parameter(torch.zeros(1, 2).to(device))

    def reset_params(self):
        with torch.no_grad():
            self.p_angle.data.zero_()
            self.p_scale.data.zero_()
            self.p_trans.data.zero_()

    def decompose_affine(self, W):
        a, b, tx, ty = W[:, 0, 0], W[:, 0, 1], W[:, 0, 2], W[:, 1, 2]
        scale = torch.sqrt(a**2 + b**2 + 1e-8)
        angle = torch.atan2(W[:, 1, 0], W[:, 0, 0])
        return angle, scale, tx, ty

    def construct_affine(self, angle, scale, tx, ty):
        cos, sin = torch.cos(angle), torch.sin(angle)
        row1 = torch.stack([scale*cos, -scale*sin, tx], dim=1)
        row2 = torch.stack([scale*sin,  scale*cos, ty], dim=1)
        return torch.stack([row1, row2], dim=1)

    def get_current_transform(self, base_params):
        b_ang, b_scl, b_tx, b_ty = base_params
        curr_ang = b_ang + self.p_angle
        curr_scl = b_scl * (1.0 + torch.tanh(self.p_scale) * 0.2)
        curr_tx = b_tx + self.p_trans[:, 0]
        curr_ty = b_ty + self.p_trans[:, 1]
        return self.construct_affine(curr_ang, curr_scl, curr_tx, curr_ty)

    def _pack_feats(self, p_tuple):
        s, v, b = p_tuple
        return {
            'sdf': s[:, :1].detach(),
            'vector': v.mean(dim=1).detach(),
            'rotor': b[2].mean(dim=1, keepdim=True).detach()
        }

    def compute_energy(self, src_feats, tgt_feats, W_curr, weights):
        w_sdf, w_vec, w_rot = weights
        B, _, H, W = src_feats['sdf'].shape
        
        grid = F.affine_grid(W_curr, [B, 1, H, W], align_corners=False)
        warped_sdf = F.grid_sample(src_feats['sdf'], grid, align_corners=False, padding_mode='border')
        warped_vec = F.grid_sample(src_feats['vector'], grid, align_corners=False, padding_mode='zeros')
        mask = (F.grid_sample(torch.ones_like(warped_sdf), grid, align_corners=False) > 0.9).float()
        
        rot_mat = W_curr[:, :2, :2]
        vec_perm = warped_vec.permute(0, 2, 3, 1)
        warped_vec_corr = torch.einsum('bij,bhwj->bhwi', rot_mat, vec_perm).permute(0, 3, 1, 2)

        diff_sdf = torch.abs(warped_sdf - tgt_feats['sdf'])
        sim_vec = F.cosine_similarity(warped_vec_corr, tgt_feats['vector'], dim=1)
        diff_vec = (1.0 - sim_vec).unsqueeze(1)
        
        if 'rotor' in src_feats:
            warped_rot = F.grid_sample(src_feats['rotor'], grid, align_corners=False, padding_mode='zeros')
            diff_rot = torch.abs(warped_rot - tgt_feats['rotor'])
        else:
            diff_rot = torch.zeros_like(diff_sdf)

        loss_map = w_sdf * diff_sdf + w_vec * diff_vec + w_rot * diff_rot
        return (loss_map * mask).sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) + 1e-6)

    def estimate_rotation_quality(self, pyramid_a, pyramid_b, W_init):
        """
        [V3 Key Innovation] 회전 오차를 Rotor Field 분석으로 간접 측정
        """
        safe_l = min(2, len(pyramid_a)-1)
        f_a = self._pack_feats(pyramid_a[safe_l])
        f_b = self._pack_feats(pyramid_b[safe_l])
        
        with torch.no_grad():
            # 1. Basic Energy
            init_energy = self.compute_energy(f_a, f_b, W_init, [0.5, 1.0, 0.5]).item()
            
            # 2. Rotor Inconsistency (회전 오차의 프록시)
            # W_init으로 warping한 후, Target과의 Rotor 차이의 "공간적 변동성" 측정
            B, _, H, W = f_a['sdf'].shape
            grid = F.affine_grid(W_init, [B, 1, H, W], align_corners=False)
            
            warped_rotor = F.grid_sample(f_a['rotor'], grid, align_corners=False, padding_mode='zeros')
            rotor_diff = torch.abs(warped_rotor - f_b['rotor'])
            
            # 공간적 분산 측정 (회전 오차가 크면 → 위치마다 다른 오차 → 분산 큼)
            rotor_variance = rotor_diff.var().item()
            
            # 3. Estimated Angle from Matrix (참고용)
            est_angle_rad = self.decompose_affine(W_init)[0].item()
            est_angle_deg = abs(est_angle_rad * 180.0 / np.pi)
            
            print(f"    Energy: {init_energy:.3f} | Rotor Var: {rotor_variance:.4f} | Est Angle: {est_angle_deg:.1f}°")
        
        # [Decision Tree]
        # High rotor variance → 큰 회전 오차 → Scouting 필요
        if rotor_variance > 0.015:  # 큰 회전 오차
            return {'scout_count': 16, 'jitter_angle': 70.0, 'jitter_trans': 0.15}
        elif rotor_variance > 0.008:  # 중간 회전 오차
            return {'scout_count': 10, 'jitter_angle': 50.0, 'jitter_trans': 0.12}
        elif init_energy > 0.6:  # Energy는 높은데 Rotor는 괜찮음 → Translation 문제
            return {'scout_count': 6, 'jitter_angle': 30.0, 'jitter_trans': 0.15}
        else:  # 작은 오차 → V1 Direct
            return {'scout_count': 0, 'jitter_angle': 0.0}

    def generate_scouts(self, W_init, params):
        count = params['scout_count']
        if count == 0:
            return W_init
        
        B = W_init.shape[0]
        device = W_init.device
        ang, scl, tx, ty = self.decompose_affine(W_init)
        
        # Stratified Angle Sampling
        angles = torch.linspace(-1, 1, count, device=device) * params['jitter_angle'] * (np.pi/180.0)
        angles = angles.unsqueeze(0).repeat(B, 1)
        
        noise_scl = (torch.rand(B, count, device=device) - 0.5) * 0.15
        noise_tx = (torch.rand(B, count, device=device) - 0.5) * 2 * params['jitter_trans']
        noise_ty = (torch.rand(B, count, device=device) - 0.5) * 2 * params['jitter_trans']
        
        angles[:, 0] = 0; noise_scl[:, 0] = 0; noise_tx[:, 0] = 0; noise_ty[:, 0] = 0
        
        scout_ang = (ang.unsqueeze(1) + angles).view(-1)
        scout_scl = (scl.unsqueeze(1) * (1.0 + noise_scl)).view(-1)
        scout_tx = (tx.unsqueeze(1) + noise_tx).view(-1)
        scout_ty = (ty.unsqueeze(1) + noise_ty).view(-1)
        
        return self.construct_affine(scout_ang, scout_scl, scout_tx, scout_ty)

    def optimize(self, pyramid_a_feats, pyramid_b_feats, W_init):
        B = W_init.shape[0]
        
        # Step 1: Rotation-Aware Scouting Decision
        scout_params = self.estimate_rotation_quality(pyramid_a_feats, pyramid_b_feats, W_init)
        
        if scout_params['scout_count'] > 0:
            print(f"\n[Dynamic V3] Scouting: {scout_params['scout_count']} scouts @ ±{scout_params['jitter_angle']:.0f}°")
            
            safe_l = min(MPC_CONFIG['levels'][0], len(pyramid_a_feats)-1)
            f_a = self._pack_feats(pyramid_a_feats[safe_l])
            f_b = self._pack_feats(pyramid_b_feats[safe_l])
            
            N = scout_params['scout_count']
            f_a_exp = {k: v.repeat_interleave(N, dim=0) for k,v in f_a.items()}
            f_b_exp = {k: v.repeat_interleave(N, dim=0) for k,v in f_b.items()}
            
            W_scouts = self.generate_scouts(W_init, scout_params)
            
            with torch.no_grad():
                energies = self.compute_energy(f_a_exp, f_b_exp, W_scouts, MPC_CONFIG['weights'][0])
                energies = energies.view(B, N)
                best_idx = torch.argmin(energies, dim=1)
                
                flat_idx = torch.arange(B, device=self.device) * N + best_idx
                W_start = W_scouts[flat_idx]
                
                # Confidence Check
                orig_energy = energies[:, 0].mean().item()
                best_energy = energies.min(dim=1).values.mean().item()
                improvement = (orig_energy - best_energy) / (orig_energy + 1e-6)
                
                if improvement < 0.05:  # 5% 미만 개선 → 원본 사용
                    print(f"  → Scout no help ({improvement*100:.1f}%), using W_init")
                    W_start = W_init
                else:
                    print(f"  → Scout improved by {improvement*100:.1f}%")
        else:
            W_start = W_init
            print(f"\n[Dynamic V3] Low rotation error → Direct V1")

        # Step 2: V1 Refinement
        with torch.no_grad():
            base_params = self.decompose_affine(W_start)

        loss_history = []
        
        for stage_idx, level in enumerate(MPC_CONFIG['levels']):
            safe_l = min(level, len(pyramid_a_feats)-1)
            f_a = self._pack_feats(pyramid_a_feats[safe_l])
            f_b = self._pack_feats(pyramid_b_feats[safe_l])
            
            self.reset_params()
            
            base_lr = MPC_CONFIG['base_lrs'][stage_idx]
            angle_mult = MPC_CONFIG['angle_boost'][stage_idx]
            
            optimizer = optim.Adam([
                {'params': [self.p_angle], 'lr': base_lr * angle_mult},
                {'params': [self.p_scale, self.p_trans], 'lr': base_lr}
            ])
            
            curr_w = MPC_CONFIG['weights'][stage_idx]
            
            for _ in range(MPC_CONFIG['iters'][stage_idx]):
                optimizer.zero_grad()
                W_pred = self.get_current_transform(base_params)
                loss = self.compute_energy(f_a, f_b, W_pred, curr_w).mean()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

            with torch.no_grad():
                b_ang, b_scl, b_tx, b_ty = base_params
                new_ang = b_ang + self.p_angle
                new_scl = b_scl * (1.0 + torch.tanh(self.p_scale) * 0.2)
                new_tx = b_tx + self.p_trans[:, 0]
                new_ty = b_ty + self.p_trans[:, 1]
                base_params = (new_ang, new_scl, new_tx, new_ty)

        W_final = self.construct_affine(*base_params)
        return W_final, loss_history