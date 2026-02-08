"""
================================================================================
Phase 4 Final: V1 (Angle Booster) + Scouting
================================================================================
[승리 공식]
1. Base Engine: V1 (Angle Parameter 분리 + Angle Boost LR) -> 가장 안정적
2. Add-on: Scouting (Jittering) -> V1이 해결 못하는 큰 초기 오차 극복

[프로세스]
1. Scout: V1 로직을 돌리기 전, 16개의 후보 위치를 찔러봄 (Energy Check)
2. Handover: 가장 에러가 낮은 위치를 선정
3. Optimize: 거기서부터 V1의 강력한 Angle Boosting으로 미세 조정
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

# =============================================================================
# [Hyperparameters] V1 Config + Scouting
# =============================================================================
MPC_CONFIG = {
    # [V1 Original Config]
    'levels': [2, 1, 0],
    'iters': [100, 50, 30],         # V1과 동일
    'base_lrs': [0.005, 0.002, 0.001],
    'angle_boost': [20.0, 5.0, 1.0], # V1의 핵심 (각도 우선)
    'weights': [[0.0, 2.0, 0.5], [0.5, 1.0, 0.5], [1.0, 0.2, 0.1]],
    
    # [Added: Scouting Config]
    'scout_count': 12,              # 12개 후보 정찰 (속도/성능 타협)
    'jitter_angle': 45.0,           # V1이 커버 못하는 45도 이상 범위 탐색
    'jitter_trans': 0.1,
    'jitter_scale': 0.1
}

class HierarchicalMPCRefiner(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # [V1] 파라미터 분리 (Angle/Scale/Trans)
        self.p_angle = nn.Parameter(torch.zeros(1).to(device))
        self.p_scale = nn.Parameter(torch.zeros(1).to(device))
        self.p_trans = nn.Parameter(torch.zeros(1, 2).to(device))

    def reset_params(self):
        with torch.no_grad():
            self.p_angle.data.zero_()
            self.p_scale.data.zero_()
            self.p_trans.data.zero_()

    # ==========================================================================
    # [V1 Logic] Decomposed Geometry
    # ==========================================================================
    def decompose_affine(self, W):
        """행렬 -> 파라미터 분해"""
        a, b, tx, ty = W[:, 0, 0], W[:, 0, 1], W[:, 0, 2], W[:, 1, 2]
        scale = torch.sqrt(a**2 + b**2 + 1e-8)
        angle = torch.atan2(W[:, 1, 0], W[:, 0, 0])
        return angle, scale, tx, ty

    def construct_affine(self, angle, scale, tx, ty):
        """파라미터 -> 행렬 조립"""
        cos, sin = torch.cos(angle), torch.sin(angle)
        row1 = torch.stack([scale*cos, -scale*sin, tx], dim=1)
        row2 = torch.stack([scale*sin,  scale*cos, ty], dim=1)
        return torch.stack([row1, row2], dim=1)

    def get_current_transform(self, base_params):
        """V1의 Update 방식 (Additive Angle, Multiplicative Scale)"""
        b_ang, b_scl, b_tx, b_ty = base_params
        
        curr_ang = b_ang + self.p_angle
        curr_scl = b_scl * (1.0 + torch.tanh(self.p_scale) * 0.2) # 안전장치
        curr_tx = b_tx + self.p_trans[:, 0]
        curr_ty = b_ty + self.p_trans[:, 1]
        
        return self.construct_affine(curr_ang, curr_scl, curr_tx, curr_ty)

    # ==========================================================================
    # [Energy Function] V1 Standard
    # ==========================================================================
    def _pack_feats(self, p_tuple):
        s, v, b = p_tuple
        return {
            'sdf': s[:, :1].detach(),
            'vector': v.mean(dim=1).detach(),
            'rotor': b[2].mean(dim=1, keepdim=True).detach() # V1은 Rotor L1 사용
        }

    def compute_energy(self, src_feats, tgt_feats, W_curr, weights):
        w_sdf, w_vec, w_rot = weights
        B, _, H, W = src_feats['sdf'].shape
        
        grid = F.affine_grid(W_curr, [B, 1, H, W], align_corners=False)
        warped_sdf = F.grid_sample(src_feats['sdf'], grid, align_corners=False, padding_mode='border')
        warped_vec = F.grid_sample(src_feats['vector'], grid, align_corners=False, padding_mode='zeros')
        mask = (F.grid_sample(torch.ones_like(warped_sdf), grid, align_corners=False) > 0.9).float()
        
        # Vector Rotation Correction
        rot_mat = W_curr[:, :2, :2]
        vec_perm = warped_vec.permute(0, 2, 3, 1)
        warped_vec_corr = torch.einsum('bij,bhwj->bhwi', rot_mat, vec_perm).permute(0, 3, 1, 2)

        # Basic Losses
        diff_sdf = torch.abs(warped_sdf - tgt_feats['sdf'])
        sim_vec = F.cosine_similarity(warped_vec_corr, tgt_feats['vector'], dim=1)
        diff_vec = (1.0 - sim_vec).unsqueeze(1)
        
        # Rotor Loss (V1 uses L1)
        if 'rotor' in src_feats:
            warped_rot = F.grid_sample(src_feats['rotor'], grid, align_corners=False, padding_mode='zeros')
            diff_rot = torch.abs(warped_rot - tgt_feats['rotor'])
        else:
            diff_rot = torch.zeros_like(diff_sdf)

        loss_map = w_sdf * diff_sdf + w_vec * diff_vec + w_rot * diff_rot
        
        # Scouting을 위해 배치별 평균 반환
        return (loss_map * mask).sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) + 1e-6)

    # ==========================================================================
    # [Scouting] 정찰병 보내기
    # ==========================================================================
    def generate_scouts(self, W_init, count):
        B = W_init.shape[0]
        device = W_init.device
        
        ang, scl, tx, ty = self.decompose_affine(W_init)
        
        # Noise (첫번째는 원본 유지)
        noise_ang = (torch.rand(B, count, device=device) - 0.5) * 2 * MPC_CONFIG['jitter_angle'] * (np.pi/180.0)
        noise_scl = (torch.rand(B, count, device=device) - 0.5) * 2 * MPC_CONFIG['jitter_scale']
        noise_tx = (torch.rand(B, count, device=device) - 0.5) * 2 * MPC_CONFIG['jitter_trans']
        noise_ty = (torch.rand(B, count, device=device) - 0.5) * 2 * MPC_CONFIG['jitter_trans']
        
        noise_ang[:, 0] = 0; noise_scl[:, 0] = 0; noise_tx[:, 0] = 0; noise_ty[:, 0] = 0
        
        scout_ang = (ang.unsqueeze(1) + noise_ang).view(-1)
        scout_scl = (scl.unsqueeze(1) * (1.0 + noise_scl)).view(-1)
        scout_tx = (tx.unsqueeze(1) + noise_tx).view(-1)
        scout_ty = (ty.unsqueeze(1) + noise_ty).view(-1)
        
        return self.construct_affine(scout_ang, scout_scl, scout_tx, scout_ty)

    # ==========================================================================
    # [Main Optimize] Scout + V1 Boost
    # ==========================================================================
    def optimize(self, pyramid_a_feats, pyramid_b_feats, W_init):
        loss_history = []
        B = W_init.shape[0]
        
        # --- Step 1: Scouting ---
        # Level 2 (Coarse) Feature로 빠르게 정찰
        safe_l = min(MPC_CONFIG['levels'][0], len(pyramid_a_feats)-1)
        f_a = self._pack_feats(pyramid_a_feats[safe_l])
        f_b = self._pack_feats(pyramid_b_feats[safe_l])
        
        N = MPC_CONFIG['scout_count']
        f_a_exp = {k: v.repeat_interleave(N, dim=0) for k,v in f_a.items()}
        f_b_exp = {k: v.repeat_interleave(N, dim=0) for k,v in f_b.items()}
        
        W_scouts = self.generate_scouts(W_init, N)
        
        with torch.no_grad():
            energies = self.compute_energy(f_a_exp, f_b_exp, W_scouts, MPC_CONFIG['weights'][0])
            energies = energies.view(B, N)
            best_idx = torch.argmin(energies, dim=1)
            
            flat_idx = torch.arange(B, device=self.device) * N + best_idx
            W_start = W_scouts[flat_idx]
            print(f"[Phase 4 V1+Scout] Best start energy: {energies.min(dim=1).values.mean().item():.4f}")

        # --- Step 2: V1 Optimization ---
        # 가장 좋은 위치(W_start)에서 V1의 Angle Boost 시작
        with torch.no_grad():
            base_params = self.decompose_affine(W_start) # Start from Scout result

        for stage_idx, level in enumerate(MPC_CONFIG['levels']):
            safe_l = min(level, len(pyramid_a_feats)-1)
            f_a = self._pack_feats(pyramid_a_feats[safe_l])
            f_b = self._pack_feats(pyramid_b_feats[safe_l])
            
            self.reset_params()
            
            # [V1 핵심] Angle Boost LR
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

            # Update Base
            with torch.no_grad():
                b_ang, b_scl, b_tx, b_ty = base_params
                new_ang = b_ang + self.p_angle
                new_scl = b_scl * (1.0 + torch.tanh(self.p_scale) * 0.2)
                new_tx = b_tx + self.p_trans[:, 0]
                new_ty = b_ty + self.p_trans[:, 1]
                base_params = (new_ang, new_scl, new_tx, new_ty)

        W_final = self.construct_affine(*base_params)
        return W_final, loss_history