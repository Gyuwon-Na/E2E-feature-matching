"""
================================================================================
Phase 4 v6r: Revised Inference-Only Refinement (Multi-Hypothesis 제거)
================================================================================
[Architecture.md §4 참조]

★ 학습 없이 기존 best model에서 즉시 성능 향상 ★
★ v6i 대비 변경: Multi-Hypothesis 제거 (Phase 3 rotor_map이 global 각도와 불일치) ★

[v6i에서 문제였던 것]
- Phase 3의 rotor_map은 Cross-Attention의 local context 출력이라
  실제 global rotation angle과 직접 대응하지 않음
- 따라서 rotor_map에서 추출한 후보 각도가 엉뚱 → Scout가 나쁜 시작점 선택
- 결과: W_init보다 더 나쁜 곳에서 출발하여 전체 성능 하락

[v6r에서 살린 개선점 (W_init 기반, 안전)]
1. [보완] Pseudo-Confidence Weighting: Rotor 공간 일관성으로 에너지 가중
   → 경계/오클루전의 잘못된 gradient를 억제
2. [보완] Clifford Rotor Distance: L1 → 1-cos(Δθ) 
   → 각도 공간의 측지선 거리, 큰 회전에서 gradient 방향 개선
3. [보완] Adaptive Angle Boost: 에너지 비례 LR 자동 조절
   → 큰 회전이 남아있으면 더 과감하게 탐색
4. [보완] 반복 횟수 증가 + Level별 가중치 미세 조정

[v5에서 유지하는 것들]
- Parameter Split: d_angle, d_scale, d_trans 개별 LR
- Coarse Strategy: Level 2에서 Vector 위주 최적화
- 단일 W_init 시작점 (Multi-Hypothesis 제거)
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
# [Hyperparameters] v6r Inference-Only Refinement
# =============================================================================
MPC_CONFIG = {
    'levels': [2, 1, 0],                          # [Hyperparameter] 최적화 레벨 순서
    'iters': [150, 80, 40],                        # [Hyperparameter] 레벨별 반복 횟수 [v6r: v5 대비 증가]
    
    # 기본 학습률 (Scale, Translation용)
    'base_lrs': [0.005, 0.002, 0.001],             # [Hyperparameter] 레벨별 기본 학습률
    
    # [v5 유지] 각도(Angle) 전용 부스트 배율
    'angle_boost': [20.0, 5.0, 1.0],               # [Hyperparameter] 레벨별 Angle LR 배율

    'weights': [
        # w_sdf(픽셀), w_vector(방향), w_rotor(회전량)
        [0.0, 2.0, 0.8],   # Level 2: 방향+Rotor 위주 [v6r: w_rot 0.5→0.8, Clifford 거리 활용]
        [0.3, 1.0, 0.5],   # Level 1: 균형 [v6r: w_sdf 0.5→0.3, Confidence가 SDF 노이즈 억제]
        [1.0, 0.2, 0.1]    # Level 0: 미세 픽셀 조정 (v5 유지)
    ]
}

# [v6r] Pseudo-Confidence 설정
ROTOR_VARIANCE_SIGMA = 0.15                        # [Hyperparameter] Rotor 분산 → Confidence 변환 온도
PSEUDO_CONF_KERNEL = 5                             # [Hyperparameter] 국소 분산 계산 커널 크기
CONFIDENCE_FLOOR = 0.1                             # [Hyperparameter] 최소 confidence (완전 무시 방지)

# [v6r] Adaptive Angle Boost 설정
ENERGY_REF = 0.5                                   # [Hyperparameter] 기준 에너지
MAX_ADAPTIVE_MULTIPLIER = 3.0                      # [Hyperparameter] 최대 adaptive 배율


def compute_pseudo_confidence(rotor_map, kernel_size=PSEUDO_CONF_KERNEL, sigma=ROTOR_VARIANCE_SIGMA):
    """
    [Phase 4 v6r — §4.0 보완] 학습 없는 Pseudo-Confidence 생성
    
    Phase 3의 Dense Rotor Map에서 '공간적 일관성(Spatial Coherence)'을 측정하여
    신뢰도를 휴리스틱으로 생성합니다.
    
    원리: 
    - 주변 KxK 이웃의 Rotor 각도가 일관되면 → 높은 신뢰도
    - Rotor 각도가 제각각이면 → 낮은 신뢰도
    
    Circular Variance = 1 - |mean(e^{iθ})| = 1 - sqrt(mean(cos)² + mean(sin)²)
    C(p) = exp(-CircVar / σ²)
    
    Args:
        rotor_map: (B, H, W, 4) — Phase 3 출력 (cos, sin, dx, dy)
        kernel_size: 국소 분산 커널 크기 [Hyperparameter]
        sigma: 분산→신뢰도 온도 [Hyperparameter]
        
    Returns:
        confidence: (B, 1, H, W) ∈ [CONFIDENCE_FLOOR, 1]
    """
    cos_map = rotor_map[..., 0].unsqueeze(1)  # (B, 1, H, W)
    sin_map = rotor_map[..., 1].unsqueeze(1)
    
    pad = kernel_size // 2
    
    # 국소 평균 cos, sin
    local_mean_cos = F.avg_pool2d(cos_map, kernel_size, stride=1, padding=pad)
    local_mean_sin = F.avg_pool2d(sin_map, kernel_size, stride=1, padding=pad)
    
    # Circular Variance
    mean_resultant = torch.sqrt(local_mean_cos**2 + local_mean_sin**2 + 1e-8)
    circular_var = 1.0 - mean_resultant
    
    # Variance → Confidence
    confidence = torch.exp(-circular_var / (sigma**2 + 1e-8))
    confidence = confidence.clamp(min=CONFIDENCE_FLOOR)
    
    return confidence


class HierarchicalMPCRefiner(nn.Module):
    """
    [Phase 4 v6r Main] Revised Inference-Only Refinement
    
    Architecture.md §4 전체 구현
    
    v6i 대비 변경: Multi-Hypothesis 제거, W_init 단일 시작점 유지
    
    [동작하는 개선점]
    1. Pseudo-Confidence Weighting (§4.3 보완)
    2. Clifford Rotor Distance (§4.3 보완)
    3. Adaptive Angle Boost (§4.4 보완)
    4. 반복 횟수 및 가중치 미세 조정
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.epsilon = 1e-6
        
        # [v5 유지] 파라미터 개별 분리
        self.p_angle = nn.Parameter(torch.zeros(1).to(device))
        self.p_scale = nn.Parameter(torch.zeros(1).to(device))
        self.p_trans = nn.Parameter(torch.zeros(1, 2).to(device))

    def reset_params(self):
        """[Helper] 최적화 파라미터 초기화"""
        with torch.no_grad():
            self.p_angle.data.zero_()
            self.p_scale.data.zero_()
            self.p_trans.data.zero_()

    def decompose_affine(self, W):
        """[§4.1] 행렬에서 기하학적 파라미터 추출"""
        a = W[:, 0, 0]
        b = W[:, 0, 1]
        tx = W[:, 0, 2]
        ty = W[:, 1, 2]
        scale = torch.sqrt(a**2 + b**2 + 1e-8)
        angle = torch.atan2(W[:, 1, 0], W[:, 0, 0])
        return angle, scale, tx, ty

    def construct_affine(self, angle, scale, tx, ty):
        """[§4.1] 파라미터 → 행렬 재조립"""
        B = angle.shape[0]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        row1_rot = torch.stack([scale * cos, -scale * sin], dim=1)
        row2_rot = torch.stack([scale * sin,  scale * cos], dim=1)
        t_x = tx.unsqueeze(1)
        t_y = ty.unsqueeze(1)
        row1 = torch.cat([row1_rot, t_x], dim=1)
        row2 = torch.cat([row2_rot, t_y], dim=1)
        return torch.stack([row1, row2], dim=1)

    def get_current_transform(self, base_params):
        """[§4.1] Base + Delta → Current W (v5 유지)"""
        base_angle, base_scale, base_tx, base_ty = base_params
        curr_angle = base_angle + self.p_angle
        scale_factor = 1.0 + torch.tanh(self.p_scale) * 0.2
        curr_scale = base_scale * scale_factor
        curr_tx = base_tx + self.p_trans[:, 0]
        curr_ty = base_ty + self.p_trans[:, 1]
        return self.construct_affine(curr_angle, curr_scale, curr_tx, curr_ty)

    def _pack_feats(self, p_tuple):
        """
        [§4.3 Helper] Phase 2 출력 패킹
        
        v6r: rotor_cos, rotor_sin 분리 추가 (Clifford Rotor Distance용)
        """
        s, v, b = p_tuple
        unit_cos, unit_sin, rotor_mag = b
        return {
            'sdf': s[:, :1, :, :].detach(),
            'vector': v.mean(dim=1).detach(),
            'rotor': rotor_mag.mean(dim=1, keepdim=True).detach(),
            # [v6r 보완] Clifford Rotor Distance용
            'rotor_cos': unit_cos.mean(dim=1, keepdim=True).detach(),
            'rotor_sin': unit_sin.mean(dim=1, keepdim=True).detach()
        }

    def compute_energy(self, src_feats, tgt_feats, W_current, weights, confidence=None):
        """
        [§4.3 보완] Confidence-Gated Clifford Energy
        
        v5 대비 변경:
        1. [보완] confidence 가중: 신뢰도 높은 영역의 gradient 우선
        2. [보완] Clifford Rotor Distance: L1 → 1-cos(Δθ)
        
        Args:
            src_feats: dict {'sdf','vector','rotor','rotor_cos','rotor_sin'}
            tgt_feats: dict (동일)
            W_current: (B, 2, 3)
            weights: [w_sdf, w_vector, w_rotor]
            confidence: (B, 1, H, W) or None [v6r 보완]
        """
        w_sdf, w_vec, w_rot = weights
        B, _, H, W = src_feats['sdf'].shape
        
        # Warping
        grid = F.affine_grid(W_current, [B, 1, H, W], align_corners=False)
        warped_sdf = F.grid_sample(src_feats['sdf'], grid, align_corners=False, padding_mode='border')
        warped_vec = F.grid_sample(src_feats['vector'], grid, align_corners=False, padding_mode='zeros')
        
        # Valid Mask (v5 유지)
        mask = (F.grid_sample(torch.ones_like(warped_sdf), grid, align_corners=False) > 0.9).float()
        
        # [v6r 보완] Confidence × Mask → 결합 가중치
        if confidence is not None:
            if confidence.shape[-2:] != (H, W):
                conf = F.interpolate(confidence, size=(H, W), mode='bilinear', align_corners=False)
            else:
                conf = confidence
            combined_weight = mask * conf
        else:
            combined_weight = mask
        
        # Vector Rotation Correction (v5 유지)
        rot_mat = W_current[:, :2, :2]
        vec_perm = warped_vec.permute(0, 2, 3, 1)
        warped_vec_corr = torch.einsum('bij,bhwj->bhwi', rot_mat, vec_perm).permute(0, 3, 1, 2)

        # [E_scalar] SDF 차이 (v5 유지)
        diff_sdf = torch.abs(warped_sdf - tgt_feats['sdf'])
        
        # [E_vector] Cosine Distance (v5 유지)
        sim_vec = F.cosine_similarity(warped_vec_corr, tgt_feats['vector'], dim=1)
        diff_vec = (1.0 - sim_vec).unsqueeze(1)
        
        # [E_rotor — v6r 보완] Clifford Rotor Distance
        # v5: diff_rot = |warped_rot - tgt_rot|  (L1, 각도 공간에서 부적합)
        # v6r: 1 - cos(θ_w - θ_t)  (Clifford Rotor 내적, 측지선 거리)
        #      cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        if 'rotor_cos' in src_feats and 'rotor_sin' in src_feats:
            warped_cos = F.grid_sample(src_feats['rotor_cos'], grid,
                                        align_corners=False, padding_mode='zeros')
            warped_sin = F.grid_sample(src_feats['rotor_sin'], grid,
                                        align_corners=False, padding_mode='zeros')
            rotor_inner = warped_cos * tgt_feats['rotor_cos'] + warped_sin * tgt_feats['rotor_sin']
            diff_rot = (1.0 - rotor_inner).clamp(min=0)
        else:
            # Fallback: v5 L1
            warped_rot = F.grid_sample(src_feats['rotor'], grid,
                                        align_corners=False, padding_mode='zeros')
            diff_rot = torch.abs(warped_rot - tgt_feats['rotor'])
        
        # 총 에너지 (Confidence-Weighted)
        loss_map = w_sdf * diff_sdf + w_vec * diff_vec + w_rot * diff_rot
        energy = (loss_map * combined_weight).sum() / (combined_weight.sum() + self.epsilon)
        return energy

    def _compute_adaptive_boost(self, initial_energy, stage_idx):
        """
        [§4.4 보완] Adaptive Angle Boost
        에너지 높으면(=큰 회전 남음) → Angle LR을 더 높임
        """
        base_boost = MPC_CONFIG['angle_boost'][stage_idx]
        energy_ratio = min(initial_energy / (ENERGY_REF + self.epsilon), MAX_ADAPTIVE_MULTIPLIER)
        return base_boost * (1.0 + max(0, energy_ratio - 1.0))

    def optimize(self, pyramid_a_feats, pyramid_b_feats, W_init, rotor_map=None):
        """
        [Phase 4 v6r Main Optimization]
        
        ★ 기존 Phase 3 best model과 100% 호환 ★
        ★ Multi-Hypothesis 제거 — W_init에서 안전하게 출발 ★
        
        rotor_map이 있으면 → Pseudo-Confidence 가중 + Clifford 거리 + Adaptive Boost
        rotor_map이 None이면 → Clifford 거리 + Adaptive Boost (Confidence 없이)
        
        Args:
            pyramid_a_feats: Phase 2 출력 피라미드 (source)
            pyramid_b_feats: Phase 2 출력 피라미드 (target)
            W_init: Phase 3/3.5의 초기 변환 행렬 (B, 2, 3)
            rotor_map: Phase 3의 Dense Rotor Map (B, H, W, 4) [Optional]
        """
        loss_history = []
        pseudo_conf = None
        
        # =================================================================
        # [§4.0 보완] Pseudo-Confidence 생성 (학습 불필요)
        # =================================================================
        if rotor_map is not None:
            print("\n[Phase 4 v6r] Confidence-Gated Clifford Refinement")
            pseudo_conf = compute_pseudo_confidence(rotor_map)
            print(f"  [Pseudo-Confidence] Mean={pseudo_conf.mean().item():.3f}, "
                  f"Std={pseudo_conf.std().item():.3f}")
        else:
            print("\n[Phase 4 v6r] Clifford Refinement (no rotor_map)")
        
        # W_init에서 출발 (v5와 동일)
        with torch.no_grad():
            base_params = self.decompose_affine(W_init)
        
        # =================================================================
        # [§4.2~4.4] Coarse-to-Fine Optimization
        # =================================================================
        for stage_idx, level in enumerate(MPC_CONFIG['levels']):
            safe_level = min(level, len(pyramid_a_feats) - 1)
            feat_a = self._pack_feats(pyramid_a_feats[safe_level])
            feat_b = self._pack_feats(pyramid_b_feats[safe_level])
            
            # Confidence를 현재 해상도에 맞춤
            stage_conf = None
            if pseudo_conf is not None:
                H_s, W_s = feat_a['sdf'].shape[-2:]
                stage_conf = F.interpolate(pseudo_conf, size=(H_s, W_s),
                                           mode='bilinear', align_corners=False)
            
            self.reset_params()
            
            # [v6r 보완] Adaptive Angle Boost
            with torch.no_grad():
                W_check = self.construct_affine(*base_params)
                init_energy = self.compute_energy(feat_a, feat_b, W_check,
                                                  MPC_CONFIG['weights'][stage_idx], stage_conf)
            
            base_lr = MPC_CONFIG['base_lrs'][stage_idx]
            angle_boost = self._compute_adaptive_boost(init_energy.item(), stage_idx)
            
            print(f"  [Stage {stage_idx}] Level={level}, Energy={init_energy.item():.6f}, "
                  f"AdaptiveBoost={angle_boost:.1f}x")
            
            # [v5 유지] Parameter Groups로 LR 차등 적용
            optimizer = optim.Adam([
                {'params': [self.p_angle], 'lr': base_lr * angle_boost},
                {'params': [self.p_scale, self.p_trans], 'lr': base_lr}
            ])
            
            curr_weights = MPC_CONFIG['weights'][stage_idx]
            
            for i in range(MPC_CONFIG['iters'][stage_idx]):
                optimizer.zero_grad()
                W_pred = self.get_current_transform(base_params)
                loss = self.compute_energy(feat_a, feat_b, W_pred, curr_weights, stage_conf)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
            
            # Update Base Params
            with torch.no_grad():
                base_angle = base_params[0] + self.p_angle
                base_scale = base_params[1] * (1.0 + torch.tanh(self.p_scale) * 0.2)
                base_tx = base_params[2] + self.p_trans[:, 0]
                base_ty = base_params[3] + self.p_trans[:, 1]
                base_params = (base_angle, base_scale, base_tx, base_ty)
        
        W_final = self.construct_affine(*base_params)
        return W_final, loss_history