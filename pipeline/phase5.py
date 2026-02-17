# =============================================================================
# [ARCHITECTURE MAPPING (Phase4 + Addendum 6.4)]
# =============================================================================
# [ARCH L0441] ## **📂  4: 기하학적 에너지 기반 MPC 정제 — 추론 단계에서만** -> GeometricMPCRefiner (class overview)
# [ARCH L0442]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0443] Phase 4는 딥러닝이 예측한 매칭 지도를 바탕으로 물리적인 에너지 함수를 최소화하여 **0.1 픽셀 단위의 초정밀 정렬**을 달성하는 단계입니다. -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0444]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0445] ### **1. 전역 필터링 및 초기화** -> GeometricMPCRefiner (class overview)
# [ARCH L0446]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0447] - **역할:** 최적화 연산이 엉뚱한 곳에서 시작하지 않도록 기준점을 잡아줌 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0448] - **전역 필터링:** Phase 2의 평균 Rotor(Sin/Cos)를 비교해 이미지 전체가 대략 몇 도 돌아갔는지 파악하여 터무니없는 후보군을 제거 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0449] - **$W_0$ 설정:** 평균 Rotor(회전)와 벡터 크기 비율(줌)을 결합하여 초기 변환 행렬 **$W_0$**를 생성 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0450]     - "대략 30도 돌아갔고 1.2배 커졌다"는 사실을 알고 최적화를 시작하므로 수렴 속도가 비약적으로 빨라짐 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0451]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0452] ### **2. 지역 탐색 (Priority Search)** -> GeometricMPCRefiner.compute_priority_map / build_priority_map_from_features
# [ARCH L0453]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0454] - **역할:** "어디부터 정밀하게 맞출 것인가?"라는 **우선순위 지도**를 만듭니다. -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0455] - **방법:** Phase 3에서 배운 **Group Conv 특징**과 지역적 Rotor 분산(Variance)을 결합 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0456]     - 회전 정보가 일관되고 기하학적 덩어리가 뚜렷한 구역(예: 건물의 모서리)에 높은 가중치를 주어, 신뢰도가 높은 지역부터 자석처럼 딱딱 들어맞게 유도합니다. -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0457]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0458] ### **3. 에너지 평면 생성** -> GeometricMPCRefiner (class overview)
# [ARCH L0459]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0460] 이 시스템의 핵심인 **에너지 함수**입니다. S, V, B 세 가지 성분을 물리적으로 결합하여 오차를 계산 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0461]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0462] $$ -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0463] E_{total} = \frac{1}{N} \sum_{p} \left( g_s(p) \cdot E_{scalar}(p) + g_v(p) \cdot E_{vector}(p) + g_b(p) \cdot E_{bivector}(p) \right) -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0464] $$ -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0465]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0466] - **$E_{scalar}$ (에너지/SDF):** Softplus로 정제된 SDF 값의 차이를 계산 (→ 미분값이 매끄러워 최적화 엔진이 '골짜기'를 타고 내려가기 좋음) -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0467] - **$E_{vector}$ (방향/흐름):** 변환(W) 후에도 벡터의 방향이 일치하는지 확인 (→ 이미지가 회전했다면 벡터도 그만큼 돌아가야 한다는 **방향 보존성**을 강제) -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0468] - **$E_{bivector}$ (Rotor 일관성):** 단순히 위치만 맞는 게 아니라, 해당 지점의 **지역적인 회전/줌 상태**가 전체 변환 행렬과 기하학적으로 일치하는지 봄 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0469] - 회전하여 검정색으로 잘린 영역에 대해서는 Loss X -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0470]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0471] ### **4. 기하학적 게이트 가중 최적화 (Gate-Guided Refinement)** -> GeometricMPCRefiner (class overview)
# [ARCH L0472]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0473] - **핵심:** Phase 3(인코딩 과정 중)의 **Geometric Descriptor Guidance**에서 나온 3개의 Gate 값($~~g_s, g_v, g_b~~$)을 최적화 가중치로 직접 사용 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0474] - **지능적 최적화:** -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0475]     - 엣지가 선명한 곳은 $g_v$(Vector)를 높여 방향 정밀도를 높입니다. -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0476]     - 텍스처가 복잡한 곳은 $g_s$(Scalar)를 높여 픽셀 일치도를 높입니다. -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0477]         - 모델이 "이 구역은 벡터 정보가 믿을만해!"라고 판단한 정보를 MPC가 적극 수용하여 루프를 돌림으로써, 단순 계산보다 훨씬 견고한 정제가 가능 -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0478]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0479] --- -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0480]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0590] ### 6.4 Phase 4.2 구현 메모 -> GeometricMPCRefiner (class overview)
# [ARCH L0591]  -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0592] - **Similarity Transform 파라미터화 최적화:**   -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0593]   `(theta, tx, ty, log_scale)` 를 Adam으로 최적화한 뒤 2×3 affine로 재구성합니다. -> GeometricMPCRefiner.optimize
# [ARCH L0594] - **Valid Mask (검정 잘림 영역 Loss 제외):**   -> GeometricMPCRefiner.compute_energy (valid_mask)
# [ARCH L0595]   warp 과정에서 in-bounds mask를 생성하고, out-of-bounds 픽셀은 energy 계산에서 제외합니다. -> GeometricMPCRefiner.compute_energy (valid_mask)
# [ARCH L0596] - **Priority Map 자동 생성 옵션:**   -> GeometricMPCRefiner.compute_priority_map / build_priority_map_from_features
# [ARCH L0597]   priority_map이 주어지지 않으면,   -> GeometricMPCRefiner.compute_priority_map / build_priority_map_from_features
# [ARCH L0598]   - rotor_map의 **지역 회전 분산(variance)**   -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0599]   - mpc_map의 **벡터장 크기(magnitude)**   -> GeometricMPCRefiner (see optimize/compute_energy)
# [ARCH L0600]   를 결합하여 priority 가중치를 만들 수 있습니다. -> GeometricMPCRefiner.compute_priority_map / build_priority_map_from_features
# [ARCH L0601]  -> GeometricMPCRefiner (see optimize/compute_energy)
# =============================================================================

"""
================================================================================
Phase 4_2 Improved: Robust Geometric MPC Refiner (Fixed)
================================================================================
수정 사항:
- get_current_W()에서 텐서 shape 불일치 문제 해결
- torch.stack 대신 직접 텐서 구성
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional


class GeometricMPCRefiner(nn.Module):
    """
    Geometric MPC Refiner - Fixed Version
    """
    
    def __init__(self, device='cuda', config: Optional[Dict] = None):
        super().__init__()
        self.device = device
        self.epsilon = 1e-6
        
        # Default configuration
        self.config = {
            'learning_rate': 0.01,
            'iterations': 150,
            'patience': 30,
            'angle_lr_mult': 2.0,
            'scale_range': 0.15,
            'trans_range': 0.1,
            'huber_delta': 0.1,
            'warmup_iters': 10,
            'use_priority': True,
            'verbose': True
        }
        
        if config:
            self.config.update(config)
        
        # Learnable parameters (decomposed) - 모두 1D 텐서로 통일
        self.p_angle = nn.Parameter(torch.zeros(1, device=device))
        self.p_scale = nn.Parameter(torch.zeros(1, device=device))
        self.p_trans = nn.Parameter(torch.zeros(2, device=device))
        
        # Base parameters
        self.base_angle = 0.0
        self.base_scale = 1.0
        
        # Tracking
        self.best_loss = float('inf')
        self.best_state = None
        self.W = None
        
    def reset_parameters(self):
        """Reset learnable parameters to zero"""
        with torch.no_grad():
            self.p_angle.zero_()
            self.p_scale.zero_()
            self.p_trans.zero_()
        self.best_loss = float('inf')
        self.best_state = None
    
    def global_filtering_init(self, mean_rotor: float, mean_scale: float = 1.0):
        """Initialize from Phase 3 predictions"""
        angle_deg = np.degrees(mean_rotor)
        
        if abs(angle_deg) > 90:
            if self.config['verbose']:
                print(f"[Phase 4] Warning: Large init angle ({angle_deg:.1f} deg), clamping to +/-90")
            mean_rotor = np.clip(mean_rotor, -np.pi/2, np.pi/2)
        
        self.base_angle = float(mean_rotor)
        self.base_scale = float(np.clip(mean_scale, 0.5, 2.0))
        
        self.reset_parameters()
        self._update_W()
        
        if self.config['verbose']:
            print(f"[Phase 4] Init: Angle={np.degrees(self.base_angle):.2f} deg, Scale={self.base_scale:.3f}")
    
    def _update_W(self):
        """Update the transformation matrix from current parameters"""
        angle = self.base_angle + self.p_angle.item()
        
        scale_delta = torch.tanh(self.p_scale).item() * self.config['scale_range']
        scale = self.base_scale * (1.0 + scale_delta)
        
        tx = torch.tanh(self.p_trans[0]).item() * self.config['trans_range']
        ty = torch.tanh(self.p_trans[1]).item() * self.config['trans_range']
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        W = torch.zeros(1, 2, 3, device=self.device)
        W[0, 0, 0] = scale * cos_a
        W[0, 0, 1] = -scale * sin_a
        W[0, 0, 2] = tx
        W[0, 1, 0] = scale * sin_a
        W[0, 1, 1] = scale * cos_a
        W[0, 1, 2] = ty
        
        self.W = W
        return W
    
    def get_current_W(self) -> torch.Tensor:
        """
        Get current transformation matrix (differentiable)
        
        Returns:
            W: [1, 2, 3] affine transformation matrix
        """
        # Current angle - p_angle is [1] tensor
        angle = self.base_angle + self.p_angle[0]  # scalar tensor
        
        # Scale with bounded delta
        scale_delta = torch.tanh(self.p_scale[0]) * self.config['scale_range']
        scale = self.base_scale * (1.0 + scale_delta)
        
        # Translation (bounded) - p_trans is [2] tensor
        tx = torch.tanh(self.p_trans[0]) * self.config['trans_range']
        ty = torch.tanh(self.p_trans[1]) * self.config['trans_range']
        
        # Construct W matrix - 직접 텐서 구성 (torch.stack 사용 안함)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # 2x3 행렬 직접 구성
        W = torch.zeros(1, 2, 3, device=self.device, dtype=torch.float32)
        W[0, 0, 0] = scale * cos_a
        W[0, 0, 1] = -scale * sin_a
        W[0, 0, 2] = tx
        W[0, 1, 0] = scale * sin_a
        W[0, 1, 1] = scale * cos_a
        W[0, 1, 2] = ty
        
        return W
    
    def _normalize_features(self, feat_dict: Dict) -> Dict:
        """Normalize input features to consistent format"""
        normalized = {}
        
        # SDF
        sdf = feat_dict['sdf']
        if sdf.dim() == 3:
            sdf = sdf.unsqueeze(0)
        if sdf.shape[1] > 1:
            sdf = sdf.mean(dim=1, keepdim=True)
        normalized['sdf'] = sdf
        
        # Vector
        vec = feat_dict['vector']
        if vec.dim() == 5:  # [B, C, 2, H, W]
            B, C, _, H, W = vec.shape
            vec = vec.mean(dim=1)  # [B, 2, H, W]
        elif vec.dim() == 4 and vec.shape[1] > 2:
            B, C2, H, W = vec.shape
            n_vecs = C2 // 2
            vec = vec.view(B, n_vecs, 2, H, W).mean(dim=1)
        normalized['vector'] = vec
        
        # Rotor
        rotor = feat_dict['rotor']
        if rotor.dim() == 3:
            rotor = rotor.unsqueeze(0)
        if rotor.shape[1] > 1:
            rotor = rotor.mean(dim=1, keepdim=True)
        normalized['rotor'] = rotor
        
        return normalized
    

    # -------------------------------------------------------------------------
    # [Architecture.md §4.2] Priority Map 자동 계산 (코드 보강)
    # -------------------------------------------------------------------------
    def compute_rotor_variance_map(self, rotor_field: torch.Tensor, window: int = 7) -> torch.Tensor:
        """Rotor(또는 rotation proxy) 채널의 지역 분산(variance) 추정.

        - Architecture.md §4.2의 '지역 rotor variance'를 구현 코드 입력 형식에 맞춰 근사합니다.
        - 여기서 rotor_field는 (B,1,H,W) 또는 (B,H,W) 스칼라로 가정합니다.

        Returns:
            var_map: (B,H,W)
        """
        if rotor_field.dim() == 3:
            rotor_field = rotor_field.unsqueeze(1)  # (B,1,H,W)

        pad = window // 2
        mean = F.avg_pool2d(rotor_field, kernel_size=window, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(rotor_field ** 2, kernel_size=window, stride=1, padding=pad)
        var = (mean_sq - mean ** 2).clamp(min=0.0)
        return var.squeeze(1)

    def compute_vector_magnitude_map(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Vector field magnitude (Architecture.md §4.2 - feature magnitude)."""
        # vector_field: (B,2,H,W)
        vx = vector_field[:, 0, :, :]
        vy = vector_field[:, 1, :, :]
        mag = torch.sqrt(vx ** 2 + vy ** 2 + self.epsilon)
        return mag

    def build_priority_map_from_features(self, src_dict: Dict, tgt_dict: Optional[Dict] = None,
                                         window: int = 7) -> torch.Tensor:
        """priority_map 자동 생성.

        - src(및 선택적으로 tgt)에서 rotor variance / vector magnitude를 계산하여
          compute_priority_map()에 투입합니다.
        - tgt_dict를 함께 주면, src/tgt의 통계치를 평균내어 더 안정적으로 만들 수 있습니다.

        Returns:
            priority_map: (B,H,W) in [0,1]
        """
        src = self._normalize_features(src_dict)
        rotor_var = self.compute_rotor_variance_map(src['rotor'], window=window)
        feat_mag = self.compute_vector_magnitude_map(src['vector'])

        if tgt_dict is not None:
            tgt = self._normalize_features(tgt_dict)
            rotor_var_t = self.compute_rotor_variance_map(tgt['rotor'], window=window)
            feat_mag_t = self.compute_vector_magnitude_map(tgt['vector'])
            rotor_var = 0.5 * (rotor_var + rotor_var_t)
            feat_mag = 0.5 * (feat_mag + feat_mag_t)

        return self.compute_priority_map(rotor_var, feat_mag)


    def compute_priority_map(self, rotor_variance: torch.Tensor, 
                             feature_magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spatial priority map for weighted optimization"""
        stability = 1.0 / (rotor_variance + 0.01)
        raw_priority = stability * feature_magnitude
        
        B = raw_priority.shape[0]
        priority_map = torch.zeros_like(raw_priority)
        
        for b in range(B):
            flat = raw_priority[b].view(-1)
            p_min = flat.min()
            p_max = torch.quantile(flat, 0.95)
            
            normalized = (raw_priority[b] - p_min) / (p_max - p_min + self.epsilon)
            priority_map[b] = torch.clamp(normalized, 0, 1)
        
        return priority_map
    
    def huber_loss(self, x: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
        """Huber loss - robust to outliers"""
        abs_x = torch.abs(x)
        quadratic = torch.clamp(abs_x, max=delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic ** 2 + delta * linear
    
    def compute_energy(self, src_dict: Dict, tgt_dict: Dict, 
                       gates: Tuple[torch.Tensor, ...],
                       priority_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute alignment energy"""
        # Get current transformation
        W = self.get_current_W()
        
        # Normalize inputs
        src = self._normalize_features(src_dict)
        tgt = self._normalize_features(tgt_dict)
        
        B, _, H, W_size = src['sdf'].shape
        
        # Create affine grid
        grid = F.affine_grid(W, [B, 1, H, W_size], align_corners=False)
        
        # Warp source features
        warped_sdf = F.grid_sample(src['sdf'], grid, align_corners=False, 
                                   padding_mode='border')
        warped_vec = F.grid_sample(src['vector'], grid, align_corners=False,
                                   padding_mode='zeros')
        warped_rotor = F.grid_sample(src['rotor'], grid, align_corners=False,
                                     padding_mode='zeros')
        
        # Valid mask
        ones = torch.ones_like(src['sdf'])
        valid_mask = F.grid_sample(ones, grid, align_corners=False, padding_mode='zeros')
        valid_mask = (valid_mask > 0.5).float()
        
        # Apply rotation to warped vectors
        rot_matrix = W[0, :2, :2]  # [2, 2]
        vec_permuted = warped_vec.permute(0, 2, 3, 1)  # [B, H, W, 2]
        vec_rotated = torch.einsum('ij,bhwj->bhwi', rot_matrix, vec_permuted)
        vec_rotated = vec_rotated.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # Energy terms
        e_scalar = self.huber_loss(warped_sdf - tgt['sdf'], self.config['huber_delta'])
        
        cos_sim = F.cosine_similarity(vec_rotated, tgt['vector'], dim=1, eps=self.epsilon)
        e_vector = (1.0 - cos_sim).unsqueeze(1)
        
        e_bivector = self.huber_loss(warped_rotor - tgt['rotor'], self.config['huber_delta'])
        
        # Apply gate weights
        g_s, g_v, g_b = gates
        
        if g_s.dim() < 4:
            g_s = g_s.view(1, 1, 1, 1)
            g_v = g_v.view(1, 1, 1, 1)
            g_b = g_b.view(1, 1, 1, 1)
        
        # Weighted sum
        total_energy = g_s * e_scalar + g_v * e_vector + g_b * e_bivector
        
        # Apply priority weighting
        if priority_map is not None and self.config['use_priority']:
            if priority_map.shape[-2:] != total_energy.shape[-2:]:
                priority_map = F.interpolate(priority_map, size=total_energy.shape[-2:],
                                            mode='bilinear', align_corners=False)
            total_energy = total_energy * (0.5 + 0.5 * priority_map)
        
        # Masked average
        masked_energy = total_energy * valid_mask
        energy = masked_energy.sum() / (valid_mask.sum() + self.epsilon)
        
        return energy
    
    def optimize(self, src_dict: Dict, tgt_dict: Dict,
                 gates: Tuple[torch.Tensor, ...],
                 priority_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """Run MPC optimization"""
        base_lr = self.config['learning_rate']
        
        optimizer = optim.Adam([
            {'params': [self.p_angle], 'lr': base_lr * self.config['angle_lr_mult']},
            {'params': [self.p_scale], 'lr': base_lr},
            {'params': [self.p_trans], 'lr': base_lr * 0.5}
        ])
        
        n_iters = self.config['iterations']
        warmup = self.config['warmup_iters']
        
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup  # +1 to avoid zero
            else:
                progress = (epoch - warmup) / max(n_iters - warmup, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        loss_history = []
        # -----------------------------------------------------------------
        # [Architecture.md §4.2] priority_map이 없으면 자동 생성 (optional)
        # -----------------------------------------------------------------
        if (priority_map is None) and self.config.get('use_priority', True):
            priority_map = self.build_priority_map_from_features(src_dict, tgt_dict, window=7)
            if self.config.get('verbose', False):
                print("  [Phase 4] priority_map auto-generated from rotor variance & vector magnitude")
        patience_counter = 0
        
        if self.config['verbose']:
            print(f"\n[Phase 4] MPC Optimization ({n_iters} iterations)")
        
        for i in range(n_iters):
            optimizer.zero_grad()
            
            loss = self.compute_energy(src_dict, tgt_dict, gates, priority_map)
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_state = {
                    'p_angle': self.p_angle.detach().clone(),
                    'p_scale': self.p_scale.detach().clone(),
                    'p_trans': self.p_trans.detach().clone()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                if self.config['verbose']:
                    print(f"  Early stop at iter {i}")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.p_angle, self.p_scale, self.p_trans], 1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            
            if self.config['verbose'] and (i % 30 == 0 or i == n_iters - 1):
                angle_deg, scale, tx, ty = self.get_transform_params()
                print(f"  [{i:3d}] Loss: {loss.item():.6f} | "
                      f"Angle: {angle_deg:.2f} deg | Scale: {scale:.3f}")
        
        # Restore best
        if self.best_state is not None:
            with torch.no_grad():
                self.p_angle.copy_(self.best_state['p_angle'])
                self.p_scale.copy_(self.best_state['p_scale'])
                self.p_trans.copy_(self.best_state['p_trans'])
        
        # Update final W
        self._update_W()
        
        if self.config['verbose']:
            print(f"[Phase 4] Done. Best Loss: {self.best_loss:.6f}")
            angle_deg, scale, tx, ty = self.get_transform_params()
            print(f"[Phase 4] Final: Angle={angle_deg:.2f} deg, Scale={scale:.3f}, "
                  f"Trans=({tx:.4f}, {ty:.4f})")
        
        return self.W.detach(), loss_history
    
    def get_transform_params(self, W: Optional[torch.Tensor] = None) -> Tuple[float, float, float, float]:
        """Extract transformation parameters"""
        if W is not None:
            W_np = W.detach().cpu().numpy()
            if W_np.ndim == 3:
                W_np = W_np[0]
        elif self.W is not None:
            W_np = self.W.detach().cpu().numpy()[0]
        else:
            angle = self.base_angle + self.p_angle.item()
            scale_delta = torch.tanh(self.p_scale).item() * self.config['scale_range']
            scale = self.base_scale * (1.0 + scale_delta)
            tx = torch.tanh(self.p_trans[0]).item() * self.config['trans_range']
            ty = torch.tanh(self.p_trans[1]).item() * self.config['trans_range']
            
            return np.degrees(angle), scale, tx, ty
        
        angle_rad = np.arctan2(W_np[1, 0], W_np[0, 0])
        scale = np.sqrt(W_np[0, 0]**2 + W_np[1, 0]**2)
        tx = W_np[0, 2]
        ty = W_np[1, 2]
        
        return np.degrees(angle_rad), scale, tx, ty


# =============================================================================
# Factory function
# =============================================================================
def create_refiner(device='cuda', **kwargs) -> GeometricMPCRefiner:
    """Factory function to create refiner with custom config"""
    return GeometricMPCRefiner(device=device, config=kwargs)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing GeometricMPCRefiner (Fixed)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    B, C, H, W = 1, 1, 64, 64
    
    src_dict = {
        'sdf': torch.randn(B, C, H, W, device=device),
        'vector': torch.randn(B, 2, H, W, device=device),
        'rotor': torch.randn(B, C, H, W, device=device)
    }
    
    tgt_dict = {
        'sdf': torch.randn(B, C, H, W, device=device),
        'vector': torch.randn(B, 2, H, W, device=device),
        'rotor': torch.randn(B, C, H, W, device=device)
    }
    
    gates = (
        torch.ones(1, device=device) * 0.5,
        torch.ones(1, device=device) * 0.3,
        torch.ones(1, device=device) * 0.2
    )
    
    refiner = GeometricMPCRefiner(device=device, config={'verbose': True, 'iterations': 50})
    
    init_angle = np.radians(30)
    refiner.global_filtering_init(mean_rotor=init_angle, mean_scale=1.0)
    
    W_final, loss_history = refiner.optimize(src_dict, tgt_dict, gates)
    
    angle, scale, tx, ty = refiner.get_transform_params()
    
    print(f"\nFinal Results:")
    print(f"  Angle: {angle:.2f} deg")
    print(f"  Scale: {scale:.3f}")
    print(f"  Translation: ({tx:.4f}, {ty:.4f})")
    print(f"  Loss reduction: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")