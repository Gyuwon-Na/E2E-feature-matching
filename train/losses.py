import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# =============================================================================
# [Loss Function] - low-angle-first + final ±90° robustness (RTX 3060 preset)
# =============================================================================

LAMBDA_SCALAR = 1.0
LAMBDA_VECTOR = 8.0
LAMBDA_BIVECTOR = 130.0

# [Final Consistency]
BETA_COORD = 1.0
LAMBDA_ANGLE = 900.0
LAMBDA_PIXEL = 0.45
LAMBDA_ROTATION_INV = 320.0

# [SmoothL1 Beta]
SMOOTH_L1_BETA = 0.3

# [Adaptive angle weighting]
SMALL_ANGLE_SIGMA_DEG = 18.0
SMALL_ANGLE_BOOST = 1.15
LARGE_ANGLE_CENTER_DEG = 32.0
LARGE_ANGLE_SHARPNESS = 5.0
LARGE_ANGLE_BOOST = 1.45

# [Iterative Stability]
GAMMA_CONVERGENCE = 0.08
GAMMA_MULTISCALE = 0.50

class GeometricAccuracyLoss(nn.Module):
    """
    [Phase 5.1] Geometric Accuracy (기하학적 정밀도)
    
    Architecture.md §5.1
    
    이미지 A와 정답 변환(W_GT)으로 되돌린 이미지 B의 특징들이 
    물리적으로 일치하는지 검사합니다.
    
    - L_s (뼈대 일치): SDF/에너지가 정답 위치에서 겹침
    - L_v (방향 정렬): 벡터가 국소 회전량만큼 회전하여 일치
    - L_b (회전 일관성): 지역적 Rotor가 전체 변환과 호응
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_local_rotation(self, W_gt, points=None):
        """
        [Helper] W_GT의 Jacobian을 통해 국소 회전량 계산

        Affine Transform의 경우, Jacobian은 전역적으로 동일 (W의 2x2 부분)

        ⚠️ 중요(컨벤션):
          - W_gt는 grid_sample/affine_grid에 들어가는 theta(out->in)이며,
            본 프로젝트에서는 Dataset에서 W_gt를 **A->B (out=A, in=B)** 로 생성합니다.
          - B에서 샘플된 V_B를 A 좌표계의 V_A와 비교하려면,
            벡터 성분은 **B->A** 로 회전시켜야 합니다.
          - 따라서 여기서는 (A->B)의 역회전(=transpose)을 반환합니다.
        """
        # W_gt: (B, 2, 3)
        # A2B 선형부
        A2B = W_gt[:, :2, :2]  # (B,2,2)
        # scale 제거 후 순수 회전 추출
        eps = 1e-6
        s = torch.sqrt(A2B[:, 0, 0]**2 + A2B[:, 1, 0]**2 + eps)  # (B,)
        R_A2B = A2B / s.view(-1, 1, 1)
        # B2A = (A2B)^T (회전행렬 가정)
        R_B2A = R_A2B.transpose(1, 2).contiguous()
        return R_B2A

    @staticmethod
    def masked_mse(a, b, mask, eps=1e-6):
        diff2 = (a - b) ** 2
        while mask.dim() < diff2.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(diff2)
        return (diff2 * mask).sum() / (mask.sum() + eps)

    def forward(self, S_A, V_A, B_A, S_B, V_B, B_B, W_gt):
        """
        Args:
            S_A, S_B: Scalar features (B, C, H, W)
            V_A, V_B: Vector features (B, C, 2, H, W)

            B_A, B_B:
                - (권장) Phase2 rotor tuple: (unit_cos, unit_sin, rotor_mag) where each is (B, C, H, W)
                - (호환) Tensor rotor_mag only: (B, C, H, W)

            W_gt: Ground Truth Transform (B, 2, 3)

        Returns:
            L_geo: 기하학적 정밀도 손실
            loss_dict: 개별 손실 딕셔너리
        """
        B, C, H, W = S_A.shape
        eps = 1e-6

        # -------------------------------------------------------------
        # Bivector/Rotor 입력 파싱 (tuple or tensor)
        # -------------------------------------------------------------
        if isinstance(B_A, (tuple, list)) and len(B_A) == 3:
            cos_A, sin_A, mag_A = B_A
        else:
            cos_A, sin_A, mag_A = None, None, B_A

        if isinstance(B_B, (tuple, list)) and len(B_B) == 3:
            cos_B, sin_B, mag_B = B_B
        else:
            cos_B, sin_B, mag_B = None, None, B_B

        # W_gt로 Sampling Grid 생성
        grid = F.affine_grid(W_gt, [B, C, H, W], align_corners=True)

        valid_mask = F.grid_sample(
            torch.ones(B, 1, H, W, device=S_A.device, dtype=S_A.dtype),
            grid,
            align_corners=True,
            mode='nearest',
            padding_mode='zeros'
        )
        valid_mask = (valid_mask > 0.999).to(dtype=S_A.dtype)

        # =====================================================================
        # [§5.1.1] L_s (뼈대 일치)
        # L_s(p) = ||S_A(p) - S_B(W_GT(p))||²
        # =====================================================================
        S_B_warped = F.grid_sample(S_B, grid, align_corners=True, mode='bilinear')
        L_s = self.masked_mse(S_A, S_B_warped, valid_mask)


        # =====================================================================
        # [§5.1.2] L_v (방향 정렬) - Local Rotation 적용
        # L_v(p) = ||V_A(p) - R_loc · V_B(W_GT(p))||²
        # =====================================================================
        V_B_flat = V_B.view(B, -1, H, W)  # (B, C*2, H, W)
        V_B_warped_flat = F.grid_sample(V_B_flat, grid, align_corners=True)
        V_B_warped = V_B_warped_flat.view(B, C, 2, H, W)

        # 국소 회전(2x2) 적용
        A_loc = self.compute_local_rotation(W_gt)  # (B, 2, 2)

        # V_B_warped의 벡터 성분에 회전 적용
        V_B_warped_perm = V_B_warped.permute(0, 1, 3, 4, 2)  # (B, C, H, W, 2)
        V_B_rotated = torch.einsum('bij,bchwj->bchwi', A_loc, V_B_warped_perm)
        V_B_rotated = V_B_rotated.permute(0, 1, 4, 2, 3)  # (B, C, 2, H, W)

        L_v = self.masked_mse(V_A, V_B_rotated, valid_mask)

        # =====================================================================
        # [§5.1.3] L_b (회전/스케일 일관성) - Local Rotor 비교
        #
        # Architecture.md 식:
        #   L_b(p) = ||Rotor_A(p) - R_loc · Rotor_B(W_GT(p))||²
        #
        # 구현:
        #   - rotor_mag: 스케일/강도 일관성 (기존 구현 유지)
        #   - unit_cos/unit_sin: 회전 방향 일관성 (신규 보강)
        # =====================================================================
        # magnitude alignment (기존과 동일)
        mag_B_warped = F.grid_sample(mag_B, grid, align_corners=True)
        L_b_mag = self.masked_mse(mag_A, mag_B_warped, valid_mask)

        # orientation alignment (가능할 때만)
        L_b_orient = torch.tensor(0.0, device=S_A.device)
        if (cos_A is not None) and (sin_A is not None) and (cos_B is not None) and (sin_B is not None):
            cos_B_warped = F.grid_sample(cos_B, grid, align_corners=True)
            sin_B_warped = F.grid_sample(sin_B, grid, align_corners=True)

            # A_loc은 scale*rotation일 수 있으므로, 회전 성분만 정규화해서 사용
            # scale = sqrt(a^2 + c^2)  (a=W[0,0], c=W[1,0])
            a = A_loc[:, 0, 0]
            c = A_loc[:, 1, 0]
            scale = torch.sqrt(a * a + c * c + eps)
            cos_r = (a / scale).view(B, 1, 1, 1)
            sin_r = (c / scale).view(B, 1, 1, 1)

            # (cos, sin) 벡터로 간주해 회전 적용
            cos_B_rot = cos_r * cos_B_warped - sin_r * sin_B_warped
            sin_B_rot = sin_r * cos_B_warped + cos_r * sin_B_warped

            L_b_cos = self.masked_mse(cos_A, cos_B_rot, valid_mask)
            L_b_sin = self.masked_mse(sin_A, sin_B_rot, valid_mask)
            L_b_orient = 0.5 * (L_b_cos + L_b_sin)

        # 최종 L_b: magnitude + orientation
        L_b = L_b_mag + L_b_orient

        # 총합
        L_geo = LAMBDA_SCALAR * L_s + LAMBDA_VECTOR * L_v + LAMBDA_BIVECTOR * L_b

        return L_geo, {
            'L_s': L_s.item(),
            'L_v': L_v.item(),
            'L_b': L_b.item(),
            'L_b_mag': L_b_mag.item(),
            'L_b_orient': L_b_orient.item() if isinstance(L_b_orient, torch.Tensor) else float(L_b_orient),
        }



class FinalConsistencyLoss(nn.Module):
    """
    [Phase 5.2] Final Consistency (뒤틀림 일관성) - v7

    핵심 변경:
    1. low-angle 정밀도와 high-angle 회전 안정성을 동시에 잡기 위해
       sample-wise adaptive weighting을 사용합니다.
    2. 단순 determinant/orthogonality 외에 rotation-matrix alignment를 직접 넣습니다.
    """

    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)

    def get_inverse_affine(self, matrix_2x3):
        """
        [Helper] 2x3 Affine 행렬의 역행렬 계산
        """
        B = matrix_2x3.shape[0]
        in_dtype = matrix_2x3.dtype
        device = matrix_2x3.device

        matrix_2x3_f = matrix_2x3.to(dtype=torch.float32)
        bottom_row = torch.tensor([0., 0., 1.], device=device, dtype=matrix_2x3_f.dtype)
        bottom_row = bottom_row.view(1, 1, 3).repeat(B, 1, 1)

        matrix_3x3 = torch.cat([matrix_2x3_f, bottom_row], dim=1)
        matrix_inv = torch.linalg.inv(matrix_3x3)
        return matrix_inv[:, :2, :].to(dtype=in_dtype)

    def build_sample_weights(self, gt_angle_rad: torch.Tensor):
        """
        low-angle 샘플에는 좌표/픽셀 정밀도를,
        high-angle 샘플에는 회전/직교성 일관성을 더 강하게 겁니다.
        """
        gt_abs_deg = torch.abs(gt_angle_rad.detach().to(dtype=torch.float32)) * 180.0 / torch.pi
        small_w = 1.0 + SMALL_ANGLE_BOOST * torch.exp(-gt_abs_deg / SMALL_ANGLE_SIGMA_DEG)
        large_w = 1.0 + LARGE_ANGLE_BOOST * torch.sigmoid(
            (gt_abs_deg - LARGE_ANGLE_CENTER_DEG) / LARGE_ANGLE_SHARPNESS
        )
        return small_w, large_w

    def compute_rotation_invariant_loss(self, pred_W, W_gt):
        """
        rotation alignment + scale consistency + orthogonality
        Returns:
            per_sample_loss: (B,)
        """
        pred_linear = pred_W[:, :2, :2].to(dtype=torch.float32)
        gt_linear = W_gt[:, :2, :2].to(dtype=torch.float32)

        eps = 1e-6
        pred_scale = torch.sqrt(pred_linear[:, 0, 0] ** 2 + pred_linear[:, 1, 0] ** 2 + eps)
        gt_scale = torch.sqrt(gt_linear[:, 0, 0] ** 2 + gt_linear[:, 1, 0] ** 2 + eps)

        R_pred = pred_linear / pred_scale.view(-1, 1, 1)
        R_gt = gt_linear / gt_scale.view(-1, 1, 1)

        rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = rel[:, 0, 0] + rel[:, 1, 1]
        rot_geo = 1.0 - 0.5 * trace.clamp(-2.0, 2.0)

        pred_det = pred_linear[:, 0, 0] * pred_linear[:, 1, 1] - pred_linear[:, 0, 1] * pred_linear[:, 1, 0]
        gt_det = gt_linear[:, 0, 0] * gt_linear[:, 1, 1] - gt_linear[:, 0, 1] * gt_linear[:, 1, 0]
        det_loss = F.smooth_l1_loss(pred_det, gt_det, reduction='none', beta=SMOOTH_L1_BETA)

        pred_rtr = torch.bmm(R_pred.transpose(1, 2), R_pred)
        identity = torch.eye(2, device=pred_W.device, dtype=pred_rtr.dtype).unsqueeze(0).repeat(pred_W.shape[0], 1, 1)
        ortho_loss = ((pred_rtr - identity) ** 2).mean(dim=(1, 2))

        scale_loss = F.smooth_l1_loss(pred_scale, gt_scale, reduction='none', beta=SMOOTH_L1_BETA)
        return rot_geo + 0.25 * det_loss + 0.25 * ortho_loss + 0.5 * scale_loss

    def forward(self, pred_W, W_gt, pred_cos, pred_sin, gt_angle_rad, S_A=None, S_B=None):
        """
        Args:
            pred_W: 예측된 변환 행렬 (B, 2, 3)
            W_gt: 정답 변환 행렬 (B, 2, 3)
            pred_cos, pred_sin: 예측된 Rotor 성분
            gt_angle_rad: 정답 각도 (라디안)
            S_A, S_B: Scalar features (Optional, for pixel loss)

        Returns:
            L_final: 최종 일관성 손실
            loss_dict: 개별 손실 딕셔너리
        """
        B = pred_W.shape[0]
        device = pred_W.device

        small_w, large_w = self.build_sample_weights(gt_angle_rad)

        # [1] L_coord (per-sample)
        pred_W_f = pred_W.to(dtype=torch.float32)
        W_gt_f = W_gt.to(dtype=torch.float32)

        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.],
            [1., 1., 1.], [-1., 1., 1.]
        ], device=device, dtype=pred_W_f.dtype)
        corners = corners.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)  # (B, 3, 4)

        pts_pred = torch.bmm(pred_W_f, corners)
        pts_gt = torch.bmm(W_gt_f, corners)
        coord_map = F.smooth_l1_loss(pts_pred, pts_gt, beta=SMOOTH_L1_BETA, reduction='none')
        L_coord_per = coord_map.mean(dim=(1, 2))
        L_coord = (L_coord_per * small_w).mean()

        # [2] L_angle
        gt_angle_f = gt_angle_rad.to(dtype=torch.float32)
        gt_cos = torch.cos(gt_angle_f)
        gt_sin = torch.sin(gt_angle_f)
        pred_cos_f = pred_cos.to(dtype=torch.float32)
        pred_sin_f = pred_sin.to(dtype=torch.float32)
        dot = (pred_cos_f * gt_cos + pred_sin_f * gt_sin).clamp(-1.0, 1.0)
        L_angle_per = 1.0 - dot
        L_angle = (L_angle_per * large_w).mean()

        # [3] L_pixel
        L_pixel = torch.tensor(0.0, device=device, dtype=torch.float32)
        if S_A is not None and S_B is not None:
            theta = self.get_inverse_affine(pred_W).to(dtype=S_B.dtype)
            grid = F.affine_grid(theta, S_B.size(), align_corners=True)

            S_A_warped = F.grid_sample(
                S_A.to(dtype=grid.dtype),
                grid,
                align_corners=True,
                padding_mode='zeros'
            )

            valid_mask = F.grid_sample(
                torch.ones_like(S_A[:, :1]).to(dtype=grid.dtype),
                grid,
                align_corners=True,
                mode='nearest',
                padding_mode='zeros'
            )
            valid_mask = (valid_mask > 0.999).to(dtype=grid.dtype)

            pixel_diff2 = (S_A_warped - S_B.to(dtype=grid.dtype)) ** 2
            pixel_map = (pixel_diff2 * valid_mask).sum(dim=(1, 2, 3)) / (
                valid_mask.sum(dim=(1, 2, 3)) * S_A.shape[1] + 1e-6
            )
            L_pixel = (pixel_map * small_w.to(dtype=pixel_map.dtype)).mean()

        # [4] L_rotation_inv / matrix alignment
        L_rot_inv_per = self.compute_rotation_invariant_loss(pred_W, W_gt)
        L_rot_inv = (L_rot_inv_per * large_w).mean()

        L_final = (
            BETA_COORD * L_coord +
            LAMBDA_ANGLE * L_angle +
            LAMBDA_PIXEL * L_pixel +
            LAMBDA_ROTATION_INV * L_rot_inv
        )

        return L_final, {
            'L_coord': L_coord.item(),
            'L_angle': L_angle.item(),
            'L_pixel': L_pixel.item(),
            'L_rot_inv': L_rot_inv.item(),
            'small_angle_weight_mean': small_w.mean().item(),
            'large_angle_weight_mean': large_w.mean().item(),
        }


class IterativeStabilityLoss(nn.Module):
    """
    [Phase 5.3] Iterative & Multi-Scale Constraint (반복 및 스케일 안정성)
    
    Architecture.md §5.3 - [보완] 신규
    
    반복 정제(Iterative Refinement)와 다중 해상도(Multi-Scale) 학습 과정에서 
    모델이 발산하지 않고 올바른 방향으로 수렴하도록 강제하는 제약 조건입니다.
    
    - L_convergence: 반복마다 ΔW가 Identity에 가까워지도록
    - L_multi_scale: 스케일 간 예측값 일관성 유지
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_convergence_loss(self, delta_W_list):
        """
        [§5.3.1] L_convergence (수렴 유도 손실)
        
        L_convergence = Σ_k w_k · ||ΔW^(k) - I||_F²
        
        Args:
            delta_W_list: list of ΔW at each iteration [(B, 2, 3), ...]
            
        Returns:
            L_conv: 수렴 손실
        """
        if len(delta_W_list) <= 1:
            return torch.tensor(0.0, device=delta_W_list[0].device if delta_W_list else 'cpu')
        
        K = len(delta_W_list)
        device = delta_W_list[0].device
        
        # Identity Matrix (2x3)
        I = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
        
        L_conv = torch.tensor(0.0, device=device)
        for k in range(1, K):  # k=1부터 시작 (첫 번째는 큰 변화 허용)
            delta_W = delta_W_list[k]
            
            # Linear Warm-up 가중치: 후반부에 더 강하게 페널티
            w_k = k / K
            
            # Frobenius Norm
            diff = delta_W - I.unsqueeze(0)
            frob_norm_sq = (diff ** 2).sum(dim=(1, 2)).mean()
            
            L_conv = L_conv + w_k * frob_norm_sq
        
        return L_conv
    
    def compute_multiscale_loss(self, W_predictions):
        """
        [§5.3.2] L_multi_scale (다중 스케일 일관성 손실)
        
        L_multi_scale = Σ_l ||W^(l) - Upsample(W^(l+1))||²
        
        Args:
            W_predictions: dict {level: W_pred} 또는 list of W_pred (coarse to fine)
            
        Returns:
            L_ms: 다중 스케일 손실
        """
        if isinstance(W_predictions, dict):
            levels = sorted(W_predictions.keys())
            W_list = [W_predictions[l] for l in levels]
        else:
            W_list = W_predictions
            
        if len(W_list) <= 1:
            return torch.tensor(0.0, device=W_list[0].device if W_list else 'cpu')
        
        device = W_list[0].device
        L_ms = torch.tensor(0.0, device=device)
        
        # Fine to Coarse 순회 (인접 레벨 비교)
        for l in range(len(W_list) - 1):
            W_fine = W_list[l]      # Level l (더 고해상도)
            W_coarse = W_list[l+1]  # Level l+1 (더 저해상도)
            
            # Affine Transform은 해상도와 무관하므로 직접 비교 가능
            diff = W_fine - W_coarse
            L_ms = L_ms + (diff ** 2).sum(dim=(1, 2)).mean()
        
        return L_ms
    
    def forward(self, delta_W_list=None, W_predictions=None):
        """
        Args:
            delta_W_list: 각 반복의 ΔW 리스트 (Optional)
            W_predictions: 각 레벨의 W 예측 (Optional)
            
        Returns:
            L_iter: 반복 안정성 손실
            loss_dict: 개별 손실 딕셔너리
        """
        device = 'cpu'
        if delta_W_list and len(delta_W_list) > 0:
            device = delta_W_list[0].device
        elif W_predictions and len(W_predictions) > 0:
            device = W_predictions[0].device if isinstance(W_predictions, list) else list(W_predictions.values())[0].device
        
        L_conv = torch.tensor(0.0, device=device)
        L_ms = torch.tensor(0.0, device=device)
        
        # 빈 리스트([])가 들어오면 compute_* 내부에서 CPU 텐서를 반환할 수 있어
        # (GPU 텐서와 더해질 때) device mismatch 오류가 날 수 있습니다.
        if delta_W_list is not None and len(delta_W_list) > 0:
            L_conv = self.compute_convergence_loss(delta_W_list)
            
        if W_predictions is not None and len(W_predictions) > 0:
            L_ms = self.compute_multiscale_loss(W_predictions)
        
        L_iter = GAMMA_CONVERGENCE * L_conv + GAMMA_MULTISCALE * L_ms
        
        return L_iter, {
            'L_convergence': L_conv.item() if isinstance(L_conv, torch.Tensor) else L_conv,
            'L_multi_scale': L_ms.item() if isinstance(L_ms, torch.Tensor) else L_ms
        }


class UnifiedGeometricLoss(nn.Module):
    """
    [Phase 5 Main] 통합 기하학적 손실 함수 - v5 수정
    
    Architecture.md §5 전체 구현
    
    [v5 수정사항]
    - FinalConsistencyLoss에 L_rotation_inv 추가
    - ±60도 회전에 최적화된 가중치
    
    L_total = α·Geometric_Accuracy + β·Final_Consistency + γ·Iterative_Stability
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        """
        Args:
            alpha: Geometric Accuracy 가중치
            beta: Final Consistency 가중치 [v5: ±60도에서 더 중요]
            gamma: Iterative Stability 가중치
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.geo_loss = GeometricAccuracyLoss()
        self.final_loss = FinalConsistencyLoss()
        self.iter_loss = IterativeStabilityLoss()
        
    def forward(self, pred_W, W_gt, pred_cos, pred_sin, gt_angle_rad,
                phase2_a_tuple=None, phase2_b_tuple=None,
                delta_W_list=None, W_predictions=None):
        """
        [통합 손실 계산]
        
        Args:
            pred_W: 예측된 변환 행렬 (B, 2, 3)
            W_gt: 정답 변환 행렬 (B, 2, 3)
            pred_cos, pred_sin: 예측된 Rotor 성분
            gt_angle_rad: 정답 각도
            phase2_a_tuple: Phase 2 출력 (S, V, B) for A
            phase2_b_tuple: Phase 2 출력 (S, V, B) for B
            delta_W_list: 반복별 ΔW (Optional, Phase 3.5용)
            W_predictions: 레벨별 W 예측 (Optional, Multi-scale용)
            
        Returns:
            total_loss: 총 손실
            loss_dict: 모든 개별 손실 딕셔너리
        """
        loss_dict = {}
        
        # =====================================================================
        # [Part 1] Final Consistency (항상 계산)
        # =====================================================================
        S_A = phase2_a_tuple[0] if phase2_a_tuple else None
        S_B = phase2_b_tuple[0] if phase2_b_tuple else None
        
        L_final, final_dict = self.final_loss(
            pred_W, W_gt, pred_cos, pred_sin, gt_angle_rad, S_A, S_B
        )
        loss_dict.update(final_dict)
        
        # =====================================================================
        # [Part 2] Geometric Accuracy (Phase 2 출력 있을 때만)
        # =====================================================================
        L_geo = torch.tensor(0.0, device=pred_W.device)
        if phase2_a_tuple is not None and phase2_b_tuple is not None:
            S_A, V_A, B_A_tuple = phase2_a_tuple
            S_B, V_B, B_B_tuple = phase2_b_tuple
            # B는 (unit_cos, unit_sin, rotor_mag) tuple 전체를 전달하여
            # L_b에서 방향(cos/sin) + 크기(mag) 일관성을 함께 강제합니다.
            L_geo, geo_dict = self.geo_loss(S_A, V_A, B_A_tuple, S_B, V_B, B_B_tuple, W_gt)
            loss_dict.update(geo_dict)
        
        # =====================================================================
        # [Part 3] Iterative Stability (해당 데이터 있을 때만)
        # =====================================================================
        L_iter = torch.tensor(0.0, device=pred_W.device)
        if delta_W_list is not None or W_predictions is not None:
            L_iter, iter_dict = self.iter_loss(delta_W_list, W_predictions)
            loss_dict.update(iter_dict)
        
        # =====================================================================
        # [Total Loss]
        # =====================================================================
        total_loss = self.alpha * L_geo + self.beta * L_final + self.gamma * L_iter
        
        loss_dict['total'] = total_loss.item()
        loss_dict['L_geo'] = L_geo.item() if isinstance(L_geo, torch.Tensor) else L_geo
        loss_dict['L_final'] = L_final.item()
        loss_dict['L_iter'] = L_iter.item() if isinstance(L_iter, torch.Tensor) else L_iter
        
        return total_loss, loss_dict


# =============================================================================
# 헬퍼 함수
# =============================================================================

def normalize_rotor_output(cos_raw, sin_raw):
    """
    [Helper] Rotor 출력 정규화
    
    예측된 cos, sin을 정규화하여 단위 원 위에 투영
    """
    magnitude = torch.sqrt(cos_raw**2 + sin_raw**2 + 1e-6)
    return cos_raw / magnitude, sin_raw / magnitude


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loss Test on: {device}")
    
    B = 2
    
    # 더미 데이터 (±60도 회전 시뮬레이션)
    angle_rad = torch.tensor([1.0, -0.8], device=device)  # ~57°, ~46°
    
    # GT 변환 행렬
    cos_gt = torch.cos(angle_rad)
    sin_gt = torch.sin(angle_rad)
    W_gt = torch.zeros(B, 2, 3, device=device)
    W_gt[:, 0, 0] = cos_gt
    W_gt[:, 0, 1] = -sin_gt
    W_gt[:, 1, 0] = sin_gt
    W_gt[:, 1, 1] = cos_gt
    
    # 예측 (약간의 오차)
    pred_cos = cos_gt + 0.05 * torch.randn(B, device=device)
    pred_sin = sin_gt + 0.05 * torch.randn(B, device=device)
    pred_cos, pred_sin = normalize_rotor_output(pred_cos, pred_sin)
    
    pred_W = torch.zeros(B, 2, 3, device=device)
    pred_W[:, 0, 0] = pred_cos
    pred_W[:, 0, 1] = -pred_sin
    pred_W[:, 1, 0] = pred_sin
    pred_W[:, 1, 1] = pred_cos
    
    # Loss 계산
    loss_fn = UnifiedGeometricLoss(alpha=1.0, beta=1.5, gamma=0.1).to(device)
    
    total_loss, loss_dict = loss_fn(
        pred_W, W_gt, pred_cos, pred_sin, angle_rad
    )
    
    print(f"\n[v5] ±90° Test Results:")
    print(f"Total Loss: {total_loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
