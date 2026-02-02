"""
================================================================================
Phase 5: 통합 기하학적 손실 함수 (Unified Geometric Loss)
================================================================================
[Architecture.md §5 참조]

모델의 최종 학습 목표는 아래의 단일 통합 수식을 최소화하는 것입니다.

L_total = α·ΣGeometric_Accuracy + β·Final_Consistency + γ·Iterative_Stability

구성:
1. Geometric Accuracy (기하학적 정밀도): L_s, L_v_local, L_b_local
2. Final Consistency (뒤틀림 일관성): L_coord, L_sdf_photo
3. Iterative Stability (반복 안정성): L_convergence, L_multi_scale [신규]
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# [Hyperparameters] Loss Function
# =============================================================================
# [Geometric Accuracy 가중치]
LAMBDA_SCALAR = 1.0              # [Hyperparameter] L_s 가중치
LAMBDA_VECTOR = 1.0              # [Hyperparameter] L_v 가중치
LAMBDA_BIVECTOR = 1.0            # [Hyperparameter] L_b 가중치

# [Final Consistency 가중치]
BETA_COORD = 10.0                # [Hyperparameter] L_coord 가중치
LAMBDA_ANGLE = 35.0              # [Hyperparameter] L_angle 가중치
LAMBDA_PIXEL = 0.3               # [Hyperparameter] L_pixel 가중치

# [Iterative Stability 가중치]
GAMMA_CONVERGENCE = 0.1          # [Hyperparameter] L_convergence 가중치
GAMMA_MULTISCALE = 0.1           # [Hyperparameter] L_multi_scale 가중치

# [SmoothL1 Beta]
SMOOTH_L1_BETA = 0.5             # [Hyperparameter] SmoothL1Loss beta


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
        """
        # W_gt: (B, 2, 3)
        # 국소 회전 행렬 = W_gt[:, :2, :2]
        return W_gt[:, :2, :2]
    
    def forward(self, S_A, V_A, B_A, S_B, V_B, B_B, W_gt):
        """
        Args:
            S_A, S_B: Scalar features (B, C, H, W)
            V_A, V_B: Vector features (B, C, 2, H, W)
            B_A, B_B: Bivector (Rotor Magnitude) (B, C, H, W)
            W_gt: Ground Truth Transform (B, 2, 3)
            
        Returns:
            L_geo: 기하학적 정밀도 손실
            loss_dict: 개별 손실 딕셔너리
        """
        B, C, H, W = S_A.shape
        
        # W_gt로 Sampling Grid 생성
        grid = F.affine_grid(W_gt, [B, C, H, W], align_corners=False)
        
        # =====================================================================
        # [§5.1.1] L_s (뼈대 일치)
        # L_s(p) = ||S_A(p) - S_B(W_GT(p))||²
        # =====================================================================
        S_B_warped = F.grid_sample(S_B, grid, align_corners=False, mode='bilinear')
        L_s = F.mse_loss(S_A, S_B_warped)
        
        # =====================================================================
        # [§5.1.2] L_v (방향 정렬) - Local Rotation 적용
        # L_v(p) = ||V_A(p) - R_loc · V_B(W_GT(p))||²
        # =====================================================================
        # V_B를 워핑
        V_B_flat = V_B.view(B, -1, H, W)  # (B, C*2, H, W)
        V_B_warped_flat = F.grid_sample(V_B_flat, grid, align_corners=False)
        V_B_warped = V_B_warped_flat.view(B, C, 2, H, W)
        
        # 국소 회전 적용
        R_loc = self.compute_local_rotation(W_gt)  # (B, 2, 2)
        
        # V_B_warped의 벡터 성분에 회전 적용
        # (B, C, 2, H, W) -> (B, C, H, W, 2) -> matmul -> (B, C, H, W, 2)
        V_B_warped_perm = V_B_warped.permute(0, 1, 3, 4, 2)  # (B, C, H, W, 2)
        V_B_rotated = torch.einsum('bij,bchwj->bchwi', R_loc, V_B_warped_perm)
        V_B_rotated = V_B_rotated.permute(0, 1, 4, 2, 3)  # (B, C, 2, H, W)
        
        L_v = F.mse_loss(V_A, V_B_rotated)
        
        # =====================================================================
        # [§5.1.3] L_b (회전 일관성) - Local Rotor와 비교
        # L_b(p) = ||Rotor_A(p) - R_loc · Rotor_B(W_GT(p))||²
        # =====================================================================
        B_B_warped = F.grid_sample(B_B, grid, align_corners=False)
        
        # Rotor는 Magnitude이므로 스케일 보정만 (회전과 무관)
        # 단순 MSE로 처리 (더 정교한 로직은 추후 추가 가능)
        L_b = F.mse_loss(B_A, B_B_warped)
        
        # 총합
        L_geo = LAMBDA_SCALAR * L_s + LAMBDA_VECTOR * L_v + LAMBDA_BIVECTOR * L_b
        
        return L_geo, {
            'L_s': L_s.item(),
            'L_v': L_v.item(),
            'L_b': L_b.item()
        }


class FinalConsistencyLoss(nn.Module):
    """
    [Phase 5.2] Final Consistency (뒤틀림 일관성)
    
    Architecture.md §5.2
    
    모델이 예측한 W*가 수학적으로 얼마나 견고한지 증명합니다.
    
    - L_coord (모서리 거리): 네 모서리 좌표의 SmoothL1 거리
    - L_angle (각도 일치): cos(pred_angle - gt_angle)
    - L_pixel (SDF 기반 복원): 복원된 이미지의 SDF 일치
    """
    
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
        
    def get_inverse_affine(self, matrix_2x3):
        """
        [Helper] 2x3 Affine 행렬의 역행렬 계산
        """
        B = matrix_2x3.shape[0]
        bottom_row = torch.tensor([0., 0., 1.], device=matrix_2x3.device).view(1, 1, 3).repeat(B, 1, 1)
        matrix_3x3 = torch.cat([matrix_2x3, bottom_row], dim=1)
        matrix_inv = torch.linalg.inv(matrix_3x3)
        return matrix_inv[:, :2, :]
    
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
        
        # =====================================================================
        # [§5.2.1] L_coord (모서리 거리) - SmoothL1
        # =====================================================================
        # 네 모서리 좌표 (정규화 좌표계: -1 ~ 1)
        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.], 
            [1., 1., 1.], [-1., 1., 1.]
        ], device=device)
        corners = corners.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)  # (B, 3, 4)
        
        pts_pred = torch.bmm(pred_W, corners)  # (B, 2, 4)
        pts_gt = torch.bmm(W_gt, corners)
        
        L_coord = self.smooth_l1(pts_pred, pts_gt)
        
        # =====================================================================
        # [§5.2.2] L_angle (각도 일치)
        # L_angle = 1 - cos(pred_angle - gt_angle)
        # =====================================================================
        pred_angle = torch.atan2(pred_sin, pred_cos)
        L_angle = 1.0 - torch.cos(pred_angle - gt_angle_rad).mean()
        
        # =====================================================================
        # [§5.2.3] L_pixel (SDF 기반 복원) - Optional
        # =====================================================================
        L_pixel = torch.tensor(0.0, device=device)
        if S_A is not None and S_B is not None:
            # 예측된 역변환으로 S_A를 워핑하여 S_B와 비교
            pred_W_inv = self.get_inverse_affine(pred_W)
            grid = F.affine_grid(pred_W_inv, S_B.size(), align_corners=False)
            S_A_warped = F.grid_sample(S_A, grid, align_corners=False, padding_mode='zeros')
            L_pixel = F.mse_loss(S_A_warped, S_B)
        
        L_final = BETA_COORD * L_coord + LAMBDA_ANGLE * L_angle + LAMBDA_PIXEL * L_pixel
        
        return L_final, {
            'L_coord': L_coord.item(),
            'L_angle': L_angle.item(),
            'L_pixel': L_pixel.item()
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
        
        if delta_W_list is not None:
            L_conv = self.compute_convergence_loss(delta_W_list)
            
        if W_predictions is not None:
            L_ms = self.compute_multiscale_loss(W_predictions)
        
        L_iter = GAMMA_CONVERGENCE * L_conv + GAMMA_MULTISCALE * L_ms
        
        return L_iter, {
            'L_convergence': L_conv.item() if isinstance(L_conv, torch.Tensor) else L_conv,
            'L_multi_scale': L_ms.item() if isinstance(L_ms, torch.Tensor) else L_ms
        }


class UnifiedGeometricLoss(nn.Module):
    """
    [Phase 5 Main] 통합 기하학적 손실 함수
    
    Architecture.md §5 전체 구현
    
    L_total = α·Geometric_Accuracy + β·Final_Consistency + γ·Iterative_Stability
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        """
        Args:
            alpha: Geometric Accuracy 가중치
            beta: Final Consistency 가중치
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
            B_A = B_A_tuple[2]  # Rotor Magnitude
            B_B = B_B_tuple[2]
            
            L_geo, geo_dict = self.geo_loss(S_A, V_A, B_A, S_B, V_B, B_B, W_gt)
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
    
    # 더미 데이터
    pred_W = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
    pred_W[:, 0, 0] = 0.95  # 약간의 차이
    
    W_gt = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
    
    pred_cos = torch.tensor([0.98, 0.99], device=device)
    pred_sin = torch.tensor([0.02, 0.01], device=device)
    gt_angle = torch.tensor([0.0, 0.0], device=device)
    
    # Loss 계산
    loss_fn = UnifiedGeometricLoss(alpha=1.0, beta=1.0, gamma=0.1).to(device)
    
    total_loss, loss_dict = loss_fn(
        pred_W, W_gt, pred_cos, pred_sin, gt_angle
    )
    
    print(f"Total Loss: {total_loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
