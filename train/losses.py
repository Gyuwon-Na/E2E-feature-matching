# =============================================================================
# [ARCHITECTURE MAPPING (Phase5 + Addendum 6.5)]
# =============================================================================
# [ARCH L0481] ## **📂 5: 통합 기하학적 손실 함수 (Unified Geometric Loss)** -> UnifiedGeometricLoss.forward
# [ARCH L0482]  -> UnifiedGeometricLoss / component losses
# [ARCH L0483] 모델의 최종 학습 목표는 아래의 **단일 통합 수식**을 최소화하는 것입니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0484]  -> UnifiedGeometricLoss / component losses
# [ARCH L0485] $$L_{total} = \alpha \sum_{p \in \Omega} \underbrace{\left( L_{s}(p) + L_{v\_local}(p) + L_{b\_local}(p) \right)}_{\text{Geometric Accuracy (Local-Aware)}} + \beta \underbrace{\left( \lambda_c L_{\text{SmoothL1}} + \lambda_s L_{\text{SDF-Photo}} \right)}_{\text{Final Consistency}} + \gamma \underbrace{\left( L_{convergence} + L_{multi\_scale} \right)}_{\text{Iterative Stability}}$$ -> FinalConstraintLoss.forward
# [ARCH L0486]  -> UnifiedGeometricLoss / component losses
# [ARCH L0487] ### **1. Geometric Accuracy (기하학적 정밀도)** -> losses.py (section entry)
# [ARCH L0488]  -> UnifiedGeometricLoss / component losses
# [ARCH L0489] 이미지 A와 정답 변환($W_{GT}$)으로 되돌린 이미지 B의 특징들이 물리적으로 일치하는지 검사 -> UnifiedGeometricLoss / component losses
# [ARCH L0490]  -> UnifiedGeometricLoss / component losses
# [ARCH L0491] - **$L_s$ (뼈대 일치):** -> GeometricAccuracyLoss.forward (L_s)
# [ARCH L0492]  -> UnifiedGeometricLoss / component losses
# [ARCH L0493]     $$  L_s(p) = \| S_A(p) - S_B(W_{GT}(p)) \|^2$$ -> GeometricAccuracyLoss.forward (L_s)
# [ARCH L0494]  -> UnifiedGeometricLoss / component losses
# [ARCH L0495]     - **의미:** Softplus로 살려낸 SDF와 에너지가 정답 위치에서 정확히 겹쳐야 함 -> UnifiedGeometricLoss / component losses
# [ARCH L0496] - **$L_v$ (방향 정렬)** -> GeometricAccuracyLoss.forward (L_v + local rotation)
# [ARCH L0497]  -> UnifiedGeometricLoss / component losses
# [ARCH L0498]     $$  L_{v\_local}(p) = \| V_A(p) - \underbrace{\mathcal{R}_{loc}(W_{GT}, p)}_{\text{Jacobian Rotation}} \cdot V_B(W_{GT}(p)) \|^2$$ -> FinalConstraintLoss.forward
# [ARCH L0499]  -> UnifiedGeometricLoss / component losses
# [ARCH L0500]     - 단순히 전체 행렬 $W_{GT}$를 곱하는 것이 아니라, $W_{GT}$**의 Jacobian(미분값)을 통해 각 픽셀 위치에서의 '국소 회전량(Local Rotation)'을 계산**하여 적용. -> FinalConstraintLoss.forward
# [ARCH L0501]     - **의미:** 이미지가 회전했다면, 그 안의 엣지(V)도 그 각도만큼 물리적으로 회전했음을 학습합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0502] - **$L_b$ (회전 일관성)** -> GeometricAccuracyLoss.forward (L_b: orientation+mag)
# [ARCH L0503]  -> UnifiedGeometricLoss / component losses
# [ARCH L0504]     $$  L_{b\_local}(p) = \| \text{Rotor}_A(p) - \mathcal{R}_{loc}(W_{GT}, p) \cdot \text{Rotor}_B(W_{GT}(p)) \|^2$$ -> GeometricAccuracyLoss.forward (L_b: orientation+mag)
# [ARCH L0505]  -> UnifiedGeometricLoss / component losses
# [ARCH L0506]     - $W_{GT}$에서 유도된 **지역적 회전(Local Rotor)** 정보와 비교 -> GeometricAccuracyLoss.forward (L_b: orientation+mag)
# [ARCH L0507]     - **의미:** 지역적인 Sin/Cos 정보가 전체 변환 행렬(W)의 회전량과 기하학적으로 호응해야 합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0508]  -> UnifiedGeometricLoss / component losses
# [ARCH L0509] ### **2. Final Consistency (뒤틀림 일관성)** -> FinalConstraintLoss.forward
# [ARCH L0510]  -> UnifiedGeometricLoss / component losses
# [ARCH L0511] 모델이 예측한 $W^*$가 수학적으로 얼마나 견고한지 증명합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0512]  -> UnifiedGeometricLoss / component losses
# [ARCH L0513] - **$L_{coord}$ (모서리 거리) — Smooth L1**을 사용하여 예측된 $W^*$로 변환한 네 모서리 좌표와 정답 좌표 사이의 거리를 줄임 -> UnifiedGeometricLoss / component losses
# [ARCH L0514]  -> UnifiedGeometricLoss / component losses
# [ARCH L0515]     $$  L_{coord}(W_{GT}, W^*) = \frac{1}{4} \sum_{k=1}^{4} \rho \left( \mathcal{T}(p_k; W_{GT}) - \mathcal{T}(p_k; W^*) \right)$$ -> UnifiedGeometricLoss / component losses
# [ARCH L0516]  -> UnifiedGeometricLoss / component losses
# [ARCH L0517]     - $\mathcal{T}(p; W)$ : 좌표 p를 호모그래피 행렬 W를 이용해 변환하는 함수 -> UnifiedGeometricLoss / component losses
# [ARCH L0518]     - $\rho(x)$ : 거리 함수 -> UnifiedGeometricLoss / component losses
# [ARCH L0519]  -> UnifiedGeometricLoss / component losses
# [ARCH L0520]         $$  \rho(x) =  -> UnifiedGeometricLoss / component losses
# [ARCH L0521]           \begin{cases}  -> UnifiedGeometricLoss / component losses
# [ARCH L0522]           0.5 x^2 / \beta & \text{if } |x| < \beta \\ -> UnifiedGeometricLoss / component losses
# [ARCH L0523]           |x| - 0.5 \beta & \text{otherwise} -> UnifiedGeometricLoss / component losses
# [ARCH L0524]           \end{cases}$$ -> UnifiedGeometricLoss / component losses
# [ARCH L0525]  -> UnifiedGeometricLoss / component losses
# [ARCH L0526] - **$L_{sdf\_photo}$ (SDF 기반 복원) —** 복원된 이미지 $\hat{A}$의 SDF가 원본 A의 SDF와 일치하는가 확인 -> UnifiedGeometricLoss / component losses
# [ARCH L0527]  -> UnifiedGeometricLoss / component losses
# [ARCH L0528]     $$  L_{sdf\_photo}={\sum_{p \in \Omega} \| SDF_A(p) - SDF_{\hat{A}}(p; W^*) \|^2}$$ -> UnifiedGeometricLoss / component losses
# [ARCH L0529]  -> UnifiedGeometricLoss / component losses
# [ARCH L0530]  -> UnifiedGeometricLoss / component losses
# [ARCH L0531] ### **3. Iterative & Multi-Scale Constraint (반복 및 스케일 안정성) — [보완] 신규** -> losses.py (section entry)
# [ARCH L0532]  -> UnifiedGeometricLoss / component losses
# [ARCH L0533] 반복 정제(Iterative Refinement)와 다중 해상도(Multi-Scale) 학습 과정에서 모델이 발산하지 않고 올바른 방향으로 수렴하도록 강제하는 제약 조건입니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0534]  -> UnifiedGeometricLoss / component losses
# [ARCH L0535] - **$L_{convergence}$ (수렴 유도 손실)** -> UnifiedGeometricLoss / component losses
# [ARCH L0536]  -> UnifiedGeometricLoss / component losses
# [ARCH L0537]     각 반복 단계($k$)에서 추정된 잔차 변환 $\Delta W^{(k)}$가 점차 Identity ($I$) 에 가까워지도록 유도하여, 불필요한 진동을 억제합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0538]  -> UnifiedGeometricLoss / component losses
# [ARCH L0539]     $$  L_{convergence} = \sum_{k=2}^{K} w_k \cdot \| \Delta W^{(k)} - I \|_F^2$$ -> UnifiedGeometricLoss / component losses
# [ARCH L0540]  -> UnifiedGeometricLoss / component losses
# [ARCH L0541]     - $w_k = k / K$: 반복 횟수가 증가할수록 가중치를 높여(Linear Warm-up), 후반부에는 큰 변화 대신 미세 조정만 수행하도록 강제합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0542]     - $\| \cdot \|_F$: Frobenius Norm (행렬 원소 간 차이의 제곱합) -> UnifiedGeometricLoss / component losses
# [ARCH L0543]     - **의미:** "첫 번째 반복에서 큰 틀을 잡고(Coarse), 이후에는 얌전히 다듬기만 해라(Fine)"는 지침을 줍니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0544] - **$L_{multi\_scale}$ (다중 스케일 일관성 손실)** -> UnifiedGeometricLoss / component losses
# [ARCH L0545]  -> UnifiedGeometricLoss / component losses
# [ARCH L0546]     저해상도(Coarse)에서 추정한 변환이 고해상도(Fine)에서도 유효하도록, 스케일 간의 예측값 일관성을 유지합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0547]  -> UnifiedGeometricLoss / component losses
# [ARCH L0548]     $$  L_{multi\_scale} = \sum_{l=0}^{L-1} \| W^{(l)} - \text{Upsample}(W^{(l+1)}) \|^2$$ -> UnifiedGeometricLoss / component losses
# [ARCH L0549]  -> UnifiedGeometricLoss / component losses
# [ARCH L0550]     - **의미:** "작은 이미지에서 30도 돌렸으면, 큰 이미지에서도 30도 돌아가야 한다"는 물리적 일관성을 보장합니다. 이는 Coarse-to-Fine 전략의 허리 역할을 합니다. -> UnifiedGeometricLoss / component losses
# [ARCH L0602] ### 6.5 Loss / Training 구현 메모 -> losses.py (section entry)
# [ARCH L0603]  -> UnifiedGeometricLoss / component losses
# [ARCH L0604] - losses.py 의 `UnifiedGeometricLoss` 는 architecture.md §5의 큰 구조(Geo + Final + Iter)를 유지하면서,   -> FinalConstraintLoss.forward
# [ARCH L0605]   학습 안정성을 위해 아래 항목을 추가/보강합니다: -> UnifiedGeometricLoss / component losses
# [ARCH L0606]   - `L_angle` (pred angle ↔ gt angle 직접 loss) -> UnifiedGeometricLoss / component losses
# [ARCH L0607]   - `L_pixel` (scalar feature consistency) -> GeometricAccuracyLoss.forward (L_s)
# [ARCH L0608]   - `L_rotation_invariant` (±60° 구간에서 회전 불변성/대칭성 강제) -> FinalConstraintLoss.forward
# [ARCH L0609] - `normalize_rotor_output()` 헬퍼로 cos/sin unit-normalization을 표준화합니다. -> normalize_rotor_output
# [ARCH L0610]  -> UnifiedGeometricLoss / component losses
# [ARCH L0611] --- -> UnifiedGeometricLoss / component losses
# [ARCH L0612]  -> UnifiedGeometricLoss / component losses
# =============================================================================

"""
================================================================================
Phase 5: 통합 기하학적 손실 함수 (Unified Geometric Loss) - v5 RTX 3090 Optimized
================================================================================
[Architecture.md §5 참조]

[v5 수정사항]
1. ±60도 회전에 대응하기 위해 LAMBDA_ANGLE 증가 (35 → 50)
2. 큰 회전에서의 안정성을 위해 SmoothL1 beta 조정
3. 회전 불변 손실 항목 추가 (L_rotation_invariant)

모델의 최종 학습 목표는 아래의 단일 통합 수식을 최소화하는 것입니다.

L_total = α·ΣGeometric_Accuracy + β·Final_Consistency + γ·Iterative_Stability
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# =============================================================================
# [Hyperparameters] Loss Function - v5 ±60° Optimized
# =============================================================================
LAMBDA_SCALAR = 1.0           # [Hyperparameter] 스칼라 부분 가중치
LAMBDA_VECTOR = 35.0          # [Hyperparameter] 벡터 부분 가중치 (평행이동 완화)
LAMBDA_BIVECTOR = 800.0       # [Hyperparameter] Bivector 가중치 (회전 제약 강화) ⭐
BETA_COORD = 8.0              # [Hyperparameter] 좌표 정확도 (회전 우선이므로 낮춤)
LAMBDA_ANGLE = 2000.0         # [Hyperparameter] 각도 손실 가중치 - 최우선! ⭐⭐⭐
LAMBDA_PIXEL = 0.03           # [Hyperparameter] 픽셀 오차 가중치 (낮춤)
LAMBDA_ROTATION_INV = 600.0   # [Hyperparameter] 회전 불변성 강화 ⭐
SMOOTH_L1_BETA = 2.0          # [Hyperparameter] 큰 각도 오차에 민감 ⭐
GAMMA_CONVERGENCE = 0.5       # [Hyperparameter] 수렴 가중치
GAMMA_MULTISCALE = 0.3        # [Hyperparameter] 멀티스케일 가중치

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
        V_B_flat = V_B.view(B, -1, H, W)  # (B, C*2, H, W)
        V_B_warped_flat = F.grid_sample(V_B_flat, grid, align_corners=False)
        V_B_warped = V_B_warped_flat.view(B, C, 2, H, W)

        # 국소 회전(2x2) 적용
        A_loc = self.compute_local_rotation(W_gt)  # (B, 2, 2)

        # V_B_warped의 벡터 성분에 회전 적용
        V_B_warped_perm = V_B_warped.permute(0, 1, 3, 4, 2)  # (B, C, H, W, 2)
        V_B_rotated = torch.einsum('bij,bchwj->bchwi', A_loc, V_B_warped_perm)
        V_B_rotated = V_B_rotated.permute(0, 1, 4, 2, 3)  # (B, C, 2, H, W)

        L_v = F.mse_loss(V_A, V_B_rotated)

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
        mag_B_warped = F.grid_sample(mag_B, grid, align_corners=False)
        L_b_mag = F.mse_loss(mag_A, mag_B_warped)

        # orientation alignment (가능할 때만)
        L_b_orient = torch.tensor(0.0, device=S_A.device)
        if (cos_A is not None) and (sin_A is not None) and (cos_B is not None) and (sin_B is not None):
            cos_B_warped = F.grid_sample(cos_B, grid, align_corners=False)
            sin_B_warped = F.grid_sample(sin_B, grid, align_corners=False)

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

            L_b_cos = F.mse_loss(cos_A, cos_B_rot)
            L_b_sin = F.mse_loss(sin_A, sin_B_rot)
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
    [Phase 5.2] Final Consistency (뒤틀림 일관성) - v5 수정
    
    Architecture.md §5.2
    
    [v5 수정사항]
    1. L_angle 가중치 증가 (±60도 대응)
    2. SmoothL1 beta 조정 (큰 오차 관용)
    3. [v5 신규] L_rotation_invariant 추가
    
    - L_coord (모서리 거리): 네 모서리 좌표의 SmoothL1 거리
    - L_angle (각도 일치): cos(pred_angle - gt_angle)
    - L_pixel (SDF 기반 복원): 복원된 이미지의 SDF 일치
    - L_rotation_inv [v5]: 회전 불변량 보존 손실
    """
    
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
        
    def get_inverse_affine(self, matrix_2x3):
        """
        [Helper] 2x3 Affine 행렬의 역행렬 계산
        """
        # linalg.inv / cat 등은 dtype이 정확히 맞아야 하고,
        # fp16에서는 수치적으로 불안정하거나 일부 디바이스에서 지원이 제한될 수 있습니다.
        # => 역행렬 계산은 fp32로 수행한 뒤, 원래 dtype으로 되돌립니다.
        B = matrix_2x3.shape[0]
        in_dtype = matrix_2x3.dtype
        device = matrix_2x3.device

        matrix_2x3_f = matrix_2x3.to(dtype=torch.float32)
        bottom_row = torch.tensor([0., 0., 1.], device=device, dtype=matrix_2x3_f.dtype)
        bottom_row = bottom_row.view(1, 1, 3).repeat(B, 1, 1)

        matrix_3x3 = torch.cat([matrix_2x3_f, bottom_row], dim=1)
        matrix_inv = torch.linalg.inv(matrix_3x3)
        return matrix_inv[:, :2, :].to(dtype=in_dtype)
    
    def compute_rotation_invariant_loss(self, pred_W, W_gt):
        """
        [v5 신규] 회전 불변량 보존 손실
        
        행렬의 고유값(특이값)은 회전에 불변 → 이를 비교하여 스케일 정확도 보장
        """
        # 2x2 선형 변환 부분 추출
        # (AMP 상황에서 dtype mismatch로 인한 bmm/mse 오류를 피하기 위해 fp32로 계산)
        pred_linear = pred_W[:, :2, :2].to(dtype=torch.float32)  # (B, 2, 2)
        gt_linear = W_gt[:, :2, :2].to(dtype=torch.float32)
        
        # 특이값 분해 (SVD) 대신 간단히 determinant 비교
        # det(R) = 1 (순수 회전) 이어야 함
        pred_det = pred_linear[:, 0, 0] * pred_linear[:, 1, 1] - pred_linear[:, 0, 1] * pred_linear[:, 1, 0]
        gt_det = gt_linear[:, 0, 0] * gt_linear[:, 1, 1] - gt_linear[:, 0, 1] * gt_linear[:, 1, 0]
        
        # Determinant 차이 (스케일 보존)
        L_det = F.mse_loss(pred_det, gt_det)
        
        # 직교성 손실: R^T @ R ≈ I
        pred_RTR = torch.bmm(pred_linear.transpose(1, 2), pred_linear)
        identity = torch.eye(2, device=pred_W.device, dtype=pred_RTR.dtype).unsqueeze(0).repeat(pred_W.shape[0], 1, 1)
        L_ortho = F.mse_loss(pred_RTR, identity)
        
        return L_det + 0.5 * L_ortho
    
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
        # AMP로 pred_W가 fp16, W_gt가 fp32일 때 torch.bmm dtype mismatch가 날 수 있어
        # 좌표/각도 관련 손실은 fp32로 계산합니다(grad는 정상적으로 전파됨).
        pred_W_f = pred_W.to(dtype=torch.float32)
        W_gt_f = W_gt.to(dtype=torch.float32)

        corners = torch.tensor([
            [-1., -1., 1.], [1., -1., 1.],
            [1., 1., 1.], [-1., 1., 1.]
        ], device=device, dtype=pred_W_f.dtype)
        corners = corners.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)  # (B, 3, 4)

        pts_pred = torch.bmm(pred_W_f, corners)  # (B, 2, 4)
        pts_gt = torch.bmm(W_gt_f, corners)
        
        L_coord = self.smooth_l1(pts_pred, pts_gt)
        
        # =====================================================================
        # [§5.2.2] L_angle (각도 일치) - v5: 가중치 증가
        # L_angle = 1 - cos(pred_angle - gt_angle)
        # =====================================================================
        # pred_cos/pred_sin은 이미 normalize_rotor_output()을 거친 값(권장)입니다.
        # atan2 기반 각도 손실은 ±180° 근처(π)에서 sin(Δ)≈0로 gradient가 약해질 수 있어,
        # cos/sin 공간에서 직접 정렬시키는 형태(내적 기반)로 변경합니다.
        gt_angle_f = gt_angle_rad.to(dtype=torch.float32)
        gt_cos = torch.cos(gt_angle_f)
        gt_sin = torch.sin(gt_angle_f)
        pred_cos_f = pred_cos.to(dtype=torch.float32)
        pred_sin_f = pred_sin.to(dtype=torch.float32)
        dot = (pred_cos_f * gt_cos + pred_sin_f * gt_sin).clamp(-1.0, 1.0)
        L_angle = 1.0 - dot.mean()
        
        # =====================================================================
        # [§5.2.3] L_pixel (SDF 기반 복원) - Optional
        # =====================================================================
        L_pixel = torch.tensor(0.0, device=device, dtype=torch.float32)
        if S_A is not None and S_B is not None:
            # grid_sample은 input/grid dtype이 동일해야 하므로 S_B dtype 기준으로 맞춥니다.
            theta = self.get_inverse_affine(pred_W).to(dtype=S_B.dtype)
            grid = F.affine_grid(theta, S_B.size(), align_corners=False)
            S_A_warped = F.grid_sample(
                S_A.to(dtype=grid.dtype),
                grid,
                align_corners=False,
                padding_mode='zeros'
            )
            L_pixel = F.mse_loss(S_A_warped, S_B.to(dtype=grid.dtype))
        
        # =====================================================================
        # [v5 신규] L_rotation_inv (회전 불변량 보존)
        # =====================================================================
        L_rot_inv = self.compute_rotation_invariant_loss(pred_W, W_gt)
        
        L_final = (BETA_COORD * L_coord + 
                   LAMBDA_ANGLE * L_angle + 
                   LAMBDA_PIXEL * L_pixel +
                   LAMBDA_ROTATION_INV * L_rot_inv)
        
        return L_final, {
            'L_coord': L_coord.item(),
            'L_angle': L_angle.item(),
            'L_pixel': L_pixel.item(),
            'L_rot_inv': L_rot_inv.item()  # [v5 신규]
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
    
    print(f"\n[v5] ±60° Test Results:")
    print(f"Total Loss: {total_loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")