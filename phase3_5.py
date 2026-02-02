"""
================================================================================
Phase 3.5: Dual-Adaptive Recurrent Refinement
================================================================================
[Architecture.md §3.5 참조] - 신규 구현

Phase 3의 단일 추정(Single-Shot)으로는 큰 변환(>15°, >20px)에서 정확도가 
떨어지는 문제를 해결하기 위해, 이중 적응형(Dual-Adaptive) 전략과 
경량 순환 신경망(Mini-GRU)을 결합한 능동적 정제 단계입니다.

핵심 메커니즘:
1. Level Selection (거시적 선택): 오차 크기에 따라 피라미드 레벨 선택
2. Feature Selection (미시적 선택): 오차 타입에 따라 S/V/B 특징 선택
3. Recurrent Memory (순환 기억): Mini-GRU가 이전 수정 방향 유지

수식:
W_final = MiniGRU(F_selected^level) ∘ ... ∘ MiniGRU(F_selected^level)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# [Hyperparameters] Phase 3.5
# =============================================================================
# [반복 정제]
NUM_ITERATIONS = 4               # [Hyperparameter] 최대 반복 횟수
GRU_HIDDEN_DIM = 16              # [Hyperparameter] GRU Hidden State 차원

# [Level Selection 임계값] (px 단위)
LEVEL_THRESHOLD_HIGH = 30        # [Hyperparameter] Level 3 선택 임계값
LEVEL_THRESHOLD_MID = 10         # [Hyperparameter] Level 2 선택 임계값
LEVEL_THRESHOLD_LOW = 5          # [Hyperparameter] Level 1 선택 임계값

# [Feature Selection 임계값]
FEATURE_POS_THRESHOLD = 15.0     # [Hyperparameter] S 선택 위치 오차 임계값 (px)
FEATURE_ANGLE_THRESHOLD = 0.25   # [Hyperparameter] V 선택 방향 오차 임계값 (rad ≈ 14.3°)

# [종료 조건]
CONVERGENCE_THRESHOLD = 0.005    # [Hyperparameter] 수렴 판정 임계값 (Frobenius Norm)
TARGET_ERROR_PX = 3.0            # [Hyperparameter] 목표 오차 (Phase 4 이관 기준)
TOLERANCE_ALPHA = 0.05           # [Hyperparameter] 발산 방지 허용 오차 (5%)

# [안전장치]
MAX_CONSECUTIVE_REJECTIONS = 3   # [Hyperparameter] 연속 Rejection 허용 횟수
LR_DECAY_FACTOR = 0.5            # [Hyperparameter] Rejection 시 LR 감소 비율


class MiniConvGRU(nn.Module):
    """
    [Phase 3.5 - Mini-ConvGRU] 경량 순환 엔진
    
    Architecture.md §3.5.2 - Mini-ConvGRU
    
    IGEV의 Full ConvGRU를 1/4 크기로 경량화하고,
    Correlation Volume을 제거하여 메모리 효율을 극대화했습니다.
    
    구조 (Minimal-GRU):
    - z_k = σ(Conv([h_{k-1}, E_diff]))        [Update Gate: 16채널]
    - h̃_k = tanh(Conv([F_selected, E_diff]))  [Candidate State]
    - h_k = (1-z_k) ⊙ h_{k-1} + z_k ⊙ h̃_k   [Linear Interpolation]
    - ΔW_k = Head(h_k)                        [16→8→4 채널]
    
    주요 개선점:
    - Reset Gate 제거: Minimal-GRU 구조로 파라미터 50% 감소
    - Correlation Volume 제거: Difference Map으로 대체
    - 16채널 Hidden State: 원본 IGEV(64채널) 대비 75% 메모리 절감
    """
    
    def __init__(self, input_dim, hidden_dim=GRU_HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # [Update Gate] z = σ(W_z * [h, E_diff])
        # Reset Gate 제거 (Minimal-GRU)
        self.conv_z = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # [Candidate State] h̃ = tanh(W_h * [F_selected, E_diff])
        self.conv_h = nn.Conv2d(input_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        # [Delta W Head] h → ΔW (16→8→4)
        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(hidden_dim // 2, 4, kernel_size=1)  # (cos, sin, dx, dy)
        )
        
    def init_hidden(self, batch, height, width, device):
        """Hidden State 초기화"""
        return torch.zeros(batch, self.hidden_dim, height, width, device=device)
    
    def forward(self, h_prev, e_diff, f_selected):
        """
        Args:
            h_prev: 이전 Hidden State (B, hidden_dim, H, W)
            e_diff: Difference Map |A' - B| (B, C, H, W)
            f_selected: 선택된 특징 (B, C, H, W)
            
        Returns:
            h_new: 새로운 Hidden State
            delta_w: 잔차 변환 (B, 4, H, W)
        """
        # [Update Gate]
        z_input = torch.cat([h_prev, e_diff], dim=1)
        z = torch.sigmoid(self.conv_z(z_input))
        
        # [Candidate State]
        h_input = torch.cat([f_selected, e_diff], dim=1)
        h_candidate = torch.tanh(self.conv_h(h_input))
        
        # [Linear Interpolation] (Minimal-GRU 핵심)
        h_new = (1 - z) * h_prev + z * h_candidate
        
        # [Delta W 출력]
        delta_w = self.delta_head(h_new)
        
        return h_new, delta_w


class ErrorDiagnostic(nn.Module):
    """
    [Phase 3.5 - Error Diagnostic] 오차 측정 모듈
    
    Architecture.md §3.5.1.A - Level Selection을 위한 오차 측정
    
    E_pos = Mean(|SDF_A(W_curr(p)) - SDF_B(p)|)  [위치 오차]
    E_angle = 1 - Mean(cos(θ_residual))          [방향 오차]
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_position_error(self, warped_sdf, target_sdf):
        """위치 오차 계산 (px 단위 근사)"""
        diff = torch.abs(warped_sdf - target_sdf)
        # SDF 차이를 픽셀 단위로 스케일링 (휴리스틱)
        error_px = diff.mean(dim=(1, 2, 3)) * 100  # 대략적인 스케일
        return error_px
    
    def compute_angle_error(self, warped_vector, target_vector):
        """방향 오차 계산 (rad 단위)"""
        # Cosine Similarity
        cos_sim = F.cosine_similarity(warped_vector, target_vector, dim=1, eps=1e-6)
        # 1 - cos(θ) ∈ [0, 2]
        error_angle = (1.0 - cos_sim.mean(dim=(1, 2))).clamp(0, 2)
        return error_angle
    
    def forward(self, warped_features, target_features):
        """
        Args:
            warped_features: dict with 'sdf', 'vector'
            target_features: dict with 'sdf', 'vector'
        
        Returns:
            e_pos: 위치 오차 (B,)
            e_angle: 방향 오차 (B,)
        """
        e_pos = self.compute_position_error(
            warped_features['sdf'], 
            target_features['sdf']
        )
        e_angle = self.compute_angle_error(
            warped_features['vector'], 
            target_features['vector']
        )
        return e_pos, e_angle


class DualAdaptiveSelector(nn.Module):
    """
    [Phase 3.5 - Dual-Adaptive Selector] 이중 선택 전략
    
    Architecture.md §3.5.1 - Dual-Adaptive Routing
    
    매 반복마다 "어느 레벨에서, 어떤 특징을 사용할 것인가?"를 동적으로 결정합니다.
    
    A. Level Selection (피라미드 레벨 선택):
       오차 크기 기준으로 적절한 수용 범위를 가진 레벨 선택
       
    B. Feature Selection (특징 선택):
       오차 타입 기준으로 가장 관련 있는 Clifford 성분 선택
    """
    
    def __init__(self):
        super().__init__()
        
    def select_level(self, e_pos):
        """
        [Level Selection]
        
        | 오차 범위 | 선택 레벨 | 수용 영역 | 목적 |
        |---------|----------|---------|------|
        | > 30px  | Level 3  | 넓음     | 큰 변환 포착 |
        | 10~30px | Level 2  | 중간     | 구조적 정렬 |
        | 5~10px  | Level 1  | 좁음     | 세부 매칭 |
        | < 5px   | Level 0  | 픽셀     | 미세 조정 |
        """
        # Batch 평균 오차 사용
        avg_error = e_pos.mean().item()
        
        if avg_error > LEVEL_THRESHOLD_HIGH:
            return 3  # Global
        elif avg_error > LEVEL_THRESHOLD_MID:
            return 2  # Structural
        elif avg_error > LEVEL_THRESHOLD_LOW:
            return 1  # Local
        else:
            return 0  # Fine
    
    def select_feature(self, e_pos, e_angle):
        """
        [Feature Selection]
        
        | 오차 타입 | 선택 특징 | 역할 |
        |---------|----------|-----|
        | 위치 불일치 (E_pos 지배적)    | S (Scalar)  | 텍스처 매칭, SDF 정렬 |
        | 방향 불일치 (E_angle 지배적)  | V (Vector)  | 그래디언트 방향 정렬 |
        | 스케일/회전 불일치 (둘 다 큼) | B (Bivector)| Rotor 보정 |
        """
        avg_pos = e_pos.mean().item()
        avg_angle = e_angle.mean().item()
        
        if avg_pos > FEATURE_POS_THRESHOLD:
            return 'S'  # 위치부터 맞춤
        elif avg_angle > FEATURE_ANGLE_THRESHOLD:
            return 'V'  # 방향 정렬
        else:
            return 'B'  # 미세 회전 보정
    
    def forward(self, e_pos, e_angle):
        """
        Returns:
            selected_level: int (0~3)
            selected_feature: str ('S', 'V', or 'B')
        """
        level = self.select_level(e_pos)
        feature = self.select_feature(e_pos, e_angle)
        return level, feature


class TransformAccumulator:
    """
    [Phase 3.5 Helper] 변환 행렬 누적 관리
    
    변환의 합성: W_current = ΔW ∘ W_prev
    """
    
    def __init__(self, device):
        self.device = device
        # Identity Matrix (2x3)
        self.W_accum = torch.eye(2, 3, device=device).unsqueeze(0)
        
    def reset(self, batch_size=1):
        """Identity로 초기화"""
        self.W_accum = torch.eye(2, 3, device=self.device).unsqueeze(0)
        if batch_size > 1:
            self.W_accum = self.W_accum.repeat(batch_size, 1, 1)
    
    def compose(self, delta_w_map):
        """
        Delta W Map을 평균내어 Global Transform에 합성
        
        Args:
            delta_w_map: (B, 4, H, W) - (cos, sin, dx, dy)
        """
        # 공간 평균
        avg_delta = delta_w_map.mean(dim=(2, 3))  # (B, 4)
        cos_d, sin_d = avg_delta[:, 0], avg_delta[:, 1]
        dx_d, dy_d = avg_delta[:, 2], avg_delta[:, 3]
        
        # Delta Matrix 구성
        B = cos_d.shape[0]
        delta_mat = torch.zeros(B, 2, 3, device=self.device)
        delta_mat[:, 0, 0] = cos_d
        delta_mat[:, 0, 1] = -sin_d
        delta_mat[:, 0, 2] = dx_d
        delta_mat[:, 1, 0] = sin_d
        delta_mat[:, 1, 1] = cos_d
        delta_mat[:, 1, 2] = dy_d
        
        # 합성: W_new = delta @ W_old
        # 3x3으로 확장 후 행렬곱
        W_aug = torch.cat([
            self.W_accum, 
            torch.tensor([[[0, 0, 1]]], device=self.device).repeat(B, 1, 1)
        ], dim=1)
        delta_aug = torch.cat([
            delta_mat,
            torch.tensor([[[0, 0, 1]]], device=self.device).repeat(B, 1, 1)
        ], dim=1)
        
        result = torch.bmm(delta_aug, W_aug)
        self.W_accum = result[:, :2, :]
        
        return self.W_accum
    
    def get_current(self):
        return self.W_accum


class IterativeRefinementLoop(nn.Module):
    """
    [Phase 3.5 Main] 반복 정제 루프
    
    Architecture.md §3.5.3 - 반복 정제 루프
    
    실제 시나리오별 동작 흐름:
    | Iter | 오차 상태 | 선택 레벨 | 선택 특징 | 목표 |
    |------|----------|----------|----------|-----|
    | 1    | 45px, 20°| Level 3  | V + B    | 20px |
    | 2    | 20px, 5° | Level 2  | S        | 8px  |
    | 3    | 8px, 1°  | Level 1  | S        | 3px  |
    | 4    | 3px, 0.3°| Level 1  | S        | 1-2px|
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # [Components]
        self.error_diagnostic = ErrorDiagnostic()
        self.selector = DualAdaptiveSelector()
        
        # [Mini-GRU] 각 특징 타입별 GRU
        self.gru_s = MiniConvGRU(input_dim=feature_dim // 3)
        self.gru_v = MiniConvGRU(input_dim=feature_dim // 3)
        self.gru_b = MiniConvGRU(input_dim=feature_dim // 3)
        
        # [Feature Extractor] S, V, B 분리용
        self.chunk_dim = feature_dim // 3
        
    def extract_feature_by_type(self, features, feature_type):
        """특징 타입에 따라 채널 분리"""
        # features: (B, C, H, W) where C = feature_dim
        s, v, b = torch.chunk(features, 3, dim=1)
        
        if feature_type == 'S':
            return s
        elif feature_type == 'V':
            return v
        else:  # 'B'
            return b
    
    def get_gru_by_type(self, feature_type):
        """특징 타입에 따른 GRU 반환"""
        if feature_type == 'S':
            return self.gru_s
        elif feature_type == 'V':
            return self.gru_v
        else:
            return self.gru_b
    
    def warp_features(self, features, W_matrix, target_size):
        """
        현재 변환 행렬로 특징 워핑
        
        Args:
            features: (B, C, H, W)
            W_matrix: (B, 2, 3)
            target_size: (H, W)
        """
        grid = F.affine_grid(W_matrix, [features.shape[0], features.shape[1], 
                                         target_size[0], target_size[1]], 
                             align_corners=False)
        warped = F.grid_sample(features, grid, align_corners=False, 
                               mode='bilinear', padding_mode='zeros')
        return warped
    
    def check_convergence(self, delta_w):
        """
        [종료 조건 A] 변화량 수렴 확인
        
        ||ΔW - I||_F < ε_conv
        """
        B = delta_w.shape[0]
        avg_delta = delta_w.mean(dim=(2, 3))
        
        # Identity와의 차이
        identity = torch.tensor([[1, 0, 0, 0]], device=delta_w.device).repeat(B, 1)
        diff = avg_delta - identity
        frobenius_norm = torch.norm(diff, dim=1).mean().item()
        
        return frobenius_norm < CONVERGENCE_THRESHOLD
    
    def check_divergence(self, e_curr, e_prev):
        """
        [안전장치] 발산 확인
        
        E_next > E_curr × (1 + α)
        """
        if e_prev is None:
            return False
        return e_curr > e_prev * (1 + TOLERANCE_ALPHA)
    
    def forward(self, pyramid_features_a, pyramid_features_b, 
                phase3_results, device):
        """
        [Phase 3.5 Forward]
        
        Args:
            pyramid_features_a: Phase 3에서 처리된 이미지 A의 피라미드 특징
            pyramid_features_b: Phase 3에서 처리된 이미지 B의 피라미드 특징
            phase3_results: Phase 3의 출력 (초기 rotor_map 포함)
            device: 연산 장치
            
        Returns:
            W_final: 최종 변환 행렬 (B, 2, 3)
            refinement_history: 각 반복의 상태 기록
        """
        # 초기화
        B = pyramid_features_a[0].shape[0]
        accumulator = TransformAccumulator(device)
        accumulator.reset(B)
        
        # Hidden States 초기화 (Level 0 해상도 기준)
        _, _, H, W = pyramid_features_a[0].shape
        h_s = self.gru_s.init_hidden(B, H, W, device)
        h_v = self.gru_v.init_hidden(B, H, W, device)
        h_b = self.gru_b.init_hidden(B, H, W, device)
        hidden_states = {'S': h_s, 'V': h_v, 'B': h_b}
        
        # 반복 기록
        history = []
        e_prev = None
        consecutive_rejections = 0
        
        print(f"[Phase 3.5] Starting Iterative Refinement (max {NUM_ITERATIONS} iterations)")
        
        for k in range(NUM_ITERATIONS):
            W_curr = accumulator.get_current()
            
            # 1. 현재 변환으로 A 워핑
            feat_a_warped = self.warp_features(
                pyramid_features_a[0], W_curr, (H, W)
            )
            
            # 2. Difference Map 계산
            e_diff = torch.abs(feat_a_warped - pyramid_features_b[0])
            
            # 3. 오차 진단
            warped_dict = {
                'sdf': feat_a_warped[:, :self.chunk_dim, :, :],
                'vector': feat_a_warped[:, self.chunk_dim:2*self.chunk_dim, :, :]
            }
            target_dict = {
                'sdf': pyramid_features_b[0][:, :self.chunk_dim, :, :],
                'vector': pyramid_features_b[0][:, self.chunk_dim:2*self.chunk_dim, :, :]
            }
            e_pos, e_angle = self.error_diagnostic(warped_dict, target_dict)
            e_curr = e_pos.mean().item()
            
            # 4. [종료 조건 B] 목표 오차 도달
            if e_curr < TARGET_ERROR_PX:
                print(f"  [Iter {k+1}] Target reached: {e_curr:.2f}px < {TARGET_ERROR_PX}px ✓")
                break
            
            # 5. [안전장치] 발산 확인
            if self.check_divergence(e_curr, e_prev):
                consecutive_rejections += 1
                print(f"  [Iter {k+1}] Divergence detected ({consecutive_rejections}/{MAX_CONSECUTIVE_REJECTIONS})")
                
                if consecutive_rejections >= 2:
                    # GRU Reset + LR Decay (여기서는 간단히 Hidden Reset만)
                    for key in hidden_states:
                        hidden_states[key] = hidden_states[key] * LR_DECAY_FACTOR
                    print(f"  [Recovery] GRU states scaled by {LR_DECAY_FACTOR}")
                
                if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                    print(f"  [Emergency] Max rejections reached, aborting...")
                    break
                continue
            else:
                consecutive_rejections = 0
            
            # 6. Dual-Adaptive Selection
            selected_level, selected_feature = self.selector(e_pos, e_angle)
            
            # 7. 해당 레벨의 특징 가져오기 (해상도 맞춤)
            level_feat_a = pyramid_features_a[min(selected_level, len(pyramid_features_a)-1)]
            level_feat_b = pyramid_features_b[min(selected_level, len(pyramid_features_b)-1)]
            
            # 현재 해상도에 맞게 리사이즈
            if level_feat_a.shape[-2:] != (H, W):
                level_feat_a = F.interpolate(level_feat_a, size=(H, W), mode='bilinear', align_corners=True)
                level_feat_b = F.interpolate(level_feat_b, size=(H, W), mode='bilinear', align_corners=True)
            
            # 8. 선택된 특징 추출
            f_selected = self.extract_feature_by_type(level_feat_a, selected_feature)
            e_diff_selected = self.extract_feature_by_type(e_diff, selected_feature)
            
            # 9. Mini-GRU 실행
            gru = self.get_gru_by_type(selected_feature)
            h_prev = hidden_states[selected_feature]
            
            # [Level Transfer] 해상도 맞춤
            if h_prev.shape[-2:] != (H, W):
                h_prev = F.interpolate(h_prev, size=(H, W), mode='bilinear', align_corners=True)
            
            h_new, delta_w = gru(h_prev, e_diff_selected, f_selected)
            hidden_states[selected_feature] = h_new
            
            # 10. [종료 조건 A] 수렴 확인
            if self.check_convergence(delta_w):
                print(f"  [Iter {k+1}] Converged: ΔW ≈ I ✓")
                break
            
            # 11. 변환 누적
            W_new = accumulator.compose(delta_w)
            
            # 기록
            history.append({
                'iteration': k + 1,
                'error_px': e_curr,
                'error_angle': e_angle.mean().item(),
                'selected_level': selected_level,
                'selected_feature': selected_feature
            })
            
            print(f"  [Iter {k+1}] Error={e_curr:.1f}px | Level={selected_level} | Feature={selected_feature}")
            
            e_prev = e_curr
        
        W_final = accumulator.get_current()
        return W_final, history


class Phase35Refiner(nn.Module):
    """
    [Phase 3.5 Wrapper] Dual-Adaptive Recurrent Refinement
    
    Architecture.md §3.5 전체 구현
    
    Phase 3와 Phase 4 사이에서 동작하며, Phase 3의 초기 추정을
    반복적으로 정제하여 큰 변환에서도 정확한 매칭을 달성합니다.
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        self.refinement_loop = IterativeRefinementLoop(feature_dim)
        
    def forward(self, pyramid_features_a, pyramid_features_b, 
                phase3_results, device=None):
        """
        Args:
            pyramid_features_a: list of (B, C, H, W) - 각 레벨의 특징
            pyramid_features_b: list of (B, C, H, W)
            phase3_results: Phase 3 출력 (rotor_map 등)
            
        Returns:
            W_refined: 정제된 변환 행렬 (B, 2, 3)
            history: 반복 기록
        """
        if device is None:
            device = pyramid_features_a[0].device
            
        W_refined, history = self.refinement_loop(
            pyramid_features_a, 
            pyramid_features_b,
            phase3_results,
            device
        )
        
        return W_refined, history


# =============================================================================
# 테스트 코드
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3.5 Test on: {device}")
    
    # 더미 데이터 생성 (실제로는 Phase 3 출력 사용)
    B, C, H, W = 1, 144, 64, 64
    
    # 4레벨 피라미드 시뮬레이션
    pyramid_a = [
        torch.randn(B, C, H // (2**i), W // (2**i), device=device)
        for i in range(4)
    ]
    pyramid_b = [
        torch.randn(B, C, H // (2**i), W // (2**i), device=device)
        for i in range(4)
    ]
    
    # Phase 3.5 실행
    refiner = Phase35Refiner(feature_dim=C).to(device)
    
    with torch.no_grad():
        W_refined, history = refiner(pyramid_a, pyramid_b, None, device)
    
    print(f"\n[Result]")
    print(f"  W_refined shape: {W_refined.shape}")
    print(f"  Iterations: {len(history)}")
    for h in history:
        print(f"    Iter {h['iteration']}: {h['error_px']:.1f}px, Level={h['selected_level']}, Feature={h['selected_feature']}")
