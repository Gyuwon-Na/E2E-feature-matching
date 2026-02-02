"""
================================================================================
Phase 4: 기하학적 에너지 기반 MPC 정제 (추론 단계에서만)
================================================================================
[Architecture.md §4 참조]

Phase 3/3.5에서 딥러닝이 예측한 매칭 지도를 바탕으로 물리적인 에너지 함수를 
최소화하여 0.1 픽셀 단위의 초정밀 정렬을 달성하는 단계입니다.

주요 단계:
1. 전역 필터링 및 초기화: Phase 2의 평균 Rotor/Scale로 초기 W0 설정
2. 지역 탐색 (Priority Search): 신뢰도 높은 영역 우선순위 지도 생성
3. 에너지 평면 생성: S, V, B 성분별 에너지 계산
4. Gate 가중 최적화: Phase 3의 Gate 값으로 가중 최적화

에너지 함수:
E_total = (1/N) Σ_p (g_s·E_scalar + g_v·E_vector + g_b·E_bivector)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from phase3 import GeometricDescriptorGuidance, FEATURE_DIM

# =============================================================================
# [Hyperparameters] Phase 4
# =============================================================================
MPC_LEARNING_RATE = 0.005        # [Hyperparameter] MPC 최적화 학습률
MPC_ITERATIONS = 300             # [Hyperparameter] MPC 루프 반복 횟수
MPC_EPSILON = 1e-6               # [Hyperparameter] 수치 안정성 엡실론
SCHEDULER_ETA_MIN = 0.001        # [Hyperparameter] Cosine Annealing 최소 LR
PRIORITY_VARIANCE_MIN = 1e-3     # [Hyperparameter] Priority Map 분산 하한
PRIORITY_PERCENTILE = 0.99       # [Hyperparameter] Priority Map 상위 백분위
VALID_MASK_THRESHOLD = 0.9       # [Hyperparameter] Warping 유효 영역 임계값


class GeometricMPCRefiner(nn.Module):
    """
    [Phase 4 메인 클래스] 기하학적 에너지 기반 MPC 정제 모듈
    
    Architecture.md §4 전체 구현
    
    Phase 3의 매칭 지도와 Gate 정보를 바탕으로 물리적 에너지를 최소화하여 
    0.1 픽셀 단위의 정밀 정렬을 수행합니다.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Hyperparameters
        self.learning_rate = MPC_LEARNING_RATE
        self.iterations = MPC_ITERATIONS
        self.epsilon = MPC_EPSILON
        
        # 변환 행렬 W (Affine Transform: 2x3) - 최적화 대상
        self.W = nn.Parameter(torch.eye(2, 3).unsqueeze(0).to(device)) 

    def get_affine_grid(self, shape):
        """
        [Phase 4 Helper] 현재 W를 기반으로 Sampling Grid 생성
        """
        B, C, H, W = shape
        grid = F.affine_grid(self.W, [B, C, H, W], align_corners=False)
        return grid

    def global_filtering_init(self, mean_rotor, mean_scale):
        """
        [Phase 4 - Step 1] 전역 필터링 및 초기화
        
        Architecture.md §4.1 - 전역 필터링 및 초기화
        
        Phase 2의 평균 Rotor와 Scale을 사용하여 초기 변환 행렬 W0를 설정합니다.
        "대략 몇 도 돌아갔고 몇 배 커졌는지" 알고 시작하여 수렴 속도 향상
        
        Args:
            mean_rotor: Phase 2 Bivector의 평균 각도 (Radian)
            mean_scale: Phase 2 Rotor Magnitude의 평균
        """
        # Rotation Matrix 구성 (2x2)
        cos_theta = torch.cos(torch.tensor(mean_rotor))
        sin_theta = torch.sin(torch.tensor(mean_rotor))
        
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ]).to(self.device)
        
        # Scale 적용 및 W0 생성
        transform_matrix = rotation_matrix * mean_scale
        
        with torch.no_grad():
            self.W[0, :2, :2] = transform_matrix
            self.W[0, :2, 2] = 0.0 
            
        print(f"[Phase 4] Global Filter Init: Rotation={np.degrees(mean_rotor):.2f}deg, Scale={mean_scale:.2f}")

    def compute_priority_map(self, rotor_variance, feature_magnitude):
        """
        [Phase 4 - Step 2] 지역 탐색 (Priority Search)
        
        Architecture.md §4.2 - 지역 탐색
        
        회전 정보가 일관되고 기하학적 덩어리가 뚜렷한 구역에 높은 가중치를 주어,
        신뢰도가 높은 지역부터 자석처럼 맞추도록 유도합니다.
        
        Args:
            rotor_variance: Rotor의 지역적 분산 (B, 1, H, W)
            feature_magnitude: 특징의 크기 (B, 1, H, W)
            
        Returns:
            priority_map: 정규화된 우선순위 맵 (B, 1, H, W)
        """
        # [Robust Logic] 분산 하한선 제한 (Clamping)
        safe_variance = torch.clamp(rotor_variance, min=PRIORITY_VARIANCE_MIN)
        stability = 1.0 / safe_variance
        
        raw_priority = stability * feature_magnitude
        
        # [Robust Min-Max Normalization]
        B = raw_priority.shape[0]
        priority_map = torch.zeros_like(raw_priority)
        
        for b in range(B):
            flat = raw_priority[b].view(-1)
            v_min = flat.min()
            v_max = torch.quantile(flat, PRIORITY_PERCENTILE)  # 상위 1% 값까지만 인정
            
            clipped = torch.clamp(raw_priority[b], min=v_min, max=v_max)
            priority_map[b] = (clipped - v_min) / (v_max - v_min + self.epsilon)
            
        return priority_map

    def compute_energy(self, src_dict, tgt_dict, gates):
        """
        [Phase 4 - Step 3 & 4] 에너지 평면 생성 및 Gate 가중 최적화
        
        Architecture.md §4.3 - 에너지 평면 생성
        Architecture.md §4.4 - 기하학적 게이트 가중 최적화
        
        E_total = (1/N) Σ_p (g_s·E_scalar + g_v·E_vector + g_b·E_bivector)
        
        Args:
            src_dict: Source 데이터 {'sdf', 'vector', 'rotor'}
            tgt_dict: Target 데이터 {'sdf', 'vector', 'rotor'}
            gates: (g_s, g_v, g_b) Gate 튜플
            
        Returns:
            total_loss: 스칼라 손실 값
        """
        grid = self.get_affine_grid(src_dict['sdf'].shape)

        # --- [Valid Mask 생성] ---
        # 회전하여 검정색으로 잘린 영역은 Loss에서 제외
        ones_mask = torch.ones_like(src_dict['sdf'])
        warped_mask = F.grid_sample(ones_mask, grid, align_corners=False)
        binary_mask = (warped_mask > VALID_MASK_THRESHOLD).float()
        
        # 1. Warping
        warped_sdf = F.grid_sample(src_dict['sdf'], grid, align_corners=False)
        warped_vector = F.grid_sample(src_dict['vector'], grid, align_corners=False)
        warped_rotor = F.grid_sample(src_dict['rotor'], grid, align_corners=False)
        
        # 2. Vector Field Rotation Correction (방향 보존성)
        # 이미지가 회전했다면 벡터도 그만큼 회전해야 함
        current_rot_matrix = self.W[0, :2, :2] 
        warped_vector_corrected = torch.einsum(
            'ij, bhwj -> bhwi', 
            current_rot_matrix, 
            warped_vector.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # =====================================================================
        # 3. Energy Calculation (Architecture.md §4.3)
        # =====================================================================
        
        # [E_scalar] SDF 차이 (Softplus로 미분 매끄럽게)
        e_scalar = torch.abs(F.softplus(warped_sdf) - F.softplus(tgt_dict['sdf']))
        
        # [E_vector] Cosine Distance (방향 일치도)
        e_vector = 1.0 - F.cosine_similarity(
            warped_vector_corrected, tgt_dict['vector'], dim=1, eps=self.epsilon
        )
        e_vector = e_vector.unsqueeze(1) 
        
        # [E_bivector] Rotor Consistency (지역적 회전/줌 상태)
        e_bivector = torch.abs(warped_rotor - tgt_dict['rotor']) 

        # =====================================================================
        # 4. Gate-Guided Refinement (Architecture.md §4.4)
        # =====================================================================
        g_s, g_v, g_b = gates
        
        total_energy_map = (g_s * e_scalar) + (g_v * e_vector) + (g_b * e_bivector)

        # --- [Masking 적용] ---
        masked_energy = total_energy_map * binary_mask
        
        # 유효 픽셀 개수로 나누어 정확한 평균
        valid_pixel_count = binary_mask.sum() + self.epsilon
        final_loss = masked_energy.sum() / valid_pixel_count
        
        return final_loss

    def optimize(self, src_dict, tgt_dict, gates, priority_map=None):
        """
        [Phase 4 - Optimization Loop] MPC 최적화 루프
        
        Args:
            src_dict: Source 데이터
            tgt_dict: Target 데이터
            gates: (g_s, g_v, g_b) Gate 튜플
            priority_map: 우선순위 맵 (Optional)
            
        Returns:
            loss_history: 손실 기록 리스트
        """
        optimizer = optim.Adam([self.W], lr=self.learning_rate)
        # Cosine Annealing: 초반 과감하게, 후반 미세 조정
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.iterations, eta_min=SCHEDULER_ETA_MIN
        )
        loss_history = []
        
        print(f"[Phase 4] MPC Refinement Start (Iter: {self.iterations})")
        
        for i in range(self.iterations):
            optimizer.zero_grad()
            loss = self.compute_energy(src_dict, tgt_dict, gates)
            
            # (Optional) Priority Map weighting
            # 추후 구현 가능
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
            
            if i % 50 == 0:
                print(f"  Step [{i}/{self.iterations}] Loss: {loss.item():.6f}")
                
        return loss_history

    def visualize_result(self, src_img, tgt_img, src_sdf, tgt_sdf, priority_map):
        """
        [Phase 4 - Visualization] 결과 시각화
        """
        with torch.no_grad():
            grid = self.get_affine_grid(src_img.shape)
            warped_img = F.grid_sample(src_img, grid, align_corners=False)
            
            w_img = warped_img[0].permute(1, 2, 0).cpu().numpy()
            t_img = tgt_img[0].permute(1, 2, 0).cpu().numpy()
            p_map = priority_map[0, 0].cpu().numpy()
            
            warped_sdf = F.grid_sample(src_sdf, grid, align_corners=False)
            diff_map = torch.abs(warped_sdf - tgt_sdf)[0, 0].cpu().numpy()

        plt.figure(figsize=(18, 5))
        plt.subplot(1, 4, 1)
        plt.title("[Phase 4] Optimized Warped Source")
        plt.imshow(w_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("[Target] Reference Image")
        plt.imshow(t_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title("[Phase 4] Priority Map")
        plt.imshow(p_map, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("[Phase 4] Energy Residual")
        plt.imshow(diff_map, cmap='magma')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# ==============================================================================
# [Integration Logic] Phase 1 → 2 → 3 → 4 Pipeline
# ==============================================================================

def run_integrated_pipeline():
    """
    [Phase 4 통합 테스트] 전체 파이프라인 실행
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pipeline running on: {device}")

    # 1. 이미지 로드
    IMG_PATH = "./img/val2017/000000000885.jpg"
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image not found at {IMG_PATH}")
        return

    img_raw = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    
    # [Simulation] Source와 Target 생성
    # Source를 15도 회전 + 1.1배 확대시켜 '틀어진 상태' 생성
    rows, cols = img_rgb.shape[:2]
    M_gt = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1.1)
    img_warped = cv2.warpAffine(img_rgb, M_gt, (cols, rows))

    # --- Phase 1: Preprocessing ---
    preprocessor = MathGeometricPreprocessor()
    pyramid_src = preprocessor.process_pyramid(img_warped, levels=3) 
    pyramid_tgt = preprocessor.process_pyramid(img_rgb, levels=3)
    
    src_dict_np = pyramid_src[0]
    tgt_dict_np = pyramid_tgt[0]
    
    # --- Phase 2: Embedding ---
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    s_src, v_src, b_src = embedder.core(src_dict_np, device)
    s_tgt, v_tgt, b_tgt = embedder.core(tgt_dict_np, device)

    # --- Phase 3 Logic (Gate 추출) ---
    with torch.no_grad():
        inv_s = torch.mean(torch.abs(s_src), dim=1, keepdim=True)
        inv_v = torch.mean(torch.norm(v_src, dim=2), dim=1, keepdim=True) 
        inv_b = torch.mean(b_src[2], dim=1, keepdim=True)
        
        # Heuristic Gate
        g_s = torch.sigmoid(inv_s)
        g_v = torch.sigmoid(inv_v)
        g_b = torch.sigmoid(inv_b)
        gates = (g_s, g_v, g_b)
        
        # Priority Map
        feature_magnitude = inv_v 
        local_avg = F.avg_pool2d(inv_v, kernel_size=3, stride=1, padding=1)
        rotor_variance = torch.abs(inv_v - local_avg)

    # --- Phase 4: MPC Refinement ---
    refiner = GeometricMPCRefiner(device=device)
    
    # 1. Init
    mean_cos = b_src[0].mean().item()
    mean_sin = b_src[1].mean().item()
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_scale = b_src[2].mean().item()
    
    refiner.global_filtering_init(mean_rotor=mean_angle, mean_scale=mean_scale)
    
    # 2. Priority Map
    p_map = refiner.compute_priority_map(rotor_variance, feature_magnitude)
    gates = tuple(g.detach() for g in gates)
    if p_map is not None:
        p_map = p_map.detach()

    # 3. Data Packing
    src_img_tensor = torch.from_numpy(img_warped).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    tgt_img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    
    src_mpc_data = {
        'sdf': torch.tensor(src_dict_np['sdf'][np.newaxis, np.newaxis, ...]).float().to(device),
        'vector': v_src.mean(dim=1).detach(),
        'rotor': b_src[2].mean(dim=1, keepdim=True).detach()
    }
    
    tgt_mpc_data = {
        'sdf': torch.tensor(tgt_dict_np['sdf'][np.newaxis, np.newaxis, ...]).float().to(device),
        'vector': v_tgt.mean(dim=1).detach(),
        'rotor': b_tgt[2].mean(dim=1, keepdim=True).detach()
    }

    # 4. Optimize
    loss_curve = refiner.optimize(src_mpc_data, tgt_mpc_data, gates, priority_map=p_map)
    
    # 5. Visualize
    refiner.visualize_result(src_img_tensor, tgt_img_tensor, src_mpc_data['sdf'], tgt_mpc_data['sdf'], p_map)
    
    plt.figure()
    plt.plot(loss_curve)
    plt.title("MPC Optimization Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    run_integrated_pipeline()
