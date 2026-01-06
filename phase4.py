import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 이전 단계 모듈 가져오기 (파일들이 같은 디렉토리에 있다고 가정)
from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder
from phase3 import Phase3Transformer, GeometricDescriptorGuidance

class GeometricMPCRefiner(nn.Module):
    """
    [Phase 4] 기하학적 에너지 기반 MPC 정제 모듈
    Phase 3의 매칭 지도와 Gate 정보를 바탕으로 물리적 에너지를 최소화하여 
    0.1 픽셀 단위의 정밀 정렬을 수행합니다.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # --- [Hyperparameters] ---
        self.learning_rate = 0.005 # 정밀 조정을 위해 LR을 조금 낮춤
        self.iterations = 300      # MPC 루프 반복 횟수
        self.epsilon = 1e-6        # 수치 안정성을 위한 작은 값
        
        # 변환 행렬 W (Affine Transform: 2x3) - 최적화 대상
        self.W = nn.Parameter(torch.eye(2, 3).unsqueeze(0).to(device)) 

    def get_affine_grid(self, shape):
        """
        [Phase 4 Helper] 현재 W를 기반으로 Sampling Grid 생성 (CUDA 가속)
        """
        B, C, H, W = shape
        grid = F.affine_grid(self.W, [B, C, H, W], align_corners=False)
        return grid

    def global_filtering_init(self, mean_rotor, mean_scale):
        """
        [Phase 4 - Step 1] 전역 필터링 및 초기화
        Phase 2의 평균 Rotor와 Scale을 사용하여 초기 변환 행렬 W0를 설정
        """
        # Rotation Matrix 구성 (2x2)
        # mean_rotor는 Phase 2 Bivector의 평균 Angle(Radian)
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
        [Phase 4 - Step 2] 지역 탐색 (Priority Search) - [Robust Logic]
        """
        # [수정 1] 분산 하한선 제한 (Clamping)
        safe_variance = torch.clamp(rotor_variance, min=1e-3)
        stability = 1.0 / safe_variance
        
        raw_priority = stability * feature_magnitude
        
        # [수정 2] Robust Min-Max Normalization
        B = raw_priority.shape[0]
        priority_map = torch.zeros_like(raw_priority)
        
        for b in range(B):
            flat = raw_priority[b].view(-1)
            v_min = flat.min()
            v_max = torch.quantile(flat, 0.99) # 상위 1% 값까지만 인정
            
            clipped = torch.clamp(raw_priority[b], min=v_min, max=v_max)
            priority_map[b] = (clipped - v_min) / (v_max - v_min + self.epsilon)
            
        return priority_map

    def compute_energy(self, src_dict, tgt_dict, gates):
        """
        [Phase 4 - Step 3 & 4] 에너지 평면 생성 및 Gate 가중 최적화
        """
        grid = self.get_affine_grid(src_dict['sdf'].shape)

        # --- [Valid Mask 생성] ---
        # Source 이미지 크기와 동일한 1.0짜리 백색 도화지를 만듭니다.
        # 이 도화지도 이미지랑 똑같이 회전(Warp)시킵니다.
        # 회전 후 검은색(0)이 된 부분은 "가짜 영역"이므로 Loss에서 제외합니다.
        ones_mask = torch.ones_like(src_dict['sdf'])
        warped_mask = F.grid_sample(ones_mask, grid, align_corners=False)
        # 마스크가 0.9 이상인 곳만 확실한 유효 영역으로 간주 (Bilinear 보간 경계 제거)
        binary_mask = (warped_mask > 0.9).float()
        
        # 1. Warping
        warped_sdf = F.grid_sample(src_dict['sdf'], grid, align_corners=False)
        warped_vector = F.grid_sample(src_dict['vector'], grid, align_corners=False)
        warped_rotor = F.grid_sample(src_dict['rotor'], grid, align_corners=False)
        
        # 2. Vector Field Rotation Correction (방향 보존성)
        current_rot_matrix = self.W[0, :2, :2] 
        warped_vector_corrected = torch.einsum('ij, bhwj -> bhwi', current_rot_matrix, warped_vector.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 3. Energy Calculation
        # E_scalar: SDF 차이
        e_scalar = torch.abs(F.softplus(warped_sdf) - F.softplus(tgt_dict['sdf']))
        
        # E_vector: Cosine Distance
        e_vector = 1.0 - F.cosine_similarity(warped_vector_corrected, tgt_dict['vector'], dim=1, eps=self.epsilon)
        e_vector = e_vector.unsqueeze(1) 
        
        # E_bivector: Rotor Consistency (Magnitude)
        e_bivector = torch.abs(warped_rotor - tgt_dict['rotor']) 

        # 4. Gate-Guided Refinement
        g_s, g_v, g_b = gates
        
        # [수식 보완] 정규화된 에너지 합 (Mean)
        total_energy_map = (g_s * e_scalar) + (g_v * e_vector) + (g_b * e_bivector)

        # --- [Masking 적용] ---
        # 빈 공간(Padding)의 에러를 0으로 만듦
        masked_energy = total_energy_map * binary_mask
        
        # 단순 mean()을 하면 0인 영역까지 포함해서 평균을 내므로 Loss가 인위적으로 낮아짐.
        # 따라서 "마스크가 1인 픽셀 개수"로 나누어야 정확한 평균 에러가 나옴.
        valid_pixel_count = binary_mask.sum() + 1e-6
        final_loss = masked_energy.sum() / valid_pixel_count
        
        return final_loss

    def optimize(self, src_dict, tgt_dict, gates, priority_map=None):
        """
        [Phase 4 - Optimization Loop]
        """
        optimizer = optim.Adam([self.W], lr=self.learning_rate)
        # [추가] Learning Rate Scheduler
        # 처음엔 과감하게(0.05) 돌리다가, 후반부엔 미세하게(0.005) 조정하여 튀는 현상 방지
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.001)
        loss_history = []
        
        print(f"[Phase 4] MPC Refinement Start (Iter: {self.iterations})")
        
        for i in range(self.iterations):
            optimizer.zero_grad()
            loss = self.compute_energy(src_dict, tgt_dict, gates)
            
            # (Optional) Priority Map weighting could be added here
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
            
            if i % 20 == 0:
                print(f"  Step [{i}/{self.iterations}] Loss: {loss.item():.6f}")
                
        return loss_history

    def visualize_result(self, src_img, tgt_img, src_sdf, tgt_sdf, priority_map):
        """
        [Phase 4 - Visualization] Result Check
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
        plt.subplot(1, 4, 1); plt.title("[Phase 4] Optimized Warped Source"); plt.imshow(w_img); plt.axis('off')
        plt.subplot(1, 4, 2); plt.title("[Target] Reference Image"); plt.imshow(t_img); plt.axis('off')
        plt.subplot(1, 4, 3); plt.title("[Phase 4] Priority Map (Robust)"); plt.imshow(p_map, cmap='jet'); plt.colorbar(); plt.axis('off')
        plt.subplot(1, 4, 4); plt.title("[Phase 4] Final Energy Residual"); plt.imshow(diff_map, cmap='magma'); plt.colorbar(); plt.axis('off')  # 붉은 영역일수록 매칭이 실패한 영역, 검을수록 매칭이 잘 되는 영역
        plt.tight_layout(); plt.show()

# ==============================================================================
# [Integration Logic] Phase 1 -> 2 -> 3 -> 4 Pipeline
# ==============================================================================
def run_integrated_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pipeline running on: {device}")

    # 1. 이미지 로드 (Phase 3에서 사용된 경로)
    IMG_PATH = "./img/val2017/000000000885.jpg"
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image not found at {IMG_PATH}")
        return

    img_raw = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    
    # [Simulation Setup] Source와 Target 생성
    # Phase 4가 할 일을 만들기 위해, Source를 Target 대비 약 15도 회전 + 1.1배 확대시켜 '틀어진 상태'를 만듭니다.
    # 목표: Phase 4가 이 틀어짐($W$)을 찾아내어 다시 맞추는지 확인
    rows, cols = img_rgb.shape[:2]
    M_gt = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1.1) # 15도 회전, 1.1배 줌
    img_warped = cv2.warpAffine(img_rgb, M_gt, (cols, rows))

    # Input: img_warped (Source), img_rgb (Target/Reference)
    
    # --- Phase 1: Preprocessing ---
    preprocessor = MathGeometricPreprocessor()
    # Level 0 (원본 해상도)만 사용하거나, 피라미드 최상단을 사용. 여기선 정밀 정렬이므로 Level 0 사용
    # process_pyramid returns list of dicts
    pyramid_src = preprocessor.process_pyramid(img_warped, levels=3) 
    pyramid_tgt = preprocessor.process_pyramid(img_rgb, levels=3)
    
    # Tensor 변환 (Level 0)
    src_dict_np = pyramid_src[0] # Finest Level
    tgt_dict_np = pyramid_tgt[0]
    
    # --- Phase 2: Embedding ---
    embedder = CliffordPyramidEmbedder(hidden_dim=64).to(device)
    # embedder.core expects dict, returns s, v, b
    # Embedding만 필요하므로 embedder.core를 직접 호출하여 Tensor 추출
    s_src, v_src, b_src = embedder.core(src_dict_np, device)
    s_tgt, v_tgt, b_tgt = embedder.core(tgt_dict_np, device)

    # --- Phase 3 Logic Extraction (Guidance & Priority) ---
    # Phase 3 전체 Transformer를 돌릴 수도 있지만, Phase 4에 필요한 건 
    # 1. Gate (from GeometricDescriptorGuidance)
    # 2. Priority (from Features)
    # 입니다. Phase 3의 Guidance 모듈을 빌려옵니다.
    
    guidance_net = GeometricDescriptorGuidance(dim=64).to(device) # Phase 2 hidden_dim=64
    
    # Gate 추출 (Source 기준)
    # Phase 3 Guidance는 입력(64ch)을 받아 3개로 쪼개고 Gate 계산
    # Embedder 출력 S(64)를 사용 (S가 가장 풍부한 정보를 담고 있음)
    # 실제 Phase 3 구조에 맞게 Concat하여 입력 생성
    # Phase 2 Output: S(64), V(64x2), B(Tuple) -> 차원 맞추기 까다로움.
    # [Shortcut] Guidance Net을 새로 정의해서 S(Scalar)맵에서 Gate를 추출하도록 함.
    # 여기서는 Phase 3 코드의 로직을 본따 직접 계산합니다.
    with torch.no_grad():
        # S, V(Magnitude), B(Magnitude)를 사용하여 Gate를 결정
        inv_s = torch.mean(torch.abs(s_src), dim=1, keepdim=True)
        # v_src는 (B, 64, 2, H, W) -> Magnitude -> (B, 64, H, W) -> Mean -> (B, 1, H, W)
        inv_v = torch.mean(torch.norm(v_src, dim=2), dim=1, keepdim=True) 
        inv_b = torch.mean(b_src[2], dim=1, keepdim=True) # Rotor Magnitude
        
        # 간단한 Heuristic Gate (학습된 Net 대신 물리적 직관 사용)
        # S가 강하면 g_s, V가 강하면 g_v ...
        g_s = torch.sigmoid(inv_s)
        g_v = torch.sigmoid(inv_v)
        g_b = torch.sigmoid(inv_b)
        gates = (g_s, g_v, g_b)
        
        # Priority Map 생성을 위한 Feature Variance & Magnitude
        # 엣지(V)가 강한 곳을 우선순위로 둠
        feature_magnitude = inv_v 
        # 분산은 지역적 평균과의 차이로 근사
        local_avg = F.avg_pool2d(inv_v, kernel_size=3, stride=1, padding=1)
        rotor_variance = torch.abs(inv_v - local_avg) # 분산 대용

    # --- Phase 4: MPC Refinement ---
    refiner = GeometricMPCRefiner(device=device)
    
    # 1. Init: Phase 2의 평균 Rotor/Scale 값 사용
    # 여기서는 우리가 정답(15도, 1.1배)을 알고 있지만, 모델은 데이터에서 추정해야 함.
    # b_src(Rotor)의 평균을 사용
    # b_src = (cos, sin, mag)
    mean_cos = b_src[0].mean().item()
    mean_sin = b_src[1].mean().item()
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_scale = b_src[2].mean().item() # Rotor Magnitude가 Scale 정보 포함
    
    # 15도 ~ 0.26 rad, 1.1 scale
    # 실제로는 Source가 Warped 되었으므로, Target과 비교했을 때 역변환이 필요하거나
    # Source 자체의 절대적 회전량을 의미할 수 있음. 
    # 여기서는 Init이 "대략적 값"을 준다고 가정.
    refiner.global_filtering_init(mean_rotor=mean_angle, mean_scale=mean_scale)
    
    # 2. Priority Map
    p_map = refiner.compute_priority_map(rotor_variance, feature_magnitude)
    gates = tuple(g.detach() for g in gates)
    if p_map is not None:
        p_map = p_map.detach()

    # 3. Data Dict Packing (MPC가 이해하는 포맷)
    # Input Image도 Tensor로 변환
    src_img_tensor = torch.from_numpy(img_warped).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    tgt_img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
    
    # Source Data Packing
    # v_src와 b_src는 embedder 출력값이므로 grad_fn이 붙어있음 -> .detach() 필수
    src_mpc_data = {
        'sdf': torch.tensor(src_dict_np['sdf'][np.newaxis, np.newaxis, ...]).float().to(device),
        'vector': v_src.mean(dim=1).detach(),       # <--- .detach() 추가
        'rotor': b_src[2].mean(dim=1, keepdim=True).detach() # <--- .detach() 추가
    }
    
    tgt_mpc_data = {
        'sdf': torch.tensor(tgt_dict_np['sdf'][np.newaxis, np.newaxis, ...]).float().to(device),
        'vector': v_tgt.mean(dim=1).detach(),
        'rotor': b_tgt[2].mean(dim=1, keepdim=True).detach()
    }

    # 4. Run Optimization
    loss_curve = refiner.optimize(src_mpc_data, tgt_mpc_data, gates, priority_map=p_map)
    
    # 5. Visualize
    refiner.visualize_result(src_img_tensor, tgt_img_tensor, src_mpc_data['sdf'], tgt_mpc_data['sdf'], p_map)
    
    plt.plot(loss_curve)
    plt.title("MPC Optimization Loss")
    plt.show()

if __name__ == "__main__":
    run_integrated_pipeline()