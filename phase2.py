import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Phase 1의 전처리 클래스 가져오기 (파일 이름이 phase1.py라고 가정)
from phase1 import MathGeometricPreprocessor

# [Custom Layer] Learnable Scaling
# Softplus를 거치면 값이 양수로만 나오는데, 값의 범위를 학습 가능하게 조절해주는 층입니다.
class LearnableScaling(nn.Module):
    def __init__(self, channels, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1)) # 편향도 추가하여 유연성 확보

    def forward(self, x):
        return x * self.scale + self.bias
    

class CliffordComponentEmbedding(nn.Module):
    def __init__(self, hidden_dim=64):
        """
        [Phase 2: Clifford Embedding Layer]
        Phase 1에서 넘어온 '물리적 원석(Raw Data)'을 딥러닝이 이해하는
        '고차원 클리포드 멀티벡터(Clifford Multi-vector)'로 변환합니다.

        Args:
            hidden_dim (int): 임베딩 차원 (기본 64). 
                              실제로는 (S:64 + V:64*2 + B:64) = 256 채널 분량의 정보를 가집니다.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.s_mixer = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1) # S값 추출 시 단순히 더하여 cos 정보가 텍스쳐 정보를 엉뚱하게 덮어씌우는 것을 방지
        
        # =====================================================================
        # 1. Scalar Embedding ($S_{in} \to S_{out}$)
        # =====================================================================
        # 입력 채널 (총 4개):
        #   - Texture (1): 재질/밝기
        #   - Structure Energy (1): 결의 선명도
        #   - Edge Magnitude (1): 엣지 세기
        #   - SDF (1): 뼈대 잠재장
        # 출력: 64차원 스칼라 특징
        # "정보의 풍부함"이 중요하므로, 0 이하를 버리지 않고 부드럽게 살림.
        self.proj_s = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            # beta=1 (기본값), threshold=20 (큰 값은 선형으로 처리하여 안정성 확보)
            nn.Softplus(beta=1, threshold=20),
            # Softplus로 인해 양수로 편향된 분포를, 모델이 원하는 범위로 스케일링
            LearnableScaling(hidden_dim, init_scale=1.0)
        )

        # =====================================================================
        # 2. Vector Embedding ($V_{in} \to V_{out}$)
        # =====================================================================
        # 입력 채널 (총 4개):
        #   - Gradient Vector (dx, dy): 2ch
        #   - Texture Flow Vector (fx, fy): 2ch
        # 출력: 128차원 (64개 채널 * 2개 성분 x,y)
        # -> Conv2d가 두 벡터(Gradient, Flow)를 섞어서 최적의 기하학적 방향을 학습합니다.
        self.proj_v = nn.Conv2d(4, hidden_dim * 2, kernel_size=1, bias=False) 
        # * Vector는 방향이므로 Bias를 쓰지 않는 것이 기하학적으로 더 자연스럽습니다.

        # =====================================================================
        # 3. Rotor Generation
        # =====================================================================
        # 기존의 Bivector 직접 생성(Tanh) 대신, 벡터 관계를 Sin/Cos 쌍으로 분해합니다.
        # - 입력: 4개 벡터 성분 (dx, dy, fx, fy)
        # - 출력: 2 * hidden_dim (채널당 Cos, Sin 쌍을 가짐)
        # 
        # [Why Bias=True?]
        # Rotor는 변환(Transformation)이므로 편향을 통해 기본 상태(Identity)나
        # 특정 초기 회전값을 학습할 수 있어야 합니다.
        self.proj_rotor = nn.Conv2d(4, hidden_dim * 2, kernel_size=1, bias=True)

        # =====================================================================
        # 4. Global Context Injection (Option)
        # =====================================================================
        # 이미지 전체의 통계(v_shape)를 스칼라에 주입하여 전역 정보를 보정합니다.
        # 입력 6개 -> 출력 64개
        self.global_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid() # 0~1 사이의 게이트(Gate) 역할
        )

    def to_tensor(self, data_dict, device):
        """
        [Helper - 수정됨] Numpy 데이터를 GPU Tensor로 변환하고 차원을 맞춥니다.
        단일 이미지(HWC)와 배치 이미지(BHWC)를 모두 처리할 수 있도록 개선되었습니다.
        """
        # 1. Scalar Group (HSI + SDF) 합치기
        hsi = data_dict['hsi'] # (H, W, 3) 또는 (B, H, W, 3)
        sdf = data_dict['sdf'] # (H, W) 또는 (B, H, W)
        
        # 배치 차원 존재 여부 확인 (ndim이 4면 배치 포함)
        has_batch = (hsi.ndim == 4)
        
        # SDF 차원 확장
        sdf = sdf[..., np.newaxis] # (..., 1) -> (B, H, W, 1) or (H, W, 1)
        scalars_np = np.concatenate([hsi, sdf], axis=-1) # (..., 4)
        vectors_np = data_dict['gradient']
        v_shape_np = data_dict['v_shape']

        if has_batch:
            # Case A: Training (Batch Input)
            # Input: (B, H, W, C) -> Output: (B, C, H, W)
            # permute(0, 3, 1, 2): Batch(0), Channel(3), Height(1), Width(2)
            s_tensor = torch.from_numpy(scalars_np).permute(0, 3, 1, 2).float().to(device)
            v_tensor = torch.from_numpy(vectors_np).permute(0, 3, 1, 2).float().to(device)
            g_tensor = torch.from_numpy(v_shape_np).float().to(device) # (B, 6)
            
        else:
            # Case B: Inference (Single Image Input)
            # Input: (H, W, C) -> Output: (1, C, H, W)
            # permute(2, 0, 1): Channel(2), Height(0), Width(1)
            # unsqueeze(0): Add Batch dim
            s_tensor = torch.from_numpy(scalars_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            v_tensor = torch.from_numpy(vectors_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            g_tensor = torch.from_numpy(v_shape_np).unsqueeze(0).float().to(device) # (1, 6)

        return s_tensor, v_tensor, g_tensor

    def forward(self, phase1_item, device=None):
        """
        [Forward Pass]
        Numpy Dictionary -> Tensor 변환 -> Clifford Projection

        Returns:
            S (Tensor): (B, 64, H, W) - Cos 성분이 추가된 스칼라
            V (Tensor): (B, 64, 2, H, W) - 벡터 필드
            B (Tuple): ((B, 64, H, W), (B, 64, H, W), (B, 64, H, W)) 
                       -> (Unit_Cos, Unit_Sin, Magnitude) 형태의 Rotor 패키지
        """
        if device is None:
            device = next(self.parameters()).device

        # 1. 데이터 준비 (GPU 전송 포함)
        s_in, v_in, g_in = self.to_tensor(phase1_item, device)
        batch, _, h, w = s_in.shape

        # 2. Scalar Embedding (S)
        # 물리적 스칼라들(재질, 에너지, SDF)을 고차원 특징으로 변환
        s_emb = self.proj_s(s_in)
        
        # [Global Context 주입]
        # 이미지 전체 분위기(g_in)를 보고 스칼라 특징을 강조/억제(Attention)
        global_gate = self.global_mlp(g_in).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)
        s_emb = s_emb * global_gate # 채널별 중요도 조절

        # 3. Vector Embedding (V)
        # (B, 4, H, W) -> (B, 128, H, W)
        v_flat = self.proj_v(v_in)
        
        # [핵심] 차원 재구성 (Reshape)
        # 128개 채널을 (64개 채널 x 2개 성분)으로 분리합니다.
        # 이렇게 해야 이후 연산에서 '벡터로서의 기하학적 성질'이 유지됩니다.
        v_emb = v_flat.view(batch, self.hidden_dim, 2, h, w)

        # =====================================================================
        # 4. Rotor Embedding & Decomposition (핵심 변경!)
        # =====================================================================
        # (1) Rotor 추론: 벡터 입력으로부터 변환 관계(Rotation+Scale)를 뽑아냄
        # Shape: (B, 128, H, W)
        rotor_raw = self.proj_rotor(v_in)
        
        # (2) Pair 분리: (B, 64, 2, H, W) 형태로 Reshape하여 Cos/Sin 쌍으로 나눔
        # - dim 2의 index 0: Cosine Part (Scale/Divergence/Dot Product)
        # - dim 2의 index 1: Sine Part (Rotation/Curl/Wedge Product)
        rotor_pair = rotor_raw.view(batch, self.hidden_dim, 2, h, w)
        
        cos_part = rotor_pair[:, :, 0, :, :] 
        sin_part = rotor_pair[:, :, 1, :, :]
        
        # (3) 정규화 및 분리 (Normalization & Separation)
        # Rotor Magnitude ($|R|$) 계산: Scale 성분 추출
        rotor_mag = torch.sqrt(cos_part**2 + sin_part**2 + 1e-6) # 0 나누기 방지
        
        # Unit Rotor ($R/|R|$) 계산: 순수 회전 성분 추출
        unit_cos = cos_part / rotor_mag
        unit_sin = sin_part / rotor_mag
        
        # [Output B] Phase 3를 위해 분리된 성분들을 튜플로 패키징
        b_emb = (unit_cos, unit_sin, rotor_mag)
        
        # (4) Scalar ($S$) 업데이트
        # Cos Part(유사도 성분)를 기존 Scalar($S$)에 더해줍니다.
        # 주의: 스칼라 업데이트에는 정규화 전의 'Raw Cosine(Scale 포함)'을 사용하여
        # '얼마나 닮았고 얼마나 큰지'에 대한 정보를 S에 반영합니다.
        s_combined = torch.cat([s_emb, cos_part], dim=1) # (B, 128, H, W)
        s_emb = self.s_mixer(s_combined)                 # (B, 64, H, W)

        # 5. 최종 멀티벡터 반환 (B는 이제 Tuple입니다)
        return s_emb, v_emb, b_emb
    
class CliffordPyramidEmbedder(nn.Module):
    """
    [Phase 2 Main Wrapper]
    Phase 1에서 온 '피라미드 리스트'를 받아서,
    Core Engine(Weights Shared)을 사용해 모든 레벨을 처리합니다.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 하나의 Core 모듈을 모든 스케일에서 재사용 (Weight Sharing)
        self.core = CliffordComponentEmbedding(hidden_dim)
        
    def forward(self, pyramid_data_list, device=None):
        """
        Args:
            pyramid_data_list (list): [Level0_Dict, Level1_Dict, ...]
        Returns:
            list of tuples: [(S0, V0, (uCos0, uSin0, Mag0)), ...]
        """
        pyramid_outputs = []
        
        # 리스트 순회 (Global to Fine or Fine to Global)
        for level_data in pyramid_data_list:
            # 동일한 물리 법칙(Core) 적용
            s, v, b = self.core(level_data, device)
            pyramid_outputs.append((s, v, b))
            
        return pyramid_outputs


# --- 시각화 함수 (피라미드 전체 시각화) ---
def visualize_clifford_pyramid(clifford_pyramid):
    """
    [Phase 2 Visualization]
    고차원(64채널) 클리포드 임베딩을 시각화합니다.
    채널 전체의 '에너지(Energy)'를 평균내어 활성화 맵(Activation Map)으로 봅니다.
    
    Args:
        S: (B, 64, H, W)
        V: (B, 64, 2, H, W)
        B: Tuple (Unit_Cos, Unit_Sin, Magnitude) - 각 (B, 64, H, W)
    """
    levels = len(clifford_pyramid)
    plt.figure(figsize=(18, 4 * levels))
    plt.suptitle("Phase 2: Multi-Scale Clifford Embedding (S, V, B)", fontsize=20, fontweight='bold')
    
    for i, (S, V, B) in enumerate(clifford_pyramid):
        # 1. Tensor -> Numpy & 배치 차원 제거
        # (64, H, W) 형태로 가져옴

        # 2. 정보 압축 (Aggregation)
    
        # [Scalar Map] 64개 채널의 평균 활성도
        # (64, H, W) -> (H, W)
        s_map = S[0].detach().cpu().numpy().mean(axis=0)          

        # [Vector Map] 64개 벡터들의 '크기(Magnitude)'의 평균
        # 먼저 각 채널별 벡터 크기 계산: sqrt(x^2 + y^2)
        # 그 다음 채널 평균
        v_vec = V[0].detach().cpu().numpy()
        v_mag = np.mean(np.sqrt(v_vec[:,0]**2 + v_vec[:,1]**2), axis=0)

        # [Bivector Map] B가 Tuple로 변경되었으므로 Unpacking 필요
        unit_cos, unit_sin, rotor_mag = B
        
        # 시각화에는 'Rotor Magnitude (Scale/Intensity)'를 사용
        # 혹은 unit_sin(Rotation Amount)을 사용할 수도 있음. 여기선 Magnitude 사용.
        b_map = np.mean(rotor_mag[0].detach().cpu().numpy(), axis=0)
        
        h, w = s_map.shape
        
        # Plotting
        base_idx = i * 3
        
        # Scalar
        plt.subplot(levels, 3, base_idx + 1)
        plt.imshow(s_map, cmap='inferno')
        plt.ylabel(f"Level {i}\n({h}x{w})", fontsize=14, fontweight='bold')
        if i==0: plt.title("Scalar Energy ($S$)\n(Texture + Similarity)")
        plt.xticks([]), plt.yticks([])
        
        # Vector
        plt.subplot(levels, 3, base_idx + 2)
        plt.imshow(v_mag, cmap='viridis')
        if i==0: plt.title("Vector Magnitude ($V$)\n(Directional Force)")
        plt.xticks([]), plt.yticks([])
        
        # Bivector
        plt.subplot(levels, 3, base_idx + 3)
        plt.imshow(b_map, cmap='magma')
        if i==0: plt.title("Bivector Intensity ($|R|$)\n(Rotation Scale)")
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

# =============================================================================
# 실행 및 검증 코드
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2 Running on: {device}")

    # 1. Load & Phase 1 (Pyramid Extraction)
    IMG_PATH = "./img/val2017/000000569972.jpg" # 이미지 경로
    img = cv2.imread(IMG_PATH)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Phase 1: Create 4-level pyramid (Level 0 ~ 3)
        preprocessor = MathGeometricPreprocessor()
        pyramid_raw = preprocessor.process_pyramid(img_rgb, levels=6)
        print(f"Phase 1 Complete. Scales generated: {len(pyramid_raw)}")

        # 2. Phase 2 (Pyramid Embedding)
        embedder = CliffordPyramidEmbedder(hidden_dim=64).to(device)
        
        with torch.no_grad():
            # 리스트 전체를 넘기면 내부에서 순회하며 처리
            clifford_pyramid = embedder(pyramid_raw, device)
            
        print(f"Phase 2 Complete. Pyramid Embeddings Created: {len(clifford_pyramid)}")
        
        # 3. Structure Check & Visualization
        for i, (S, V, B) in enumerate(clifford_pyramid):
            # [수정] B는 이제 Tuple이므로 B.shape 대신 내부 요소의 shape을 확인합니다.
            # B = (Unit_Cos, Unit_Sin, Magnitude)
            # 가장 중요한 Magnitude(인덱스 2)의 shape을 출력합니다.
            rotor_mag_shape = B[2].shape 
            print(f"   [Level {i}] Res: {S.shape[-2:]} | S:{S.shape} V:{V.shape} B(Mag):{rotor_mag_shape}")
            
        visualize_clifford_pyramid(clifford_pyramid)

    else:
        print("Image Not Found.")