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
        [Helper] Numpy 데이터를 GPU Tensor로 변환하고 차원을 맞춥니다.
        HWC(OpenCV) -> CHW(PyTorch) -> Batch 추가 -> CUDA 전송
        """
        # 1. Scalar Group (HSI + SDF) 합치기
        # hsi: (H, W, 3), sdf: (H, W)
        hsi = data_dict['hsi']
        sdf = data_dict['sdf'][..., np.newaxis] # (H, W, 1)로 확장
        
        # (H, W, 4) 형태로 결합
        scalars_np = np.concatenate([hsi, sdf], axis=-1)
        
        # 2. Vector Group
        # gradient: (H, W, 4) -> 이미 dx, dy, fx, fy 들어있음
        vectors_np = data_dict['gradient']
        
        # 3. Global Stats
        v_shape_np = data_dict['v_shape']

        # 4. To Tensor & Permute & Batch
        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        s_tensor = torch.from_numpy(scalars_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        v_tensor = torch.from_numpy(vectors_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        g_tensor = torch.from_numpy(v_shape_np).unsqueeze(0).float().to(device) # (1, 6)

        return s_tensor, v_tensor, g_tensor

    def forward(self, phase1_output, device=None):
        """
        [Forward Pass]
        Numpy Dictionary -> Tensor 변환 -> Clifford Projection

        Returns:
            S (Tensor): (B, 64, H, W) - Cos 성분이 추가된 스칼라
            V (Tensor): (B, 64, 2, H, W) - 벡터 필드
            B (Tensor): (B, 64, H, W) - Sin 성분으로 구성된 바이벡터
        """
        if device is None:
            device = next(self.parameters()).device

        # 1. 데이터 준비 (GPU 전송 포함)
        s_in, v_in, g_in = self.to_tensor(phase1_output, device)

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
        batch, _, h, w = v_flat.shape
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
        
        # (3) Bivector ($B$) 할당
        # Sin Part(회전 성분)를 그대로 Bivector로 사용합니다.
        # Tanh를 쓰지 않는 이유: 아핀 변환의 '크기' 정보까지 살리기 위함 (사용자 의도 반영)
        b_emb = sin_part 
        
        # (4) Scalar ($S$) 업데이트
        # Cos Part(스케일/유사도 성분)를 기존 Scalar($S$)에 더해줍니다.
        # 이로써 S는 단순 텍스처 정보뿐만 아니라, '얼마나 확대/축소되었는지'의 기하학적 정보도 가집니다.
        s_emb = s_emb + cos_part

        # 5. 최종 멀티벡터 반환
        return s_emb, v_emb, b_emb
    

def visualize_embedding(S, V, B):
    """
    [Phase 2 Visualization]
    고차원(64채널) 클리포드 임베딩을 시각화합니다.
    채널 전체의 '에너지(Energy)'를 평균내어 활성화 맵(Activation Map)으로 봅니다.
    
    Args:
        S: (B, 64, H, W)
        V: (B, 64, 2, H, W)
        B: (B, 64, H, W)
    """
    # 1. Tensor -> Numpy & 배치 차원 제거
    # (64, H, W) 형태로 가져옴
    s_map = S[0].detach().cpu().numpy()
    v_map = V[0].detach().cpu().numpy() # (64, 2, H, W)
    b_map = B[0].detach().cpu().numpy()
    
    # 2. 정보 압축 (Aggregation)
    
    # [Scalar Map] 64개 채널의 평균 활성도
    # (64, H, W) -> (H, W)
    s_vis = np.mean(s_map, axis=0)
    
    # [Vector Map] 64개 벡터들의 '크기(Magnitude)'의 평균
    # 먼저 각 채널별 벡터 크기 계산: sqrt(x^2 + y^2)
    v_mag = np.sqrt(v_map[:, 0, :, :]**2 + v_map[:, 1, :, :]**2)
    # 그 다음 채널 평균
    v_vis = np.mean(v_mag, axis=0)
    
    # [Bivector Map] 64개 채널의 '회전 강도(절대값)'의 평균
    # 음수 회전(-), 양수 회전(+) 모두 '회전이 있다'는 뜻이므로 절대값 취함
    b_vis = np.mean(np.abs(b_map), axis=0)
    
    # 3. 시각화 (Plotting)
    plt.figure(figsize=(18, 5))
    plt.suptitle(f"Phase 2: Clifford Embedding Analysis (Hidden Dim: {s_map.shape[0]})", fontsize=16, fontweight='bold')
    
    # Scalar Embedding
    plt.subplot(1, 3, 1)
    plt.imshow(s_vis, cmap='inferno')
    plt.title("1. Scalar Activation ($S_{emb}$)\n(Mean intensity of features)")
    plt.colorbar()
    plt.axis('off')
    
    # Vector Embedding
    plt.subplot(1, 3, 2)
    plt.imshow(v_vis, cmap='viridis')
    plt.title("2. Vector Magnitude ($V_{emb}$)\n(Mean strength of directional forces)")
    plt.colorbar()
    plt.axis('off')
    
    # Bivector Embedding (New!)
    plt.subplot(1, 3, 3)
    plt.imshow(b_vis, cmap='magma')
    plt.title("3. Bivector Intensity ($B_{emb}$)\n(Detected Rotations/Curls)")
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 실행 및 검증 코드
# =============================================================================
if __name__ == "__main__":
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2 Initialized on Device: {device}")

    # 1. Phase 1 실행 (전처리)
    # 이미지 경로 설정 (사용자 환경에 맞게 수정)
    IMG_PATH = "./img/val2017/000000569972.jpg" 
    
    # 이미지 로드
    img = torch.from_numpy(cv2.imread(IMG_PATH)).numpy() # Dummy load ensure cv2
    if img is None:
        print("이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        exit()
    
    img_rgb = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)

    # Phase 1 Preprocessor 호출
    preprocessor = MathGeometricPreprocessor()
    phase1_data = preprocessor.process_from_array(img_rgb)
    
    print("\nPhase 1 Output Keys:", phase1_data.keys())
    print(f"   - HSI Shape: {phase1_data['hsi'].shape}")       # (H, W, 3)
    print(f"   - Grad Shape: {phase1_data['gradient'].shape}") # (H, W, 4)
    print(f"   - SDF Shape: {phase1_data['sdf'].shape}")       # (H, W)

    # 2. Phase 2 실행 (임베딩)
    # 모델 초기화 및 GPU 이동
    clifford_embedder = CliffordComponentEmbedding(hidden_dim=64).to(device)
    
    # Forward Pass (Numpy dict를 넣으면 알아서 Tensor 변환 후 처리)
    with torch.no_grad(): # 추론 모드 (메모리 절약)
        S, V, B = clifford_embedder(phase1_data, device)

    # 3. 결과 확인
    print("\nPhase 2 Embedding Complete (Clifford Multi-vector Created)")
    print("-" * 50)
    print(f"Scalar Part ($S$):   {S.shape}")     
    # Expected: (1, 64, H, W) -> 텍스처와 에너지 정보가 융합됨
    
    print(f"Vector Part ($V$):   {V.shape}")     
    # Expected: (1, 64, 2, H, W) -> [Gradient + Flow]가 섞여서 64개의 기하학적 벡터 생성
    # 마지막 차원 2는 (x, y) 성분을 의미함
    
    print(f"Bivector Part ($B$): {B.shape}")     
    # Expected: (1, 64, H, W) -> 벡터들의 상호작용으로 생성된 회전 정보
    print("-" * 50)
    
    visualize_embedding(S, V, B)
    print("이 (S, V, B) 튜플이 이제 Clifford Convolution Layer의 입력으로 들어갑니다.")