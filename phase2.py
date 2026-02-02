"""
================================================================================
Phase 2: Clifford Embedding & Pyramid 생성
================================================================================
[Architecture.md §2 참조]

Phase 1에서 준비된 해상도별 물리량(S, V, B 후보)을 입력받아, 고차원 공간에서의
멀티벡터 임베딩을 완성합니다.

출력:
- S (Scalar): 64차원 스칼라 특징 (Texture + Similarity)
- V (Vector): 64채널 x 2성분 벡터 필드
- B (Rotor): (Unit_Cos, Unit_Sin, Magnitude) 튜플

핵심 설계:
- Scalar: Softplus + Learnable Scaling (정보 보존)
- Vector: Linear Projection (No Bias) - 방향의 순수성 유지
- Bivector: Sin/Cos Pair Output - 회전의 정석
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from phase1 import MathGeometricPreprocessor

# =============================================================================
# [Hyperparameters] Phase 2
# =============================================================================
HIDDEN_DIM = 48                  # [Hyperparameter] 임베딩 차원
SOFTPLUS_BETA = 1                # [Hyperparameter] Softplus 기울기
SOFTPLUS_THRESHOLD = 20          # [Hyperparameter] Softplus 선형 전환 임계값
INIT_SCALE = 1.0                 # [Hyperparameter] Learnable Scaling 초기값
GLOBAL_MLP_HIDDEN = None         # None이면 HIDDEN_DIM과 동일하게 설정


class LearnableScaling(nn.Module):
    """
    [Phase 2 Custom Layer] Learnable Scaling
    
    Architecture.md §2.1 - Scalar Embedding의 일부
    
    Softplus를 거치면 값이 양수로만 나오는데, 
    값의 범위를 학습 가능하게 조절해주는 층입니다.
    """
    def __init__(self, channels, init_scale=INIT_SCALE):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias
    

class CliffordComponentEmbedding(nn.Module):
    """
    [Phase 2 Core Engine] Clifford Embedding Layer
    
    Architecture.md §2 전체 구현
    
    Phase 1에서 넘어온 '물리적 원석(Raw Data)'을 딥러닝이 이해하는
    '고차원 클리포드 멀티벡터(Clifford Multi-vector)'로 변환합니다.
    
    Args:
        hidden_dim: 임베딩 차원 (기본 48)
                   실제로는 (S:48 + V:48*2 + B:48) = 192 채널 분량의 정보
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # S값 추출 시 cos 정보가 텍스쳐 정보를 덮어씌우는 것을 방지
        self.s_mixer = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        
        # =====================================================================
        # [§2.1] Scalar Embedding (S_in → S_out)
        # =====================================================================
        # 입력 채널 (총 4개):
        #   - Texture (1): 재질/밝기
        #   - Structure Energy (1): 결의 선명도
        #   - Edge Magnitude (1): 엣지 세기
        #   - SDF (1): 뼈대 잠재장
        # 출력: hidden_dim차원 스칼라 특징
        # 
        # [Architecture.md §2.1]
        # Softplus: 정보 보존력 - 음수/미세 신호를 0으로 깎지 않음
        # Learnable Scaling: Softplus로 양수 편향된 분포를 원하는 범위로 조절
        self.proj_s = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.Softplus(beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD),
            LearnableScaling(hidden_dim, init_scale=INIT_SCALE)
        )

        # =====================================================================
        # [§2.2] Vector Embedding (V_in → V_out)
        # =====================================================================
        # 입력 채널 (총 4개):
        #   - Gradient Vector (dx, dy): 2ch
        #   - Texture Flow Vector (fx, fy): 2ch
        # 출력: hidden_dim*2 차원 (hidden_dim개 채널 * 2개 성분 x,y)
        # 
        # [Architecture.md §2.2]
        # No Bias: 벡터는 방향이므로 Bias 없이 선형성과 기하학적 성질 보존
        self.proj_v = nn.Conv2d(4, hidden_dim * 2, kernel_size=1, bias=False)

        # =====================================================================
        # [§2.3] Rotor Generation (Bivector)
        # =====================================================================
        # [Architecture.md §2.3]
        # 기존의 Bivector 직접 생성(Tanh) 대신, 벡터 관계를 Sin/Cos 쌍으로 분해
        # - 입력: 4개 벡터 성분 (dx, dy, fx, fy)
        # - 출력: 2 * hidden_dim (채널당 Cos, Sin 쌍)
        # 
        # Bias=True 이유: Rotor는 변환(Transformation)이므로 편향을 통해 
        # 기본 상태(Identity)나 특정 초기 회전값을 학습 가능
        self.proj_rotor = nn.Conv2d(4, hidden_dim * 2, kernel_size=1, bias=True)

        # =====================================================================
        # [§2.4] Global Context Injection (Option)
        # =====================================================================
        # 이미지 전체의 통계(v_shape)를 스칼라에 주입하여 전역 정보를 보정
        # 입력 6개 -> 출력 hidden_dim개
        global_hidden = GLOBAL_MLP_HIDDEN if GLOBAL_MLP_HIDDEN else hidden_dim
        self.global_mlp = nn.Sequential(
            nn.Linear(6, global_hidden),
            nn.ReLU(),
            nn.Linear(global_hidden, hidden_dim),
            nn.Sigmoid()  # 0~1 사이의 게이트(Gate) 역할
        )

    def to_tensor(self, data_dict, device):
        """
        [Phase 2 Helper] Numpy 데이터를 GPU Tensor로 변환
        
        단일 이미지(HWC)와 배치 이미지(BHWC)를 모두 처리합니다.
        """
        # 1. Scalar Group (HSI + SDF) 합치기
        hsi = data_dict['hsi']  # (H, W, 3) 또는 (B, H, W, 3)
        sdf = data_dict['sdf']  # (H, W) 또는 (B, H, W)
        
        # 배치 차원 존재 여부 확인
        has_batch = (hsi.ndim == 4)
        
        # SDF 차원 확장
        sdf = sdf[..., np.newaxis]  # (..., 1)
        scalars_np = np.concatenate([hsi, sdf], axis=-1)  # (..., 4)
        vectors_np = data_dict['gradient']
        v_shape_np = data_dict['v_shape']

        if has_batch:
            # Case A: Training (Batch Input)
            # Input: (B, H, W, C) -> Output: (B, C, H, W)
            s_tensor = torch.from_numpy(scalars_np).permute(0, 3, 1, 2).float().to(device)
            v_tensor = torch.from_numpy(vectors_np).permute(0, 3, 1, 2).float().to(device)
            g_tensor = torch.from_numpy(v_shape_np).float().to(device)
        else:
            # Case B: Inference (Single Image Input)
            # Input: (H, W, C) -> Output: (1, C, H, W)
            s_tensor = torch.from_numpy(scalars_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            v_tensor = torch.from_numpy(vectors_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            g_tensor = torch.from_numpy(v_shape_np).unsqueeze(0).float().to(device)

        return s_tensor, v_tensor, g_tensor

    def forward(self, phase1_item, device=None):
        """
        [Phase 2 Forward Pass]
        
        Architecture.md §2 전체 흐름
        
        Args:
            phase1_item: Phase 1의 출력 딕셔너리
            device: 연산 장치
            
        Returns:
            S (Tensor): (B, hidden_dim, H, W) - Cos 성분이 추가된 스칼라
            V (Tensor): (B, hidden_dim, 2, H, W) - 벡터 필드
            B (Tuple): (Unit_Cos, Unit_Sin, Magnitude) 형태의 Rotor 패키지
        """
        if device is None:
            device = next(self.parameters()).device

        # 1. 데이터 준비 (GPU 전송)
        s_in, v_in, g_in = self.to_tensor(phase1_item, device)
        batch, _, h, w = s_in.shape

        # =====================================================================
        # [§2.1] Scalar Embedding
        # =====================================================================
        # 물리적 스칼라들(재질, 에너지, SDF)을 고차원 특징으로 변환
        s_emb = self.proj_s(s_in)
        
        # Global Context 주입 (Attention-like)
        global_gate = self.global_mlp(g_in).unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)
        s_emb = s_emb * global_gate  # 채널별 중요도 조절

        # =====================================================================
        # [§2.2] Vector Embedding
        # =====================================================================
        # (B, 4, H, W) -> (B, hidden_dim*2, H, W)
        v_flat = self.proj_v(v_in)
        
        # 차원 재구성: 128개 채널을 (64개 채널 x 2개 성분)으로 분리
        # 이렇게 해야 '벡터로서의 기하학적 성질'이 유지됨
        v_emb = v_flat.view(batch, self.hidden_dim, 2, h, w)

        # =====================================================================
        # [§2.3] Rotor Embedding & Decomposition
        # =====================================================================
        # (1) Rotor 추론: 벡터 입력으로부터 변환 관계(Rotation+Scale) 추출
        rotor_raw = self.proj_rotor(v_in)  # (B, hidden_dim*2, H, W)
        
        # (2) Pair 분리: Cos/Sin 쌍으로 나눔
        rotor_pair = rotor_raw.view(batch, self.hidden_dim, 2, h, w)
        cos_part = rotor_pair[:, :, 0, :, :]  # 닮음(내적) 성분
        sin_part = rotor_pair[:, :, 1, :, :]  # 회전(외적) 성분
        
        # (3) 정규화 및 분리 (Architecture.md §2.3 핵심)
        # Rotor Magnitude (|R|): Scale 성분
        rotor_mag = torch.sqrt(cos_part**2 + sin_part**2 + 1e-6)
        
        # Unit Rotor (R/|R|): 순수 회전 성분
        unit_cos = cos_part / rotor_mag
        unit_sin = sin_part / rotor_mag
        
        # Output B: Phase 3를 위해 분리된 성분들을 튜플로 패키징
        b_emb = (unit_cos, unit_sin, rotor_mag)
        
        # (4) Scalar 업데이트
        # Cos Part(유사도 성분)를 기존 Scalar에 더해줌
        # Raw Cosine(Scale 포함)을 사용하여 '닮음+크기' 정보를 S에 반영
        s_combined = torch.cat([s_emb, cos_part], dim=1)  # (B, hidden_dim*2, H, W)
        s_emb = self.s_mixer(s_combined)                  # (B, hidden_dim, H, W)

        # 5. 최종 멀티벡터 반환
        return s_emb, v_emb, b_emb
    

class CliffordPyramidEmbedder(nn.Module):
    """
    [Phase 2 Main Wrapper]
    
    Architecture.md §2 마무리
    
    Phase 1에서 온 '피라미드 리스트'를 받아서,
    Core Engine(Weights Shared)을 사용해 모든 레벨을 처리합니다.
    
    Weight Sharing: 동일한 물리 법칙이 모든 스케일에 적용
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        # 하나의 Core 모듈을 모든 스케일에서 재사용
        self.core = CliffordComponentEmbedding(hidden_dim)
        
    def forward(self, pyramid_data_list, device=None):
        """
        Args:
            pyramid_data_list: [Level0_Dict, Level1_Dict, ...]
        Returns:
            list of tuples: [(S0, V0, (uCos0, uSin0, Mag0)), ...]
        """
        pyramid_outputs = []
        
        for level_data in pyramid_data_list:
            s, v, b = self.core(level_data, device)
            pyramid_outputs.append((s, v, b))
            
        return pyramid_outputs


# =============================================================================
# [시각화 함수]
# =============================================================================

def visualize_clifford_pyramid(clifford_pyramid):
    """
    [Phase 2 시각화]
    
    고차원(hidden_dim 채널) 클리포드 임베딩을 시각화합니다.
    채널 전체의 '에너지(Energy)'를 평균내어 활성화 맵으로 표시합니다.
    """
    levels = len(clifford_pyramid)
    plt.figure(figsize=(18, 4 * levels))
    plt.suptitle("Phase 2: Multi-Scale Clifford Embedding (S, V, B)", 
                 fontsize=20, fontweight='bold')
    
    for i, (S, V, B) in enumerate(clifford_pyramid):
        # [Scalar Map] 채널의 평균 활성도
        s_map = S[0].detach().cpu().numpy().mean(axis=0)

        # [Vector Map] 벡터들의 크기(Magnitude)의 평균
        v_vec = V[0].detach().cpu().numpy()
        v_mag = np.mean(np.sqrt(v_vec[:,0]**2 + v_vec[:,1]**2), axis=0)

        # [Bivector Map] Rotor Magnitude
        unit_cos, unit_sin, rotor_mag = B
        b_map = np.mean(rotor_mag[0].detach().cpu().numpy(), axis=0)
        
        h, w = s_map.shape
        base_idx = i * 3
        
        plt.subplot(levels, 3, base_idx + 1)
        plt.imshow(s_map, cmap='inferno')
        plt.ylabel(f"Level {i}\n({h}x{w})", fontsize=14, fontweight='bold')
        if i==0: plt.title("Scalar Energy ($S$)\n(Texture + Similarity)")
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(levels, 3, base_idx + 2)
        plt.imshow(v_mag, cmap='viridis')
        if i==0: plt.title("Vector Magnitude ($V$)\n(Directional Force)")
        plt.xticks([]), plt.yticks([])
        
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

    IMG_PATH = "./img/val2017/000000569972.jpg"
    img = cv2.imread(IMG_PATH)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Phase 1
        preprocessor = MathGeometricPreprocessor()
        pyramid_raw = preprocessor.process_pyramid(img_rgb, levels=6)
        print(f"Phase 1 Complete. Scales generated: {len(pyramid_raw)}")

        # Phase 2
        embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
        
        with torch.no_grad():
            clifford_pyramid = embedder(pyramid_raw, device)
            
        print(f"Phase 2 Complete. Pyramid Embeddings Created: {len(clifford_pyramid)}")
        
        # Structure Check
        for i, (S, V, B) in enumerate(clifford_pyramid):
            rotor_mag_shape = B[2].shape 
            print(f"   [Level {i}] Res: {S.shape[-2:]} | S:{S.shape} V:{V.shape} B(Mag):{rotor_mag_shape}")
            
        visualize_clifford_pyramid(clifford_pyramid)
    else:
        print("Image Not Found.")
