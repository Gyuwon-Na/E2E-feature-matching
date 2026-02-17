# =============================================================================
# Architecture.md Line Mapping (원본 architecture.md 기준)
# - 각 라인이 코드의 어느 부분에 해당하는지 anchor로 명시합니다.
# - (요청사항) architecture.md 내용을 스킵/생략하지 않기 위해, 빈 줄도 포함합니다.
# =============================================================================
# [ARCH L0055] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) ## **📂  2: Clifford Embedding & Pyramid 생성**
# [ARCH L0056] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) 
# [ARCH L0057] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) <aside>
# [ARCH L0058] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) 
# [ARCH L0059] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) - **입력:** Phase 1에서 리사이즈된 이미지들
# [ARCH L0060] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) - **출력:** 각 해상도에 최적화된 **Clifford 특징 세트 (S, V, B)**
# [ARCH L0061] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) - **용도:**
# [ARCH L0062] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward)     - **Encoder Input:** Phase 3 인코더가 이미지의 맥락을 파악하는 기초 자료로 활용.
# [ARCH L0063] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward)     - **Decoder Skip-Connection:** 업샘플링 과정에서 원본의 날카로운 기하학을 다시 수혈하기 위한 대조군으로 활용.
# [ARCH L0064] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) </aside>
# [ARCH L0065] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) 
# [ARCH L0066] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) Phase 1에서 준비된 해상도별 물리량(S, V, B 후보)을 입력받아, 고차원 공간에서의 **멀티벡터 임베딩**을 완성. 이 결과물은 Phase 3의 디코더가 필요할 때마다 즉시 꺼내 쓸 수 있는 '기하학적 정답 창고(Pyramid)'가 됩니다.
# [ARCH L0067] (CliffordPyramidEmbedder.forward / CliffordComponentEmbedding.forward) 
# [ARCH L0068] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding)) ### **1. S (Scalar) 임베딩: 에너지 보존과 확률적 해석**
# [ARCH L0069] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding)) 
# [ARCH L0070] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding)) - **핵심 기법:** **Softplus + Learnable Scaling**
# [ARCH L0071] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding)) - **선택 이유:**
# [ARCH L0072] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding))     - **정보 보존력:** 음수 입력이나 미세한 신호도 0으로 깎아버리지 않고 부드럽게 살려두어, 아주 미약한 기하학적 단서라도 모델이 판단 근거로 삼을 수 있게 함
# [ARCH L0073] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding))     - **미분 특성:** 미분 시 Sigmoid 형태가 되어, 학습 과정에서 게이팅(Gating) 효과를 자연스럽게 유도합니다.
# [ARCH L0074] (CliffordComponentEmbedding: proj_s + s_mixer (scalar embedding)) 
# [ARCH L0075] (CliffordComponentEmbedding: proj_v (vector embedding)) ### **2. V (Vector) 임베딩: 방향의 순수성 유지**
# [ARCH L0076] (CliffordComponentEmbedding: proj_v (vector embedding)) 
# [ARCH L0077] (CliffordComponentEmbedding: proj_v (vector embedding)) - **핵심 기법:** **Linear Projection (No Bias)**
# [ARCH L0078] (CliffordComponentEmbedding: proj_v (vector embedding)) - **의미:** 벡터는 '어디로 향하는가'라는 방향 정보가 본질입니다.
# [ARCH L0079] (CliffordComponentEmbedding: proj_v (vector embedding)) - **설계 의도:** 활성화 함수로 방향을 왜곡하지 않고, Phase 1에서 추출된 그레이디언트와 텍스처 흐름을 고차원 채널로 확장하여 **벡터 특유의 선형성과 기하학적 성질**을 그대로 보존
# [ARCH L0080] (CliffordComponentEmbedding: proj_v (vector embedding)) 
# [ARCH L0081] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) ### **3. B (Bivector) Generation: 회전과 닮음의 수치화**
# [ARCH L0082] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0083] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) - **핵심 기법:** **Sin/Cos Pair Output**
# [ARCH L0084] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) - **선택 이유:**
# [ARCH L0085] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)     - **회전의 정석:** 단순히 Tanh()로 값을 제한하는 대신, 수학적으로 완벽한 회전을 표현하는 `Sin/Cos` 쌍을 직접 생성.
# [ARCH L0086] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)     - **설계 의도:** 아핀 변환(Affine Transform)이나 줌(Zoom)이 발생한 이미지 매칭에서 압도적인 성능을 보일 것이라 예측
# [ARCH L0087] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)     - **기하학적 분리**
# [ARCH L0088] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)         - **sin 성분:** 두 특징 사이의 **'다름(외적/회전)'**을 나타내며 Bivector의 핵심 값이 됩니다.
# [ARCH L0089] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)         - **cos 성분:** 두 특징 사이의 **'닮음(내적/일치)'**을 나타내며 Scalar에 더해져 유사도 판정을 돕습니다.
# [ARCH L0090] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0091] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)         - **Unit Rotor 분리 (Magnitude Normalization):**
# [ARCH L0092] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0093] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)         $$\text{Rotor Magnitude} = |R| = \sqrt{\cos^2 + \sin^2 + \epsilon}$$
# [ARCH L0094] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0095] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)         $$\text{Unit Rotor} = \frac{R}{|R|} = \left( \frac{\cos}{|R|}, \frac{\sin}{|R|} \right)$$
# [ARCH L0096] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0097] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)         - **정규화:** 출력된 Rotor를 Unit Rotor ($R/|R|$, 순수 회전)과 Magnitude ($|R|$, 스케일) 로 **분리하여 제공**할 수 있도록 설계.
# [ARCH L0098] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0099] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)             → Phase 3에서 회전은 같은데 크기만 다른 경우를 명확히 구분
# [ARCH L0100] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0101] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)             **분리 이유:**
# [ARCH L0102] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)             1. **Unit Rotor ($R/|R|$)**: 순수 회전 방향만 표현 → Phase 3의 Path A (Rotation Invariant Matching)에서 사용
# [ARCH L0103] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)             2. **Magnitude ($|R|$)**: 스케일 정보만 표현 → Phase 3의 Path B (Scale Bias)에서 사용
# [ARCH L0104] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate)             3. **기하학적 독립성**: 회전과 스케일을 분리하여 각각 독립적으로 처리 가능
# [ARCH L0105] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0106] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# [ARCH L0107] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) ---
# [ARCH L0108] (CliffordComponentEmbedding: proj_rotor + unit rotor normalization + global gate) 
# =============================================================================
# =============================================================================
# [ARCH ADDENDUM §6.1-6.2 MAPPING]
# =============================================================================
# [ARCH L0554] ## 📌 6: Code Implementation Notes (v5 / Code-Architecture Sync) -> phase2.py (implementation detail)
# [ARCH L0555]  -> phase2.py (implementation detail)
# [ARCH L0556] > 이 섹션은 **architecture.md(개념 설계)** 와 **현재 코드 구현(phase1~phase4_2, losses, fine_tune/fast_finetune)** 사이의 차이를 없애기 위해,   -> phase2.py (implementation detail)
# [ARCH L0557] > 코드에 존재하지만 본문에 상세히 없던 구현 포인트/하이퍼파라미터를 문서화한 **"Implementation Addendum"** 입니다. -> phase2.py (implementation detail)
# [ARCH L0558]  -> phase2.py (implementation detail)
# [ARCH L0559] ### 6.1 공통 하이퍼파라미터 (코드 기본값) -> phase2.py (addendum entry)
# [ARCH L0560]  -> phase2.py (implementation detail)
# [ARCH L0561] - `HIDDEN_DIM = 48`  *(Phase 2 Clifford Embedding 기본 채널 수)* -> phase2.py/phase3.py hyperparams (HIDDEN_DIM/FEATURE_DIM/levels)
# [ARCH L0562] - `FEATURE_DIM = 144 (= 3 × 48)` *(Phase 3 Transformer 내부 S/V/B concat 특징 차원)* -> phase2.py/phase3.py hyperparams (HIDDEN_DIM/FEATURE_DIM/levels)
# [ARCH L0563] - `NUM_ENCODER_LAYERS = 3`, `NUM_ATTENTION_HEADS = 4` -> phase2.py/phase3.py hyperparams (HIDDEN_DIM/FEATURE_DIM/levels)
# [ARCH L0564] - `pyramid levels = 5` *(fine_tune.py 기본 학습 설정: 큰 회전(±60°) 대응을 위해 4→5로 확장)* -> phase2.py/phase3.py hyperparams (HIDDEN_DIM/FEATURE_DIM/levels)
# [ARCH L0565]  -> phase2.py (implementation detail)
# [ARCH L0566] ### 6.2 Phase 2 구현 메모 -> phase2.py (addendum entry)
# [ARCH L0567]  -> phase2.py (implementation detail)
# [ARCH L0568] - **Rotor 생성 입력 채널 확장 (5채널):**   -> phase2.py CliffordRotorLayer.forward (rotor_in concat)
# [ARCH L0569]   `(dx, dy, fx, fy)`(Phase1의 V1/V2) 에 더해, Phase1에서 계산한 **Bivector 후보 `bivector = v1 ∧ v2`** 를 추가하여   -> phase1.py (bivector candidate) + phase2.py rotor_in
# [ARCH L0570]   `rotor_in = concat([v_in(4ch), b_in(1ch)])` 형태로 Rotor Conv에 투입합니다. -> phase2.py (implementation detail)
# [ARCH L0571] - **Scalar 업데이트 방식(s_mixer):**   -> phase2.py CliffordComponentEmbedding.forward (s_mixer)
# [ARCH L0572]   Cos 파트를 scalar embedding에 단순 가산하기보다, `concat([s_emb, cos_part]) → 1×1 Conv(s_mixer)` 로 **혼합**하여   -> phase2.py CliffordComponentEmbedding.forward (s_mixer)
# [ARCH L0573]   과도한 덮어쓰기(override)와 스케일 폭주를 줄입니다. -> phase2.py (implementation detail)
# [ARCH L0574]  -> phase2.py (implementation detail)
# =============================================================================


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


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from pipeline.phase1 import MathGeometricPreprocessor

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
        self.proj_rotor = nn.Conv2d(5, hidden_dim * 2, kernel_size=1, bias=True)  # (dx,dy,fx,fy,bivector)

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

        # 2. Bivector Candidate (Phase 1) - optional
        # Architecture.md §1.3 - B (Bivector 후보)
        # Phase 1에서 제공되는 wedge-product 기반 회전 씨앗이 있는 경우 Rotor 생성에 함께 사용합니다.
        b_np = data_dict.get('bivector', None)
        if b_np is None:
            # backward compatibility: B 후보가 없으면 0으로 채움
            if has_batch:
                b_np = np.zeros((hsi.shape[0], hsi.shape[1], hsi.shape[2]), dtype=np.float32)
            else:
                b_np = np.zeros((hsi.shape[0], hsi.shape[1]), dtype=np.float32)

        v_shape_np = data_dict['v_shape']

        if has_batch:
            # Case A: Training (Batch Input)
            # Input: (B, H, W, C) -> Output: (B, C, H, W)
            s_tensor = torch.from_numpy(scalars_np).permute(0, 3, 1, 2).float().to(device)
            v_tensor = torch.from_numpy(vectors_np).permute(0, 3, 1, 2).float().to(device)
            b_tensor = torch.from_numpy(b_np).unsqueeze(1).float().to(device)  # (B,1,H,W)
            g_tensor = torch.from_numpy(v_shape_np).float().to(device)
        else:
            # Case B: Inference (Single Image Input)
            # Input: (H, W, C) -> Output: (1, C, H, W)
            s_tensor = torch.from_numpy(scalars_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            v_tensor = torch.from_numpy(vectors_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            b_tensor = torch.from_numpy(b_np).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
            g_tensor = torch.from_numpy(v_shape_np).unsqueeze(0).float().to(device)

        return s_tensor, v_tensor, b_tensor, g_tensor

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
        s_in, v_in, b_in, g_in = self.to_tensor(phase1_item, device)
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
        rotor_in = torch.cat([v_in, b_in], dim=1)  # (B,5,H,W)
        rotor_raw = self.proj_rotor(rotor_in)  # (B, hidden_dim*2, H, W)
        
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
