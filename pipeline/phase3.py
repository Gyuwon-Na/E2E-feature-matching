# =============================================================================
# Architecture.md Line Mapping (원본 architecture.md 기준)
# - 각 라인이 코드의 어느 부분에 해당하는지 anchor로 명시합니다.
# - (요청사항) architecture.md 내용을 스킵/생략하지 않기 위해, 빈 줄도 포함합니다.
# =============================================================================
# [ARCH L0109] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) ## **📂  3: Geometric Transformer & Decoder**
# [ARCH L0110] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0111] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 이 단계는 Phase 2에서 준비된 **해상도별 S, V, B 꾸러미**를 입력받아, 이미지 간의 기하학적 대응 관계를 추론하고 초정밀 매칭 지도를 생성하는 최종 공정입니다.
# [ARCH L0112] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0113] (GeometricTokenizer + Phase3Transformer.prepare_input) ### **1. 토큰화 (Tokenization)**
# [ARCH L0114] (GeometricTokenizer + Phase3Transformer.prepare_input) 
# [ARCH L0115] (GeometricTokenizer + Phase3Transformer.prepare_input) 1. Input Alignment
# [ARCH L0116] (GeometricTokenizer + Phase3Transformer.prepare_input)     - Phase 2에서 넘어온 S,V,B의 차원이 제각각이므로 1X1 Conv를 통해 동일한 차원으로 투영한 후 결합
# [ARCH L0117] (GeometricTokenizer + Phase3Transformer.prepare_input) 2. **Group Conv**
# [ARCH L0118] (GeometricTokenizer + Phase3Transformer.prepare_input)     - 정렬된 입력을 S그룹, V그룹, B그룹으로 나누어 Group=3인 Conv 적용하여 물리적 성질이 섞이지 않게 초기 특징 추출
# [ARCH L0119] (GeometricTokenizer + Phase3Transformer.prepare_input) 
# [ARCH L0120] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) ### **2. 인코더 (Encoder)**
# [ARCH L0121] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 
# [ARCH L0122] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 1. **Q, K, V 변환**
# [ARCH L0123] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     1. **S,V,B 각각을 위한 3개의 병렬 Independent Linear →** 각 성분이 가진 고유한 기하학적 정체성을 유지하며 고차원 특징으로 투영
# [ARCH L0124] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 
# [ARCH L0125] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     | **입력 성분** | **생성되는 쿼리/키/값** | **주요 역할** |
# [ARCH L0126] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     | --- | --- | --- |
# [ARCH L0127] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     | **Scalar ($S$)** | $Q_S, K_S, V_S$ | 존재 유무 및 에너지 유사도 판단 |
# [ARCH L0128] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     | **Vector ($V$)** | $Q_V, K_V, V_V$ | 방향 정렬 및 스케일 차이 계산 |
# [ARCH L0129] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     | **Bivector ($B$)** | $Q_B, K_B, V_B$ | 회전 일관성 및 아핀 변환 대응 |
# [ARCH L0130] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 2. **위치 정보 주입**
# [ARCH L0131] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     1. **Group Convolution 기반 CPE**를 사용하여 픽셀 단위가 아닌 '기하학적 덩어리' 단위로 위치를 파악
# [ARCH L0132] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 3. **Rotor-Scale Attention**
# [ARCH L0133] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     - Transformer의 hidden state 뿐 아니라 Phase 2의 Rotor Tuple을 보조 입력으로 직접 받아 어텐션 점수 계산에 활용
# [ARCH L0134] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     1. Path A
# [ARCH L0135] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)         - Phase 2에서 분리한 Unit Rotor (정규화된 순수 회전 방향, $R/|R|$) 참조
# [ARCH L0136] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)         - 학습된 Q,K의 코사인 유사도에 더해 물리적인 회전 방향의 일치도를 어텐션 점수에 반영 (Rotation Invariant Matching)
# [ARCH L0137] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     2. Path B
# [ARCH L0138] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)         - Phase 2에서 분리한 Magnitude (벡터/Rotor  크기 정보, $|R|$)  참조
# [ARCH L0139] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)         - Q와 K 사이의 상대적인 스케일 비율 $\log{|K|} - \log{|Q|}$ 를 계산
# [ARCH L0140] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - 이 스케일 차이 값은 Attention Bias로 작용하며 V에 Injection 되어 매칭된 K가 원본 Q보다 몇 배 크거나 작은지에 대한 물리적 수치를 디코더로 전달
# [ARCH L0141] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 4. **Injection Fusion**
# [ARCH L0142] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     1. Path A에 대해서만 어텐션 점수를 매김
# [ARCH L0143] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     2. Path B에서 나온 스케일 차이를 Value에 직접 주입
# [ARCH L0144] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - 이로써 V는 K 본연의 모습 뿐만 아니라 Q 대비 크기 비율을 가짐
# [ARCH L0145] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0146] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)             Ex. "나보다 2배 큰 짝꿍"이라는 정보를 벡터에 담아 좌표 계산 시 반영되도록 함
# [ARCH L0147] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0148] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0149] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         **Global Descriptor ($G \in \mathbb{R}^6$) 구성:**
# [ARCH L0150] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0151] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         ```python
# [ARCH L0152] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         # 코드: phase1.py lines 198-204
# [ARCH L0153] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         v_shape = np.array([
# [ARCH L0154] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)             np.mean(edge_mag), np.std(edge_mag),           # [0-1] 엣지 분포
# [ARCH L0155] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)             np.mean(struct_energy), np.std(struct_energy), # [2-3] 구조 분포
# [ARCH L0156] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)             np.mean(texture), np.std(texture)              # [4-5] 밝기 분포
# [ARCH L0157] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         ], dtype=np.float32)
# [ARCH L0158] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         ```
# [ARCH L0159] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0160] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         **MLP Gate 생성:**
# [ARCH L0161] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0162] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         $$\text{Gate}_{global} = \sigma\left( \text{Linear}_{hidden\_dim} \circ \text{ReLU} \circ \text{Linear}_{6 \to hidden\_dim}(G) \right)$$
# [ARCH L0163] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0164] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         **Scalar 조정:**
# [ARCH L0165] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0166] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         $$S_{adjusted} = S_{base} \odot \text{Gate}_{global}$$
# [ARCH L0167] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0168] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         **물리적 의미:**
# [ARCH L0169] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - 엣지가 선명한 이미지($\sigma_{edge}$ 높음) → Gate ↑ → Scalar 강조
# [ARCH L0170] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - 균일한 이미지($\sigma_{texture}$ 낮음) → Gate ↓ → 과적합 방지
# [ARCH L0171] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0172] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 5. **Geometric Descriptor Guidance**
# [ARCH L0173] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     1. Fast Lane (독립 처리)
# [ARCH L0174] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - S,V,B가 각각 독립적인 Layer를 통과하여 고유 특징만 빠르게 추출 (채널 섞임 방지)
# [ARCH L0175] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     2. Descriptor 생성
# [ARCH L0176] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - 각 성분에서 회전/방향에 상관없는 불변량($S, ||V||, ||B||$) 를 뽑아 3차원 요약 벡터(descriptor) 생성
# [ARCH L0177] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     3. Gate Modulation
# [ARCH L0178] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - 요약 벡터를 MLP에 통과시켜 3개의 Gate 값($~~g_s, g_v, g_b~~$)을 얻어 각 Gate값들을 S,V,B에 곱해 볼륨(중요도)를 동적으로 조절
# [ARCH L0179] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) - Encoder Block의 `forward()`
# [ARCH L0180] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance)     - 정규화: LayerNorm을 통해 평균을 잡아 기하학적 안정성을 유지
# [ARCH L0181] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 
# [ARCH L0182] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) ### **3. 디코더 (Decoder) — Coarse-to-Fine Transform Propagation**
# [ARCH L0183] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0184] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 인코더가 만든 맥락이 담긴 피라미드와 Phase 2의 원본 피라미드를 결합하여 최종 지도를 완성.
# [ARCH L0185] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 저해상도에서 추정한 Global Transform을 고해상도로 전파하여 "큰 변환 → 작은 잔차" 순으로 처리합니다.
# [ARCH L0186] (GeometricEncoderBlock / RotorScaleAttention / GeometricDescriptorGuidance) 
# [ARCH L0187] (Phase3Transformer.rotor_map_to_theta + compose_theta) #### **3.1 Global Transform Estimation (Coarsest Level)**
# [ARCH L0188] (Phase3Transformer.rotor_map_to_theta + compose_theta) 
# [ARCH L0189] (Phase3Transformer.rotor_map_to_theta + compose_theta) - **역할:** 가장 저해상도(Level N-1)에서 이미지 전체의 대략적인 변환 $W_{global}$을 먼저 추정
# [ARCH L0190] (Phase3Transformer.rotor_map_to_theta + compose_theta) 
# [ARCH L0191] (Phase3Transformer.rotor_map_to_theta + compose_theta) - **방법:** 
# [ARCH L0192] (GeometricCrossAttention.forward)     1. Coarsest Level의 Phase 2 특징으로 Cross-Attention 수행
# [ARCH L0193] (GeometricCrossAttention.forward)     2. Dense Rotor Map의 **공간 평균**을 계산하여 단일 Global Transform 행렬 생성
# [ARCH L0194] (GeometricCrossAttention.forward)     3. 이 $W_{global}$은 "전체 이미지가 대략 몇 도 돌아갔고, 얼마나 이동했는지"를 나타냄
# [ARCH L0195] (GeometricCrossAttention.forward) 
# [ARCH L0196] (GeometricCrossAttention.forward) - **수식:**
# [ARCH L0197] (GeometricCrossAttention.forward)     $$
# [ARCH L0198] (GeometricCrossAttention.forward)     W_{global} = \text{Avg}_{p \in \Omega_{coarse}} \begin{bmatrix} \cos\theta(p) & -\sin\theta(p) & dx(p) \\ \sin\theta(p) & \cos\theta(p) & dy(p) \end{bmatrix}
# [ARCH L0199] (GeometricCrossAttention.forward)     $$
# [ARCH L0200] (GeometricCrossAttention.forward) 
# [ARCH L0201] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) #### **3.1.1 Transform-Guided Feature Warping**
# [ARCH L0202] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) 
# [ARCH L0203] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) - **핵심 아이디어:** 다음 레벨로 내려가기 전, 이전 레벨에서 추정한 변환으로 이미지 A의 특징을 미리 워핑
# [ARCH L0204] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) 
# [ARCH L0205] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) - **효과:** 고해상도에서는 이미 대충 맞춰진 상태에서 작은 잔차(Residual)만 추정하면 됨
# [ARCH L0206] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) 
# [ARCH L0207] (Phase3Transformer.warp_with_theta + warp_rotor_tuple) - **과정:**
# [ARCH L0208] (Phase3Transformer.warp_with_theta + warp_rotor_tuple)     1. $W_{prev}$를 현재 해상도에 맞게 업샘플링
# [ARCH L0209] (Phase3Transformer.warp_with_theta + warp_rotor_tuple)     2. `F.affine_grid` + `F.grid_sample`로 Phase 2 특징(A)을 워핑
# [ARCH L0210] (GeometricCrossAttention.forward)     3. 워핑된 A와 원본 B 사이의 잔차 변환 $\Delta W$만 Cross-Attention으로 추정
# [ARCH L0211] (GeometricCrossAttention.forward)     4. 최종 변환: $W_{current} = \Delta W \circ W_{prev}$ (변환의 합성)
# [ARCH L0212] (GeometricCrossAttention.forward) 
# [ARCH L0213] (GeometricCrossAttention.forward) #### **3.2 Cross-Attention**
# [ARCH L0214] (GeometricCrossAttention.forward) 
# [ARCH L0215] (GeometricCrossAttention.forward) - 이미지 A와 B를 대조하여 "A의 이 지점이 B로 가기 위한 회전/변환 값(Rotor)"을 추출
# [ARCH L0216] (GeometricCrossAttention.forward) - 고해상도 레벨에서는 워핑된 A와 B를 비교하므로, 추정해야 할 변환량이 작아져 정확도 향상
# [ARCH L0217] (GeometricCrossAttention.forward)     - Dense Rotor Regression Head:
# [ARCH L0218] (GeometricCrossAttention.forward)         - **(B, 4, H, W) 형태의 Pixel-wise Dense Rotor Map** 출력: $(\cos, \sin, dx, dy)$
# [ARCH L0219] (GeometricCrossAttention.forward)         - **이유:** 원근감이 있거나 비평면 물체인 경우, 픽셀마다 변환량이 다름
# [ARCH L0220] (GeometricCrossAttention.forward) 
# [ARCH L0221] (GeometricCrossAttention.forward) 2. **Clifford Interpolation**
# [ARCH L0222] (GeometricCrossAttention.forward)     - 이전 단계의 저해상도 결과를 업샘플링할 때, 스칼라는 부드럽게 늘리고 벡터의 방향성을 보존하며 회전 보간을 수행
# [ARCH L0223] (GeometricCrossAttention.forward)         - Scalar: 부드러운 선형 보간
# [ARCH L0224] (GeometricCrossAttention.forward)         - Vector: 방향성을 유지하며 보간 (일반 Bilinear 허용)
# [ARCH L0225] (GeometricCrossAttention.forward)         - Rotor(Bivector): 회전 성질 보존을 위해 보간 후 정규화(NLERP) 수행
# [ARCH L0226] (GeometricCrossAttention.forward) 3. **Geometric Skip-Connection**
# [ARCH L0227] (GeometricCrossAttention.forward)     1. **정렬:** 3.1(Cross Attention)에서 예측한 Dense Rotor Map($\cos,\sin,dx,dy$)을 이용해 픽셀별 Affine Grid 생성
# [ARCH L0228] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - Phase 2의 원본 S,V,B 피쳐를 이 Grid에 맞춰 샘플링 (`grid_sample`) 하여 현재 디코더의 시점에 맞게 뒤틈
# [ARCH L0229] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     2. **융합:** 업샘플링된 문맥과 정렬된 원본 디테일을 Concat (채널 방향으로 합침)
# [ARCH L0230] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         - Gated Injection 방식→이 상황에서는 스케일 정보를 얼마나 믿을까?를 입력 데이터로부터 판단
# [ARCH L0231] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 4. **Feature Map 생성**
# [ARCH L0232] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     1. CNN이 합쳐진 특징을 훑으며 노이즈를 제거 & 정교한 S,V,B 출력
# [ARCH L0233] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)     2. 다음 상위 레벨로 전달되거나 최종 단계일 경우 MPC 에너지를 계산할 최종 Feature map이 됨
# [ARCH L0234] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection)         1. 해당 Feature Map은 에너지 포텐셜 (Scalar Field)와 벡터 필드를 포함하는 다채널 구조로 출력
# [ARCH L0235] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) 
# [ARCH L0236] (Phase3Transformer.forward (Decoder) + Cross-Attn + SkipConnection) ---
# =============================================================================
# =============================================================================
# [ARCH ADDENDUM §6.3 MAPPING]
# =============================================================================
# [ARCH L0575] ### 6.3 Phase 3 구현 메모 (v5 메모리/속도 최적화 포함) -> phase3.py (addendum entry)
# [ARCH L0576]  -> phase3.py (Phase3Transformer implementation)
# [ARCH L0577] - **Chunked Attention (RTX 3090 24GB 대응):**   -> Phase3Transformer.forward (chunked attention path) + SAFE_* const
# [ARCH L0578]   `SAFE_N_LIMIT`, `SAFE_ELEMENTS` 기준으로 픽셀 수(N)가 큰 레벨에서 attention을 chunk 단위로 수행합니다. -> Phase3Transformer.forward (chunked attention path) + SAFE_* const
# [ARCH L0579] - **Optional High-Res Attention Skip:**   -> HIGH_RES_SKIP_LEVEL + forward() high-res skip branch
# [ARCH L0580]   `HIGH_RES_SKIP_LEVEL` 을 통해 level 0~1(고해상도)에서 self/cross-attention을 생략하고   -> HIGH_RES_SKIP_LEVEL + forward() high-res skip branch
# [ARCH L0581]   **이전 레벨 rotor/context를 업샘플링**하여 속도/메모리를 확보할 수 있습니다. -> phase3.py (Phase3Transformer implementation)
# [ARCH L0582] - **Transform-Guided Warping + Residual Composition:**   -> Phase3Transformer.warp_features + forward() (coarse-to-fine warping)
# [ARCH L0583]   coarse 레벨에서 얻은 `W_prev`로 A 특징을 먼저 warp한 뒤, 다음 레벨에서 잔차 `ΔW`만 추정하여   -> Phase3Transformer.warp_features + forward() (coarse-to-fine warping)
# [ARCH L0584]   `W_current = ΔW ∘ W_prev` 로 누적합니다. -> phase3.py (Phase3Transformer implementation)
# [ARCH L0585] - **Skip-Connection 실제 적용 강화:**   -> HIGH_RES_SKIP_LEVEL + forward() high-res skip branch
# [ARCH L0586]   rotor_map 기반 warp → gated injection → refinement block(ResBlock)을 **실제 forward path에 반영**합니다. -> Phase3Transformer.warp_features + forward() (coarse-to-fine warping)
# [ARCH L0587] - **Phase 4 사용을 위한 Gate Map 노출:**   -> GeometricGateHead.forward + results['gates']
# [ARCH L0588]   Phase3 내부에서 계산 가능한 `g_s, g_v, g_b` gate map을 **출력 dict에 포함**하여 Phase4 MPC 에너지 가중치로 사용 가능합니다. -> GeometricGateHead.forward + results['gates']
# [ARCH L0589]  -> phase3.py (Phase3Transformer implementation)
# =============================================================================


"""
================================================================================
Phase 3: Geometric Transformer & Decoder - Architecture.md 완전 반영 버전
================================================================================
[Architecture.md §3 참조]

이 버전은 기존 phase3.py(v5 RTX3090 최적화)에 더해,
Architecture.md에 존재하지만 코드에 없던 항목들을 "삭제하지 않고" 그대로 구현합니다.

추가/보강된 구현 포인트 (Architecture.md 기준):
1) Encoder Rotor-Scale Attention의 Path A: Unit Rotor(회전 방향) 정렬을 Attention Bias에 반영
2) Decoder Coarse-to-Fine Transform Propagation:
   - Coarsest Level에서 Global Transform(W_global) 추정
   - 이전 레벨 변환(W_prev)으로 A 특징을 먼저 워핑한 뒤 잔차 ΔW만 추정
   - 변환 합성: W_current = ΔW ∘ W_prev
3) Rotor Map 업샘플 시 NLERP 정규화( cos/sin 정규화 ) 적용
4) Geometric Skip-Connection: Gated Injection(신뢰도 기반 디테일 주입) + Refinement Block 실제 적용

출력:
- per-level dict:
  - level
  - delta_rotor_map: (B,H,W,4) = (cos, sin, dx, dy)  [Residual ΔW at this level]
  - W_global: (B,2,3)                               [Accumulated transform after composition]
  - refined_feature: (B,C,H,W)                      [Refined S/V/B feature volume]
  - mpc_map: (B,4,H,W)                              [Energy, Vx, Vy, Rotation proxy]
================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM

# =============================================================================
# [Hyperparameters] Phase 3
# =============================================================================
FEATURE_DIM = 144                # Transformer 내부 연산 차원 (S/V/B 각 48ch)
NUM_ENCODER_LAYERS = 3           # Encoder 블록 수
NUM_ATTENTION_HEADS = 4          # Multi-Head Attention 헤드 수
SE_REDUCTION = 16                # SE Block 축소 비율

# RTX 3090 메모리 최적화 (기존 코드 유지)
SAFE_N_LIMIT = 8192              # Chunking 없이 처리할 최대 픽셀 수
SAFE_ELEMENTS = 2**22            # Chunk당 최대 요소 수 (~400만)

# [Architecture.md 정합] 기본은 "모든 레벨"에서 Cross-Attention 수행
# (필요 시 사용자가 HIGH_RES_SKIP_LEVEL을 >0 으로 올려 최적화 가능)
HIGH_RES_SKIP_LEVEL = 2          # >=2이면 level0~1(high-res) cross-attn 스킵 (v5 옵션)

# [Architecture.md §3.2.3 Path A] 회전 정렬 Bias 스케일
ROTATION_BIAS_SCALE = 0.5


# =============================================================================
# [공통 빌딩 블록]
# =============================================================================

class Mish(nn.Module):
    """Mish Activation: f(x) = x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SEBlock(nn.Module):
    """Channel Attention (Squeeze-and-Excitation)"""
    def __init__(self, channel, reduction=SE_REDUCTION):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GeometricResBlock(nn.Module):
    """Residual + Dilated Conv + SE Block"""
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=max(1, dim // 3))
        self.norm1 = nn.GroupNorm(max(1, dim // 16), dim)
        self.act = Mish()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(max(1, dim // 16), dim)
        self.se = SEBlock(dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        return out + residual


# =============================================================================
# [Stage 1] Tokenization & Alignment
# =============================================================================

class GeometricTokenizer(nn.Module):
    """
    Architecture.md §3.1 - 토큰화
    - Input Alignment은 Phase3Transformer.prepare_input()에서 수행
    - Group Conv(groups=3): S/V/B 성분의 물리적 성질이 섞이지 않도록 초기 특징 추출
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.group_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, groups=3)
        self.norm = nn.GroupNorm(3, hidden_dim)

    def forward(self, x):
        return self.norm(self.group_conv(x))


# =============================================================================
# [Stage 2] Encoder Components
# =============================================================================

class GeometricCPE(nn.Module):
    """Architecture.md §3.2.2 - Group Convolution 기반 CPE"""
    def __init__(self, dim):
        super().__init__()
        self.pos_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.pos_conv(x)


class IndependentLinear(nn.Module):
    """Architecture.md §3.2.1 - S/V/B 독립 Q/K/V 투영"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // 3
        self.lin_s = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.lin_v = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.lin_b = nn.Linear(self.chunk_dim, self.chunk_dim)

    def forward(self, x):
        # x: (..., C)
        s, v, b = torch.chunk(x, 3, dim=-1)
        return torch.cat([self.lin_s(s), self.lin_v(v), self.lin_b(b)], dim=-1)


class RotorScaleAttention(nn.Module):
    """
    Architecture.md §3.2.3 & §3.2.4 - Rotor-Scale Attention + Injection Fusion

    Path A (Rotation): Unit Rotor 정렬(회전 방향 일치) 정보를 Attention Bias에 추가
    Path B (Scale): Rotor Magnitude 차이를 Attention Bias로 사용 + Value Injection
    """
    def __init__(self, dim, num_heads=NUM_ATTENTION_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = IndependentLinear(dim)
        self.to_k = IndependentLinear(dim)
        self.to_v = IndependentLinear(dim)

        self.proj = nn.Linear(dim, dim)

        # Scale Injection 게이트(0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Mish(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, rotor_tuple):
        """
        Args:
            x: (B, H, W, C)
            rotor_tuple: (unit_cos, unit_sin, rotor_mag) from Phase2
        """
        B, H, W, C = x.shape
        N = H * W

        # 1) Q,K,V
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) Q,K normalize (cosine similarity)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        unit_cos, unit_sin, rotor_mag = rotor_tuple  # (B, hidden_dim, H, W) each

        # Rotor magnitude(Scale) - per-pixel mean
        r_mag = rotor_mag.mean(dim=1, keepdim=True)  # (B,1,H,W)
        r_mag = r_mag.view(B, 1, N, 1)

        # Unit rotor (Rotation direction) - per-pixel mean 후 정규화
        r_cos = unit_cos.mean(dim=1, keepdim=True)  # (B,1,H,W)
        r_sin = unit_sin.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # (cos,sin) 정규화 (NLERP 형태)
        r_norm = torch.sqrt(r_cos**2 + r_sin**2 + 1e-6)
        r_cos = (r_cos / r_norm).view(B, 1, N, 1)
        r_sin = (r_sin / r_norm).view(B, 1, N, 1)

        # Gate Weight: scale injection 신뢰도
        gate_weight = self.gate_net(x).view(B, 1, N, 1)

        # Chunk size 결정
        if N <= SAFE_N_LIMIT:
            CHUNK_SIZE = N
        else:
            CHUNK_SIZE = max(1, SAFE_ELEMENTS // N)

        output_chunks = []

        # Key-side tensor view
        r_mag_k = r_mag.transpose(-2, -1)  # (B,1,1,N)
        r_cos_k = r_cos.transpose(-2, -1)  # (B,1,1,N)
        r_sin_k = r_sin.transpose(-2, -1)  # (B,1,1,N)

        r_mag_v = r_mag.expand(B, self.num_heads, N, 1)

        pbar = tqdm(
            range(0, N, CHUNK_SIZE),
            desc=f"  [Attn] Chunks (N={N})",
            leave=False,
            disable=(N <= SAFE_N_LIMIT)
        )

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for i in pbar:
                q_chunk = q[:, :, i:i + CHUNK_SIZE, :]  # (B,heads,M,hd)
                r_mag_q = r_mag[:, :, i:i + CHUNK_SIZE, :]  # (B,1,M,1)
                gate_c = gate_weight[:, :, i:i + CHUNK_SIZE, :]  # (B,1,M,1)

                # ------------------------------
                # Path B: Scale bias (log scale diff)
                # ------------------------------
                scale_diff = torch.abs(
                    torch.log(r_mag_q + 1e-6) - torch.log(r_mag_k + 1e-6)
                )  # (B,1,M,N)
                scale_bias = -scale_diff.to(q.dtype)

                # ------------------------------
                # Path A: Rotation bias (Unit rotor alignment)
                # rot_sim = cos(Δθ) = cos_q*cos_k + sin_q*sin_k
                # ------------------------------
                r_cos_q = r_cos[:, :, i:i + CHUNK_SIZE, :]  # (B,1,M,1)
                r_sin_q = r_sin[:, :, i:i + CHUNK_SIZE, :]
                rot_sim = (r_cos_q * r_cos_k) + (r_sin_q * r_sin_k)  # (B,1,M,N)
                rot_bias = (ROTATION_BIAS_SCALE * rot_sim).to(q.dtype)

                # 총 bias
                attn_mask = scale_bias + rot_bias  # (B,1,M,N)

                # Main attention
                out_chunk = F.scaled_dot_product_attention(
                    q_chunk, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0
                )

                # Scale injection (Value에 scale ratio 주입)
                r_mag_att = F.scaled_dot_product_attention(
                    q_chunk, k, r_mag_v,
                    attn_mask=attn_mask,
                    dropout_p=0.0
                )
                injection = r_mag_att / (r_mag_q + 1e-6)
                out_chunk = out_chunk * (1.0 + gate_c * injection)

                output_chunks.append(out_chunk)

                del scale_diff, scale_bias, rot_sim, rot_bias, attn_mask, r_mag_att, injection

        out = torch.cat(output_chunks, dim=2)  # (B,heads,N,hd)
        out = out.transpose(1, 2).reshape(B, H, W, C)

        return self.proj(out)


class GeometricDescriptorGuidance(nn.Module):
    """Architecture.md §3.2.5 - (g_s,g_v,g_b) Gate 기반 모듈레이션"""
    def __init__(self, dim):
        super().__init__()
        self.descriptor_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Mish(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x, return_gates: bool = False):
        # x: (B,H,W,C)
        s, v, b = torch.chunk(x, 3, dim=-1)

        inv_s = torch.mean(torch.abs(s), dim=-1, keepdim=True)
        inv_v = torch.norm(v, dim=-1, keepdim=True)
        inv_b = torch.norm(b, dim=-1, keepdim=True)

        descriptor = torch.cat([inv_s, inv_v, inv_b], dim=-1)  # (B,H,W,3)
        gates = self.descriptor_net(descriptor)                # (B,H,W,3)

        g_s, g_v, g_b = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]

        s_mod = s * g_s
        v_mod = v * g_v
        b_mod = b * g_b

        x_mod = torch.cat([s_mod, v_mod, b_mod], dim=-1)

        if return_gates:
            return x_mod, (g_s, g_v, g_b)
        return x_mod


class GeometricGateHead(nn.Module):
    """Phase 3 출력으로 (g_s, g_v, g_b) gate map을 노출하기 위한 Head.

    - Architecture.md §3.2.5(Geometric Descriptor Guidance)의 게이트 정의를 그대로 사용
    - Phase 4(MPC Refiner)에서 energy 가중치로 사용 가능하도록 (B,H,W) 게이트를 출력

    입력:
        feat_chw: (B, C, H, W), C는 3의 배수(= S/V/B chunk concat)여야 함
    출력:
        (g_s, g_v, g_b): 각 (B, H, W), 값 범위 [0,1]
    """
    def __init__(self):
        super().__init__()
        self.descriptor_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Mish(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, feat_chw: torch.Tensor):
        x = feat_chw.permute(0, 2, 3, 1)  # (B,H,W,C)
        s, v, b = torch.chunk(x, 3, dim=-1)

        inv_s = torch.mean(torch.abs(s), dim=-1, keepdim=True)
        inv_v = torch.norm(v, dim=-1, keepdim=True)
        inv_b = torch.norm(b, dim=-1, keepdim=True)

        descriptor = torch.cat([inv_s, inv_v, inv_b], dim=-1)  # (B,H,W,3)
        gates = self.descriptor_net(descriptor)                # (B,H,W,3)

        g_s = gates[..., 0]
        g_v = gates[..., 1]
        g_b = gates[..., 2]
        return g_s, g_v, g_b



class GeometricEncoderBlock(nn.Module):
    """Architecture.md §3.2 - CPE -> Rotor-Scale Attention -> Guidance -> FFN"""
    def __init__(self, dim):
        super().__init__()
        self.cpe = GeometricCPE(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotorScaleAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.guidance = GeometricDescriptorGuidance(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, rotor_tuple):
        # x: (B,H,W,C)
        x_cpe = self.cpe(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = x_cpe + self.attn(self.norm1(x_cpe), rotor_tuple)
        x_guided = self.guidance(x)
        x = x + self.ffn(self.norm2(x_guided))
        return x


# =============================================================================
# [Stage 3] Decoder Components
# =============================================================================

class DenseRotorHead(nn.Module):
    """Architecture.md §3.3.1 - Dense Rotor Regression Head"""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Mish(),
            nn.Linear(in_dim // 2, 4)
        )

    def forward(self, x):
        return self.net(x)


class GeometricCrossAttention(nn.Module):
    """Architecture.md §3.3.2 - Cross-Attention (A query, B key/value)"""
    def __init__(self, dim, num_heads=NUM_ATTENTION_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.to_q = IndependentLinear(dim)
        self.to_k = IndependentLinear(dim)
        self.to_v = IndependentLinear(dim)

        self.rotor_head = DenseRotorHead(dim)
        self.proj = IndependentLinear(dim)

    def forward(self, x_a, x_b):
        # x_a, x_b: (B,H,W,C)
        B, H, W, C = x_a.shape
        N = H * W

        q = self.to_q(x_a.view(B, N, C)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.to_k(x_b.view(B, N, C)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.to_v(x_b.view(B, N, C)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        context = out.transpose(1, 2).reshape(B, H, W, C)

        dense_rotor = self.rotor_head(context)  # (B,H,W,4)

        return self.proj(context), dense_rotor


class CliffordInterpolation(nn.Module):
    """Architecture.md §3.3.2 - Clifford Interpolation (S/V/B 분리 업샘플 + B 정규화)"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (B,C,H,W), C는 3의 배수
        s, v, b = torch.chunk(x, 3, dim=1)
        s_up = F.interpolate(s, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        v_up = F.interpolate(v, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        b_up = F.interpolate(b, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        b_up = F.normalize(b_up, dim=1)  # NLERP 형태 정규화
        return torch.cat([s_up, v_up, b_up], dim=1)


class RotorMapInterpolator(nn.Module):
    """
    Architecture.md §3.3.2 - Rotor(Bivector) 보간 후 정규화(NLERP)
    delta_rotor_map: (B,H,W,4) where (cos,sin,dx,dy)
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, rotor_map, target_hw=None):
        # rotor_map: (B,H,W,4)
        rotor_chw = rotor_map.permute(0, 3, 1, 2)  # (B,4,H,W)
        up = F.interpolate(rotor_chw, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        if target_hw is not None and up.shape[-2:] != target_hw:
            up = F.interpolate(up, size=target_hw, mode='bilinear', align_corners=True)

        # NLERP: cos/sin 정규화
        cos_t = up[:, 0:1, :, :]
        sin_t = up[:, 1:2, :, :]
        norm = torch.sqrt(cos_t**2 + sin_t**2 + 1e-6)
        cos_t = cos_t / norm
        sin_t = sin_t / norm

        out = torch.cat([cos_t, sin_t, up[:, 2:3, :, :], up[:, 3:4, :, :]], dim=1)
        return out.permute(0, 2, 3, 1)  # (B,H,W,4)


class GeometricSkipConnection(nn.Module):
    """
    Architecture.md §3.3.3 - Geometric Skip-Connection

    1) Rotor Map 기반 Warping
    2) 업샘플링된 문맥(dec_feat) + 정렬된 디테일(enc_feat) 융합
       - Gated Injection: 디테일을 얼마나 주입할지(신뢰도) 결정
    """
    def __init__(self, dim):
        super().__init__()
        self.compress = nn.Conv2d(dim * 2, dim, 1)

        # [Gated Injection] (B,2C,H,W) -> (B,C,H,W)
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Mish(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # [Refinement] 실제 적용 (기존 코드에서 정의만 하고 사용하지 않던 부분 보강)
        self.refine = GeometricResBlock(dim, dilation=2)

    def get_warp_grid(self, rotor_map, B, H, W, device):
        # rotor_map: (B,H,W,4) (cos,sin,dx,dy)
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        cos_t = rotor_map[..., 0].unsqueeze(-1)
        sin_t = rotor_map[..., 1].unsqueeze(-1)
        dx_t = rotor_map[..., 2].unsqueeze(-1)
        dy_t = rotor_map[..., 3].unsqueeze(-1)

        x_new = cos_t * base_grid[..., 0:1] - sin_t * base_grid[..., 1:2] + dx_t
        y_new = sin_t * base_grid[..., 0:1] + cos_t * base_grid[..., 1:2] + dy_t

        return torch.cat([x_new, y_new], dim=-1)

    def forward(self, dec_feat, enc_feat, rotor_map):
        """
        Args:
            dec_feat: (B,C,H,W) - decoder context
            enc_feat: (B,C,H,W) - encoder(Phase2 원본 or Transform-guided warped) feature
            rotor_map: (B,H,W,4) - residual ΔW (warped A -> B)
        """
        B, C, H, W = enc_feat.shape

        # 1) Warping (A -> B)
        grid = self.get_warp_grid(rotor_map, B, H, W, enc_feat.device)
        warped_enc = F.grid_sample(enc_feat, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

        # 2) Concat & Gated Injection
        concat = torch.cat([dec_feat, warped_enc], dim=1)  # (B,2C,H,W)
        gate = self.gate_net(concat)                       # (B,C,H,W)
        injected = torch.cat([dec_feat, warped_enc * gate], dim=1)

        fused = self.compress(injected)
        fused = self.refine(fused)
        return fused


# =============================================================================
# [Main Wrapper] Phase 3 Transformer
# =============================================================================

class Phase3Transformer(nn.Module):
    """
    Architecture.md §3 전체 구현 (Coarse-to-Fine Transform Propagation 포함)
    """
    def __init__(self, feature_dim=FEATURE_DIM, num_layers=NUM_ENCODER_LAYERS, embed_dim=HIDDEN_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # Input Alignment (Phase2 출력의 S/V/B를 Phase3 feature_dim/3로 정렬)
        self.adapt_s = nn.Conv2d(embed_dim, feature_dim // 3, 1)
        self.adapt_v = nn.Conv2d(embed_dim * 2, feature_dim // 3, 1)

        # B stream은 Architecture 취지에 맞게 "회전 성분" 중심으로 사용
        # Phase2 B tuple: (unit_cos, unit_sin, mag)
        # - cos 성분은 Phase2에서 scalar로 섞이므로, 여기서는 unit_sin(회전) 중심으로 전달
        self.adapt_b = nn.Conv2d(embed_dim, feature_dim // 3, 1)

        # Stage 1: Tokenization
        self.tokenizer = GeometricTokenizer(in_channels=feature_dim, hidden_dim=feature_dim)

        # Stage 2: Encoder
        self.encoder_layers = nn.ModuleList([GeometricEncoderBlock(feature_dim) for _ in range(num_layers)])

        # Stage 3: Decoder blocks
        self.cross_attn = GeometricCrossAttention(feature_dim)
        self.upsampler = CliffordInterpolation(scale_factor=2)
        self.rotor_upsampler = RotorMapInterpolator(scale_factor=2)
        self.skip_conn = GeometricSkipConnection(feature_dim)

        # Feature refinement + MPC head
        self.refine_net = GeometricResBlock(feature_dim, dilation=1)
        self.head_mpc = nn.Conv2d(feature_dim, 4, 1)  # (Energy, Vx, Vy, Rotation proxy)
        # Phase4(MPC)에서 사용할 gate map 노출용
        self.gate_head = GeometricGateHead()

    # ---------------------------------------------------------------------
    # Helper: Phase2 Output -> Phase3 Input Volume (S/V/B concat)
    # ---------------------------------------------------------------------
    def prepare_input(self, p2_out):
        """
        p2_out: (S, V, (unit_cos, unit_sin, mag))
          - S: (B,embed_dim,H,W)
          - V: (B,embed_dim,2,H,W)
          - B: tuple of (B,embed_dim,H,W)
        """
        s, v, b = p2_out
        unit_cos, unit_sin, rotor_mag = b

        # V: (B,embed_dim,2,H,W) -> (B,embed_dim*2,H,W)
        v_flat = v.view(v.shape[0], -1, v.shape[-2], v.shape[-1])

        s_feat = self.adapt_s(s)
        v_feat = self.adapt_v(v_flat)
        b_feat = self.adapt_b(unit_sin)  # 회전 성분 중심

        return torch.cat([s_feat, v_feat, b_feat], dim=1)  # (B,feature_dim,H,W)

    # ---------------------------------------------------------------------
    # Helper: rotor map 평균 -> 2x3 matrix (Global transform)
    # ---------------------------------------------------------------------
    @staticmethod
    def rotor_map_to_theta(rotor_map):
        """
        rotor_map: (B,H,W,4) = (cos,sin,dx,dy)
        Returns:
            theta: (B,2,3)
        """
        avg = rotor_map.mean(dim=(1, 2))  # (B,4)
        cos_t, sin_t, dx_t, dy_t = avg[:, 0], avg[:, 1], avg[:, 2], avg[:, 3]

        # cos/sin 정규화 (NLERP)
        norm = torch.sqrt(cos_t**2 + sin_t**2 + 1e-6)
        cos_t = cos_t / norm
        sin_t = sin_t / norm

        B = rotor_map.shape[0]
        theta = torch.zeros(B, 2, 3, device=rotor_map.device, dtype=rotor_map.dtype)
        theta[:, 0, 0] = cos_t
        theta[:, 0, 1] = -sin_t
        theta[:, 0, 2] = dx_t
        theta[:, 1, 0] = sin_t
        theta[:, 1, 1] = cos_t
        theta[:, 1, 2] = dy_t
        return theta

    @staticmethod
    def compose_theta(delta, prev):
        """
        W_current = ΔW ∘ W_prev
        delta, prev: (B,2,3)
        """
        B = delta.shape[0]
        device = delta.device
        dtype = delta.dtype
        bottom = torch.tensor([0, 0, 1], device=device, dtype=dtype).view(1, 1, 3).repeat(B, 1, 1)

        prev_aug = torch.cat([prev, bottom], dim=1)    # (B,3,3)
        delta_aug = torch.cat([delta, bottom], dim=1)  # (B,3,3)

        out = torch.bmm(delta_aug, prev_aug)
        return out[:, :2, :]  # (B,2,3)

    @staticmethod
    def warp_with_theta(feat, theta, align_corners=True):
        """
        feat: (B,C,H,W)
        theta: (B,2,3)
        """
        grid = F.affine_grid(theta, feat.size(), align_corners=align_corners)
        return F.grid_sample(feat, grid, align_corners=align_corners, mode='bilinear', padding_mode='zeros')

    @staticmethod
    def warp_rotor_tuple(rotor_tuple, theta):
        """
        rotor_tuple: (unit_cos, unit_sin, mag) each (B, C, H, W)
        """
        unit_cos, unit_sin, mag = rotor_tuple
        cos_w = Phase3Transformer.warp_with_theta(unit_cos, theta, align_corners=True)
        sin_w = Phase3Transformer.warp_with_theta(unit_sin, theta, align_corners=True)
        mag_w = Phase3Transformer.warp_with_theta(mag, theta, align_corners=True)

        # unit 재정규화 (NLERP)
        norm = torch.sqrt(cos_w**2 + sin_w**2 + 1e-6)
        cos_w = cos_w / norm
        sin_w = sin_w / norm
        return (cos_w, sin_w, mag_w)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, pyramid_a, pyramid_b):
        """
        Args:
            pyramid_a, pyramid_b:
              list of Phase2 outputs per level: [(S,V,Btuple), ...]
              level 0: finest, level N-1: coarsest
        Returns:
            results: list of dict (coarse->fine 순서로 append됨)
        """
        results = []

        W_prev = None            # 누적 Global Transform (B,2,3)
        dec_feat_prev = None     # decoder feature volume (coarse level)

        num_levels = len(pyramid_a)

        for level_idx in reversed(range(num_levels)):
            # -----------------------------
            # 0) Prepare features
            # -----------------------------
            raw_a = self.prepare_input(pyramid_a[level_idx])  # (B,C,H,W)
            raw_b = self.prepare_input(pyramid_b[level_idx])  # (B,C,H,W)

            rotor_a = pyramid_a[level_idx][2]
            rotor_b = pyramid_b[level_idx][2]

            B, C, H, W = raw_a.shape

            # -----------------------------
            # 1) Transform-Guided Warping (Architecture.md §3.1.1)
            # -----------------------------
            if W_prev is not None:
                raw_a_warped = self.warp_with_theta(raw_a, W_prev, align_corners=True)
                rotor_a_warped = self.warp_rotor_tuple(rotor_a, W_prev)
            else:
                raw_a_warped = raw_a
                rotor_a_warped = rotor_a

            # -----------------------------
            # 2) Tokenize (GroupConv)
            # -----------------------------
            tok_a = self.tokenizer(raw_a_warped).permute(0, 2, 3, 1)  # (B,H,W,C)
            tok_b = self.tokenizer(raw_b).permute(0, 2, 3, 1)

            # -----------------------------
            # 3) Encoder (Self-Attn with rotor tuple)
            # -----------------------------
            for layer in self.encoder_layers:
                tok_a = checkpoint(layer, tok_a, rotor_a_warped, use_reentrant=False)
                tok_b = checkpoint(layer, tok_b, rotor_b, use_reentrant=False)

            # -----------------------------
            # 4) Cross-Attention -> Residual ΔW (dense rotor map)
            # -----------------------------
            if level_idx >= HIGH_RES_SKIP_LEVEL:
                ctx, delta_rotor = self.cross_attn(tok_a, tok_b)  # delta_rotor: (B,H,W,4)
            else:
                # 최적화 모드: high-res skip (기본은 사용하지 않음)
                ctx = tok_a
                delta_rotor = results[-1]['delta_rotor_map']  # 이전 레벨 결과 업샘플
                delta_rotor = self.rotor_upsampler(delta_rotor, target_hw=(H, W))

            ctx_chw = ctx.permute(0, 3, 1, 2)  # (B,C,H,W)

            # -----------------------------
            # 5) Decoder context propagation (coarse->fine)
            # -----------------------------
            if dec_feat_prev is not None:
                dec_up = self.upsampler(dec_feat_prev)
                if dec_up.shape[-2:] != (H, W):
                    dec_up = F.interpolate(dec_up, size=(H, W), mode='bilinear', align_corners=True)
                dec_context = ctx_chw + dec_up
            else:
                dec_context = ctx_chw

            # -----------------------------
            # 6) Skip-Connection (Warp + Gated Injection + Refinement)
            #    - enc_feat는 "W_prev로 1차 워핑된 raw_a_warped"를 사용
            #    - rotor_map은 "잔차 ΔW" 사용
            # -----------------------------
            fused = self.skip_conn(dec_context, raw_a_warped, delta_rotor)

            # -----------------------------
            # 7) Feature Map 생성 (Refine) + MPC map head
            # -----------------------------
            refined_feature = self.refine_net(fused)     # (B,C,H,W)
            mpc_map = self.head_mpc(refined_feature)     # (B,4,H,W)

            # Phase4 가중치용 gate map (g_s,g_v,g_b)
            g_s, g_v, g_b = self.gate_head(refined_feature)

            # -----------------------------
            # 8) Global Transform update: W_current = ΔW ∘ W_prev
            # -----------------------------
            delta_theta = self.rotor_map_to_theta(delta_rotor)  # (B,2,3)
            if W_prev is None:
                W_prev = delta_theta
            else:
                W_prev = self.compose_theta(delta_theta, W_prev)

            # 다음 레벨로 전달할 decoder feature
            dec_feat_prev = refined_feature

            # 기록
            results.append({
                'level': level_idx,
                'delta_rotor_map': delta_rotor,
                'W_global': W_prev,
                'refined_feature': refined_feature,
                'mpc_map': mpc_map,
            })

        # (중요) 외부 코드/학습 루프가 level=0(최고해상도)을 results[0]로 기대하는 경우가 많아
        # 반환 직전에 level 오름차순(0->coarse)으로 정렬합니다.
        results = sorted(results, key=lambda d: d.get('level', 0))
        return results


# =============================================================================
# 시각화 & 테스트
# =============================================================================

def visualize_phase3_results(results):
    """
    Phase 3 결과 시각화:
    - Rotor(ΔW)에서 dx/dy 흐름 색상 표시
    - MPC energy, vector field 표시
    """
    levels = len(results)
    plt.figure(figsize=(15, 4 * levels))
    plt.suptitle("Phase 3: Coarse-to-Fine Geometric Matching Analysis (ArchFull)", fontsize=18, fontweight='bold')

    for idx, res in enumerate(results):
        lvl = res['level']
        rotor = res['delta_rotor_map'].detach().cpu().numpy()[0]
        mpc = res['mpc_map'].detach().cpu().numpy()[0]

        h, w = rotor.shape[:2]

        dx, dy = rotor[..., 2], rotor[..., 3]
        rotor_img = np.zeros((h, w, 3))
        mag, ang = cv2.cartToPolar(dx, dy)
        rotor_img[..., 0] = ang / (2 * np.pi)
        rotor_img[..., 1] = 1.0
        rotor_img[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        import matplotlib.colors as mcolors
        rotor_rgb = mcolors.hsv_to_rgb(rotor_img)

        energy = mpc[0]
        vx, vy = mpc[1], mpc[2]
        vec_img = np.zeros((h, w, 3))
        mag_v, ang_v = cv2.cartToPolar(vx, vy)
        vec_img[..., 0] = ang_v / (2 * np.pi)
        vec_img[..., 1] = 1.0
        vec_img[..., 2] = cv2.normalize(mag_v, None, 0, 1, cv2.NORM_MINMAX)
        vec_rgb = mcolors.hsv_to_rgb(vec_img)

        base = idx * 3

        plt.subplot(levels, 3, base + 1)
        plt.imshow(rotor_rgb)
        plt.title(f"Level {lvl}: Residual Rotor (ΔW)")
        plt.axis('off')

        plt.subplot(levels, 3, base + 2)
        plt.imshow(energy, cmap='inferno')
        plt.title(f"Level {lvl}: MPC Energy")
        plt.axis('off')

        plt.subplot(levels, 3, base + 3)
        plt.imshow(vec_rgb)
        plt.title(f"Level {lvl}: MPC Vector Field")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3 ArchFull Running on: {device}")

    IMG_PATH = "./img/val2017/000000569972.jpg"
    img = cv2.imread(IMG_PATH)

    if img is None:
        print("Image Not Found.")
        sys.exit(0)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Phase 1
    preprocessor = MathGeometricPreprocessor()
    pyramid_raw = preprocessor.process_pyramid(img_rgb, levels=5)

    # Phase 2
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        pyramid_a = embedder(pyramid_raw, device)
        pyramid_b = embedder(pyramid_raw, device)

    # Phase 3
    model = Phase3Transformer(feature_dim=FEATURE_DIM, num_layers=NUM_ENCODER_LAYERS, embed_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        results = model(pyramid_a, pyramid_b)

    print(f"Phase 3 Complete. Levels processed: {len(results)}")
    visualize_phase3_results(results)
