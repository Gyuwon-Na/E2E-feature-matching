## **📂  1: Geometry Raw Data 추출 (Physical Raw Data Extraction)**

이 단계에서는 원본 이미지를 다양한 해상도로 리사이즈한 후, 각 층에서 클리포드 멀티벡터의 재료가 될 물리량들을 추출

- **1/32 ~ 1/16 (Global):** 이미지의 아주 거친 형태만 보입니다. 큰 건물이나 산의 위치처럼 **전역적인 배치**를 파악하기 위함입니다.
- **1/8 ~ 1/4 (Structural):** 물체의 구체적인 윤곽과 구조가 드러납니다. **주요 특징점들의 기하학적 관계**를 학습합니다.
- **1/2 ~ 1 (Fine):** 아주 세밀한 텍스처와 0.1 픽셀 단위의 엣지가 보입니다. **최종적인 초정밀 정렬**을 수행하는 단계입니다.

### **1. S (Scalar): 존재의 강도와 뼈대**

스칼라는 방향성은 없지만, 해당 지점에 **"무엇이 얼마나 강하게 있는가"**를 나타냅니다.

- **Texture (재질/밝기)**
    - 이미지의 국부적인 밝기 변화를 의미
    - 물체의 표면 특성을 반영하며, 나중에 두 이미지 사이의 **색상 유사도**를 비교하는 기초 값이 됨
- **Structure Energy (구조 에너지)**
    - 주변 픽셀들과 비교했을 때 정보가 얼마나 밀집되어 있는지를 나타냄
    - 매칭할 때 "믿을만한 특징점인가?"를 판단하는 **신뢰도 가중치**로 쓰임
- **Edge Magnitude (엣지 세기)**
    - 경계선이 얼마나 뚜렷한지 나타내는 수치
    - V(Vector)가 방향을 가리킨다면, 이 값은 그 **방향의 확신도**를 결정
- **SDF (Signed Distance Field)**
    - 물체의 뼈대로부터의 거리
    - **MPC(제어) 단계에서 가장 핵심**적인 정보로, 두 물체가 얼마나 떨어져 있는지 '에너지'로 계산할 수 있게 해주는 **잠재적인 중력장** 역할

    
    **하이퍼파라미터:**
    - `SDF_SKELETON_POWER = 8.0`: 뼈대 선명도 (높을수록 얇아짐)
    - `SDF_FIELD_POWER = 2.0`: 장 부드러움 (낮을수록 넓게 퍼짐)
    - `SDF_FIELD_WEIGHT = 0.4`: 장 가중치 (뼈대 대비 영향력)

    **물리적 의미:**
    - **Skeleton Component**: 정확한 엣지 위치 표현 (위치 정밀도)
    - **Field Component**: 넓은 탐색 범위 제공 (수렴 Basin 확대)
    - **Max Fusion**: 두 장점을 모두 활용 (날카로움 + 부드러움)

### **2. V (Vector): 변화의 방향과 흐름**

벡터는 이미지 내에서 **"어느 쪽으로 움직이는가"**라는 동적인 정보를 담습니다.

- **Gradient (경계 변화)**
    - 픽셀 값이 가장 급격하게 변하는 방향($dx, dy$)
    - 물체의 **윤곽선에 수직인 방향**을 가리키며, 두 이미지가 정렬될 때 "선의 방향이 일치하는지" 확인하는 기준이 됨
- **Texture Flow (텍스처 흐름)**
    - 질감(결)이 반복되거나 흐르는 방향($fx, fy$)
    - 엣지가 없는 매끄러운 표면에서도 **"결의 방향"**을 알 수 있게 하여, 특징이 부족한 영역에서도 매칭의 단서를 제공

### **3. B (Bivector 후보): 회전의 잠재력**

- **Bivector Candidate:** 벡터와 벡터 사이의 외적(Wedge Product)을 통해 생성될 **회전 성분의 씨앗**
- **의미:** 단순히 점이 이동하는 것을 넘어, "이 구역은 시계 방향으로 돌아가 있는가?"를 판단하기 위한 재료. 나중에 Phase 2에서 **Rotor(회전 연산자)**를 만드는 결정적인 근거가 됩니다.

---

## **📂  2: Clifford Embedding & Pyramid 생성**

<aside>

- **입력:** Phase 1에서 리사이즈된 이미지들
- **출력:** 각 해상도에 최적화된 **Clifford 특징 세트 (S, V, B)**
- **용도:**
    - **Encoder Input:** Phase 3 인코더가 이미지의 맥락을 파악하는 기초 자료로 활용.
    - **Decoder Skip-Connection:** 업샘플링 과정에서 원본의 날카로운 기하학을 다시 수혈하기 위한 대조군으로 활용.
</aside>

Phase 1에서 준비된 해상도별 물리량(S, V, B 후보)을 입력받아, 고차원 공간에서의 **멀티벡터 임베딩**을 완성. 이 결과물은 Phase 3의 디코더가 필요할 때마다 즉시 꺼내 쓸 수 있는 '기하학적 정답 창고(Pyramid)'가 됩니다.

### **1. S (Scalar) 임베딩: 에너지 보존과 확률적 해석**

- **핵심 기법:** **Softplus + Learnable Scaling**
- **선택 이유:**
    - **정보 보존력:** 음수 입력이나 미세한 신호도 0으로 깎아버리지 않고 부드럽게 살려두어, 아주 미약한 기하학적 단서라도 모델이 판단 근거로 삼을 수 있게 함
    - **미분 특성:** 미분 시 Sigmoid 형태가 되어, 학습 과정에서 게이팅(Gating) 효과를 자연스럽게 유도합니다.

### **2. V (Vector) 임베딩: 방향의 순수성 유지**

- **핵심 기법:** **Linear Projection (No Bias)**
- **의미:** 벡터는 '어디로 향하는가'라는 방향 정보가 본질입니다.
- **설계 의도:** 활성화 함수로 방향을 왜곡하지 않고, Phase 1에서 추출된 그레이디언트와 텍스처 흐름을 고차원 채널로 확장하여 **벡터 특유의 선형성과 기하학적 성질**을 그대로 보존

### **3. B (Bivector) Generation: 회전과 닮음의 수치화**

- **핵심 기법:** **Sin/Cos Pair Output**
- **선택 이유:**
    - **회전의 정석:** 단순히 Tanh()로 값을 제한하는 대신, 수학적으로 완벽한 회전을 표현하는 `Sin/Cos` 쌍을 직접 생성.
    - **설계 의도:** 아핀 변환(Affine Transform)이나 줌(Zoom)이 발생한 이미지 매칭에서 압도적인 성능을 보일 것이라 예측
    - **기하학적 분리**
        - **sin 성분:** 두 특징 사이의 **'다름(외적/회전)'**을 나타내며 Bivector의 핵심 값이 됩니다.
        - **cos 성분:** 두 특징 사이의 **'닮음(내적/일치)'**을 나타내며 Scalar에 더해져 유사도 판정을 돕습니다.
        
        - **Unit Rotor 분리 (Magnitude Normalization):**

        $$\text{Rotor Magnitude} = |R| = \sqrt{\cos^2 + \sin^2 + \epsilon}$$

        $$\text{Unit Rotor} = \frac{R}{|R|} = \left( \frac{\cos}{|R|}, \frac{\sin}{|R|} \right)$$

        - **정규화:** 출력된 Rotor를 Unit Rotor ($R/|R|$, 순수 회전)과 Magnitude ($|R|$, 스케일) 로 **분리하여 제공**할 수 있도록 설계.
            
            → Phase 3에서 회전은 같은데 크기만 다른 경우를 명확히 구분
            
            **분리 이유:**
            1. **Unit Rotor ($R/|R|$)**: 순수 회전 방향만 표현 → Phase 3의 Path A (Rotation Invariant Matching)에서 사용
            2. **Magnitude ($|R|$)**: 스케일 정보만 표현 → Phase 3의 Path B (Scale Bias)에서 사용
            3. **기하학적 독립성**: 회전과 스케일을 분리하여 각각 독립적으로 처리 가능
            

---

## **📂  3: Geometric Transformer & Decoder**

이 단계는 Phase 2에서 준비된 **해상도별 S, V, B 꾸러미**를 입력받아, 이미지 간의 기하학적 대응 관계를 추론하고 초정밀 매칭 지도를 생성하는 최종 공정입니다.

### **1. 토큰화 (Tokenization)**

1. Input Alignment
    - Phase 2에서 넘어온 S,V,B의 차원이 제각각이므로 1X1 Conv를 통해 동일한 차원으로 투영한 후 결합
2. **Group Conv**
    - 정렬된 입력을 S그룹, V그룹, B그룹으로 나누어 Group=3인 Conv 적용하여 물리적 성질이 섞이지 않게 초기 특징 추출

### **2. 인코더 (Encoder)**

1. **Q, K, V 변환**
    1. **S,V,B 각각을 위한 3개의 병렬 Independent Linear →** 각 성분이 가진 고유한 기하학적 정체성을 유지하며 고차원 특징으로 투영
    
    | **입력 성분** | **생성되는 쿼리/키/값** | **주요 역할** |
    | --- | --- | --- |
    | **Scalar ($S$)** | $Q_S, K_S, V_S$ | 존재 유무 및 에너지 유사도 판단 |
    | **Vector ($V$)** | $Q_V, K_V, V_V$ | 방향 정렬 및 스케일 차이 계산 |
    | **Bivector ($B$)** | $Q_B, K_B, V_B$ | 회전 일관성 및 아핀 변환 대응 |
2. **위치 정보 주입**
    1. **Group Convolution 기반 CPE**를 사용하여 픽셀 단위가 아닌 '기하학적 덩어리' 단위로 위치를 파악
3. **Rotor-Scale Attention**
    - Transformer의 hidden state 뿐 아니라 Phase 2의 Rotor Tuple을 보조 입력으로 직접 받아 어텐션 점수 계산에 활용
    1. Path A
        - Phase 2에서 분리한 Unit Rotor (정규화된 순수 회전 방향, $R/|R|$) 참조
        - 학습된 Q,K의 코사인 유사도에 더해 물리적인 회전 방향의 일치도를 어텐션 점수에 반영 (Rotation Invariant Matching)
    2. Path B
        - Phase 2에서 분리한 Magnitude (벡터/Rotor  크기 정보, $|R|$)  참조
        - Q와 K 사이의 상대적인 스케일 비율 $\log{|K|} - \log{|Q|}$ 를 계산
        - 이 스케일 차이 값은 Attention Bias로 작용하며 V에 Injection 되어 매칭된 K가 원본 Q보다 몇 배 크거나 작은지에 대한 물리적 수치를 디코더로 전달
4. **Injection Fusion**
    1. Path A에 대해서만 어텐션 점수를 매김
    2. Path B에서 나온 스케일 차이를 Value에 직접 주입
        - 이로써 V는 K 본연의 모습 뿐만 아니라 Q 대비 크기 비율을 가짐
            
            Ex. "나보다 2배 큰 짝꿍"이라는 정보를 벡터에 담아 좌표 계산 시 반영되도록 함
            

        **Global Descriptor ($G \in \mathbb{R}^6$) 구성:**

        ```python
        # 코드: phase1.py lines 198-204
        v_shape = np.array([
            np.mean(edge_mag), np.std(edge_mag),           # [0-1] 엣지 분포
            np.mean(struct_energy), np.std(struct_energy), # [2-3] 구조 분포
            np.mean(texture), np.std(texture)              # [4-5] 밝기 분포
        ], dtype=np.float32)
        ```

        **MLP Gate 생성:**

        $$\text{Gate}_{global} = \sigma\left( \text{Linear}_{hidden\_dim} \circ \text{ReLU} \circ \text{Linear}_{6 \to hidden\_dim}(G) \right)$$

        **Scalar 조정:**

        $$S_{adjusted} = S_{base} \odot \text{Gate}_{global}$$

        **물리적 의미:**
        - 엣지가 선명한 이미지($\sigma_{edge}$ 높음) → Gate ↑ → Scalar 강조
        - 균일한 이미지($\sigma_{texture}$ 낮음) → Gate ↓ → 과적합 방지
            
5. **Geometric Descriptor Guidance**
    1. Fast Lane (독립 처리)
        - S,V,B가 각각 독립적인 Layer를 통과하여 고유 특징만 빠르게 추출 (채널 섞임 방지)
    2. Descriptor 생성
        - 각 성분에서 회전/방향에 상관없는 불변량($S, ||V||, ||B||$) 를 뽑아 3차원 요약 벡터(descriptor) 생성
    3. Gate Modulation
        - 요약 벡터를 MLP에 통과시켜 3개의 Gate 값($~~g_s, g_v, g_b~~$)을 얻어 각 Gate값들을 S,V,B에 곱해 볼륨(중요도)를 동적으로 조절
- Encoder Block의 `forward()`
    - 정규화: LayerNorm을 통해 평균을 잡아 기하학적 안정성을 유지

### **3. 디코더 (Decoder) — Coarse-to-Fine Transform Propagation**

인코더가 만든 맥락이 담긴 피라미드와 Phase 2의 원본 피라미드를 결합하여 최종 지도를 완성.
저해상도에서 추정한 Global Transform을 고해상도로 전파하여 "큰 변환 → 작은 잔차" 순으로 처리합니다.

#### **3.1 Global Transform Estimation (Coarsest Level)**

- **역할:** 가장 저해상도(Level N-1)에서 이미지 전체의 대략적인 변환 $W_{global}$을 먼저 추정

- **방법:** 
    1. Coarsest Level의 Phase 2 특징으로 Cross-Attention 수행
    2. Dense Rotor Map의 **공간 평균**을 계산하여 단일 Global Transform 행렬 생성
    3. 이 $W_{global}$은 "전체 이미지가 대략 몇 도 돌아갔고, 얼마나 이동했는지"를 나타냄

- **수식:**
    $$
    W_{global} = \text{Avg}_{p \in \Omega_{coarse}} \begin{bmatrix} \cos\theta(p) & -\sin\theta(p) & dx(p) \\ \sin\theta(p) & \cos\theta(p) & dy(p) \end{bmatrix}
    $$

#### **3.1.1 Transform-Guided Feature Warping**

- **핵심 아이디어:** 다음 레벨로 내려가기 전, 이전 레벨에서 추정한 변환으로 이미지 A의 특징을 미리 워핑

- **효과:** 고해상도에서는 이미 대충 맞춰진 상태에서 작은 잔차(Residual)만 추정하면 됨

- **과정:**
    1. $W_{prev}$를 현재 해상도에 맞게 업샘플링
    2. `F.affine_grid` + `F.grid_sample`로 Phase 2 특징(A)을 워핑
    3. 워핑된 A와 원본 B 사이의 잔차 변환 $\Delta W$만 Cross-Attention으로 추정
    4. 최종 변환: $W_{current} = \Delta W \circ W_{prev}$ (변환의 합성)

#### **3.2 Cross-Attention**

- 이미지 A와 B를 대조하여 "A의 이 지점이 B로 가기 위한 회전/변환 값(Rotor)"을 추출
- 고해상도 레벨에서는 워핑된 A와 B를 비교하므로, 추정해야 할 변환량이 작아져 정확도 향상
    - Dense Rotor Regression Head:
        - **(B, 4, H, W) 형태의 Pixel-wise Dense Rotor Map** 출력: $(\cos, \sin, dx, dy)$
        - **이유:** 원근감이 있거나 비평면 물체인 경우, 픽셀마다 변환량이 다름

2. **Clifford Interpolation**
    - 이전 단계의 저해상도 결과를 업샘플링할 때, 스칼라는 부드럽게 늘리고 벡터의 방향성을 보존하며 회전 보간을 수행
        - Scalar: 부드러운 선형 보간
        - Vector: 방향성을 유지하며 보간 (일반 Bilinear 허용)
        - Rotor(Bivector): 회전 성질 보존을 위해 보간 후 정규화(NLERP) 수행
3. **Geometric Skip-Connection**
    1. **정렬:** 3.1(Cross Attention)에서 예측한 Dense Rotor Map($\cos,\sin,dx,dy$)을 이용해 픽셀별 Affine Grid 생성
        - Phase 2의 원본 S,V,B 피쳐를 이 Grid에 맞춰 샘플링 (`grid_sample`) 하여 현재 디코더의 시점에 맞게 뒤틈
    2. **융합:** 업샘플링된 문맥과 정렬된 원본 디테일을 Concat (채널 방향으로 합침)
        - Gated Injection 방식→이 상황에서는 스케일 정보를 얼마나 믿을까?를 입력 데이터로부터 판단
4. **Feature Map 생성**
    1. CNN이 합쳐진 특징을 훑으며 노이즈를 제거 & 정교한 S,V,B 출력
    2. 다음 상위 레벨로 전달되거나 최종 단계일 경우 MPC 에너지를 계산할 최종 Feature map이 됨
        1. 해당 Feature Map은 에너지 포텐셜 (Scalar Field)와 벡터 필드를 포함하는 다채널 구조로 출력

---
## **📂 Phase 4: Dual-Adaptive Recurrent Refinement**

Phase 3의 단일 추정(Single-Shot)으로는 큰 변환(>15°, >20px)에서 정확도가 떨어지는 문제를 해결하기 위해,
**이중 적응형(Dual-Adaptive) 전략**과 **경량 순환 신경망(Mini-GRU)**을 결합한 능동적 정제 단계입니다.

### **핵심 아이디어: Smart Traversal with Momentum**

$$
W_{final} = \text{MiniGRU}(F_{selected}^{level}) \circ \cdots \circ \text{MiniGRU}(F_{selected}^{level})
$$

**3가지 핵심 메커니즘:**
1. **Level Selection (거시적 선택):** 오차 크기에 따라 Coarse(Level 3) ↔ Fine(Level 0) 피라미드 레벨 선택
2. **Feature Selection (미시적 선택):** 오차 타입에 따라 S(Scalar) / V(Vector) / B(Bivector) 특징 선택
3. **Recurrent Memory (순환 기억):** Mini-GRU가 이전 수정 방향(Momentum)을 유지하여 진동 없이 수렴

---

### **1. Dual-Adaptive Routing (이중 선택 전략)**

매 반복마다 **"어느 레벨에서, 어떤 특징을 사용할 것인가?"**를 동적으로 결정합니다.

#### **A. Level Selection (피라미드 레벨 선택)**

**오차 크기**를 기준으로 적절한 수용 범위(Receptive Field)를 가진 레벨을 선택합니다.

| 오차 범위 | 선택 레벨 | 수용 영역 | 목적 |
|---------|----------|---------|------|
| > 30px | **Level 3** (Global) | 넓음 (32px) | 큰 변환 포착 |
| 10~30px | **Level 2** (Structural) | 중간 (16px) | 구조적 정렬 |
| 5~10px | **Level 1** (Local) | 좁음 (8px) | 세부 매칭 |
| < 5px | **Level 0** (Fine) | 픽셀 단위 | 미세 조정 |

**오차 측정 (Error Diagnosis):**

$$
E_{pos} = \text{Mean}(|SDF_A(W_{curr}(p)) - SDF_B(p)|) \quad \text{[위치 오차]}
$$

$$
E_{angle} = 1 - \text{Mean}(\cos(\theta_{residual})) \quad \text{[방향 오차]}
$$

- **[Hyperparameter]** Level 전환 임계값: `[30, 10, 5]` px

#### **B. Feature Selection (특징 선택)**

**오차 타입**을 기준으로 가장 관련 있는 Clifford 성분을 선택합니다.

| 오차 타입 | 선택 특징 | 차원 | 역할 |
|---------|----------|-----|-----|
| 위치 불일치<br>($E_{pos}$ 지배적) | **S (Scalar)** | (B, 64, H, W) | 텍스처 매칭, SDF 정렬 |
| 방향 불일치<br>($E_{angle}$ 지배적) | **V (Vector)** | (B, 64, 2, H, W) | 그래디언트 방향 정렬 |
| 스케일/회전 불일치<br>(둘 다 큼) | **B (Bivector)** | (B, 64, H, W) | Rotor 보정 |

**선택 기준:**
```python
if E_pos > 15.0:
    selected = S  # 위치부터 맞춤
elif E_angle > 0.25:  # ≈ 14.3°
    selected = V  # 방향 정렬
else:
    selected = B  # 미세 회전 보정
```

- **[Hyperparameter]** 특징 선택 임계값: `E_pos = 15.0` px, `E_angle = 0.25` rad

---

### **2. Mini-ConvGRU (경량 순환 엔진)**

IGEV의 Full ConvGRU를 **1/4 크기로 경량화**하고, Correlation Volume을 제거하여 메모리 효율을 극대화했습니다.

#### **A. 구조 (Minimal-GRU)**

$$
\begin{aligned}
z_k &= \sigma(\text{Conv}_{3x3}([h_{k-1}, E_{diff}])) \quad \text{[Update Gate: 16채널]} \\
\tilde{h}_k &= \tanh(\text{Conv}_{3x3}([F_{selected}, E_{diff}])) \quad \text{[Candidate State]} \\
h_k &= (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k \quad \text{[Linear Interpolation]} \\
\Delta W_k &= \text{Head}_{2\text{-layer}}(h_k) \quad \text{[16→8→4 채널]}
\end{aligned}
$$

**주요 개선점:**
- **Reset Gate 제거:** Minimal-GRU 구조로 파라미터 50% 감소
- **Correlation Volume 제거:** Difference Map ($E_{diff} = |A' - B|$)으로 대체하여 메모리 절약
- **16채널 Hidden State:** 원본 IGEV(64채널) 대비 75% 메모리 절감

#### **B. Level Transfer (해상도 전환 시)**

피라미드 레벨이 바뀔 때(예: Level 3 → Level 2), Hidden State의 해상도를 조정합니다.

$$
h_k^{l} = \text{MiniGRU}(\text{Upsample}(h_{k-1}^{l+1}), [E_{diff}^{l}, F_{selected}^{l}])
$$

**구현:**
```python
if h_prev.shape[-2:] != target_size:
    h_prev = F.interpolate(h_prev, size=target_size, mode='bilinear')
```

- **목적:** 저해상도(Coarse)에서 학습한 "큰 흐름"을 고해상도(Fine)로 전달
- **효과:** 각 레벨이 독립적으로 시작하는 것보다 **2배 빠른 수렴**

---

### **3. 반복 정제 루프 (Iteration Strategy)**

실제 시나리오별 동작 흐름입니다.

| Iter | 오차 상태 | 선택 레벨 | 선택 특징 | GRU 동작 | 목표 |
|------|----------|----------|----------|---------|-----|
| **1** | 45px, 20° | Level 3 | **V + B** | 큰 회전 감지 → Momentum 축적 | 20px |
| **2** | 20px, 5° | Level 2 | **S** | 이전 방향 유지 + 텍스처 매칭 | 8px |
| **3** | 8px, 1° | Level 1 | **S** | 디테일 엣지 정렬 | 3px |
| **4** | 3px, 0.3° | Level 1 | **S** | 미세 조정 | 1-2px |

**[Hyperparameter]** `num_iterations = 4` (레벨 수와 동일)

**시각화 예시:**
```
[Iter 1] Error=45.2px | Level=3 (Global) | Feature=V+B (Rotation) → 22.1px
[Iter 2] Error=22.1px | Level=2 (Struct)  | Feature=S (Texture)   → 9.4px
[Iter 3] Error=9.4px  | Level=1 (Local)   | Feature=S (Edge)      → 3.8px
[Iter 4] Error=3.8px  | Level=1 (Fine)    | Feature=S (Detail)    → 1.5px ✓
```

---

### **4. 종료 조건 및 안전장치**

#### **A. 조기 종료 (Convergence)**

다음 조건 **중 하나라도** 만족하면 즉시 종료:

1. **변화량 수렴:**
   $$
   \|\Delta W_k - I\|_F < \epsilon_{\text{conv}}
   $$
   - **[Hyperparameter]** $\epsilon_{\text{conv}} = 0.005$ (Frobenius Norm)

2. **오차 충분히 작음:**
   $$
   E_{curr} < \epsilon_{\text{target}}
   $$
   - **[Hyperparameter]** $\epsilon_{\text{target}} = 3.0$ px (Phase 5가 해결 가능한 범위)

#### **B. 발산 방지 (Bounded Safety Lock)**

**조건:** 새로운 변환이 이전보다 5% 이상 악화된 경우

$$
E_{next} > E_{curr} \times (1 + \alpha)
$$

- **[Hyperparameter]** $\alpha = 0.05$ (5% Tolerance)

**대응 전략 (3단계):**

1. **1차 시도: Update Rejection**
```python
   if E_next > E_curr * 1.05:
       W_accum = W_prev  # 이전 상태로 롤백
       continue
```

2. **2차 시도: GRU Reset + LR Decay**
```python
   if consecutive_rejections >= 2:
       mini_gru.reset()           # Hidden State 초기화
       learning_rate *= 0.5       # 학습률 반감
       print("[Recovery] GRU Reset + LR Decay")
```

3. **3차 시도: 강제 종료**
```python
   if consecutive_rejections >= 3:
       print("[Emergency] Divergence detected, aborting...")
       break
```

**효과:**
- Local Minima(작은 언덕)에 갇혔을 때 **GRU Reset**으로 탈출 가능
- 학습률을 낮춰 **과도한 변화(Overshoot)**를 방지
- Phase 5의 강력한 수렴 능력(Basin ≈ 10px)을 신뢰하므로, 약간의 악화는 허용

---

### **7. 하이퍼파라미터 요약**

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|-----|------|
| `num_iterations` | **4** | 2~6 | 최대 반복 횟수 |
| `gru_hidden_dim` | **16** | 8~32 | GRU Hidden State 차원 |
| `convergence_threshold` | **0.005** | 0.001~0.01 | 수렴 판정 임계값 |
| `target_error_px` | **3.0** | 1.0~5.0 | 목표 오차 (Phase 5 이관 기준) |
| `tolerance_alpha` | **0.05** | 0.03~0.1 | 발산 방지 허용 오차 (5%) |
| `level_thresholds` | **[30, 10, 5]** | - | Level 전환 임계값 (px) |
| `feature_thresholds` | **[10.0, 0.1]** | - | S/V/B 선택 임계값 |


---
## **📂 Phase 5: 기하학적 에너지 기반 MPC 정제 — 추론 단계에서만**

Phase 5는 딥러닝이 예측한 매칭 지도를 바탕으로 물리적인 에너지 함수를 최소화하여 **0.1 픽셀 단위의 초정밀 정렬**을 달성하는 단계입니다.

### **1. 전역 필터링 및 초기화**

- **역할:** 최적화 연산이 엉뚱한 곳에서 시작하지 않도록 기준점을 잡아줌
- **전역 필터링:** Phase 2의 평균 Rotor(Sin/Cos)를 비교해 이미지 전체가 대략 몇 도 돌아갔는지 파악하여 터무니없는 후보군을 제거
- **$W_0$ 설정:** 평균 Rotor(회전)와 벡터 크기 비율(줌)을 결합하여 초기 변환 행렬 **$W_0$**를 생성
    - "대략 30도 돌아갔고 1.2배 커졌다"는 사실을 알고 최적화를 시작하므로 수렴 속도가 비약적으로 빨라짐

### **2. 지역 탐색 (Priority Search)**

- **역할:** "어디부터 정밀하게 맞출 것인가?"라는 **우선순위 지도**를 만듭니다.
- **방법:** Phase 3에서 배운 **Group Conv 특징**과 지역적 Rotor 분산(Variance)을 결합
    - 회전 정보가 일관되고 기하학적 덩어리가 뚜렷한 구역(예: 건물의 모서리)에 높은 가중치를 주어, 신뢰도가 높은 지역부터 자석처럼 딱딱 들어맞게 유도합니다.

### **3. 에너지 평면 생성**

이 시스템의 핵심인 **에너지 함수**입니다. S, V, B 세 가지 성분을 물리적으로 결합하여 오차를 계산

$$
E_{total} = \frac{1}{N} \sum_{p} \left( g_s(p) \cdot E_{scalar}(p) + g_v(p) \cdot E_{vector}(p) + g_b(p) \cdot E_{bivector}(p) \right)
$$

- **$E_{scalar}$ (에너지/SDF):** Softplus로 정제된 SDF 값의 차이를 계산 (→ 미분값이 매끄러워 최적화 엔진이 '골짜기'를 타고 내려가기 좋음)
- **$E_{vector}$ (방향/흐름):** 변환(W) 후에도 벡터의 방향이 일치하는지 확인 (→ 이미지가 회전했다면 벡터도 그만큼 돌아가야 한다는 **방향 보존성**을 강제)
- **$E_{bivector}$ (Rotor 일관성):** 단순히 위치만 맞는 게 아니라, 해당 지점의 **지역적인 회전/줌 상태**가 전체 변환 행렬과 기하학적으로 일치하는지 봄
- 회전하여 검정색으로 잘린 영역에 대해서는 Loss X

### **4. 기하학적 게이트 가중 최적화 (Gate-Guided Refinement)**

- **핵심:** Phase 3(인코딩 과정 중)의 **Geometric Descriptor Guidance**에서 나온 3개의 Gate 값($~~g_s, g_v, g_b~~$)을 최적화 가중치로 직접 사용
- **지능적 최적화:**
    - 엣지가 선명한 곳은 $g_v$(Vector)를 높여 방향 정밀도를 높입니다.
    - 텍스처가 복잡한 곳은 $g_s$(Scalar)를 높여 픽셀 일치도를 높입니다.
        - 모델이 "이 구역은 벡터 정보가 믿을만해!"라고 판단한 정보를 MPC가 적극 수용하여 루프를 돌림으로써, 단순 계산보다 훨씬 견고한 정제가 가능

---

## **📂 5: 통합 기하학적 손실 함수 (Unified Geometric Loss)**

모델의 최종 학습 목표는 아래의 **단일 통합 수식**을 최소화하는 것입니다.

$$L_{total} = \alpha \sum_{p \in \Omega} \underbrace{\left( L_{s}(p) + L_{v\_local}(p) + L_{b\_local}(p) \right)}_{\text{Geometric Accuracy (Local-Aware)}} + \beta \underbrace{\left( \lambda_c L_{\text{SmoothL1}} + \lambda_s L_{\text{SDF-Photo}} \right)}_{\text{Final Consistency}} + \gamma \underbrace{\left( L_{convergence} + L_{multi\_scale} \right)}_{\text{Iterative Stability}}$$

### **1. Geometric Accuracy (기하학적 정밀도)**

이미지 A와 정답 변환($W_{GT}$)으로 되돌린 이미지 B의 특징들이 물리적으로 일치하는지 검사

- **$L_s$ (뼈대 일치):**
    
    $$  L_s(p) = \| S_A(p) - S_B(W_{GT}(p)) \|^2$$
    
    - **의미:** Softplus로 살려낸 SDF와 에너지가 정답 위치에서 정확히 겹쳐야 함
- **$L_v$ (방향 정렬)**
    
    $$  L_{v\_local}(p) = \| V_A(p) - \underbrace{\mathcal{R}_{loc}(W_{GT}, p)}_{\text{Jacobian Rotation}} \cdot V_B(W_{GT}(p)) \|^2$$
    
    - 단순히 전체 행렬 $W_{GT}$를 곱하는 것이 아니라, $W_{GT}$**의 Jacobian(미분값)을 통해 각 픽셀 위치에서의 '국소 회전량(Local Rotation)'을 계산**하여 적용.
    - **의미:** 이미지가 회전했다면, 그 안의 엣지(V)도 그 각도만큼 물리적으로 회전했음을 학습합니다.
- **$L_b$ (회전 일관성)**
    
    $$  L_{b\_local}(p) = \| \text{Rotor}_A(p) - \mathcal{R}_{loc}(W_{GT}, p) \cdot \text{Rotor}_B(W_{GT}(p)) \|^2$$
    
    - $W_{GT}$에서 유도된 **지역적 회전(Local Rotor)** 정보와 비교
    - **의미:** 지역적인 Sin/Cos 정보가 전체 변환 행렬(W)의 회전량과 기하학적으로 호응해야 합니다.

### **2. Final Consistency (뒤틀림 일관성)**

모델이 예측한 $W^*$가 수학적으로 얼마나 견고한지 증명합니다.

- **$L_{coord}$ (모서리 거리) — Smooth L1**을 사용하여 예측된 $W^*$로 변환한 네 모서리 좌표와 정답 좌표 사이의 거리를 줄임
    
    $$  L_{coord}(W_{GT}, W^*) = \frac{1}{4} \sum_{k=1}^{4} \rho \left( \mathcal{T}(p_k; W_{GT}) - \mathcal{T}(p_k; W^*) \right)$$
    
    - $\mathcal{T}(p; W)$ : 좌표 p를 호모그래피 행렬 W를 이용해 변환하는 함수
    - $\rho(x)$ : 거리 함수
        
        $$  \rho(x) = 
          \begin{cases} 
          0.5 x^2 / \beta & \text{if } |x| < \beta \\
          |x| - 0.5 \beta & \text{otherwise}
          \end{cases}$$
        
- **$L_{sdf\_photo}$ (SDF 기반 복원) —** 복원된 이미지 $\hat{A}$의 SDF가 원본 A의 SDF와 일치하는가 확인
    
    $$  L_{sdf\_photo}={\sum_{p \in \Omega} \| SDF_A(p) - SDF_{\hat{A}}(p; W^*) \|^2}$$
    

### **3. Iterative & Multi-Scale Constraint (반복 및 스케일 안정성) — [보완] 신규**

반복 정제(Iterative Refinement)와 다중 해상도(Multi-Scale) 학습 과정에서 모델이 발산하지 않고 올바른 방향으로 수렴하도록 강제하는 제약 조건입니다.

- **$L_{convergence}$ (수렴 유도 손실)**
    
    각 반복 단계($k$)에서 추정된 잔차 변환 $\Delta W^{(k)}$가 점차 Identity ($I$) 에 가까워지도록 유도하여, 불필요한 진동을 억제합니다.
    
    $$  L_{convergence} = \sum_{k=2}^{K} w_k \cdot \| \Delta W^{(k)} - I \|_F^2$$
    
    - $w_k = k / K$: 반복 횟수가 증가할수록 가중치를 높여(Linear Warm-up), 후반부에는 큰 변화 대신 미세 조정만 수행하도록 강제합니다.
    - $\| \cdot \|_F$: Frobenius Norm (행렬 원소 간 차이의 제곱합)
    - **의미:** "첫 번째 반복에서 큰 틀을 잡고(Coarse), 이후에는 얌전히 다듬기만 해라(Fine)"는 지침을 줍니다.
- **$L_{multi\_scale}$ (다중 스케일 일관성 손실)**
    
    저해상도(Coarse)에서 추정한 변환이 고해상도(Fine)에서도 유효하도록, 스케일 간의 예측값 일관성을 유지합니다.
    
    $$  L_{multi\_scale} = \sum_{l=0}^{L-1} \| W^{(l)} - \text{Upsample}(W^{(l+1)}) \|^2$$
    
    - **의미:** "작은 이미지에서 30도 돌렸으면, 큰 이미지에서도 30도 돌아가야 한다"는 물리적 일관성을 보장합니다. 이는 Coarse-to-Fine 전략의 허리 역할을 합니다.

---

## 📌 6: Code Implementation Notes (v5 / Code-Architecture Sync)

> 이 섹션은 **architecture.md(개념 설계)** 와 **현재 코드 구현(phase1~phase4_2, losses, fine_tune/fast_finetune)** 사이의 차이를 없애기 위해,  
> 코드에 존재하지만 본문에 상세히 없던 구현 포인트/하이퍼파라미터를 문서화한 **"Implementation Addendum"** 입니다.

### 6.1 공통 하이퍼파라미터 (코드 기본값)

- `HIDDEN_DIM = 48`  *(Phase 2 Clifford Embedding 기본 채널 수)*
- `FEATURE_DIM = 144 (= 3 × 48)` *(Phase 3 Transformer 내부 S/V/B concat 특징 차원)*
- `NUM_ENCODER_LAYERS = 3`, `NUM_ATTENTION_HEADS = 4`
- `pyramid levels = 5` *(fine_tune.py 기본 학습 설정: 큰 회전(±60°) 대응을 위해 4→5로 확장)*

### 6.2 Phase 2 구현 메모

- **Rotor 생성 입력 채널 확장 (5채널):**  
  `(dx, dy, fx, fy)`(Phase1의 V1/V2) 에 더해, Phase1에서 계산한 **Bivector 후보 `bivector = v1 ∧ v2`** 를 추가하여  
  `rotor_in = concat([v_in(4ch), b_in(1ch)])` 형태로 Rotor Conv에 투입합니다.
- **Scalar 업데이트 방식(s_mixer):**  
  Cos 파트를 scalar embedding에 단순 가산하기보다, `concat([s_emb, cos_part]) → 1×1 Conv(s_mixer)` 로 **혼합**하여  
  과도한 덮어쓰기(override)와 스케일 폭주를 줄입니다.

### 6.3 Phase 3 구현 메모 (v5 메모리/속도 최적화 포함)

- **Chunked Attention (RTX 3090 24GB 대응):**  
  `SAFE_N_LIMIT`, `SAFE_ELEMENTS` 기준으로 픽셀 수(N)가 큰 레벨에서 attention을 chunk 단위로 수행합니다.
- **Optional High-Res Attention Skip:**  
  `HIGH_RES_SKIP_LEVEL` 을 통해 level 0~1(고해상도)에서 self/cross-attention을 생략하고  
  **이전 레벨 rotor/context를 업샘플링**하여 속도/메모리를 확보할 수 있습니다.
- **Transform-Guided Warping + Residual Composition:**  
  coarse 레벨에서 얻은 `W_prev`로 A 특징을 먼저 warp한 뒤, 다음 레벨에서 잔차 `ΔW`만 추정하여  
  `W_current = ΔW ∘ W_prev` 로 누적합니다.
- **Skip-Connection 실제 적용 강화:**  
  rotor_map 기반 warp → gated injection → refinement block(ResBlock)을 **실제 forward path에 반영**합니다.
- **Phase 5 사용을 위한 Gate Map 노출:**  
  Phase3 내부에서 계산 가능한 `g_s, g_v, g_b` gate map을 **출력 dict에 포함**하여 Phase4 MPC 에너지 가중치로 사용 가능합니다.

### 6.4 Phase 4.2 구현 메모

- **Similarity Transform 파라미터화 최적화:**  
  `(theta, tx, ty, log_scale)` 를 Adam으로 최적화한 뒤 2×3 affine로 재구성합니다.
- **Valid Mask (검정 잘림 영역 Loss 제외):**  
  warp 과정에서 in-bounds mask를 생성하고, out-of-bounds 픽셀은 energy 계산에서 제외합니다.
- **Priority Map 자동 생성 옵션:**  
  priority_map이 주어지지 않으면,  
  - rotor_map의 **지역 회전 분산(variance)**  
  - mpc_map의 **벡터장 크기(magnitude)**  
  를 결합하여 priority 가중치를 만들 수 있습니다.

### 6.5 Loss / Training 구현 메모

- losses.py 의 `UnifiedGeometricLoss` 는 architecture.md §5의 큰 구조(Geo + Final + Iter)를 유지하면서,  
  학습 안정성을 위해 아래 항목을 추가/보강합니다:
  - `L_angle` (pred angle ↔ gt angle 직접 loss)
  - `L_pixel` (scalar feature consistency)
  - `L_rotation_invariant` (±60° 구간에서 회전 불변성/대칭성 강제)
- `normalize_rotor_output()` 헬퍼로 cos/sin unit-normalization을 표준화합니다.

---

## 📌 7: Training & Fine-Tuning Pipeline (fine_tune.py / fast_finetune.py)

### 7.1 Dataset: GeometricRotationDataset

- 입력 이미지 B에 대해 랜덤 회전(커리큘럼 범위) + (옵션) 스케일 jitter 를 적용하여 A를 생성합니다.
- GT는 `A → B` 로 되돌리는 **역변환 affine** 를 계산하여 `W_gt (normalized coord)` 로 제공합니다.
- Phase1 전처리를 통해 `levels=5` 피라미드(raw scalar/vector/bivector 등)를 생성합니다.

### 7.2 CurriculumScheduler (3-Stage)

- `CURRICULUM_STAGES` 로 에폭 구간별 회전 범위를 점진적으로 확장(또는 이동)합니다.
- Dataset은 `set_epoch(epoch)` 에서 현재 회전 범위를 갱신합니다.

### 7.3 Dataloader / Collate

- 피라미드 레벨별 dict 구조를 유지한 채 batch stacking을 수행합니다.
- phase2 embedder는 (HWC)/(BHWC) 모두 처리할 수 있도록 to_tensor를 구현합니다.

### 7.4 Training Loop 핵심

- Mixed Precision(cuDNN/AMP) + Gradient Accumulation 으로 3090 24GB 환경에서 안정적으로 학습합니다.
- Phase2 → Phase3 forward 결과에서 `W_pred` 를 구성하고, `UnifiedGeometricLoss` 로 supervised fine-tuning 합니다.
- MetricTracker로 각도 오차/픽셀 오차(코너 기준)를 추적합니다.

### 7.5 fast_finetune.py

- 빠른 실험을 위해 fine_tune의 일부 상수(에폭, 샘플 수, 회전 범위 등)를 monkey patch하여 재사용합니다.
- 핵심 학습 루프 로직은 fine_tune.py 와 동일한 손실/지표 구조를 유지합니다.
