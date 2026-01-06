# 아키텍쳐 설계

## **📂 Phase 1: Geometry Raw Data 추출 (Physical Raw Data Extraction)**

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

## **📂 Phase 2: Clifford Embedding & Pyramid 생성**

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
        - **정규화:** 출력된 Rotor를 Unit Rotor ($R/|R|$, 순수 회전)과 Magnitude ($|R|$, 스케일) 로 **분리하여 제공**할 수 있도록 설계.
            
            → Phase 3에서 회전은 같은데 크기만 다른 경우를 명확히 구분
            

---

## **📂 Phase 3: Geometric Transformer & Decoder**

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
            
5. **Geometric Descriptor Guidance**
    1. Fast Lane (독립 처리)
        - S,V,B가 각각 독립적인 Layer를 통과하여 고유 특징만 빠르게 추출 (채널 섞임 방지)
    2. Descriptor 생성
        - 각 성분에서 회전/방향에 상관없는 불변량($S, ||V||, ||B||$) 를 뽑아 3차원 요약 벡터(descriptor) 생성
    3. Gate Modulation
        - 요약 벡터를 MLP에 통과시켜 3개의 Gate 값($~~g_s, g_v, g_b~~$)을 얻어 각 Gate값들을 S,V,B에 곱해 볼륨(중요도)를 동적으로 조절
- Encoder Block의 `forward()`
    - 정규화: LayerNorm을 통해 평균을 잡아 기하학적 안정성을 유지

### **3. 디코더 (Decoder)**

인코더가 만든 맥락이 담긴 피라미드와 Phase 2의 원본 피라미드를 결합하여 최종 지도를 완성

1. **Cross-Attention**
    - 이미지 A와 B를 대조하여 "A의 이 지점이 B로 가기 위한 회전/변환 값(Rotor)"을 추출
        - Dense Rotor Regression Head:
            - 단순히 이미지 전체의 글로벌 회전값을 뽑는 것이 아니라, **(B, 2, H, W) 형태의 Pixel-wise Dense Rotor Map**을 출력해야 함
            - **이유:** 원근감이 있거나 비평면 물체인 경우, 픽셀마다 회전/스케일 변환량이 다르므로 **국소적인(Local) 변환 정보**가 필수적
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

## **📂 Phase 4: 기하학적 에너지 기반 MPC 정제**

Phase 4는 딥러닝이 예측한 매칭 지도를 바탕으로 물리적인 에너지 함수를 최소화하여 **0.1 픽셀 단위의 초정밀 정렬**을 달성하는 단계입니다.

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

## **📂 Phase 5: 통합 기하학적 손실 함수 (Unified Geometric Loss)**

모델의 최종 학습 목표는 아래의 **단일 통합 수식**을 최소화하는 것입니다.

$$
L_{total} = \alpha \sum_{p \in \Omega} \underbrace{\left( L_{s}(p) + L_{v\_local}(p) + L_{b\_local}(p) \right)}_{\text{Geometric Accuracy (Local-Aware)}} + \beta \underbrace{\left( \lambda_c L_{\text{SmoothL1}} + \lambda_s L_{\text{SDF-Photo}} \right)}_{\text{Final Consistency}}
$$

### **1. Geometric Accuracy (기하학적 정밀도)**

이미지 A와 정답 변환($W_{GT}$)으로 되돌린 이미지 B의 특징들이 물리적으로 일치하는지 검사

- **$L_s$ (뼈대 일치):**
    
    $$
    L_s(p) = \| S_A(p) - S_B(W_{GT}(p)) \|^2
    $$
    
    - **의미:** Softplus로 살려낸 SDF와 에너지가 정답 위치에서 정확히 겹쳐야 함
- **$L_v$ (방향 정렬)**
    
    $$
    L_{v\_local}(p) = \| V_A(p) - \underbrace{\mathcal{R}_{loc}(W_{GT}, p)}_{\text{Jacobian Rotation}} \cdot V_B(W_{GT}(p)) \|^2
    $$
    
    - 단순히 전체 행렬 $W_{GT}$를 곱하는 것이 아니라, $W_{GT}$**의 Jacobian(미분값)을 통해 각 픽셀 위치에서의 '국소 회전량(Local Rotation)'을 계산**하여 적용.
    - **의미:** 이미지가 회전했다면, 그 안의 엣지(V)도 그 각도만큼 물리적으로 회전했음을 학습합니다.
- **$L_b$ (회전 일관성)**
    
    $$
    L_{b\_local}(p) = \| \text{Rotor}_A(p) - \mathcal{R}_{loc}(W_{GT}, p) \cdot \text{Rotor}_B(W_{GT}(p)) \|^2
    $$
    
    - $W_{GT}$에서 유도된 **지역적 회전(Local Rotor)** 정보와 비교
    - **의미:** 지역적인 Sin/Cos 정보가 전체 변환 행렬(W)의 회전량과 기하학적으로 호응해야 합니다.

### **2. Final Consistency (뒤틀림 일관성)**

모델이 예측한 $W^*$가 수학적으로 얼마나 견고한지 증명합니다.

- **$L_{coord}$ (모서리 거리) — Smooth L1**을 사용하여 예측된  $W^*$로 변환한 네 모서리 좌표와 정답 좌표 사이의 거리를 줄임
    
    
    $$
    L_{coord}(W_{GT}, W^*) = \frac{1}{4} \sum_{k=1}^{4} \rho \left( \mathcal{T}(p_k; W_{GT}) - \mathcal{T}(p_k; W^*) \right)
    $$
    
    - $\mathcal{T}(p; W)$ : 좌표 p를 호모그래피 행렬 W를 이용해 변환하는 함수
    - $\rho(x)$ : 거리 함수
        
        $$
        \rho(x) = 
        \begin{cases} 
        0.5 x^2 / \beta & \text{if } |x| < \beta \\
        |x| - 0.5 \beta & \text{otherwise}
        \end{cases}
        $$
        
- **$L_{sdf\_photo}$ (SDF 기반 복원) —** 복원된 이미지 $\hat{A}$의 SDF가 원본 A의 SDF와 일치하는가 확인
    
    
    $$
    L_{sdf\_photo}={\sum_{p \in \Omega} \| SDF_A(p) - SDF_{\hat{A}}(p; W^*) \|^2}
    $$