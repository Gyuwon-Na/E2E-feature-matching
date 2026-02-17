# =============================================================================
# Architecture.md Line Mapping (원본 architecture.md 기준)
# - 각 라인이 코드의 어느 부분에 해당하는지 anchor로 명시합니다.
# - (요청사항) architecture.md 내용을 스킵/생략하지 않기 위해, 빈 줄도 포함합니다.
# =============================================================================
# [ARCH L0237] (Phase35Refiner / IterativeRefinementLoop.forward) ## **📂 Phase 3.5: Dual-Adaptive Recurrent Refinement**
# [ARCH L0238] (Phase35Refiner / IterativeRefinementLoop.forward) 
# [ARCH L0239] (Phase35Refiner / IterativeRefinementLoop.forward) Phase 3의 단일 추정(Single-Shot)으로는 큰 변환(>15°, >20px)에서 정확도가 떨어지는 문제를 해결하기 위해,
# [ARCH L0240] (Phase35Refiner / IterativeRefinementLoop.forward) **이중 적응형(Dual-Adaptive) 전략**과 **경량 순환 신경망(Mini-GRU)**을 결합한 능동적 정제 단계입니다.
# [ARCH L0241] (Phase35Refiner / IterativeRefinementLoop.forward) 
# [ARCH L0242] (Phase35Refiner / IterativeRefinementLoop.forward) ### **핵심 아이디어: Smart Traversal with Momentum**
# [ARCH L0243] (Phase35Refiner / IterativeRefinementLoop.forward) 
# [ARCH L0244] (Phase35Refiner / IterativeRefinementLoop.forward) $$
# [ARCH L0245] (Phase35Refiner / IterativeRefinementLoop.forward) W_{final} = \text{MiniGRU}(F_{selected}^{level}) \circ \cdots \circ \text{MiniGRU}(F_{selected}^{level})
# [ARCH L0246] (Phase35Refiner / IterativeRefinementLoop.forward) $$
# [ARCH L0247] (Phase35Refiner / IterativeRefinementLoop.forward) 
# [ARCH L0248] (Phase35Refiner / IterativeRefinementLoop.forward) **3가지 핵심 메커니즘:**
# [ARCH L0249] (DualAdaptiveSelector.select_level) 1. **Level Selection (거시적 선택):** 오차 크기에 따라 Coarse(Level 3) ↔ Fine(Level 0) 피라미드 레벨 선택
# [ARCH L0250] (DualAdaptiveSelector.select_feature) 2. **Feature Selection (미시적 선택):** 오차 타입에 따라 S(Scalar) / V(Vector) / B(Bivector) 특징 선택
# [ARCH L0251] (DualAdaptiveSelector.select_feature) 3. **Recurrent Memory (순환 기억):** Mini-GRU가 이전 수정 방향(Momentum)을 유지하여 진동 없이 수렴
# [ARCH L0252] (DualAdaptiveSelector.select_feature) 
# [ARCH L0253] (DualAdaptiveSelector.select_feature) ---
# [ARCH L0254] (DualAdaptiveSelector.select_feature) 
# [ARCH L0255] (DualAdaptiveSelector (level/feature selection)) ### **1. Dual-Adaptive Routing (이중 선택 전략)**
# [ARCH L0256] (DualAdaptiveSelector (level/feature selection)) 
# [ARCH L0257] (DualAdaptiveSelector (level/feature selection)) 매 반복마다 **"어느 레벨에서, 어떤 특징을 사용할 것인가?"**를 동적으로 결정합니다.
# [ARCH L0258] (DualAdaptiveSelector (level/feature selection)) 
# [ARCH L0259] (DualAdaptiveSelector.select_level) #### **A. Level Selection (피라미드 레벨 선택)**
# [ARCH L0260] (DualAdaptiveSelector.select_level) 
# [ARCH L0261] (DualAdaptiveSelector.select_level) **오차 크기**를 기준으로 적절한 수용 범위(Receptive Field)를 가진 레벨을 선택합니다.
# [ARCH L0262] (DualAdaptiveSelector.select_level) 
# [ARCH L0263] (DualAdaptiveSelector.select_level) | 오차 범위 | 선택 레벨 | 수용 영역 | 목적 |
# [ARCH L0264] (DualAdaptiveSelector.select_level) |---------|----------|---------|------|
# [ARCH L0265] (DualAdaptiveSelector.select_level) | > 30px | **Level 3** (Global) | 넓음 (32px) | 큰 변환 포착 |
# [ARCH L0266] (DualAdaptiveSelector.select_level) | 10~30px | **Level 2** (Structural) | 중간 (16px) | 구조적 정렬 |
# [ARCH L0267] (DualAdaptiveSelector.select_level) | 5~10px | **Level 1** (Local) | 좁음 (8px) | 세부 매칭 |
# [ARCH L0268] (DualAdaptiveSelector.select_level) | < 5px | **Level 0** (Fine) | 픽셀 단위 | 미세 조정 |
# [ARCH L0269] (DualAdaptiveSelector.select_level) 
# [ARCH L0270] (DualAdaptiveSelector.select_level) **오차 측정 (Error Diagnosis):**
# [ARCH L0271] (DualAdaptiveSelector.select_level) 
# [ARCH L0272] (DualAdaptiveSelector.select_level) $$
# [ARCH L0273] (DualAdaptiveSelector.select_level) E_{pos} = \text{Mean}(|SDF_A(W_{curr}(p)) - SDF_B(p)|) \quad \text{[위치 오차]}
# [ARCH L0274] (DualAdaptiveSelector.select_level) $$
# [ARCH L0275] (DualAdaptiveSelector.select_level) 
# [ARCH L0276] (DualAdaptiveSelector.select_level) $$
# [ARCH L0277] (DualAdaptiveSelector.select_level) E_{angle} = 1 - \text{Mean}(\cos(\theta_{residual})) \quad \text{[방향 오차]}
# [ARCH L0278] (DualAdaptiveSelector.select_level) $$
# [ARCH L0279] (DualAdaptiveSelector.select_level) 
# [ARCH L0280] (DualAdaptiveSelector.select_level) - **[Hyperparameter]** Level 전환 임계값: `[30, 10, 5]` px
# [ARCH L0281] (DualAdaptiveSelector.select_level) 
# [ARCH L0282] (DualAdaptiveSelector.select_feature) #### **B. Feature Selection (특징 선택)**
# [ARCH L0283] (DualAdaptiveSelector.select_feature) 
# [ARCH L0284] (DualAdaptiveSelector.select_feature) **오차 타입**을 기준으로 가장 관련 있는 Clifford 성분을 선택합니다.
# [ARCH L0285] (DualAdaptiveSelector.select_feature) 
# [ARCH L0286] (DualAdaptiveSelector.select_feature) | 오차 타입 | 선택 특징 | 차원 | 역할 |
# [ARCH L0287] (DualAdaptiveSelector.select_feature) |---------|----------|-----|-----|
# [ARCH L0288] (DualAdaptiveSelector.select_feature) | 위치 불일치<br>($E_{pos}$ 지배적) | **S (Scalar)** | (B, 64, H, W) | 텍스처 매칭, SDF 정렬 |
# [ARCH L0289] (DualAdaptiveSelector.select_feature) | 방향 불일치<br>($E_{angle}$ 지배적) | **V (Vector)** | (B, 64, 2, H, W) | 그래디언트 방향 정렬 |
# [ARCH L0290] (DualAdaptiveSelector.select_feature) | 스케일/회전 불일치<br>(둘 다 큼) | **B (Bivector)** | (B, 64, H, W) | Rotor 보정 |
# [ARCH L0291] (DualAdaptiveSelector.select_feature) 
# [ARCH L0292] (DualAdaptiveSelector.select_feature) **선택 기준:**
# [ARCH L0293] (DualAdaptiveSelector.select_feature) ```python
# [ARCH L0294] (DualAdaptiveSelector.select_feature) if E_pos > 15.0:
# [ARCH L0295] (DualAdaptiveSelector.select_feature)     selected = S  # 위치부터 맞춤
# [ARCH L0296] (DualAdaptiveSelector.select_feature) elif E_angle > 0.25:  # ≈ 14.3°
# [ARCH L0297] (DualAdaptiveSelector.select_feature)     selected = V  # 방향 정렬
# [ARCH L0298] (DualAdaptiveSelector.select_feature) else:
# [ARCH L0299] (DualAdaptiveSelector.select_feature)     selected = B  # 미세 회전 보정
# [ARCH L0300] (DualAdaptiveSelector.select_feature) ```
# [ARCH L0301] (DualAdaptiveSelector.select_feature) 
# [ARCH L0302] (DualAdaptiveSelector.select_feature) - **[Hyperparameter]** 특징 선택 임계값: `E_pos = 15.0` px, `E_angle = 0.25` rad
# [ARCH L0303] (DualAdaptiveSelector.select_feature) 
# [ARCH L0304] (DualAdaptiveSelector.select_feature) ---
# [ARCH L0305] (DualAdaptiveSelector.select_feature) 
# [ARCH L0306] (MiniConvGRU (+ delta_head)) ### **2. Mini-ConvGRU (경량 순환 엔진)**
# [ARCH L0307] (MiniConvGRU (+ delta_head)) 
# [ARCH L0308] (MiniConvGRU (+ delta_head)) IGEV의 Full ConvGRU를 **1/4 크기로 경량화**하고, Correlation Volume을 제거하여 메모리 효율을 극대화했습니다.
# [ARCH L0309] (MiniConvGRU (+ delta_head)) 
# [ARCH L0310] (MiniConvGRU (+ delta_head)) #### **A. 구조 (Minimal-GRU)**
# [ARCH L0311] (MiniConvGRU (+ delta_head)) 
# [ARCH L0312] (MiniConvGRU (+ delta_head)) $$
# [ARCH L0313] (MiniConvGRU (+ delta_head)) \begin{aligned}
# [ARCH L0314] (MiniConvGRU (+ delta_head)) z_k &= \sigma(\text{Conv}_{3x3}([h_{k-1}, E_{diff}])) \quad \text{[Update Gate: 16채널]} \\
# [ARCH L0315] (MiniConvGRU (+ delta_head)) \tilde{h}_k &= \tanh(\text{Conv}_{3x3}([F_{selected}, E_{diff}])) \quad \text{[Candidate State]} \\
# [ARCH L0316] (MiniConvGRU (+ delta_head)) h_k &= (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k \quad \text{[Linear Interpolation]} \\
# [ARCH L0317] (MiniConvGRU (+ delta_head)) \Delta W_k &= \text{Head}_{2\text{-layer}}(h_k) \quad \text{[16→8→4 채널]}
# [ARCH L0318] (MiniConvGRU (+ delta_head)) \end{aligned}
# [ARCH L0319] (MiniConvGRU (+ delta_head)) $$
# [ARCH L0320] (MiniConvGRU (+ delta_head)) 
# [ARCH L0321] (MiniConvGRU (+ delta_head)) **주요 개선점:**
# [ARCH L0322] (MiniConvGRU (+ delta_head)) - **Reset Gate 제거:** Minimal-GRU 구조로 파라미터 50% 감소
# [ARCH L0323] (MiniConvGRU (+ delta_head)) - **Correlation Volume 제거:** Difference Map ($E_{diff} = |A' - B|$)으로 대체하여 메모리 절약
# [ARCH L0324] (MiniConvGRU (+ delta_head)) - **16채널 Hidden State:** 원본 IGEV(64채널) 대비 75% 메모리 절감
# [ARCH L0325] (MiniConvGRU (+ delta_head)) 
# [ARCH L0326] (IterativeRefinementLoop: interpolate features/hidden across levels) #### **B. Level Transfer (해상도 전환 시)**
# [ARCH L0327] (IterativeRefinementLoop: interpolate features/hidden across levels) 
# [ARCH L0328] (IterativeRefinementLoop: interpolate features/hidden across levels) 피라미드 레벨이 바뀔 때(예: Level 3 → Level 2), Hidden State의 해상도를 조정합니다.
# [ARCH L0329] (IterativeRefinementLoop: interpolate features/hidden across levels) 
# [ARCH L0330] (IterativeRefinementLoop: interpolate features/hidden across levels) $$
# [ARCH L0331] (IterativeRefinementLoop: interpolate features/hidden across levels) h_k^{l} = \text{MiniGRU}(\text{Upsample}(h_{k-1}^{l+1}), [E_{diff}^{l}, F_{selected}^{l}])
# [ARCH L0332] (IterativeRefinementLoop: interpolate features/hidden across levels) $$
# [ARCH L0333] (IterativeRefinementLoop: interpolate features/hidden across levels) 
# [ARCH L0334] (IterativeRefinementLoop: interpolate features/hidden across levels) **구현:**
# [ARCH L0335] (IterativeRefinementLoop: interpolate features/hidden across levels) ```python
# [ARCH L0336] (IterativeRefinementLoop: interpolate features/hidden across levels) if h_prev.shape[-2:] != target_size:
# [ARCH L0337] (IterativeRefinementLoop: interpolate features/hidden across levels)     h_prev = F.interpolate(h_prev, size=target_size, mode='bilinear')
# [ARCH L0338] (IterativeRefinementLoop: interpolate features/hidden across levels) ```
# [ARCH L0339] (IterativeRefinementLoop: interpolate features/hidden across levels) 
# [ARCH L0340] (IterativeRefinementLoop: interpolate features/hidden across levels) - **목적:** 저해상도(Coarse)에서 학습한 "큰 흐름"을 고해상도(Fine)로 전달
# [ARCH L0341] (IterativeRefinementLoop: interpolate features/hidden across levels) - **효과:** 각 레벨이 독립적으로 시작하는 것보다 **2배 빠른 수렴**
# [ARCH L0342] (IterativeRefinementLoop: interpolate features/hidden across levels) 
# [ARCH L0343] (IterativeRefinementLoop: interpolate features/hidden across levels) ---
# [ARCH L0344] (IterativeRefinementLoop: interpolate features/hidden across levels) 
# [ARCH L0345] (IterativeRefinementLoop.forward (iteration strategy)) ### **3. 반복 정제 루프 (Iteration Strategy)**
# [ARCH L0346] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0347] (IterativeRefinementLoop.forward (iteration strategy)) 실제 시나리오별 동작 흐름입니다.
# [ARCH L0348] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0349] (IterativeRefinementLoop.forward (iteration strategy)) | Iter | 오차 상태 | 선택 레벨 | 선택 특징 | GRU 동작 | 목표 |
# [ARCH L0350] (IterativeRefinementLoop.forward (iteration strategy)) |------|----------|----------|----------|---------|-----|
# [ARCH L0351] (IterativeRefinementLoop.forward (iteration strategy)) | **1** | 45px, 20° | Level 3 | **V + B** | 큰 회전 감지 → Momentum 축적 | 20px |
# [ARCH L0352] (IterativeRefinementLoop.forward (iteration strategy)) | **2** | 20px, 5° | Level 2 | **S** | 이전 방향 유지 + 텍스처 매칭 | 8px |
# [ARCH L0353] (IterativeRefinementLoop.forward (iteration strategy)) | **3** | 8px, 1° | Level 1 | **S** | 디테일 엣지 정렬 | 3px |
# [ARCH L0354] (IterativeRefinementLoop.forward (iteration strategy)) | **4** | 3px, 0.3° | Level 1 | **S** | 미세 조정 | 1-2px |
# [ARCH L0355] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0356] (IterativeRefinementLoop.forward (iteration strategy)) **[Hyperparameter]** `num_iterations = 4` (레벨 수와 동일)
# [ARCH L0357] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0358] (IterativeRefinementLoop.forward (iteration strategy)) **시각화 예시:**
# [ARCH L0359] (IterativeRefinementLoop.forward (iteration strategy)) ```
# [ARCH L0360] (IterativeRefinementLoop.forward (iteration strategy)) [Iter 1] Error=45.2px | Level=3 (Global) | Feature=V+B (Rotation) → 22.1px
# [ARCH L0361] (IterativeRefinementLoop.forward (iteration strategy)) [Iter 2] Error=22.1px | Level=2 (Struct)  | Feature=S (Texture)   → 9.4px
# [ARCH L0362] (IterativeRefinementLoop.forward (iteration strategy)) [Iter 3] Error=9.4px  | Level=1 (Local)   | Feature=S (Edge)      → 3.8px
# [ARCH L0363] (IterativeRefinementLoop.forward (iteration strategy)) [Iter 4] Error=3.8px  | Level=1 (Fine)    | Feature=S (Detail)    → 1.5px ✓
# [ARCH L0364] (IterativeRefinementLoop.forward (iteration strategy)) ```
# [ARCH L0365] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0366] (IterativeRefinementLoop.forward (iteration strategy)) ---
# [ARCH L0367] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0368] (IterativeRefinementLoop.forward (iteration strategy)) ### **4. 종료 조건 및 안전장치**
# [ARCH L0369] (IterativeRefinementLoop.forward (iteration strategy)) 
# [ARCH L0370] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) #### **A. 조기 종료 (Convergence)**
# [ARCH L0371] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 
# [ARCH L0372] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 다음 조건 **중 하나라도** 만족하면 즉시 종료:
# [ARCH L0373] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 
# [ARCH L0374] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 1. **변화량 수렴:**
# [ARCH L0375] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    $$
# [ARCH L0376] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    \|\Delta W_k - I\|_F < \epsilon_{\text{conv}}
# [ARCH L0377] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    $$
# [ARCH L0378] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    - **[Hyperparameter]** $\epsilon_{\text{conv}} = 0.005$ (Frobenius Norm)
# [ARCH L0379] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 
# [ARCH L0380] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 2. **오차 충분히 작음:**
# [ARCH L0381] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    $$
# [ARCH L0382] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    E_{curr} < \epsilon_{\text{target}}
# [ARCH L0383] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    $$
# [ARCH L0384] (IterativeRefinementLoop.check_convergence + TARGET_ERROR)    - **[Hyperparameter]** $\epsilon_{\text{target}} = 3.0$ px (Phase 4가 해결 가능한 범위)
# [ARCH L0385] (IterativeRefinementLoop.check_convergence + TARGET_ERROR) 
# [ARCH L0386] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) #### **B. 발산 방지 (Bounded Safety Lock)**
# [ARCH L0387] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0388] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) **조건:** 새로운 변환이 이전보다 5% 이상 악화된 경우
# [ARCH L0389] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0390] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) $$
# [ARCH L0391] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) E_{next} > E_{curr} \times (1 + \alpha)
# [ARCH L0392] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) $$
# [ARCH L0393] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0394] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) - **[Hyperparameter]** $\alpha = 0.05$ (5% Tolerance)
# [ARCH L0395] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0396] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) **대응 전략 (3단계):**
# [ARCH L0397] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0398] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 1. **1차 시도: Update Rejection**
# [ARCH L0399] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ```python
# [ARCH L0400] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))    if E_next > E_curr * 1.05:
# [ARCH L0401] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        W_accum = W_prev  # 이전 상태로 롤백
# [ARCH L0402] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        continue
# [ARCH L0403] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ```
# [ARCH L0404] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0405] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 2. **2차 시도: GRU Reset + LR Decay**
# [ARCH L0406] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ```python
# [ARCH L0407] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))    if consecutive_rejections >= 2:
# [ARCH L0408] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        mini_gru.reset()           # Hidden State 초기화
# [ARCH L0409] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        learning_rate *= 0.5       # 학습률 반감
# [ARCH L0410] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        print("[Recovery] GRU Reset + LR Decay")
# [ARCH L0411] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ```
# [ARCH L0412] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0413] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 3. **3차 시도: 강제 종료**
# [ARCH L0414] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ```python
# [ARCH L0415] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))    if consecutive_rejections >= 3:
# [ARCH L0416] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        print("[Emergency] Divergence detected, aborting...")
# [ARCH L0417] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit))        break
# [ARCH L0418] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ```
# [ARCH L0419] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0420] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) **효과:**
# [ARCH L0421] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) - Local Minima(작은 언덕)에 갇혔을 때 **GRU Reset**으로 탈출 가능
# [ARCH L0422] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) - 학습률을 낮춰 **과도한 변화(Overshoot)**를 방지
# [ARCH L0423] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) - Phase 4의 강력한 수렴 능력(Basin ≈ 10px)을 신뢰하므로, 약간의 악화는 허용
# [ARCH L0424] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0425] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ---
# [ARCH L0426] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0427] (Phase3.5 constants (hyperparameters)) ### **7. 하이퍼파라미터 요약**
# [ARCH L0428] (Phase3.5 constants (hyperparameters)) 
# [ARCH L0429] (Phase3.5 constants (hyperparameters)) | 파라미터 | 기본값 | 범위 | 설명 |
# [ARCH L0430] (Phase3.5 constants (hyperparameters)) |---------|-------|-----|------|
# [ARCH L0431] (Phase3.5 constants (hyperparameters)) | `num_iterations` | **4** | 2~6 | 최대 반복 횟수 |
# [ARCH L0432] (Phase3.5 constants (hyperparameters)) | `gru_hidden_dim` | **16** | 8~32 | GRU Hidden State 차원 |
# [ARCH L0433] (Phase3.5 constants (hyperparameters)) | `convergence_threshold` | **0.005** | 0.001~0.01 | 수렴 판정 임계값 |
# [ARCH L0434] (Phase3.5 constants (hyperparameters)) | `target_error_px` | **3.0** | 1.0~5.0 | 목표 오차 (Phase 4 이관 기준) |
# [ARCH L0435] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) | `tolerance_alpha` | **0.05** | 0.03~0.1 | 발산 방지 허용 오차 (5%) |
# [ARCH L0436] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) | `level_thresholds` | **[30, 10, 5]** | - | Level 전환 임계값 (px) |
# [ARCH L0437] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) | `feature_thresholds` | **[10.0, 0.1]** | - | S/V/B 선택 임계값 |
# [ARCH L0438] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0439] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) 
# [ARCH L0440] (IterativeRefinementLoop.forward (Update rejection / GRU reset / step_scale decay / emergency exit)) ---
# =============================================================================
"""
================================================================================
Phase 3.5: Dual-Adaptive Recurrent Refinement (Architecture.md 완전 반영 버전)
================================================================================
[Architecture.md §3.5 참조]

이 버전은 기존 phase3_5.py 구현을 유지하면서, Architecture.md에 있으나
코드에 없던 "안전장치(3단계)"와 하이퍼파라미터(Feature Threshold) 불일치를
정확히 반영했습니다.

보강된 구현 포인트:
1) Feature Threshold 정합:
   - feature_thresholds = [10.0, 0.1] (px, rad)  ← Architecture.md 기준
2) Safety Lock (3-Stage):
   Stage 1) Update Rejection: E_next > E_curr*(1+α) 이면 업데이트 거부(rollback)
   Stage 2) GRU Reset + LR Decay(학습률 개념을 step_scale로 구현): 2회 연속 거부 시 적용
   Stage 3) Emergency Exit: 3회 연속 거부 시 중단
3) Phase 3 초기 추정 사용:
   - phase3_results가 주어지면, 가장 coarse 레벨의 rotor_map 평균으로 W_accum 초기화

수식:
W_final = MiniGRU(F_selected^level) ∘ ... ∘ MiniGRU(F_selected^level)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# [Hyperparameters] Phase 3.5 (Architecture.md 기준 정합)
# =============================================================================
NUM_ITERATIONS = 4

GRU_HIDDEN_DIM = 16

# Level Selection thresholds (px)
LEVEL_THRESHOLD_HIGH = 30
LEVEL_THRESHOLD_MID = 10
LEVEL_THRESHOLD_LOW = 5

# Feature Selection thresholds (Architecture.md: [10.0, 0.1])
FEATURE_POS_THRESHOLD = 10.0      # px
FEATURE_ANGLE_THRESHOLD = 0.1     # rad (~5.7°)

# Convergence & Target
CONVERGENCE_THRESHOLD = 0.005
TARGET_ERROR_PX = 3.0
TOLERANCE_ALPHA = 0.05

# Safety Lock
MAX_CONSECUTIVE_REJECTIONS = 3
LR_DECAY_FACTOR = 0.5

# LR(학습률) 개념을 "업데이트 스케일(step_scale)"로 구현
STEP_SCALE_INIT = 1.0


# =============================================================================
# [Mini-ConvGRU]
# =============================================================================

class MiniConvGRU(nn.Module):
    """
    Architecture.md §3.5.2 - Mini-ConvGRU

    Minimal-GRU 구조:
    - z_k = σ(Conv([h_{k-1}, E_diff]))
    - h̃_k = tanh(Conv([F_selected, E_diff]))
    - h_k = (1-z_k) ⊙ h_{k-1} + z_k ⊙ h̃_k
    - ΔW_k = Head(h_k)  (16→8→4)
    """
    def __init__(self, input_dim, hidden_dim=GRU_HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv_z = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(input_dim * 2, hidden_dim, kernel_size=3, padding=1)

        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(hidden_dim // 2, 4, kernel_size=1)  # (cos, sin, dx, dy)
        )

    def init_hidden(self, batch, height, width, device):
        return torch.zeros(batch, self.hidden_dim, height, width, device=device)

    def forward(self, h_prev, e_diff, f_selected):
        z_input = torch.cat([h_prev, e_diff], dim=1)
        z = torch.sigmoid(self.conv_z(z_input))

        h_input = torch.cat([f_selected, e_diff], dim=1)
        h_candidate = torch.tanh(self.conv_h(h_input))

        h_new = (1 - z) * h_prev + z * h_candidate
        delta_w = self.delta_head(h_new)

        return h_new, delta_w


# =============================================================================
# [Error Diagnostic]
# =============================================================================

class ErrorDiagnostic(nn.Module):
    """
    Architecture.md §3.5.1.A - 오차 측정

    E_pos = Mean(|SDF_A(W_curr(p)) - SDF_B(p)|)
    E_angle = 1 - Mean(cos(θ_residual))
    """
    def compute_position_error(self, warped_sdf, target_sdf):
        diff = torch.abs(warped_sdf - target_sdf)
        error_px = diff.mean(dim=(1, 2, 3)) * 100  # heuristic scale
        return error_px

    def compute_angle_error(self, warped_vector, target_vector):
        cos_sim = F.cosine_similarity(warped_vector, target_vector, dim=1, eps=1e-6)
        error_angle = (1.0 - cos_sim.mean(dim=(1, 2))).clamp(0, 2)
        return error_angle

    def forward(self, warped_features, target_features):
        e_pos = self.compute_position_error(warped_features['sdf'], target_features['sdf'])
        e_angle = self.compute_angle_error(warped_features['vector'], target_features['vector'])
        return e_pos, e_angle


# =============================================================================
# [Dual-Adaptive Selector]
# =============================================================================

class DualAdaptiveSelector(nn.Module):
    """Architecture.md §3.5.1 - Level Selection + Feature Selection"""
    def select_level(self, e_pos):
        avg_error = e_pos.mean().item()
        if avg_error > LEVEL_THRESHOLD_HIGH:
            return 3
        elif avg_error > LEVEL_THRESHOLD_MID:
            return 2
        elif avg_error > LEVEL_THRESHOLD_LOW:
            return 1
        else:
            return 0

    def select_feature(self, e_pos, e_angle):
        """Architecture.md §3.5 - Feature Selection

        - pos, angle 모두 큰 경우: 'B' (scale/rotation mismatch)
        - pos만 큰 경우: 'S' (translation/position mismatch)
        - angle만 큰 경우: 'V' (rotation mismatch)
        - 둘 다 작으면: 'B' (fine scale/rotation residual)
        """
        avg_pos = e_pos.mean().item()
        avg_angle = e_angle.mean().item()

        if (avg_pos > FEATURE_POS_THRESHOLD) and (avg_angle > FEATURE_ANGLE_THRESHOLD):
            return 'B'
        elif avg_pos > FEATURE_POS_THRESHOLD:
            return 'S'
        elif avg_angle > FEATURE_ANGLE_THRESHOLD:
            return 'V'
        else:
            return 'B'

    def forward(self, e_pos, e_angle):
        return self.select_level(e_pos), self.select_feature(e_pos, e_angle)


# =============================================================================
# [Transform Accumulator]
# =============================================================================

class TransformAccumulator:
    """
    W_current = ΔW ∘ W_prev

    내부 표현: 2x3 affine(theta) in normalized coord
    """
    def __init__(self, device):
        self.device = device
        self.W_accum = torch.eye(2, 3, device=device).unsqueeze(0)

    def reset(self, batch_size=1, init_W=None):
        if init_W is not None:
            self.W_accum = init_W.to(self.device)
            if batch_size > 1 and self.W_accum.shape[0] == 1:
                self.W_accum = self.W_accum.repeat(batch_size, 1, 1)
            return

        self.W_accum = torch.eye(2, 3, device=self.device).unsqueeze(0)
        if batch_size > 1:
            self.W_accum = self.W_accum.repeat(batch_size, 1, 1)

    @staticmethod
    def _to_aug(W_2x3):
        B = W_2x3.shape[0]
        bottom = torch.tensor([0, 0, 1], device=W_2x3.device, dtype=W_2x3.dtype).view(1, 1, 3).repeat(B, 1, 1)
        return torch.cat([W_2x3, bottom], dim=1)  # (B,3,3)

    def compose_from_delta_map(self, delta_w_map, step_scale: float = 1.0):
        """
        delta_w_map: (B,4,H,W) = (cos,sin,dx,dy)
        step_scale: (0~1) 업데이트 강도(= learning rate 역할)

        Architecture.md Safety Lock Stage2의 LR decay를 구현하기 위해,
        ΔW를 Identity에 가까워지도록 스케일링합니다.
        """
        avg_delta = delta_w_map.mean(dim=(2, 3))  # (B,4)
        cos_d, sin_d, dx_d, dy_d = avg_delta[:, 0], avg_delta[:, 1], avg_delta[:, 2], avg_delta[:, 3]

        # --- Step scaling (Identity로 수렴시키는 형태) ---
        # cos' = 1 + s*(cos-1)
        # sin' = s*sin
        # dx'  = s*dx
        # dy'  = s*dy
        s = float(step_scale)
        cos_d = 1.0 + s * (cos_d - 1.0)
        sin_d = s * sin_d
        dx_d = s * dx_d
        dy_d = s * dy_d

        # NLERP 정규화
        norm = torch.sqrt(cos_d**2 + sin_d**2 + 1e-6)
        cos_d = cos_d / norm
        sin_d = sin_d / norm

        B = cos_d.shape[0]
        delta_mat = torch.zeros(B, 2, 3, device=self.device, dtype=delta_w_map.dtype)
        delta_mat[:, 0, 0] = cos_d
        delta_mat[:, 0, 1] = -sin_d
        delta_mat[:, 0, 2] = dx_d
        delta_mat[:, 1, 0] = sin_d
        delta_mat[:, 1, 1] = cos_d
        delta_mat[:, 1, 2] = dy_d

        # 합성
        W_aug = self._to_aug(self.W_accum)
        d_aug = self._to_aug(delta_mat)
        out = torch.bmm(d_aug, W_aug)
        return out[:, :2, :]

    def set_current(self, W_2x3):
        self.W_accum = W_2x3

    def get_current(self):
        return self.W_accum


# =============================================================================
# [Iterative Refinement Loop]
# =============================================================================

class IterativeRefinementLoop(nn.Module):
    """
    Architecture.md §3.5.3 - 반복 정제 루프 + Safety Lock(§3.5.4)
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.error_diagnostic = ErrorDiagnostic()
        self.selector = DualAdaptiveSelector()

        self.chunk_dim = feature_dim // 3

        self.gru_s = MiniConvGRU(input_dim=self.chunk_dim)
        self.gru_v = MiniConvGRU(input_dim=self.chunk_dim)
        self.gru_b = MiniConvGRU(input_dim=self.chunk_dim)

    def extract_feature_by_type(self, features, feature_type):
        s, v, b = torch.chunk(features, 3, dim=1)
        if feature_type == 'S':
            return s
        if feature_type == 'V':
            return v
        return b

    def get_gru_by_type(self, feature_type):
        if feature_type == 'S':
            return self.gru_s
        if feature_type == 'V':
            return self.gru_v
        return self.gru_b

    @staticmethod
    def warp_features(features, W_matrix, target_size):
        grid = F.affine_grid(
            W_matrix,
            [features.shape[0], features.shape[1], target_size[0], target_size[1]],
            align_corners=False
        )
        warped = F.grid_sample(features, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
        return warped

    def compute_error(self, feat_a, feat_b, W_matrix):
        """
        현재 변환 W에서의 오차(E_pos, E_angle) 계산
        """
        B, C, H, W = feat_b.shape
        feat_a_warped = self.warp_features(feat_a, W_matrix, (H, W))
        e_diff = torch.abs(feat_a_warped - feat_b)

        warped_dict = {
            'sdf': feat_a_warped[:, :self.chunk_dim, :, :],
            'vector': feat_a_warped[:, self.chunk_dim:2*self.chunk_dim, :, :],
        }
        target_dict = {
            'sdf': feat_b[:, :self.chunk_dim, :, :],
            'vector': feat_b[:, self.chunk_dim:2*self.chunk_dim, :, :],
        }
        e_pos, e_angle = self.error_diagnostic(warped_dict, target_dict)
        return e_pos, e_angle, e_diff, feat_a_warped

    def check_convergence(self, delta_w):
        """
        Architecture.md 종료 조건 A: ||ΔW - I||_F < ε_conv
        """
        B = delta_w.shape[0]
        avg_delta = delta_w.mean(dim=(2, 3))
        identity = torch.tensor([[1, 0, 0, 0]], device=delta_w.device, dtype=delta_w.dtype).repeat(B, 1)
        diff = avg_delta - identity
        frob = torch.norm(diff, dim=1).mean().item()
        return frob < CONVERGENCE_THRESHOLD

    @staticmethod
    def rotor_map_to_theta(rotor_map):
        """
        rotor_map: (B,H,W,4) -> (B,2,3)
        """
        avg = rotor_map.mean(dim=(1, 2))
        cos_t, sin_t, dx_t, dy_t = avg[:, 0], avg[:, 1], avg[:, 2], avg[:, 3]
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

    def forward(self, pyramid_features_a, pyramid_features_b, phase3_results, device):
        """
        Args:
            pyramid_features_a/b: list of (B,C,H,W) per level (Phase3 refined_feature 추천)
            phase3_results: Phase3 output list (delta_rotor_map 포함)
        Returns:
            W_final, history
        """
        B = pyramid_features_a[0].shape[0]
        _, _, H, W = pyramid_features_a[0].shape

        # -------------------------
        # Init transform
        # -------------------------
        init_W = None
        if phase3_results is not None and len(phase3_results) > 0:
            # Phase3 결과 중 가장 coarse(level index가 가장 큰 것)를 사용
            coarse_res = max(phase3_results, key=lambda d: d.get('level', 0))
            coarse_rotor = coarse_res.get('delta_rotor_map', None)
            if coarse_rotor is None:
                coarse_rotor = coarse_res.get('rotor_map', None)
            if coarse_rotor is not None:
                init_W = self.rotor_map_to_theta(coarse_rotor).detach()

        accumulator = TransformAccumulator(device)
        accumulator.reset(batch_size=B, init_W=init_W)

        # Hidden state init (Level0 resolution)
        h_s = self.gru_s.init_hidden(B, H, W, device)
        h_v = self.gru_v.init_hidden(B, H, W, device)
        h_b = self.gru_b.init_hidden(B, H, W, device)
        hidden_states = {'S': h_s, 'V': h_v, 'B': h_b}

        # Safety lock state
        consecutive_rejections = 0
        step_scale = STEP_SCALE_INIT

        history = []

        print(f"[Phase 3.5] Starting Iterative Refinement (max {NUM_ITERATIONS} iterations)")

        for k in range(NUM_ITERATIONS):
            W_curr = accumulator.get_current()

            # 1) Current error
            e_pos, e_angle, e_diff, feat_a_warped = self.compute_error(
                pyramid_features_a[0], pyramid_features_b[0], W_curr
            )
            e_curr = e_pos.mean().item()

            # 2) Target reached
            if e_curr < TARGET_ERROR_PX:
                print(f"  [Iter {k+1}] Target reached: {e_curr:.2f}px < {TARGET_ERROR_PX}px ✓")
                break

            # 3) Dual selection
            selected_level, selected_feature = self.selector(e_pos, e_angle)

            level_feat_a = pyramid_features_a[min(selected_level, len(pyramid_features_a)-1)]
            level_feat_b = pyramid_features_b[min(selected_level, len(pyramid_features_b)-1)]

            # Resize to Level0 resolution for GRU stability
            if level_feat_a.shape[-2:] != (H, W):
                level_feat_a = F.interpolate(level_feat_a, size=(H, W), mode='bilinear', align_corners=True)
                level_feat_b = F.interpolate(level_feat_b, size=(H, W), mode='bilinear', align_corners=True)

            # Selected feature + difference
            f_selected = self.extract_feature_by_type(level_feat_a, selected_feature)
            e_diff_selected = self.extract_feature_by_type(e_diff, selected_feature)

            # 4) Mini-GRU step
            gru = self.get_gru_by_type(selected_feature)
            h_prev = hidden_states[selected_feature]
            if h_prev.shape[-2:] != (H, W):
                h_prev = F.interpolate(h_prev, size=(H, W), mode='bilinear', align_corners=True)

            h_new, delta_w = gru(h_prev, e_diff_selected, f_selected)
            hidden_states[selected_feature] = h_new

            # 5) Convergence check (ΔW ≈ I)
            if self.check_convergence(delta_w):
                print(f"  [Iter {k+1}] Converged: ΔW ≈ I ✓")
                break

            # 6) Candidate update (do NOT commit yet)
            W_candidate = accumulator.compose_from_delta_map(delta_w, step_scale=step_scale)

            # 7) Safety Lock Stage 1: Update Rejection
            e_pos_next, e_angle_next, _, _ = self.compute_error(
                pyramid_features_a[0], pyramid_features_b[0], W_candidate
            )
            e_next = e_pos_next.mean().item()

            if e_next > e_curr * (1.0 + TOLERANCE_ALPHA):
                consecutive_rejections += 1
                print(f"  [Iter {k+1}] Update Rejected: {e_next:.2f}px > {e_curr:.2f}px*(1+{TOLERANCE_ALPHA}) "
                      f"({consecutive_rejections}/{MAX_CONSECUTIVE_REJECTIONS})")

                # Stage 2: GRU Reset + LR Decay (2회 연속 거부 시)
                if consecutive_rejections >= 2:
                    step_scale *= LR_DECAY_FACTOR
                    # Hidden state reset (전부 0으로)
                    hidden_states['S'] = self.gru_s.init_hidden(B, H, W, device)
                    hidden_states['V'] = self.gru_v.init_hidden(B, H, W, device)
                    hidden_states['B'] = self.gru_b.init_hidden(B, H, W, device)
                    print(f"  [Recovery] GRU reset + step_scale *= {LR_DECAY_FACTOR} -> {step_scale:.4f}")

                # Stage 3: Emergency Exit
                if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                    print(f"  [Emergency] Max rejections reached, aborting.")
                    break

                # Rollback: accumulator는 commit 하지 않았으므로 W_curr 유지
                continue

            # Accept update
            accumulator.set_current(W_candidate)
            consecutive_rejections = 0

            history.append({
                'iteration': k + 1,
                'error_px': e_curr,
                'error_angle': e_angle.mean().item(),
                'selected_level': selected_level,
                'selected_feature': selected_feature,
                'step_scale': step_scale,
                'error_next_px': e_next
            })

            print(f"  [Iter {k+1}] Error={e_curr:.1f}px → {e_next:.1f}px | "
                  f"Level={selected_level} | Feature={selected_feature} | step_scale={step_scale:.3f}")

        W_final = accumulator.get_current()
        return W_final, history


class Phase35Refiner(nn.Module):
    """Architecture.md §3.5 - Wrapper"""
    def __init__(self, feature_dim):
        super().__init__()
        self.refinement_loop = IterativeRefinementLoop(feature_dim)

    def forward(self, pyramid_features_a, pyramid_features_b, phase3_results=None, device=None):
        if device is None:
            device = pyramid_features_a[0].device
        return self.refinement_loop(pyramid_features_a, pyramid_features_b, phase3_results, device)


# =============================================================================
# 테스트 코드
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3.5 ArchFull Test on: {device}")

    B, C, H, W = 1, 144, 64, 64

    pyramid_a = [torch.randn(B, C, H // (2**i), W // (2**i), device=device) for i in range(4)]
    pyramid_b = [torch.randn(B, C, H // (2**i), W // (2**i), device=device) for i in range(4)]

    refiner = Phase35Refiner(feature_dim=C).to(device)

    with torch.no_grad():
        W_refined, history = refiner(pyramid_a, pyramid_b, phase3_results=None, device=device)

    print(f"\n[Result]")
    print(f"  W_refined shape: {W_refined.shape}")
    print(f"  Iterations: {len(history)}")
    for h in history:
        print(f"    Iter {h['iteration']}: {h['error_px']:.1f}px -> {h['error_next_px']:.1f}px "
              f"| Level={h['selected_level']} Feature={h['selected_feature']} step={h['step_scale']:.3f}")
