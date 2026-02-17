# =============================================================================
# Architecture.md Line Mapping (원본 architecture.md 기준)
# - 각 라인이 코드의 어느 부분에 해당하는지 anchor로 명시합니다.
# - (요청사항) architecture.md 내용을 스킵/생략하지 않기 위해, 빈 줄도 포함합니다.
# =============================================================================
# [ARCH L0001] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) ## **📂  1: Geometry Raw Data 추출 (Physical Raw Data Extraction)**
# [ARCH L0002] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) 
# [ARCH L0003] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) 이 단계에서는 원본 이미지를 다양한 해상도로 리사이즈한 후, 각 층에서 클리포드 멀티벡터의 재료가 될 물리량들을 추출
# [ARCH L0004] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) 
# [ARCH L0005] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) - **1/32 ~ 1/16 (Global):** 이미지의 아주 거친 형태만 보입니다. 큰 건물이나 산의 위치처럼 **전역적인 배치**를 파악하기 위함입니다.
# [ARCH L0006] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) - **1/8 ~ 1/4 (Structural):** 물체의 구체적인 윤곽과 구조가 드러납니다. **주요 특징점들의 기하학적 관계**를 학습합니다.
# [ARCH L0007] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) - **1/2 ~ 1 (Fine):** 아주 세밀한 텍스처와 0.1 픽셀 단위의 엣지가 보입니다. **최종적인 초정밀 정렬**을 수행하는 단계입니다.
# [ARCH L0008] (MathGeometricPreprocessor.process_pyramid / _extract_raw_features) 
# [ARCH L0009] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) ### **1. S (Scalar): 존재의 강도와 뼈대**
# [ARCH L0010] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 
# [ARCH L0011] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 스칼라는 방향성은 없지만, 해당 지점에 **"무엇이 얼마나 강하게 있는가"**를 나타냅니다.
# [ARCH L0012] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 
# [ARCH L0013] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) - **Texture (재질/밝기)**
# [ARCH L0014] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - 이미지의 국부적인 밝기 변화를 의미
# [ARCH L0015] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - 물체의 표면 특성을 반영하며, 나중에 두 이미지 사이의 **색상 유사도**를 비교하는 기초 값이 됨
# [ARCH L0016] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) - **Structure Energy (구조 에너지)**
# [ARCH L0017] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - 주변 픽셀들과 비교했을 때 정보가 얼마나 밀집되어 있는지를 나타냄
# [ARCH L0018] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - 매칭할 때 "믿을만한 특징점인가?"를 판단하는 **신뢰도 가중치**로 쓰임
# [ARCH L0019] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) - **Edge Magnitude (엣지 세기)**
# [ARCH L0020] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - 경계선이 얼마나 뚜렷한지 나타내는 수치
# [ARCH L0021] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - V(Vector)가 방향을 가리킨다면, 이 값은 그 **방향의 확신도**를 결정
# [ARCH L0022] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) - **SDF (Signed Distance Field)**
# [ARCH L0023] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - 물체의 뼈대로부터의 거리
# [ARCH L0024] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - **MPC(제어) 단계에서 가장 핵심**적인 정보로, 두 물체가 얼마나 떨어져 있는지 '에너지'로 계산할 수 있게 해주는 **잠재적인 중력장** 역할
# [ARCH L0025] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 
# [ARCH L0026] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 
# [ARCH L0027] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     **하이퍼파라미터:**
# [ARCH L0028] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - `SDF_SKELETON_POWER = 8.0`: 뼈대 선명도 (높을수록 얇아짐)
# [ARCH L0029] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - `SDF_FIELD_POWER = 2.0`: 장 부드러움 (낮을수록 넓게 퍼짐)
# [ARCH L0030] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - `SDF_FIELD_WEIGHT = 0.4`: 장 가중치 (뼈대 대비 영향력)
# [ARCH L0031] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 
# [ARCH L0032] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     **물리적 의미:**
# [ARCH L0033] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - **Skeleton Component**: 정확한 엣지 위치 표현 (위치 정밀도)
# [ARCH L0034] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - **Field Component**: 넓은 탐색 범위 제공 (수렴 Basin 확대)
# [ARCH L0035] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF))     - **Max Fusion**: 두 장점을 모두 활용 (날카로움 + 부드러움)
# [ARCH L0036] (MathGeometricPreprocessor._extract_raw_features (Scalar: HSI/Structure/SDF)) 
# [ARCH L0037] (MathGeometricPreprocessor.get_flow_features (Vector field)) ### **2. V (Vector): 변화의 방향과 흐름**
# [ARCH L0038] (MathGeometricPreprocessor.get_flow_features (Vector field)) 
# [ARCH L0039] (MathGeometricPreprocessor.get_flow_features (Vector field)) 벡터는 이미지 내에서 **"어느 쪽으로 움직이는가"**라는 동적인 정보를 담습니다.
# [ARCH L0040] (MathGeometricPreprocessor.get_flow_features (Vector field)) 
# [ARCH L0041] (MathGeometricPreprocessor.get_flow_features (Vector field)) - **Gradient (경계 변화)**
# [ARCH L0042] (MathGeometricPreprocessor.get_flow_features (Vector field))     - 픽셀 값이 가장 급격하게 변하는 방향($dx, dy$)
# [ARCH L0043] (MathGeometricPreprocessor.get_flow_features (Vector field))     - 물체의 **윤곽선에 수직인 방향**을 가리키며, 두 이미지가 정렬될 때 "선의 방향이 일치하는지" 확인하는 기준이 됨
# [ARCH L0044] (MathGeometricPreprocessor.get_flow_features (Vector field)) - **Texture Flow (텍스처 흐름)**
# [ARCH L0045] (MathGeometricPreprocessor.get_flow_features (Vector field))     - 질감(결)이 반복되거나 흐르는 방향($fx, fy$)
# [ARCH L0046] (MathGeometricPreprocessor.get_flow_features (Vector field))     - 엣지가 없는 매끄러운 표면에서도 **"결의 방향"**을 알 수 있게 하여, 특징이 부족한 영역에서도 매칭의 단서를 제공
# [ARCH L0047] (MathGeometricPreprocessor.get_flow_features (Vector field)) 
# [ARCH L0048] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) ### **3. B (Bivector 후보): 회전의 잠재력**
# [ARCH L0049] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) 
# [ARCH L0050] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) - **Bivector Candidate:** 벡터와 벡터 사이의 외적(Wedge Product)을 통해 생성될 **회전 성분의 씨앗**
# [ARCH L0051] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) - **의미:** 단순히 점이 이동하는 것을 넘어, "이 구역은 시계 방향으로 돌아가 있는가?"를 판단하기 위한 재료. 나중에 Phase 2에서 **Rotor(회전 연산자)**를 만드는 결정적인 근거가 됩니다.
# [ARCH L0052] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) 
# [ARCH L0053] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) ---
# [ARCH L0054] (MathGeometricPreprocessor.get_flow_features (Bivector candidate wedge)) 
# =============================================================================
"""
================================================================================
Phase 1: Geometry Raw Data 추출 (Physical Raw Data Extraction)
================================================================================
[Architecture.md §1 참조]

원본 이미지를 다양한 해상도로 리사이즈한 후, 각 층에서 클리포드 멀티벡터의 
재료가 될 물리량들을 추출합니다.

출력 구성:
- S (Scalar): Texture, Structure Energy, Edge Magnitude, SDF
- V (Vector): Gradient Vector (dx, dy), Texture Flow Vector (fx, fy)
- B (Bivector 후보): Wedge Product 기반 회전 잠재력 (bivector_candidate)
- Global Context: 이미지 전체의 통계 요약 (v_shape)

피라미드 스케일 의미:
- 1/32 ~ 1/16 (Global): 전역적인 배치 파악
- 1/8 ~ 1/4 (Structural): 주요 특징점들의 기하학적 관계
- 1/2 ~ 1 (Fine): 초정밀 정렬용 세밀한 텍스처/엣지
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# [Hyperparameters] Phase 1
# =============================================================================
CLAHE_CLIP_LIMIT = 3.0           # [Hyperparameter] CLAHE 클리핑 한계
CLAHE_TILE_SIZE = (8, 8)         # [Hyperparameter] CLAHE 타일 크기
STRUCTURE_TENSOR_SIGMA = 1.0     # [Hyperparameter] Structure Tensor 적분 스케일
STRUCTURE_TENSOR_KSIZE = (5, 5)  # [Hyperparameter] Structure Tensor 커널 크기
EDGE_MAG_GAMMA = 0.7             # [Hyperparameter] Edge Magnitude 감마 보정
STRUCT_ENERGY_GAMMA = 0.4        # [Hyperparameter] Structure Energy 감마 보정
SDF_SKELETON_POWER = 8.0         # [Hyperparameter] SDF 뼈대 선명도
SDF_FIELD_POWER = 2.0            # [Hyperparameter] SDF 필드 부드러움
SDF_FIELD_WEIGHT = 0.4           # [Hyperparameter] SDF 필드 가중치
CANNY_LOW_THRESHOLD = 30         # [Hyperparameter] Canny 엣지 하한
CANNY_HIGH_THRESHOLD = 100       # [Hyperparameter] Canny 엣지 상한


class MathGeometricPreprocessor:
    """
    [Phase 1 메인 클래스]
    이미지로부터 클리포드 대수의 기초가 되는 물리량(S, V, B 후보)을 추출합니다.
    
    Architecture.md §1 전체에 해당
    """
    
    def __init__(self, device="cuda"):
        """
        [Phase 1 초기화]
        CLAHE(Contrast Limited Adaptive Histogram Equalization)를 설정하여
        조명 변화에 강건한 텍스처 추출을 준비합니다.
        
        Architecture.md §1.1 - Scalar: Texture 추출 준비
        """
        self.device = device
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, 
            tileGridSize=CLAHE_TILE_SIZE
        )
        print("Phase 1 Preprocessor Initialized. (Physical Raw Data Extraction)")

    def normalize_minmax(self, img_data):
        """
        [Phase 1 Helper] Min-Max 정규화
        모든 물리량을 0~1 범위로 통일합니다.
        """
        img_min = img_data.min()
        img_max = img_data.max()
        if img_max - img_min < 1e-6:
            return img_data 
        return (img_data - img_min) / (img_max - img_min)

    def get_flow_features(self, gray_img):
        """
        [Phase 1 - §1.2 Vector 추출] Dual-Vector System
        
        자연물(털, 풀 등)에 강한 '두 가지 벡터'를 추출합니다.
        모델에게 '밝기 변화 방향'과 '무늬의 결 방향'을 동시에 제공합니다.
        
        Returns:
            - edge_magnitude (Scalar): 엣지의 세기 [Architecture.md §1.1 - Edge Magnitude]
            - structure_energy (Scalar): 결의 선명도 [Architecture.md §1.1 - Structure Energy]
            - v1_x, v1_y (Vector 1): Gradient Vector [Architecture.md §1.2 - Gradient]
            - v2_x, v2_y (Vector 2): Texture Flow Vector [Architecture.md §1.2 - Texture Flow]
            - bivector_candidate (Bivector 후보): v1 ∧ v2 (Wedge Product) [Architecture.md §1.3 - B]
        """
        img_float = gray_img.astype(np.float32) / 255.0

        # =========================================================
        # [Part 1] Gradient Vector (V1): "경계선을 찾는 화살표"
        # Architecture.md §1.2 - Gradient (경계 변화)
        # =========================================================
        # 의미: 밝기가 급격하게 변하는 방향 (Edge의 수직 방향)
        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # 벡터의 크기(Magnitude) - Edge Magnitude로 사용
        mag = np.sqrt(gx**2 + gy**2)
        edge_magnitude = np.power(self.normalize_minmax(mag), EDGE_MAG_GAMMA)

        # 단위 벡터로 정규화 (순수 '방향'만 보존)
        v1_x = gx / (mag + 1e-6)
        v1_y = gy / (mag + 1e-6)

        # =========================================================
        # [Part 2] Structure Tensor (J): "흐름의 통계적 분석"
        # Architecture.md §1.2 - Texture Flow 계산을 위한 중간 단계
        # =========================================================
        Ixx = cv2.GaussianBlur(gx**2, STRUCTURE_TENSOR_KSIZE, STRUCTURE_TENSOR_SIGMA)
        Iyy = cv2.GaussianBlur(gy**2, STRUCTURE_TENSOR_KSIZE, STRUCTURE_TENSOR_SIGMA)
        Ixy = cv2.GaussianBlur(gx*gy, STRUCTURE_TENSOR_KSIZE, STRUCTURE_TENSOR_SIGMA)

        # =========================================================
        # [Part 3] Structure Energy (Scalar): "결의 선명도"
        # Architecture.md §1.1 - Structure Energy
        # =========================================================
        # 값이 큼: 털이나 풀처럼 결이 뚜렷함
        # 값이 작음: 민무늬거나 무작위 노이즈
        structure_energy = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
        structure_energy = self.normalize_minmax(structure_energy)
        structure_energy = np.power(structure_energy, STRUCT_ENERGY_GAMMA)

        # =========================================================
        # [Part 4] Texture Flow Vector (V2): "결을 따라가는 화살표"
        # Architecture.md §1.2 - Texture Flow
        # =========================================================
        # Gradient(V1)와 수직인 방향 - 엣지나 무늬가 '흘러가는' 방향
        angle = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        v2_x = -np.sin(angle)
        v2_y = np.cos(angle)
        
        # V2에 에너지를 곱하여 '선명한 결'과 '노이즈' 구분
        v2_x = v2_x * structure_energy
        v2_y = v2_y * structure_energy

        # =========================================================
        # [Part 5] Bivector Candidate (B): "회전의 잠재력 (Wedge Product)"
        # Architecture.md §1.3 - B (Bivector 후보)
        # =========================================================
        # 2D wedge product: v1 ∧ v2 = v1_x * v2_y - v1_y * v2_x
        # 의미: 단순 이동이 아닌, 지역적 회전(시계/반시계)의 씨앗
        bivector_candidate = (v1_x * v2_y) - (v1_y * v2_x)
        # [-1, 1]로 정규화 (부호=회전 방향, 크기=신뢰도)
        b_max = np.max(np.abs(bivector_candidate)) + 1e-6
        bivector_candidate = bivector_candidate / b_max

        return edge_magnitude, structure_energy, v1_x, v1_y, v2_x, v2_y, bivector_candidate

    def get_edge_sdf(self, gray_img):
        """
        [Phase 1 - §1.1 SDF 추출] Hybrid SDF Generator
        
        Architecture.md §1.1 - SDF (Signed Distance Field)
        
        뼈대(Skeleton)의 선명함과 잠재장(Potential Field)의 부드러움을 융합합니다.
        - Core: 얇고 진한 뼈대 (위치 정확성)
        - Aura: 넓게 퍼지는 장 (탐색 범위 확대)
        
        MPC(제어) 단계에서 가장 핵심적인 정보로, 두 물체가 얼마나 떨어져 있는지
        '에너지'로 계산할 수 있게 해주는 잠재적인 중력장 역할
        """
        # 1. Canny Edge 검출
        edges = cv2.Canny(gray_img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        
        # 2. Distance Transform (L2가 물리적으로 더 부드러운 원형)
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # 3. Base SDF (0.0 ~ 1.0, 엣지 위는 1.0)
        max_val = dist.max() + 1e-6
        base_sdf = 1.0 - (dist / max_val)
        
        # --- 4. Fusion Strategy ---
        # (A) Skeleton Component: 얇고 날카로운 뼈대
        skeleton = np.power(base_sdf, SDF_SKELETON_POWER)
        
        # (B) Field Component: 부드러운 경사
        field = np.power(base_sdf, SDF_FIELD_POWER)
        
        # (C) Hybrid Fusion: 뼈대 선명함 유지 + 주변에 약한 장
        sdf = np.maximum(skeleton, field * SDF_FIELD_WEIGHT)
        
        return sdf

    def _extract_raw_features(self, img_rgb):
        """
        [Phase 1 Core] 단일 이미지에서 모든 물리량 추출
        
        Architecture.md §1 전체 구현
        
        Returns:
            dict: {
                'rgb': 원본 이미지,
                'hsi': [Texture, Structure Energy, Edge Magnitude] (Scalars),
                'sdf': Signed Distance Field (Scalar),
                'gradient': [dx, dy, fx, fy] (Vectors),
                'v_shape': Global Context 통계
            }
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # [§1.1 Scalar] Texture 추출
        texture = self.clahe.apply(gray).astype(np.float32) / 255.0
        
        # [§1.2 Vector] + [§1.1 Scalar: Edge Mag, Struct Energy]
        edge_mag, struct_energy, dx, dy, fx, fy, b_candidate = self.get_flow_features(gray)
        
        # [§1.1 Scalar] SDF 추출
        sdf_map = self.get_edge_sdf(gray)
        
        # --- Scalar 패키징 ---
        # Texture, Structure Energy, Edge Magnitude (모두 방향 없는 스칼라)
        hsi_replacement = np.stack([texture, struct_energy, edge_mag], axis=-1)
        
        # --- Vector 패키징 ---
        # V1 (Gradient) + V2 (Texture Flow) = 4 channels
        vector_field = np.stack([dx, dy, fx, fy], axis=-1)
        
        # --- Global Context ($G$) ---
        # 이미지 전체의 통계 요약 (6차원)
        v_shape = np.array([
            np.mean(edge_mag), np.std(edge_mag),           # 엣지 분포
            np.mean(struct_energy), np.std(struct_energy), # 구조 분포
            np.mean(texture), np.std(texture)              # 밝기 분포
        ], dtype=np.float32)

        return {
            'rgb': img_rgb,
            'hsi': hsi_replacement,
            'sdf': sdf_map,
            'gradient': vector_field,
            'bivector': b_candidate,  # [Architecture.md §1.3] Bivector 후보 (Wedge Product)
            'v_shape': v_shape
        }
    
    def process_pyramid(self, img_rgb, levels=6):
        """
        [Phase 1 Main Pipeline] 이미지 피라미드 생성
        
        Architecture.md §1 서두 - 다양한 해상도로 리사이즈
        
        Args:
            img_rgb: 입력 RGB 이미지
            levels: 피라미드 레벨 수 (기본 6: 원본 ~ 1/32)
        
        Returns:
            list of dicts: [Level0_Data, Level1_Data, ..., Level(n-1)_Data]
        """
        pyramid_data = []
        current_img = img_rgb.copy()
        
        for i in range(levels):
            # 1. 현재 스케일 특징 추출
            features = self._extract_raw_features(current_img)
            features['level_index'] = i
            features['resolution'] = current_img.shape[:2]
            pyramid_data.append(features)
            
            # 2. 다운샘플링 (Gaussian Pyramid - 앨리어싱 방지)
            if i < levels - 1:
                current_img = cv2.pyrDown(current_img)
                
        return pyramid_data


# =============================================================================
# [시각화 헬퍼 함수]
# =============================================================================

def vector_to_rgb(vx, vy):
    """
    [Helper] 벡터 필드를 HSV 색상 공간을 이용해 RGB 이미지로 변환
    - 색상(Hue): 벡터의 방향
    - 밝기(Value): 벡터의 크기
    """
    magnitude, angle = cv2.cartToPolar(vx, vy)
    hsv = np.zeros((vx.shape[0], vx.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_pyramid_detailed(pyramid_results):
    """
    [Phase 1 시각화] 피라미드 전체 시각화
    """
    levels = len(pyramid_results)
    cols = 8
    
    plt.figure(figsize=(24, 3.5 * levels))
    plt.suptitle("Phase 1: Multi-Scale Geometric Pyramid Analysis", 
                 fontsize=24, fontweight='bold', y=0.99)
    
    for i, data in enumerate(pyramid_results):
        h, w = data['resolution']
        img_rgb = data['rgb']
        hsi = data['hsi']
        texture = hsi[:,:,0]
        struct_energy = hsi[:,:,1]
        edge_mag = hsi[:,:,2]
        sdf = data['sdf']
        
        vec = data['gradient']
        v1_x, v1_y = vec[..., 0], vec[..., 1]
        v2_x, v2_y = vec[..., 2], vec[..., 3]
        
        rgb_v1 = vector_to_rgb(v1_x, v1_y)
        rgb_v2 = vector_to_rgb(v2_x, v2_y)
        
        importance = (texture * 0.2) + (struct_energy * 0.5) + (edge_mag * 0.3)
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

        base_idx = i * cols
        
        plt.subplot(levels, cols, base_idx + 1)
        plt.imshow(img_rgb)
        plt.ylabel(f"Level {i}\n({h}x{w})", fontsize=14, fontweight='bold')
        if i == 0: plt.title("1. Original", fontsize=12, fontweight='bold')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(levels, cols, base_idx + 2)
        plt.imshow(texture, cmap='gray')
        if i == 0: plt.title("2. Texture (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(levels, cols, base_idx + 3)
        plt.imshow(edge_mag, cmap='viridis')
        if i == 0: plt.title("3. Edge Mag (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(levels, cols, base_idx + 4)
        plt.imshow(struct_energy, cmap='inferno')
        if i == 0: plt.title("4. Struct Energy (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(levels, cols, base_idx + 5)
        plt.imshow(rgb_v1)
        if i == 0: plt.title("5. Gradient Vec (V1)", fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(levels, cols, base_idx + 6)
        plt.imshow(rgb_v2)
        if i == 0: plt.title("6. Flow Vec (V2)", fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(levels, cols, base_idx + 7)
        plt.imshow(sdf, cmap='coolwarm')
        if i == 0: plt.title("7. SDF Potential (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.subplot(levels, cols, base_idx + 8)
        plt.imshow(img_rgb)
        plt.imshow(importance, cmap='jet', alpha=0.5)
        if i == 0: plt.title("8. Attention Map", fontsize=12, fontweight='bold', color='red')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 실행 및 테스트
# =============================================================================
if __name__ == "__main__":
    IMG_PATH = "./img/val2017/000000569972.jpg"
    img = cv2.imread(IMG_PATH)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processor = MathGeometricPreprocessor()
        pyramid_results = processor.process_pyramid(img_rgb, levels=6)
        visualize_pyramid_detailed(pyramid_results)
    else:
        print("Image Not Found!")
