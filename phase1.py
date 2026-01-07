import cv2
import numpy as np
import matplotlib.pyplot as plt

class MathGeometricPreprocessor:
    def __init__(self, device="cuda"):
        self.device = device
        # [Scalar Part] 조명 불변성 (유지)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        print("Phase 1 Preprocessor Initialized. (Physical Raw Data Extraction)")

    def normalize_minmax(self, img_data):
        img_min = img_data.min()
        img_max = img_data.max()
        if img_max - img_min < 1e-6:
            return img_data 
        return (img_data - img_min) / (img_max - img_min)

    def get_flow_features(self, gray_img):
        """
        [Vector Generator V_Final: Dual-Vector System]
        
        기존의 코너(Corner)를 제거하고, 자연물(털, 풀 등)에 강한 '두 가지 벡터'를 추출합니다.
        모델에게 '밝기 변화 방향'과 '무늬의 결 방향'을 동시에 제공하여 입체적인 분석을 유도합니다.
        
        Returns:
            1. edge_magnitude (Scalar): 엣지의 세기 (힘의 크기 1)
            2. structure_energy (Scalar): 결의 선명도 (힘의 크기 2)
            3. v1_x, v1_y (Vector 1): Gradient Vector (가파른 경사 방향)
            4. v2_x, v2_y (Vector 2): Texture Flow Vector (결이 흐르는 방향)
        """
        # 이미지 정규화 (0.0 ~ 1.0)
        img_float = gray_img.astype(np.float32) / 255.0

        # =========================================================
        # [Part 1] Gradient Vector ($V_1$): "경계선을 찾는 화살표"
        # =========================================================
        # 의미: 밝기가 어두운 곳에서 밝은 곳으로 가장 급격하게 변하는 방향 (Edge의 수직 방향)
        # 역할: 물체의 외곽선이나 뚜렷한 경계를 감지함.
        
        # 1차 미분 (Sobel)
        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3) # 가로 변화량
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3) # 세로 변화량
        
        # 벡터의 크기(Magnitude) 계산
        # -> 나중에 Scalar 가중치로 쓰임 (Edge가 센 곳은 V1을 강하게 반영)
        mag = np.sqrt(gx**2 + gy**2)
        edge_magnitude = np.power(self.normalize_minmax(mag), 0.7)

        # 단위 벡터(Unit Vector)로 정규화 ($e_1, e_2$ 성분)
        # 크기 정보는 edge_magnitude에 줬으니, 여기선 순수 '방향'만 남김
        # (0으로 나누기 방지를 위해 엡실론 1e-6 추가)
        v1_x = gx / (mag + 1e-6)
        v1_y = gy / (mag + 1e-6)

        # =========================================================
        # [Part 2] Structure Tensor ($J$): "흐름의 통계적 분석"
        # =========================================================
        # 의미: 픽셀 하나만 보는 게 아니라, 주변 이웃들의 흐름을 행렬로 요약함.
        # J = [[ Ixx, Ixy ], 
        #      [ Ixy, Iyy ]]
        
        sigma = 1.0  # 적분 스케일 (주변을 얼마나 넓게 볼 것인가)
        ksize = (5,5)
        # 미분값의 제곱을 블러링(Gaussian)하여 '평균적인 경향성'을 파악
        Ixx = cv2.GaussianBlur(gx**2, ksize, sigma)
        Iyy = cv2.GaussianBlur(gy**2, ksize, sigma)
        Ixy = cv2.GaussianBlur(gx*gy, ksize, sigma)

        # =========================================================
        # [Part 3] Structure Energy (Scalar): "결의 선명도"
        # =========================================================
        # 의미: 이 구역의 무늬가 '한 방향으로 잘 정렬되어 있는가?' (Coherence)
        # - 값이 큼: 털이나 풀처럼 결이 뚜렷함 -> V2(결 방향)를 신뢰함
        # - 값이 작음: 민무늬거나 무작위 노이즈 -> V2를 무시함
        
        # 고유값(Eigenvalue) 차이를 이용한 이방성(Anisotropy) 계산
        # sqrt((Ixx - Iyy)^2 + 4*Ixy^2) 공식은 (lambda1 - lambda2)와 비례함
        structure_energy = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
        structure_energy = self.normalize_minmax(structure_energy)
        structure_energy = np.power(structure_energy, 0.4)

        # =========================================================
        # [Part 4] Texture Flow Vector ($V_2$): "결을 따라가는 화살표"
        # =========================================================
        # 의미: Gradient($V_1$)와 수직인 방향. 즉, 엣지나 무늬가 '흘러가는' 방향.
        # 역할: 끊어진 엣지를 연결하거나, 털/풀밭의 자라난 방향을 추적함. (Clifford 회전 불변성에 기여)
        
        # 구조 텐서의 각도(Angle) 계산 (Gradient 방향의 각도)
        # 범위: -pi/2 ~ pi/2
        angle = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        
        # $V_2$ 생성: Gradient 각도에 90도를 더하거나, 수직 벡터(-sin, cos)를 취함
        # 이것이 바로 '결(Texture Flow)'의 방향 벡터임
        v2_x = -np.sin(angle)
        v2_y = np.cos(angle)
        
        # [중요] V2에 에너지(강도)를 곱해서 리턴합니다.
        # 이렇게 해야 시각화할 때 '선명한 결'과 '노이즈'가 구분됩니다.
        # (단위 벡터만 보내면 시각화 함수가 헷갈려 함)
        v2_x = v2_x * structure_energy
        v2_y = v2_y * structure_energy
        # ---------------------------------------------------------
        # [최종 반환]
        # Scalars (2개): [엣지세기, 구조에너지] -> 가중치(Magnitude)용
        # Vectors (4개): [Grad_X, Grad_Y, Flow_X, Flow_Y] -> 기하학적 방향($V$)용
        # ---------------------------------------------------------
        return edge_magnitude, structure_energy, v1_x, v1_y, v2_x, v2_y

    def get_edge_sdf(self, gray_img):
        """
        [Hybrid SDF Generator]
        뼈대(Skeleton)의 선명함과 잠재장(Potential Field)의 부드러움을 융합합니다.
        - Core: 얇고 진한 뼈대 (위치 정확성)
        - Aura: 넓게 퍼지는 장 (탐색 범위 확대)
        """
        # 1. Canny Edge (기존 유지)
        edges = cv2.Canny(gray_img, 30, 100) 
        
        # 2. Distance Transform
        # L1(Diamond)보다는 L2(Euclidean)가 물리적으로 더 부드러운 원형을 만듭니다.
        # MPC나 벡터 필드에는 L2가 더 유리합니다.
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # 3. Base SDF (0.0 ~ 1.0)
        # 엣지 위는 1.0, 멀어질수록 0.0
        max_val = dist.max() + 1e-6
        base_sdf = 1.0 - (dist / max_val)
        
        # --- 4. Fusion Strategy (핵심) ---
        
        # (A) Skeleton Component (이미지 1 스타일)
        # 거듭제곱을 높게(8~10) 주면, 1.0 근처만 살아남고 나머지는 급격히 0이 됨.
        # 결과: 아주 얇고 날카로운 뼈대
        skeleton = np.power(base_sdf, 8.0)
        
        # (B) Field Component (이미지 2 스타일)
        # 거듭제곱을 낮게(1~2) 주면, 멀리까지 값이 살아있음.
        # 결과: 부드러운 경사(Gradient)
        field = np.power(base_sdf, 2.0)
        
        # (C) Hybrid Fusion
        # 뼈대의 선명함을 유지하면서(Skeleton), 주변에 약한 장(Field)을 깐다.
        # np.maximum을 쓰면 뼈대가 흐려지지 않고 그대로 유지됨!
        # Field에 0.5를 곱해서 배경은 은은하게 만듦.
        sdf = np.maximum(skeleton, field * 0.4)
        
        return sdf

    def _extract_raw_features(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Texture (Scalar S) - 유지
        texture = self.clahe.apply(gray).astype(np.float32) / 255.0
        
        # 2. Vector Components (V) - 대폭 강화!
        # Corner는 이제 없습니다. 대신 Flow 정보들이 들어옵니다.
        edge_mag, struct_energy, dx, dy, fx, fy = self.get_flow_features(gray)
        sdf_map = self.get_edge_sdf(gray)
        
        # --- 3. 채널 포장 (Packing: Scalar Components) ---
        # 여기 들어가는 3가지는 모두 '방향'이 없는 [스칼라(Scalar)] 값들입니다.
        # 나중에 Clifford Embedding에서 벡터의 '길이(Magnitude)'나 '가중치(Weight)'로 쓰입니다.
        # - texture: 픽셀의 밝기/색상 정보 (0차원 정보)
        # - struct_energy: 털이나 결이 '얼마나 선명한지' 세기 (방향 X, 강도 O)
        # - edge_mag: 엣지가 '얼마나 강한지' 세기 (방향 X, 강도 O)
        hsi_replacement = np.stack([texture, struct_energy, edge_mag], axis=-1)
        
        # 4. Vector Packing (Total 4 Channels)
        # V1 (Gradient) + V2 (Texture Flow)
        # 나중에 모델은 이 4개를 받아서 (2개 벡터)로 인식하게 설계하면 됨
        vector_field = np.stack([dx, dy, fx, fy], axis=-1)
        
        # 4. Global Context ($G$)
        # 이미지 전체의 분위기(통계)를 요약합니다.
        # 코너(점) 대신, 이제는 '흐름(Structure)'과 '엣지(Edge)'의 분포를 봅니다.
        v_shape = np.array([
            np.mean(edge_mag), np.std(edge_mag),       # 전체적으로 선이 많은가?
            np.mean(struct_energy), np.std(struct_energy), # 털/결이 전체적으로 뚜렷한가? (코너 대체)
            np.mean(texture), np.std(texture)          # 배경이 밝은가 어두운가?
        ], dtype=np.float32)

        return {
            'rgb': img_rgb,                # [Original Image]: 원본 이미지 -> (B, 3, H, W)
            'hsi': hsi_replacement,   # [Scalar Bundle]: 텍스처, 구조세기, 엣지세기 -> (B, 3, H, W)
            'sdf': sdf_map,           # [Potential Field]: 뼈대 에너지 -> (B, 1, H, W)
            'gradient': vector_field, # [Vector Core]: 진짜 방향 정보 (dx, dy, fx, fy) ->(B, H, W, 4) **핵심**
            'v_shape': v_shape        # [Global Context]: 전역 통계 -> (B, 6)
        }
    
    def process_pyramid(self, img_rgb, levels=6):
        """
        [Main Pipeline]
        이미지 피라미드를 생성하고(1 ~ 1/32), 각 레벨별 물리량을 추출합니다.
        
        Returns:
            list of dicts: [Level0_Data, Level1_Data, ..., Level5_Data]
        """
        pyramid_data = []
        current_img = img_rgb.copy()
        
        # print(f"Processing Pyramid (Levels={levels})...")
        
        for i in range(levels):
            # 1. 현재 스케일 특징 추출
            features = self._extract_raw_features(current_img)
            features['level_index'] = i
            features['resolution'] = current_img.shape[:2]
            pyramid_data.append(features)
            
            # 2. 다음 스케일을 위한 다운샘플링 (Gaussian Pyramid)
            # cv2.pyrDown은 가우시안 블러 후 해상도를 1/2로 줄이므로 앨리어싱 방지에 탁월
            if i < levels - 1:
                current_img = cv2.pyrDown(current_img)
                
        return pyramid_data

# --- 벡터 시각화 헬퍼 함수 (필수 추가) ---
def vector_to_rgb(vx, vy):
    """
    벡터 필드(vx, vy)를 HSV 색상 공간을 이용해 RGB 이미지로 변환합니다.
    - 색상(Hue): 벡터의 방향 (어디를 가리키는지)
    - 밝기(Value): 벡터의 크기 (얼마나 센지)
    """
    # 1. 극좌표계 변환 (Cartesian -> Polar)
    magnitude, angle = cv2.cartToPolar(vx, vy)

    # 2. HSV 이미지 생성
    hsv = np.zeros((vx.shape[0], vx.shape[1], 3), dtype=np.uint8)
    
    # Hue: 각도 (0~360도 매핑) -> OpenCV Hue range is [0, 179]
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Saturation: 255 (색을 진하게)
    hsv[..., 1] = 255
    
    # Value: 벡터 크기 (정규화)
    # 노이즈 제거를 위해 최소/최대값으로 클리핑 후 스케일링
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm

    # 3. HSV -> RGB 변환
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

# --- 메인 시각화 함수 업데이트 ---
def visualize_pyramid_detailed(pyramid_results):
    levels = len(pyramid_results)
    cols = 8 # 보여줄 항목 수
    
    # Figure 높이를 레벨 수에 비례하게 설정 (한 레벨당 높이 3인치)
    plt.figure(figsize=(24, 3.5 * levels))
    plt.suptitle("Phase 1: Multi-Scale Geometric Pyramid Analysis", fontsize=24, fontweight='bold', y=0.99)
    
    for i, data in enumerate(pyramid_results):
        h, w = data['resolution']
        
        # --- 데이터 추출 ---
        # 1. Image
        img_rgb = data['rgb']

        # 2. Scalars
        hsi = data['hsi']
        texture = hsi[:,:,0]       # Texture
        struct_energy = hsi[:,:,1] # Structure Energy
        edge_mag = hsi[:,:,2]      # Edge Magnitude
        sdf = data['sdf']          # SDF
        
        # 3. Vectors
        vec = data['gradient']
        v1_x, v1_y = vec[..., 0], vec[..., 1] # Gradient
        v2_x, v2_y = vec[..., 2], vec[..., 3] # Flow
        
        # 4. Vector to RGB 변환
        rgb_v1 = vector_to_rgb(v1_x, v1_y)
        rgb_v2 = vector_to_rgb(v2_x, v2_y)
        
        # 5. Attention Map 생성
        importance = (texture * 0.2) + (struct_energy * 0.5) + (edge_mag * 0.3)
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

        # --- Plotting (Row: Level, Col: Features) ---
        base_idx = i * cols
        
        # Column 1: Original Image
        plt.subplot(levels, cols, base_idx + 1)
        plt.imshow(img_rgb)
        plt.ylabel(f"Level {i}\n({h}x{w})", fontsize=14, fontweight='bold')
        if i == 0: plt.title("1. Original", fontsize=12, fontweight='bold')
        plt.xticks([]), plt.yticks([])
        
        # Column 2: Texture
        plt.subplot(levels, cols, base_idx + 2)
        plt.imshow(texture, cmap='gray')
        if i == 0: plt.title("2. Texture (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Column 3: Edge Magnitude
        plt.subplot(levels, cols, base_idx + 3)
        plt.imshow(edge_mag, cmap='viridis')
        if i == 0: plt.title("3. Edge Mag (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Column 4: Structure Energy
        plt.subplot(levels, cols, base_idx + 4)
        plt.imshow(struct_energy, cmap='inferno')
        if i == 0: plt.title("4. Struct Energy (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Column 5: Gradient Vector (V1)
        plt.subplot(levels, cols, base_idx + 5)
        plt.imshow(rgb_v1)
        if i == 0: plt.title("5. Gradient Vec (V1)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Column 6: Texture Flow Vector (V2)
        plt.subplot(levels, cols, base_idx + 6)
        plt.imshow(rgb_v2)
        if i == 0: plt.title("6. Flow Vec (V2)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Column 7: SDF
        plt.subplot(levels, cols, base_idx + 7)
        plt.imshow(sdf, cmap='coolwarm')
        if i == 0: plt.title("7. SDF Potential (S)", fontsize=12, fontweight='bold')
        plt.axis('off')

        # Column 8: Attention Map
        plt.subplot(levels, cols, base_idx + 8)
        plt.imshow(img_rgb)
        plt.imshow(importance, cmap='jet', alpha=0.5)
        if i == 0: plt.title("8. Attention Map", fontsize=12, fontweight='bold', color='red')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 실행 ---
if __name__ == "__main__":
    IMG_PATH = "./img/val2017/000000569972.jpg"  # 이미지 경로 수정 필요
    img = cv2.imread(IMG_PATH)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        processor = MathGeometricPreprocessor()
        
        # 4단계 피라미드 생성 (Level 0 ~ Level 3)
        pyramid_results = processor.process_pyramid(img_rgb, levels=6)
        
        # 상세 시각화 실행
        visualize_pyramid_detailed(pyramid_results)
    else:
        print("Image Not Found!")