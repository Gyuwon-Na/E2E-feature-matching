import cv2
import numpy as np
import matplotlib.pyplot as plt

class MathGeometricPreprocessor:
    def __init__(self, device="cuda"):
        self.device = device
        # [Scalar Part] 조명 불변성 (유지)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        print("📐 Mathematical Geometry Preprocessor V_Final (Dense Flow & Structure).")

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
        edge_magnitude = self.normalize_minmax(mag)

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
        ksize = (5, 5)
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

        # ---------------------------------------------------------
        # [최종 반환]
        # Scalars (2개): [엣지세기, 구조에너지] -> 가중치(Magnitude)용
        # Vectors (4개): [Grad_X, Grad_Y, Flow_X, Flow_Y] -> 기하학적 방향($V$)용
        # ---------------------------------------------------------
        return edge_magnitude, structure_energy, v1_x, v1_y, v2_x, v2_y

    def get_edge_sdf(self, gray_img):
        # [SDF 유지] 뼈대 추출 로직 그대로 사용
        edges = cv2.Canny(gray_img, 30, 100) 
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        max_val = dist.max() + 1e-6
        sdf = 1.0 - (dist / max_val)
        sdf = np.power(sdf, 2.0) 
        return sdf

    def process_from_array(self, img_rgb):
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
            'hsi': hsi_replacement,   # [Scalar Bundle]: 텍스처, 구조세기, 엣지세기 -> (B, 3, H, W)
            'sdf': sdf_map,           # [Potential Field]: 뼈대 에너지 -> (B, 1, H, W)
            'gradient': vector_field, # [Vector Core]: 진짜 방향 정보 (dx, dy, fx, fy) ->(B, H, W, 4) **핵심**
            'v_shape': v_shape        # [Global Context]: 전역 통계 -> (B, 6)
        }

# --- 시각화 함수 업데이트 (코너 대신 Structure Energy 확인) ---
def visualize_phase1_outputs(img_path):
    print(f"🔹 Analyzing Phase 1 (Dense Flow & Structure) for: {img_path}")
    img = cv2.imread(img_path)
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    preprocessor = MathGeometricPreprocessor()
    data = preprocessor.process_from_array(img_rgb)
    
    hsi = data['hsi']
    texture = hsi[:,:,0]
    struct_energy = hsi[:,:,1] # 구 Corner 자리
    edge_mag = hsi[:,:,2]
    sdf = data['sdf']
    
    plt.figure(figsize=(20, 10))
    plt.suptitle("Phase 1: Flow-Based Vector Analysis (No Corner)", fontsize=22, fontweight='bold')
    
    plt.subplot(2,3,1); plt.imshow(img_rgb); plt.title("1. Original")
    plt.subplot(2,3,2); plt.imshow(texture, cmap='gray'); plt.title("2. Texture (Scalar)")
    
    # 여기가 핵심 변화!
    plt.subplot(2,3,3); plt.imshow(struct_energy, cmap='inferno')
    plt.title("3. Structure Energy (Flow Strength)\n*Replaces Corner*")
    
    plt.subplot(2,3,4); plt.imshow(edge_mag, cmap='viridis'); plt.title("4. Edge Magnitude")
    plt.subplot(2,3,5); plt.imshow(sdf, cmap='coolwarm'); plt.title("5. SDF (Skeleton)")
    
    # Overlay 확인
    plt.subplot(2,3,6)
    # 텍스처(배경) + 구조(흐름) + SDF(뼈대)
    importance = (texture * 0.3) + (struct_energy * 0.5) + (sdf * 0.2)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)
    
    plt.imshow(img_rgb)
    plt.imshow(importance, cmap='jet', alpha=0.6)
    plt.title("6. Final V-Field Visualization", fontsize=15, fontweight='bold', color='red')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 실행 부
if __name__ == "__main__":
    # 사용자 이미지 경로
    IMG_PATH = "./img/val2017/000000569972.jpg" 
    visualize_phase1_outputs(IMG_PATH)