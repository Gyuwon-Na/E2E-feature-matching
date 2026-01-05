import cv2
import numpy as np
import matplotlib.pyplot as plt

class MathGeometricPreprocessor:
    def __init__(self, device="cuda"):
        self.device = device
        # [Scalar Part 준비] 조명 불변성(Invariant) 확보
        # CLAHE는 빛의 영향을 제거하고 순수 '텍스처(무늬)'만 남깁니다.
        # -> 나중에 Clifford Scalar($S$)의 핵심 재료가 됩니다.
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) 
        print("📐 Mathematical Geometry Preprocessor V2 (Signal Boosted).")

    def normalize_minmax(self, img_data):
        """
        [신호 증폭기]
        데이터의 범위를 강제로 0~1로 늘려줍니다.
        아무리 작은 값이라도 가장 큰 놈은 1, 작은 놈은 0이 됩니다.
        """
        img_min = img_data.min()
        img_max = img_data.max()
        if img_max - img_min < 1e-6:
            return img_data # 변화 없으면 그대로 리턴
        return (img_data - img_min) / (img_max - img_min)

    def get_geometric_features(self, gray_img):
        """
        [수학적 특징 추출기]
        딥러닝 없이 수학 공식(Sobel, Harris)만으로 
        Vector($V$)와 Keypoint($S/B$)의 원료를 뽑아냅니다.
        """
        img_float = gray_img.astype(np.float32) / 255.0

        # --- 1. 엣지 강도 (Gradient Magnitude) ---
        # 역할: 나중에 [Vector($V$)]를 만드는 핵심 재료
        # Sobel 필터로 가로(gx), 세로(gy) 변화량을 감지 -> 엣지의 세기 계산
        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(gx**2 + gy**2)
        edge_magnitude = self.normalize_minmax(edge_magnitude) # 증폭!

        # --- 2. 코너 강도 (Harris Corner Response) ---
        # 역할: 나중에 [Scalar($S$)]의 가중치 혹은 [Bivector($B$)]의 후보
        # Harris Corner 알고리즘으로 '꺾이는 부분(특징점)'을 찾음
        dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
        
        # 코너 값을 Log 스케일로 변환해서 미세한 코너도 살려냄
        dst = np.abs(dst)
        dst = np.log1p(dst) 
        corner_strength = self.normalize_minmax(dst) # 증폭!
        
        # [신호 부스팅] 중간값들을 끌어올려 코너가 더 잘 보이게 함
        corner_strength = np.power(corner_strength, 0.5)

        # --- 3. 곡률/복잡도 (Laplacian Variance) ---
        # 역할: [Global Context($G$)]를 계산하기 위한 통계용 데이터
        # 평평한 곳은 0, 텍스처가 복잡한 곳은 높은 값을 가짐
        lap = cv2.Laplacian(img_float, cv2.CV_32F)
        complexity = np.abs(lap)
        complexity = self.normalize_minmax(complexity) # 증폭!

        return edge_magnitude, corner_strength, complexity

    def get_edge_sdf(self, gray_img):
        """
        [거리 필드 생성기]
        역할: [Vector($V$)]의 '위치/형상 정보'를 담당.
        Canny로 딴 뼈대로부터 얼마나 떨어져 있는지를 계산합니다.
        """
        # Canny 임계값을 좀 더 낮춰서(민감하게) 엣지를 더 많이 잡게 설정
        edges = cv2.Canny(gray_img, 30, 100) 
        
        # 뼈대(255)는 거리 0, 배경(0)은 거리가 늘어남 -> 반전시켜서 계산
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # 정규화: 뼈대 위가 1.0, 멀어질수록 0.0
        max_val = dist.max() + 1e-6
        sdf = 1.0 - (dist / max_val)
        
        # SDF 명암 대비를 높임 (Pow) -> 뼈대를 더 얇고 진하게 (Vector 위치 특정에 유리)
        sdf = np.power(sdf, 2.0) 
        return sdf

    def process_from_array(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Texture (Invariant)
        # -> [Scalar Part ($S$)]의 재료: 변하지 않는 무늬 정보
        texture = self.clahe.apply(gray).astype(np.float32) / 255.0
        
        # 2. Geometry Features (Updated)
        edge_mag, corner, complexity = self.get_geometric_features(gray)
        sdf_map = self.get_edge_sdf(gray)
        
        # 3. 채널 포장 (Packing)
        # 다음 단계(Clifford Embedding)에서 아래 순서대로 채널을 꺼내 씁니다.
        # hsi[:,:,0]: Texture  -> Scalar ($S$)
        # hsi[:,:,1]: Corner   -> Scalar ($S$) (Keypoints)
        # hsi[:,:,2]: Edge Mag -> Vector ($V$) Magnitude
        hsi_replacement = np.stack([texture, corner, edge_mag], axis=-1)
        
        # 4. Global Context ($G$) 생성
        # v_shape 통계 (Logits)
        # -> [Global Context($G$)]로 쓰임 (모델 전체에 이미지 분위기 전달)
        v_shape = np.array([
            np.mean(edge_mag), np.std(edge_mag),   # 엣지가 많은가?
            np.mean(corner), np.std(corner),       # 코너가 많은가?
            np.mean(complexity), np.std(complexity)# 텍스처가 복잡한가?
        ], dtype=np.float32)

        return {'hsi': hsi_replacement, 'sdf': sdf_map, 'v_shape': v_shape}
    
def visualize_phase1_outputs(img_path):
    print(f"🔹 Analyzing Phase 1 (Math-Based Geometry) for: {img_path}")

    # 1. 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. 전처리 실행 (Math-based)
    preprocessor = MathGeometricPreprocessor()
    data = preprocessor.process_from_array(img_rgb)
    
    # 3. 데이터 분리
    hsi_feat = data['hsi']
    sdf_map = data['sdf']
    
    texture = hsi_feat[:, :, 0]
    corner_score = hsi_feat[:, :, 1]
    coherence = hsi_feat[:, :, 2]
    
    # 4. 시각화
    plot_math_features(img_rgb, texture, corner_score, coherence, sdf_map)

def plot_math_features(img_rgb, texture, corner, edge_mag, sdf):
    # (앞부분 서브플롯 1~5는 동일하게 유지하거나 필요시 수정)
    # 여기서는 Overlay 부분만 집중적으로 수정합니다.
    
    plt.figure(figsize=(20, 10))
    plt.suptitle("Phase 1 Analysis V3: Robust Normalization Output", fontsize=22, fontweight='bold')

    # ... (1~5번 plot 코드 생략 - 위와 동일하게 배치) ...
    # 편의를 위해 다시 적어드립니다.
    
    plt.subplot(2, 3, 1); plt.imshow(img_rgb); plt.title("1. Original")
    plt.subplot(2, 3, 2); plt.imshow(texture, cmap='gray'); plt.title("2. Texture (Surface)")
    plt.subplot(2, 3, 3); plt.imshow(corner, cmap='inferno'); plt.title("3. Corner (Keypoints)")
    plt.subplot(2, 3, 4); plt.imshow(edge_mag, cmap='viridis'); plt.title("4. Edge Mag (Structure)")
    plt.subplot(2, 3, 5); plt.imshow(sdf, cmap='coolwarm'); plt.title("5. SDF (Skeleton)")

    # --- 6. 🔥 Final Importance Overlay (Logic Changed) ---
    plt.subplot(2, 3, 6)
    
    # [수정] 중요도 = 텍스처(면) + 코너(점) + SDF(뼈대)
    # 텍스처가 0.4 비중으로 들어가야 전체적으로 밝아집니다.
    importance_map = (texture * 0.4) + (corner * 0.4) + (sdf * 0.2)
    
    # 0~1로 다시 맞춤
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-6)
    
    plt.imshow(img_rgb)
    # 투명도(alpha)를 0.7로 높여서 히트맵이 더 잘 보이게 함
    plt.imshow(importance_map, cmap='jet', alpha=0.7) 
    plt.title("6. Final Input Visualized\n(Should show Red/Yellow on Owl)", fontsize=15, fontweight='bold', color='red')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # 이미지 경로를 본인의 경로로 수정하세요
    IMG_PATH = "./img/val2017/000000581206.jpg" 
    
    visualize_phase1_outputs(IMG_PATH)