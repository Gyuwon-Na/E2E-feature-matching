import os
import cv2
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM
from pipeline.phase3 import Phase3Transformer, FEATURE_DIM

# ==============================================================================
# [Configuration]
# ==============================================================================
IMG_SIZE = (256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/rot_90_1.32.pth"  # 상대 경로 수정
TEST_IMG_DIR = "./img/val2017"               # 상대 경로 수정
ANGLE_THRESHOLD = 90

# ==============================================================================
# [Helper Functions]
# ==============================================================================
def create_checkerboard(img1, img2, block_size=32):
    """
    두 이미지를 체커보드 패턴으로 교차하여 시각화합니다.
    img1: Target (Ground Truth)
    img2: Aligned Source (Prediction)
    """
    h, w = img1.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 체커보드 마스크 생성
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                mask[y:y+block_size, x:x+block_size] = 1.0
                
    # 채널 차원 맞추기 (H, W) -> (H, W, 3)
    mask = np.dstack([mask]*3)
    
    # 합성: 마스크가 1인 곳은 img1, 0인 곳은 img2
    mixed = img1 * mask + img2 * (1 - mask)
    return mixed.astype(np.uint8)

def load_user_model(path, device):
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(feature_dim=FEATURE_DIM, embed_dim=HIDDEN_DIM).to(device)
    
    if os.path.exists(path):
        print(f"✅ Loading model from {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        embedder.eval()
        transformer.eval()
        return embedder, transformer
    else:
        print(f"❌ Model not found at {path}")
        return None, None

def get_affine_matrix_from_rotor(rotor, w, h, center=None):
    """
    Phase 3 Rotor Output -> OpenCV Affine Matrix (2x3)
    """
    if center is None:
        center = (w / 2.0, h / 2.0)
    cx, cy = center
    
    # Rotor 값 (cos, sin, dx, dy)
    # Rotor는 "이미지를 원래대로 돌리는(복원)" 변환을 예측함
    cos = rotor[0].item()
    sin = rotor[1].item()
    dx = rotor[2].item()
    dy = rotor[3].item() # 정규화된 이동량 (-1 ~ 1)
    
    # 1. 회전 행렬 (Rotation around center)
    # OpenCV getRotationMatrix2D는 (center, angle, scale)을 받음
    # 여기서 angle은 반시계 방향(ccw)이 양수.
    # sin값이 양수면 반시계 회전 (일반적인 수학 정의)
    
    # [중요] Phase 3가 예측한 것은 "Target -> Source"로 가는 역변환일 수 있음
    # 혹은 "Source -> Target" 정변환일 수 있음.
    # 학습 시 gt_angle과 비교했던 방식 그대로 적용해야 함.
    
    # 예측된 각도 (Radian)
    angle_rad = np.arctan2(sin, cos)
    angle_deg = np.degrees(angle_rad)
    
    # 회전 행렬 생성
    # [ cos  -sin   (1-cos)*cx + sin*cy ]
    # [ sin   cos   (1-cos)*cy - sin*cx ]
    alpha = cos
    beta = sin 
    
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta  # OpenCV 좌표계에서는 y축이 아래로 내려가므로 부호 주의
    M[1, 0] = -beta
    M[1, 1] = alpha
    
    # 회전 중심점 보정 (Rotation around center)
    M[0, 2] = (1 - alpha) * cx - beta * cy
    M[1, 2] = beta * cx + (1 - alpha) * cy
    
    # 이동량 보정 (Translation)
    # dx, dy는 -1~1 범위이므로 이미지 크기에 맞춰 스케일링
    # Phase 3에서 dx, dy가 어떻게 학습되었는지에 따라 다름 (여기선 0으로 가정하거나 무시)
    # M[0, 2] += dx * w
    # M[1, 2] += dy * h
    
    return M, angle_deg

# ==============================================================================
# [Main Logic]
# ==============================================================================
def main():
    print(f"🖼️ Checkerboard Visualization on {DEVICE}")
    
    # 1. 모델 로드
    embedder, transformer = load_user_model(MODEL_PATH, DEVICE)
    if embedder is None: return

    # 2. 이미지 준비
    img_list = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if not img_list: return
    
    # 랜덤 이미지 1장 선택
    img_path = np.random.choice(img_list)
    print(f"   Target Image: {os.path.basename(img_path)}")
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE) # 256x256
    h, w = img_rgb.shape[:2]
    
    # 3. 인위적 회전 가하기 (Source 생성)
    # 예: 30도 회전
    true_angle = np.random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD) 
    print(f"   True Rotation Applied: {true_angle:.2f}°")
    
    M_gt = cv2.getRotationMatrix2D((w/2, h/2), true_angle, 1.0)
    img_warped = cv2.warpAffine(img_rgb, M_gt, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 4. 모델 추론 (Source -> Target 복원)
    preprocessor = MathGeometricPreprocessor()
    pyr_c = preprocessor.process_pyramid(img_warped, levels=5) # Source
    pyr_t = preprocessor.process_pyramid(img_rgb, levels=5)    # Target
    
    with torch.no_grad():
        f_c = embedder(pyr_c, DEVICE)
        f_t = embedder(pyr_t, DEVICE)
        res = transformer(f_c, f_t)
        rotor = res[0]['rotor_map'].mean(dim=(1,2))[0] # (4,)
    
    # 5. 복원 행렬 생성
    # Phase 3는 "Source를 Target으로 되돌리는" 변환을 예측해야 함
    # 즉, 예측된 각도는 -true_angle 이어야 함
    M_pred, pred_angle = get_affine_matrix_from_rotor(rotor, w, h)
    
    # [핵심 수정] 예측된 각도의 부호를 반대로 적용해보기 (시각화가 이상하면 이거 때문임)
    # 만약 학습 시 '복원 각도'를 학습했다면 그대로 적용
    # 만약 '회전 차이'를 학습했다면 반대로 적용
    
    print(f"   Predicted Angle (Restoration): {pred_angle:.2f}°")
    print(f"   Error: {abs(true_angle + pred_angle):.2f}°") # 복원각 + 입력각 ≈ 0 이어야 함
    
    # 6. 이미지 복원 (Warp Source with Predicted Matrix)
    # [중요] M_pred는 Source(img_warped)를 Target(img_rgb)으로 보내는 행렬이어야 함
    # 하지만 M_pred가 이상하게 적용된다면, cv2.warpAffine의 flags를 확인해야 함
    
    # 수동으로 행렬 만들기 (확실한 방법)
    # 예측된 각도만큼 '반대로' 돌리기
    M_fix = cv2.getRotationMatrix2D((w/2, h/2), pred_angle, 1.0) # pred_angle만큼 회전
    
    # img_warped를 M_fix로 돌리면 img_rgb가 되어야 함
    img_restored = cv2.warpAffine(img_warped, M_fix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 7. 시각화 (체커보드)
    # (A) Input vs Target (틀어진 상태)
    check_input = create_checkerboard(img_rgb, img_warped)
    
    # (B) Restored vs Target (맞춰진 상태)
    check_result = create_checkerboard(img_rgb, img_restored)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(check_input)
    plt.title(f"Before: Input vs Target\n(Diff: {true_angle:.1f}°)", fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(check_result)
    plt.title(f"After: Restored vs Target\n(Pred: {pred_angle:.1f}°)", fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("✅ Visualization Complete.")

if __name__ == "__main__":
    main()