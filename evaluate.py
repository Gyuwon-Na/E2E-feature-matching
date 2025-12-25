import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import kornia
from kornia.feature import LoFTR
from train import CliffordNetwork, Phase1Preprocessor, Phase4Refinement, HomographyAugmentor

# ==========================================
# 🔵 LoFTR 매칭 결과 추출 함수
# ==========================================
def get_loftr_matches(img_src, img_tgt, device):
    """
    Kornia를 사용하여 LoFTR 매칭 결과를 가져옵니다.
    """
    matcher = LoFTR(pretrained='outdoor').to(device).eval()
    
    # 그레이스케일 변환 및 텐서화
    img0 = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    img1 = cv2.cvtColor(img_tgt, cv2.COLOR_RGB2GRAY)
    
    timg0 = torch.from_numpy(img0)[None, None].float().to(device) / 255.
    timg1 = torch.from_numpy(img1)[None, None].float().to(device) / 255.
    
    with torch.no_grad():
        correspondences = matcher({'image0': timg0, 'image1': timg1})
        
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    return mkpts0, mkpts1

# ==========================================
# 📏 통합 비교 시각화 (LoFTR vs Ours)
# ==========================================
def visualize_with_error_heatmap(
    img_src, img_tgt, ours_data, H_mat, threshold=5.0, conf_thresh=0.5
):
    """
    1. 왼쪽: 매칭 결과 (Green/Red) - 이미지 B 밖으로 나가는 점은 제외
    2. 오른쪽: 에러 히트맵 - 어디서 오차가 큰지 시각화
    """
    H, W, _ = img_src.shape
    init_verts, final_verts, confidences = ours_data
    
    # 1. 좌표 계산 및 변환
    src_pts_norm = init_verts.detach().cpu().numpy()
    src_x = (src_pts_norm[:, 0] + 1) * 0.5 * W
    src_y = (src_pts_norm[:, 1] + 1) * 0.5 * H
    
    pred_pts_norm = final_verts.detach().cpu().numpy()
    pred_x = (pred_pts_norm[:, 0] + 1) * 0.5 * W
    pred_y = (pred_pts_norm[:, 1] + 1) * 0.5 * H
    
    # 정답 위치 계산
    src_pts_pixel = np.stack([src_x, src_y], axis=1).reshape(-1, 1, 2)
    gt_pts = cv2.perspectiveTransform(src_pts_pixel, H_mat).reshape(-1, 2)
    
    # 2. 마스크 생성 (유효 영역 + 신뢰도)
    mask_range = (gt_pts[:, 0] >= 0) & (gt_pts[:, 0] < W) & \
                 (gt_pts[:, 1] >= 0) & (gt_pts[:, 1] < H)
    mask_conf = confidences >= conf_thresh
    final_mask = mask_range & mask_conf
    
    # 3. 에러 계산 (픽셀 거리 오차)
    # 모든 점에 대해 계산하되, 마스크되지 않은 곳은 나중에 필터링
    errors = np.linalg.norm(np.stack([pred_x, pred_y], axis=1) - gt_pts, axis=1)
    
    plt.figure(figsize=(20, 10))
    
    # --- [왼쪽] 매칭 결과 시각화 ---
    plt.subplot(1, 2, 1)
    canvas = np.hstack((img_src, img_tgt))
    plt.imshow(canvas)
    
    valid_indices = np.where(final_mask)[0]
    if len(valid_indices) > 0:
        # 가독성을 위해 일부 점만 샘플링하여 그리기
        show_idx = np.random.choice(valid_indices, min(150, len(valid_indices)), replace=False)
        for i in show_idx:
            color = 'lime' if errors[i] < threshold else 'red'
            # 오타 수정: pred_py -> pred_y
            plt.plot([src_x[i], pred_x[i] + W], [src_y[i], pred_y[i]], color=color, lw=0.8, alpha=0.6)
            plt.scatter(src_x[i], src_y[i], c=color, s=5)
            plt.scatter(pred_x[i] + W, pred_y[i], c=color, s=5)
            
        acc = (errors[valid_indices] < threshold).mean() * 100
        plt.text(10, 30, f"Acc: {acc:.1f}% ({len(valid_indices)} pts)", 
                 color='white', backgroundcolor='black', fontsize=12)

    plt.title(f"Matching Results (Green < {threshold}px)", fontsize=15)
    plt.axis('off')

    # --- [오른쪽] 에러 히트맵 (Error Heatmap) ---
    plt.subplot(1, 2, 2)
    
    # 격자 해상도 복원 (예: 32x32)
    grid_res = int(np.sqrt(len(errors)))
    # 에러 맵 초기화 (유효하지 않은 곳은 에러 0으로 처리하거나 배경색 처리)
    error_map_flat = np.zeros_like(errors)
    error_map_flat[final_mask] = errors[final_mask]
    
    error_map = error_map_flat.reshape(grid_res, grid_res)
    
    # 히트맵을 원본 이미지 크기에 맞게 확대
    # INTER_CUBIC을 써야 에러 분포가 부드럽게 보입니다.
    error_heatmap = cv2.resize(error_map, (W, H), interpolation=cv2.INTER_CUBIC)
    
    plt.imshow(img_src)
    # 'jet' 맵: 파란색(에러 낮음) -> 빨간색(에러 높음)
    im = plt.imshow(error_heatmap, cmap='jet', alpha=0.5)
    plt.colorbar(im, label='Pixel Error Distance')
    plt.title("Error Heatmap (Red = High Error Area)", fontsize=15)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 🚀 메인 테스트 로직
# ==========================================
def run_evaluation(img_path, model_path, sam_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔹 Running evaluation on {device}...")

    # 1. 모델 로드 (내 모델)
    model = CliffordNetwork().to(device)
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    preprocessor = Phase1Preprocessor(sam_checkpoint=sam_path, model_type="vit_b", device=device)
    augmentor = HomographyAugmentor(128, 128)

    # 2. 이미지 준비
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print("❌ Image load failed.")
        return
    img_orig = cv2.resize(img_orig, (128, 128))
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    # 전처리 및 Target 생성
    data = preprocessor.process_from_array(img_rgb)
    hsi_A, sdf_A, hsi_B, sdf_B, _, H_t = augmentor(None, data['hsi'], data['sdf'])
    
    hsi_A = hsi_A.unsqueeze(0).to(device)
    sdf_A = sdf_A.unsqueeze(0).to(device)
    hsi_B = hsi_B.unsqueeze(0).to(device)
    sdf_B = sdf_B.unsqueeze(0).to(device)

    H_mat = H_t.numpy()
    img_tgt_rgb = cv2.warpPerspective(img_rgb, H_mat, (128, 128))

    # 3. 내 모델 추론
    print("🧠 Extracting Ours Features...")
    with torch.no_grad():
        feat_A, feat_B = model(hsi_A, sdf_A, hsi_B, sdf_B)
        # 🔥 graph 분리
        feat_A = feat_A.detach().clone()
        feat_B = feat_B.detach().clone()

    # 4. Phase 4 정제 (내 모델)
    print("🔧 Phase 4 Refinement...")
    refiner = Phase4Refinement(128, 128).to(device)  # 🔥 device 전달
    refiner.solver.lambdas['reg'] = 0.0001
    refiner.solver.lambdas['data'] = 100.0
    
    # 🔥 source_feat_init도 detach + clone
    with torch.no_grad():
        init_mesh_verts = refiner.mesh.vertices.detach().clone()
        source_feat_init = refiner.solver.sample_features(feat_A, init_mesh_verts)
        source_feat_init = source_feat_init.detach().clone()
    
    # 300 steps 최적화
    final_verts, _ = refiner(feat_B, source_feat_init, steps=300)

    # 🔥 5. 신뢰도(Confidence) 계산 (장치 할당 수정)
    print("📊 Computing Confidence Scores...")
    with torch.no_grad():
        # final_verts는 이미 cuda에 있지만, 
        # refiner.mesh.initial_vertices는 cpu일 수 있으므로 .to(device) 추가
        src_init_verts = refiner.mesh.initial_vertices.to(device) 
        
        pred_feat = refiner.solver.sample_features(feat_B, final_verts)
        init_feat = refiner.solver.sample_features(feat_A, src_init_verts)
        
        # 코사인 유사도 계산
        confidences = F.cosine_similarity(pred_feat, init_feat, dim=1).cpu().numpy()
    
    print(f"   Confidence Stats: min={confidences.min():.3f}, max={confidences.max():.3f}, mean={confidences.mean():.3f}")

    # 6. LoFTR 매칭 수행
    print("🥊 Running LoFTR Baseline...")
    loftr_kpts = get_loftr_matches(img_rgb, img_tgt_rgb, device)

    # 7. 결과 비교 시각화
    print("🎨 Visualizing Comparison...")
    visualize_with_error_heatmap(
        img_rgb, img_tgt_rgb, 
        ours_data=(refiner.mesh.initial_vertices, final_verts, confidences),  # 🔥 confidences 추가
        H_mat=H_mat,
        threshold=15.0,
        conf_thresh=0.5  # 🔥 신뢰도 임계값 설정
    )

if __name__ == "__main__":
    TEST_IMAGE = "./img/val2017/000000579893.jpg" 
    MODEL_PATH = "clifford_model_final.pth"
    SAM_PATH = "sam_vit_b_01ec64.pth"
    
    run_evaluation(TEST_IMAGE, MODEL_PATH, SAM_PATH)