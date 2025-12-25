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
def visualize_loftr_vs_ours(
    img_src, img_tgt, ours_data, loftr_data, H_mat, threshold=10.0, num_show=150, conf_thresh=0.5
):
    """
    상단: LoFTR 매칭 결과
    하단: Ours(Clifford + Phase4) 매칭 결과
    🔥 검은색 영역 및 신뢰도 필터링 적용
    """
    H, W, _ = img_src.shape
    canvas = np.hstack((img_src, img_tgt))
    plt.figure(figsize=(20, 14))

    # -------------------------------------------
    # 1. 상단: LoFTR 매칭 결과
    # -------------------------------------------
    plt.subplot(2, 1, 1)
    plt.imshow(canvas)
    plt.title(f"Baseline: LoFTR Matching Results (Thresh={threshold}px)", fontsize=16, fontweight='bold')
    plt.axis('off')

    mkpts0, mkpts1 = loftr_data
    if len(mkpts0) > 0:
        # GT 계산 (이미지 B 내 유효성 검사)
        src_pts_loftr = mkpts0.reshape(-1, 1, 2)
        gt_pts_loftr = cv2.perspectiveTransform(src_pts_loftr, H_mat).reshape(-1, 2)
        
        # 필터링: 이미지 B 범위 안 + 검은색이 아닌 영역
        mask_loftr = []
        for i in range(len(gt_pts_loftr)):
            gx, gy = int(gt_pts_loftr[i, 0]), int(gt_pts_loftr[i, 1])
            if 0 <= gx < W and 0 <= gy < H and np.any(img_tgt[gy, gx] > 0):
                mask_loftr.append(True)
            else:
                mask_loftr.append(False)
        mask_loftr = np.array(mask_loftr)
        
        valid_idx_loftr = np.where(mask_loftr)[0]
        if len(valid_idx_loftr) > 0:
            show_idx = np.random.choice(valid_idx_loftr, min(num_show, len(valid_idx_loftr)), replace=False)
            correct = 0
            for i in valid_idx_loftr:
                err = np.linalg.norm(mkpts1[i] - gt_pts_loftr[i])
                if err < threshold: correct += 1
                
                if i in show_idx:
                    color = 'lime' if err < threshold else 'red'
                    plt.plot([mkpts0[i, 0], mkpts1[i, 0] + W], [mkpts0[i, 1], mkpts1[i, 1]], color=color, lw=1, alpha=0.6)
                    plt.scatter(mkpts0[i, 0], mkpts0[i, 1], c=color, s=10)
                    plt.scatter(mkpts1[i, 0] + W, mkpts1[i, 1], c=color, s=10)
            
            acc_loftr = (correct / len(valid_idx_loftr)) * 100
            plt.text(10, 40, f"LoFTR Acc: {acc_loftr:.1f}% ({correct}/{len(valid_idx_loftr)} pts)", 
                     color='white', backgroundcolor='black', fontsize=14, fontweight='bold')

    # -------------------------------------------
    # 2. 하단: Ours 매칭 결과
    # -------------------------------------------
    plt.subplot(2, 1, 2)
    plt.imshow(canvas)
    plt.title(f"Ours: Clifford + Phase4 Refinement (Conf > {conf_thresh})", fontsize=16, fontweight='bold')
    plt.axis('off')

    init_verts, final_verts, confidences = ours_data
    # 픽셀 좌표 복원
    src_pts_norm = init_verts.detach().cpu().numpy()
    src_x = (src_pts_norm[:, 0] + 1) * 0.5 * W
    src_y = (src_pts_norm[:, 1] + 1) * 0.5 * H
    pred_pts_norm = final_verts.detach().cpu().numpy()
    pred_x = (pred_pts_norm[:, 0] + 1) * 0.5 * W
    pred_y = (pred_pts_norm[:, 1] + 1) * 0.5 * H
    
    # GT 계산 및 필터링
    src_pts_pixel = np.stack([src_x, src_y], axis=1).reshape(-1, 1, 2)
    gt_pts_ours = cv2.perspectiveTransform(src_pts_pixel, H_mat).reshape(-1, 2)
    
    mask_ours = []
    for i in range(len(gt_pts_ours)):
        gx, gy = int(gt_pts_ours[i, 0]), int(gt_pts_ours[i, 1])
        # 조건: 이미지 범위 안 + 검은색 아님 + 신뢰도 통과
        if 0 <= gx < W and 0 <= gy < H and np.any(img_tgt[gy, gx] > 0) and confidences[i] >= conf_thresh:
            mask_ours.append(True)
        else:
            mask_ours.append(False)
    mask_ours = np.array(mask_ours)
    
    valid_idx_ours = np.where(mask_ours)[0]
    if len(valid_idx_ours) > 0:
        show_idx_ours = np.random.choice(valid_idx_ours, min(num_show, len(valid_idx_ours)), replace=False)
        correct_ours = 0
        for i in valid_idx_ours:
            err = np.linalg.norm(np.array([pred_x[i], pred_y[i]]) - gt_pts_ours[i])
            if err < threshold: correct_ours += 1
            
            if i in show_idx_ours:
                color = 'lime' if err < threshold else 'red'
                plt.plot([src_x[i], pred_x[i] + W], [src_y[i], pred_y[i]], color=color, lw=1, alpha=0.6)
                plt.scatter(src_x[i], src_y[i], c=color, s=10)
                plt.scatter(pred_x[i] + W, pred_y[i], c=color, s=10)
                
        acc_ours = (correct_ours / len(valid_idx_ours)) * 100
        plt.text(10, 40, f"Ours Acc: {acc_ours:.1f}% ({correct_ours}/{len(valid_idx_ours)} pts)", 
                 color='white', backgroundcolor='black', fontsize=14, fontweight='bold')

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
    visualize_loftr_vs_ours(
        img_rgb, img_tgt_rgb, 
        ours_data=(refiner.mesh.initial_vertices, final_verts, confidences),
        loftr_data=loftr_kpts,
        H_mat=H_mat,
        threshold=10.0, # 10픽셀 이내면 정답(Green)
        num_show=200,    # 화면에 보여줄 선의 개수
        conf_thresh=0.5  # 신뢰도 0.5 이상만 표시
    )

if __name__ == "__main__":
    TEST_IMAGE = "./img/val2017/000000581482.jpg" 
    MODEL_PATH = "clifford_model_final.pth"
    SAM_PATH = "sam_vit_b_01ec64.pth"
    
    run_evaluation(TEST_IMAGE, MODEL_PATH, SAM_PATH)