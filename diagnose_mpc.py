"""
================================================================================
MPC Diagnostic Tool - MPC가 왜 안 되는지 진단
================================================================================
진단 항목:
1. Transformer 출력이 MPC 수렴 범위 내인가?
2. MPC 에너지 함수가 제대로 감소하는가?
3. MPC iterations가 충분한가?
4. 각 에너지 성분(SDF, Vector, Rotor)의 기여도는?

사용법:
    python diagnose_mpc.py --checkpoint ./checkpoints/best_model.pth
================================================================================
"""

import sys
import numpy as np

# [긴급 패치] Colab(NumPy 2.x)에서 만든 모델을 로컬(NumPy 1.x)에서 억지로 열기
# 로컬엔 'numpy._core'가 없으므로, 기존 'numpy.core'를 가리키도록 사기(?)를 칩니다.
try:
    import numpy._core
except ImportError:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

import os
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# [Hyperparameters]
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = (256, 256)


def diagnose_mpc(checkpoint_path: str, data_dir: str, num_samples: int = 10):
    """MPC 진단 실행"""
    
    from phase1 import MathGeometricPreprocessor
    from phase2 import CliffordPyramidEmbedder
    from phase3 import Phase3Transformer
    from phase4 import GeometricMPCRefiner
    
    # 모델 로드
    preprocessor = MathGeometricPreprocessor()
    embedder = CliffordPyramidEmbedder(hidden_dim=48).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=144, embed_dim=48).to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
        print(f"✅ Loaded: {checkpoint_path}")
    
    embedder.eval()
    transformer.eval()
    
    # 이미지 로드
    img_paths = list(Path(data_dir).glob('*.jpg'))[:num_samples]
    
    # 진단 결과 저장
    diagnostics = {
        'transformer_angle_errors': [],
        'mpc_angle_errors': [],
        'mpc_improvements': [],
        'mpc_loss_curves': [],
        'energy_components': [],  # [e_scalar, e_vector, e_bivector] per sample
        'convergence_status': [],  # 'converged', 'stuck', 'diverged'
    }
    
    test_angles = [-20, -15, -10, -5, 5, 10, 15, 20]
    
    for i, img_path in enumerate(tqdm(img_paths, desc="Diagnosing")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        gt_angle = test_angles[i % len(test_angles)]
        H, W = img.shape[:2]
        
        # 변환 적용
        M = cv2.getRotationMatrix2D((W/2, H/2), gt_angle, 1.0)
        img_warped = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        
        # GT 역변환
        M_aug = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_aug)[:2]
        N = np.array([[2.0/W, 0, -1], [0, 2.0/H, -1], [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_inv_aug = np.vstack([M_inv, [0, 0, 1]])
        gt_H_norm = (N @ M_inv_aug @ N_inv)[:2]
        
        # Phase 1-3
        def add_batch(pyramid):
            return [{k: v[np.newaxis, ...] if isinstance(v, np.ndarray) else v 
                    for k, v in level.items()} for level in pyramid]
        
        with torch.no_grad():
            pyramid_a = add_batch(preprocessor.process_pyramid(img_warped, levels=4))
            pyramid_b = add_batch(preprocessor.process_pyramid(img, levels=4))
            
            phase2_a = embedder(pyramid_a, DEVICE)
            phase2_b = embedder(pyramid_b, DEVICE)
            
            results = transformer(phase2_a, phase2_b)
            rotor = results[0]['rotor_map']
            avg_rotor = rotor.mean(dim=(1, 2))
            
            cos_t, sin_t = avg_rotor[0, 0].item(), avg_rotor[0, 1].item()
            dx, dy = avg_rotor[0, 2].item(), avg_rotor[0, 3].item()
            
            mag = np.sqrt(cos_t**2 + sin_t**2 + 1e-6)
            cos_t, sin_t = cos_t / mag, sin_t / mag
            
            transformer_H = np.array([[cos_t, -sin_t, dx], [sin_t, cos_t, dy]])
        
        # Transformer 에러 계산
        pred_angle = np.degrees(np.arctan2(transformer_H[1, 0], transformer_H[0, 0]))
        gt_angle_from_H = np.degrees(np.arctan2(gt_H_norm[1, 0], gt_H_norm[0, 0]))
        transformer_error = abs(pred_angle - gt_angle_from_H)
        diagnostics['transformer_angle_errors'].append(transformer_error)
        
        # =====================================================================
        # MPC 진단 (상세 로깅)
        # =====================================================================
        mpc = GeometricMPCRefiner(device=DEVICE)
        mpc.iterations = 200  # 충분히 많이
        
        # 초기화
        angle_init = np.arctan2(sin_t, cos_t)
        scale_init = mag
        mpc.global_filtering_init(mean_rotor=angle_init, mean_scale=scale_init)
        
        # 데이터 준비
        s_a, v_a, b_a = phase2_a[0]
        s_b, v_b, b_b = phase2_b[0]
        
        # [수정] pyramid_a는 이미 add_batch로 인해 (1, H, W) 형태입니다.
        # 채널 차원(C) 하나만 추가하여 (1, 1, H, W)로 만들어야 합니다.
        sdf_src = torch.tensor(pyramid_a[0]['sdf']).float().to(DEVICE)
        if sdf_src.ndim == 3:
            sdf_src = sdf_src.unsqueeze(1)
            
        sdf_tgt = torch.tensor(pyramid_b[0]['sdf']).float().to(DEVICE)
        if sdf_tgt.ndim == 3:
            sdf_tgt = sdf_tgt.unsqueeze(1)

        src_data = {
            'sdf': sdf_src,
            # 주의: vector와 rotor 차원도 (B, C, H, W) 형태인지 확인이 필요할 수 있습니다.
            # v_a.mean(dim=1)을 사용하는 것으로 보아 Transformer 출력 포맷에 맞춰져 있는 듯 합니다.
            # 만약 여기서도 차원 에러가 난다면 v_a의 shape을 print해보세요.
            'vector': v_a.mean(dim=1).detach().clone(),
            'rotor': b_a[2].mean(dim=1, keepdim=True).detach().clone()
        }
        tgt_data = {
            'sdf': sdf_tgt,
            'vector': v_b.mean(dim=1).detach().clone(),
            'rotor': b_b[2].mean(dim=1, keepdim=True).detach().clone()
        }
        
        g_s = torch.sigmoid(torch.mean(torch.abs(s_a), dim=1, keepdim=True)).detach()
        g_v = torch.sigmoid(torch.mean(torch.norm(v_a, dim=2), dim=1, keepdim=True)).detach()
        g_b = torch.sigmoid(torch.mean(b_a[2], dim=1, keepdim=True)).detach()
        gates = (g_s, g_v, g_b)
        
        # 에너지 성분 분석
        with torch.no_grad():
            grid = mpc.get_affine_grid(src_data['sdf'].shape)
            
            warped_sdf = F.grid_sample(src_data['sdf'], grid, align_corners=False)
            warped_vector = F.grid_sample(src_data['vector'], grid, align_corners=False)
            warped_rotor = F.grid_sample(src_data['rotor'], grid, align_corners=False)
            
            e_scalar = torch.abs(F.softplus(warped_sdf) - F.softplus(tgt_data['sdf'])).mean().item()
            e_vector = (1.0 - F.cosine_similarity(warped_vector, tgt_data['vector'], dim=1)).mean().item()
            e_bivector = torch.abs(warped_rotor - tgt_data['rotor']).mean().item()
            
            diagnostics['energy_components'].append([e_scalar, e_vector, e_bivector])
        
        # MPC 실행 (loss curve 기록)
        try:
            loss_history = mpc.optimize(src_data, tgt_data, gates)
            diagnostics['mpc_loss_curves'].append(loss_history)
            
            # 수렴 상태 판단
            if len(loss_history) > 10:
                initial_loss = loss_history[0]
                final_loss = loss_history[-1]
                mid_loss = loss_history[len(loss_history)//2]
                
                if final_loss < initial_loss * 0.5:
                    status = 'converged'
                elif abs(final_loss - initial_loss) < initial_loss * 0.1:
                    status = 'stuck'
                elif final_loss > initial_loss:
                    status = 'diverged'
                else:
                    status = 'partial'
            else:
                status = 'unknown'
            
            diagnostics['convergence_status'].append(status)
            
            # MPC 결과
            mpc_H = mpc.W.detach().cpu().numpy()[0]
            mpc_angle = np.degrees(np.arctan2(mpc_H[1, 0], mpc_H[0, 0]))
            mpc_error = abs(mpc_angle - gt_angle_from_H)
            
        except Exception as e:
            print(f"MPC failed: {e}")
            mpc_error = transformer_error
            diagnostics['convergence_status'].append('failed')
            diagnostics['mpc_loss_curves'].append([])
        
        diagnostics['mpc_angle_errors'].append(mpc_error)
        diagnostics['mpc_improvements'].append(transformer_error - mpc_error)
    
    # =====================================================================
    # 진단 결과 시각화
    # =====================================================================
    visualize_diagnostics(diagnostics)
    
    return diagnostics


def visualize_diagnostics(diagnostics):
    """진단 결과 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Transformer vs MPC 에러 비교
    ax1 = axes[0, 0]
    x = range(len(diagnostics['transformer_angle_errors']))
    ax1.bar([i - 0.2 for i in x], diagnostics['transformer_angle_errors'], 
            width=0.4, label='Transformer', color='steelblue', alpha=0.8)
    ax1.bar([i + 0.2 for i in x], diagnostics['mpc_angle_errors'], 
            width=0.4, label='After MPC', color='coral', alpha=0.8)
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Angle Error (°)')
    ax1.set_title('Transformer vs MPC Error')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. MPC 개선량
    ax2 = axes[0, 1]
    improvements = diagnostics['mpc_improvements']
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Improvement (°)')
    ax2.set_title(f'MPC Improvement\n(Green=Better, Mean: {np.mean(improvements):.2f}°)')
    ax2.grid(alpha=0.3)
    
    # 3. Loss Curves
    ax3 = axes[0, 2]
    for i, curve in enumerate(diagnostics['mpc_loss_curves'][:5]):  # 처음 5개만
        if curve:
            ax3.plot(curve, label=f'Sample {i}', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_title('MPC Loss Curves (First 5 Samples)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. 에너지 성분 분석
    ax4 = axes[1, 0]
    energy = np.array(diagnostics['energy_components'])
    if len(energy) > 0:
        ax4.bar(['E_scalar\n(SDF)', 'E_vector\n(Direction)', 'E_bivector\n(Rotor)'],
                energy.mean(axis=0), 
                yerr=energy.std(axis=0),
                color=['blue', 'green', 'orange'], alpha=0.7, capsize=5)
    ax4.set_ylabel('Energy Value')
    ax4.set_title('Energy Components (Mean ± Std)')
    ax4.grid(alpha=0.3)
    
    # 5. 수렴 상태 분포
    ax5 = axes[1, 1]
    status_counts = {}
    for s in diagnostics['convergence_status']:
        status_counts[s] = status_counts.get(s, 0) + 1
    
    colors_map = {'converged': 'green', 'partial': 'yellow', 'stuck': 'orange', 
                  'diverged': 'red', 'failed': 'gray', 'unknown': 'gray'}
    ax5.bar(status_counts.keys(), status_counts.values(),
            color=[colors_map.get(k, 'gray') for k in status_counts.keys()], alpha=0.8)
    ax5.set_ylabel('Count')
    ax5.set_title('MPC Convergence Status')
    ax5.grid(alpha=0.3)
    
    # 6. 진단 요약
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    t_errs = diagnostics['transformer_angle_errors']
    m_errs = diagnostics['mpc_angle_errors']
    imps = diagnostics['mpc_improvements']
    
    summary = f"""
    ╔══════════════════════════════════════════════════╗
    ║              MPC DIAGNOSTIC SUMMARY              ║
    ╠══════════════════════════════════════════════════╣
    ║                                                  ║
    ║  Transformer Output:                             ║
    ║    Mean Error: {np.mean(t_errs):>6.2f}° ± {np.std(t_errs):.2f}°              ║
    ║    Max Error:  {np.max(t_errs):>6.2f}°                        ║
    ║                                                  ║
    ║  After MPC:                                      ║
    ║    Mean Error: {np.mean(m_errs):>6.2f}° ± {np.std(m_errs):.2f}°              ║
    ║    Max Error:  {np.max(m_errs):>6.2f}°                        ║
    ║                                                  ║
    ║  MPC Effect:                                     ║
    ║    Mean Improvement: {np.mean(imps):>+6.2f}°                  ║
    ║    Helped: {sum(1 for i in imps if i > 0):>3d}/{len(imps)} samples                   ║
    ║    Hurt:   {sum(1 for i in imps if i < 0):>3d}/{len(imps)} samples                   ║
    ║                                                  ║
    ╠══════════════════════════════════════════════════╣
    ║  DIAGNOSIS:                                      ║
    """
    
    # 진단 결과
    if np.mean(t_errs) > 12:
        summary += "║  ⚠️  Transformer error too high (>12°)          ║\n"
        summary += "║      → MPC cannot recover from bad init         ║\n"
    elif np.mean(imps) < 0:
        summary += "║  ⚠️  MPC making things worse!                   ║\n"
        summary += "║      → Check energy function / learning rate    ║\n"
    elif np.mean(imps) < 1:
        summary += "║  ⚠️  MPC not helping much                       ║\n"
        summary += "║      → May need more iterations or tuning       ║\n"
    else:
        summary += "║  ✅  MPC is helping!                             ║\n"
    
    summary += "╚══════════════════════════════════════════════════╝"
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('MPC Diagnostic Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./mpc_diagnostic.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved: ./mpc_diagnostic.png")
    plt.show()


# =============================================================================
# 추가 진단: Transformer만의 문제인지 확인
# =============================================================================

def diagnose_transformer_only(checkpoint_path: str, data_dir: str):
    """
    Transformer 출력만 분석 (MPC 없이)
    
    Transformer가 특정 각도에서만 실패하는지 확인
    """
    from phase1 import MathGeometricPreprocessor
    from phase2 import CliffordPyramidEmbedder
    from phase3 import Phase3Transformer
    
    preprocessor = MathGeometricPreprocessor()
    embedder = CliffordPyramidEmbedder(hidden_dim=48).to(DEVICE)
    transformer = Phase3Transformer(feature_dim=144, embed_dim=48).to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        embedder.load_state_dict(ckpt['embedder'])
        transformer.load_state_dict(ckpt['transformer'])
    
    embedder.eval()
    transformer.eval()
    
    img_paths = list(Path(data_dir).glob('*.jpg'))[:5]
    
    # 다양한 각도 테스트
    test_angles = list(range(-25, 26, 5))  # -25 to 25, step 5
    
    results = {angle: [] for angle in test_angles}
    
    for img_path in tqdm(img_paths, desc="Testing angles"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        H, W = img.shape[:2]
        
        for gt_angle in test_angles:
            M = cv2.getRotationMatrix2D((W/2, H/2), gt_angle, 1.0)
            img_warped = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            
            # GT
            M_aug = np.vstack([M, [0, 0, 1]])
            M_inv = np.linalg.inv(M_aug)[:2]
            N = np.array([[2.0/W, 0, -1], [0, 2.0/H, -1], [0, 0, 1]])
            N_inv = np.linalg.inv(N)
            gt_H = (N @ np.vstack([M_inv, [0, 0, 1]]) @ N_inv)[:2]
            
            def add_batch(pyramid):
                return [{k: v[np.newaxis, ...] if isinstance(v, np.ndarray) else v 
                        for k, v in level.items()} for level in pyramid]
            
            with torch.no_grad():
                pyramid_a = add_batch(preprocessor.process_pyramid(img_warped, levels=4))
                pyramid_b = add_batch(preprocessor.process_pyramid(img, levels=4))
                
                phase2_a = embedder(pyramid_a, DEVICE)
                phase2_b = embedder(pyramid_b, DEVICE)
                
                res = transformer(phase2_a, phase2_b)
                rotor = res[0]['rotor_map'].mean(dim=(1, 2))
                
                cos_t, sin_t = rotor[0, 0].item(), rotor[0, 1].item()
                mag = np.sqrt(cos_t**2 + sin_t**2 + 1e-6)
                
                pred_angle = np.degrees(np.arctan2(sin_t/mag, cos_t/mag))
                gt_angle_h = np.degrees(np.arctan2(gt_H[1, 0], gt_H[0, 0]))
                
                error = abs(pred_angle - gt_angle_h)
                results[gt_angle].append(error)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    
    angles = list(results.keys())
    means = [np.mean(results[a]) for a in angles]
    stds = [np.std(results[a]) for a in angles]
    
    ax.errorbar(angles, means, yerr=stds, fmt='o-', capsize=5, 
                color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(angles, 
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2)
    
    ax.axhline(y=5, color='green', linestyle='--', label='Good (<5°)')
    ax.axhline(y=10, color='orange', linestyle='--', label='OK (<10°)')
    ax.axhline(y=15, color='red', linestyle='--', label='Bad (>15°)')
    
    ax.set_xlabel('Ground Truth Angle (°)', fontsize=12)
    ax.set_ylabel('Prediction Error (°)', fontsize=12)
    ax.set_title('Transformer Error vs GT Angle\n(Without MPC)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./transformer_angle_profile.png', dpi=150)
    print("\n📊 Saved: ./transformer_angle_profile.png")
    plt.show()
    
    # 요약
    print("\n" + "="*50)
    print("TRANSFORMER ANGLE PROFILE SUMMARY")
    print("="*50)
    for angle in angles:
        mean_err = np.mean(results[angle])
        status = "✅" if mean_err < 5 else ("⚠️" if mean_err < 10 else "❌")
        print(f"  {angle:>3d}° → Error: {mean_err:>5.2f}° {status}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='./val2017')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'transformer', 'both'])
    parser.add_argument('--num_samples', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.mode in ['full', 'both']:
        print("\n" + "="*60)
        print("FULL MPC DIAGNOSTIC")
        print("="*60)
        diagnose_mpc(args.checkpoint, args.data_dir, args.num_samples)
    
    if args.mode in ['transformer', 'both']:
        print("\n" + "="*60)
        print("TRANSFORMER-ONLY DIAGNOSTIC")
        print("="*60)
        diagnose_transformer_only(args.checkpoint, args.data_dir)