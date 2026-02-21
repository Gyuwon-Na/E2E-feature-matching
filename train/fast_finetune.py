"""
================================================================================
Fast Fine-Tuning: Large Angle Specialist (RTX 3090 24GB Optimized Ver.)
================================================================================
기존 학습 로직은 유지하되, **RTX 3090 24GB vRAM에 최적화**된 설정으로 변경했습니다.
매 10 Step마다 [Loss, Angle Error, Pixel Error]를 출력합니다.

[RTX 3090 최적화 포인트]
1. BATCH_SIZE: 2 → 8 (4배 증가, vRAM 활용도 극대화)
2. ACCUM_STEPS: 16 → 8 (절반으로 감소, 학습 속도 향상)
3. Effective Batch: 32 → 64 (2배 증가, gradient 안정성 향상)
4. LEARNING_RATE: 5e-5 → 8e-5 (배치 증가에 따른 Linear Scaling)
5. WARMUP_EPOCHS: 10 → 15 (높은 학습률 안정화)

[확인 포인트]
1. 초기 Angle Error: 30~60도 근처에서 시작할 것입니다.
2. 학습 진행: 이 숫자가 10도, 5도 이하로 뚝뚝 떨어지는지 확인하세요.
3. GPU 메모리 사용량: nvidia-smi로 ~20-22GB 사용 확인
"""

import sys

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import torch
torch.cuda.empty_cache()

from tqdm import tqdm
import fine_tune as fine_tune  # v6(±180°) fine_tune 모듈 로드

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# [Custom Logging Function] 상세 로그를 위해 함수 덮어쓰기
# =============================================================================
def train_one_epoch_detailed(embedder, transformer, refiner, dataloader, optimizer, criterion, 
                             device_info, scaler=None, current_rotation_range=None, 
                             enable_phase4=False, phase4_weight=1.0, 
                             phase4_backprop_to_phase23=False, **kwargs):
    """
    기존 train_one_epoch 함수를 대체하여, 
    매 Step마다 상세한 로그(Angle, Pixel Error)를 출력합니다.
    """
    device = device_info['device']
    is_tpu = device_info['type'] == 'tpu'
    
    embedder.train()
    transformer.train()
    
    metric_tracker = fine_tune.MetricTracker() # 기존 MetricTracker 재사용
    optimizer.zero_grad()
    
    rot_str = f"[{current_rotation_range[0]:.0f}°~{current_rotation_range[1]:.0f}°]" if current_rotation_range else ""
    # pbar 설정
    pbar = tqdm(dataloader, desc=f"Training {rot_str}", leave=False)
    
    # [설정] 로그 출력 주기 (Batch 단위)
    LOG_STEP = 10 
    
    for batch_idx, batch in enumerate(pbar):
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)
        
        use_amp = (device_info['type'] == 'cuda') and (scaler is not None)
        
        # --- Forward ---
        with torch.amp.autocast('cuda', enabled=use_amp):
            phase2_a = embedder(pyramid_a_raw, device)
            phase2_b = embedder(pyramid_b_raw, device)
            results = transformer(phase2_a, phase2_b)

            # results list의 순서는 구현/버전에 따라 달라질 수 있어 level 기준으로 정렬 후
            # level0(Fine)을 명시적으로 선택합니다.
            results_sorted = fine_tune.sort_results_by_level(results)
            finest_res = results_sorted[0] if len(results_sorted) > 0 else results[0]
            # Phase3 결과에서 A->B 변환 추출 (W_AB 우선, 없으면 inverse(W_global))
            pred_W = fine_tune.get_W_AB_from_phase3_result(finest_res)
            if pred_W is None:
                raise ValueError('Phase3 result missing W_AB/W_global/rotor_map')
            cos_raw = pred_W[:, 0, 0]
            sin_raw = pred_W[:, 1, 0]
            dx = pred_W[:, 0, 2]
            dy = pred_W[:, 1, 2]
            from losses import normalize_rotor_output
            cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
            
            loss, _ = criterion(
                pred_W, w_gt, cos_t, sin_t, gt_angle,
                phase2_a[0], phase2_b[0]
            )
            loss = loss / fine_tune.ACCUM_STEPS
        
        # --- Backward ---
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # --- Optimizer Step ---
        if (batch_idx + 1) % fine_tune.ACCUM_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(embedder.parameters()) + list(transformer.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(embedder.parameters()) + list(transformer.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
            optimizer.zero_grad()
        
        # --- Metrics Update ---
        pred_angle = torch.atan2(sin_t, cos_t)
        metric_tracker.update(pred_W, w_gt, pred_angle, gt_angle, 
                            loss.item() * fine_tune.ACCUM_STEPS)
        
        # [핵심 변경] 실시간 상세 로그 출력
        if batch_idx % LOG_STEP == 0:
            metrics = metric_tracker.get_metrics()
            
            # Progress Bar에 표시 (간략히)
            pbar.set_postfix({
                'L': f"{metrics['loss_mean']:.3f}",
                'Ang': f"{metrics['angle_error_mean']:.1f}°",
                'Pix': f"{metrics['pixel_error_mean']:.1f}px"
            })
            
    return metric_tracker.get_metrics()

# =============================================================================
# [Main Execution Setup] - RTX 3090 Optimized Strategy
# =============================================================================

# 커리큘럼: ±90 → ±120도 (대담한 점프!)
NEW_CURRICULUM_STAGES = [
    (0, 50, 5, 15),    # Stage 1: ±5° → ±15°
    (50, 120, 15, 30),  # Stage 2: ±15° → ±30°
    (120, 250, 30, 45),  # Stage 3: ±30° → ±45°
    (250, 400, 45, 60),  # Stage 4: ±45° → ±60°
]

# =============================================================================
# [Hyperparameters] RTX 3090 24GB Optimized
# =============================================================================
NEW_NUM_EPOCHS = 400              # [Hyperparameter] 총 350 에폭
NEW_LEARNING_RATE = 5e-5          # [Hyperparameter] 5e-5 → 8e-5 (배치 증가로 상향)
NEW_WARMUP_EPOCHS = 12            # [Hyperparameter] Warmup 에폭
NEW_VAL_INTERVAL = 2              # [Hyperparameter] 2 에폭마다 검증
NEW_BATCH_SIZE = 8                # [Hyperparameter] RTX 3090: 2 → 8 (4배 증가, vRAM 활용도 극대화)
NEW_ACCUM_STEPS = 2               # [Hyperparameter] Gradient Accumulation: 16 → 8 (effective batch = 64)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./img/val2017')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📢 RTX 3090 Optimized Fast Fine-Tuning Started")
    print("   - Target: Large Angle (curriculum range)")
    print("   - Logging: Every 10 batches (Angle & Pixel Error)")
    print("   - GPU: RTX 3090 24GB")
    print(f"   - Batch Size: {NEW_BATCH_SIZE} (Physical)")
    print(f"   - Accum Steps: {NEW_ACCUM_STEPS}")
    print(f"   - Effective Batch: {NEW_BATCH_SIZE * NEW_ACCUM_STEPS}")
    print(f"   - Learning Rate: {NEW_LEARNING_RATE}")
    print("="*70 + "\n")
    
    if args.resume and not os.path.exists(args.resume):
        print(f"❌ Error: Checkpoint not found at {args.resume}")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # [Monkey Patching] 함수 교체 및 설정 덮어쓰기
    # ------------------------------------------------------------------
    # 1. 우리가 만든 상세 로그 함수로 교체
    fine_tune.train_one_epoch = train_one_epoch_detailed
    
    # 2. [수정] RTX 3090 최적화 설정 덮어쓰기
    fine_tune.CURRICULUM_STAGES = NEW_CURRICULUM_STAGES
    fine_tune.NUM_EPOCHS = NEW_NUM_EPOCHS
    fine_tune.LEARNING_RATE = NEW_LEARNING_RATE          # 수정: 8e-5
    fine_tune.WARMUP_EPOCHS = NEW_WARMUP_EPOCHS          # 수정: 15
    fine_tune.VAL_INTERVAL = NEW_VAL_INTERVAL            # 수정: 3
    fine_tune.BATCH_SIZE = NEW_BATCH_SIZE                # 수정: 8
    fine_tune.ACCUM_STEPS = NEW_ACCUM_STEPS              # 수정: 8
    
    # 3. 학습 시작
    fine_tune.train(args.img_dir, resume_from=args.resume, debug_mode=False)