"""
================================================================================
Fast Fine-Tuning: Large Angle Specialist (Detailed Logging Ver.)
================================================================================
기존 학습 로직은 유지하되, **실시간 로그**를 강화했습니다.
매 10 Step마다 [Loss, Angle Error, Pixel Error]를 출력합니다.

[확인 포인트]
1. 초기 Angle Error: 30~60도 근처에서 시작할 것입니다.
2. 학습 진행: 이 숫자가 10도, 5도 이하로 뚝뚝 떨어지는지 확인하세요.
"""

import sys
import os
import torch
from tqdm import tqdm
import fine_tune  # 기존 fine_tune.py 모듈 로드

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# [Custom Logging Function] 상세 로그를 위해 함수 덮어쓰기
# =============================================================================
def train_one_epoch_detailed(embedder, transformer, dataloader, optimizer, criterion, 
                             device_info, scaler=None, current_rotation_range=None):
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
            
            finest_res = results[0]
            dense_rotor = finest_res['rotor_map']
            avg_rotor = dense_rotor.mean(dim=(1, 2))
            
            cos_raw = avg_rotor[:, 0]
            sin_raw = avg_rotor[:, 1]
            dx = avg_rotor[:, 2]
            dy = avg_rotor[:, 3]
            
            from losses import normalize_rotor_output
            cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)
            
            row1 = torch.stack([cos_t, -sin_t, dx], dim=1)
            row2 = torch.stack([sin_t, cos_t, dy], dim=1)
            pred_W = torch.stack([row1, row2], dim=1)
            
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
# [Main Execution Setup] - BOLD Strategy
# =============================================================================

# 커리큘럼: ±90 → ±120도 (대담한 점프!)
NEW_CURRICULUM_STAGES = [
    (0, 25, 90.0, 120.0)  # ✅ 수정: ±30도 점프
]

# 하이퍼파라미터
NEW_NUM_EPOCHS = 25           # ✅ 수정: 100 → 25
NEW_LEARNING_RATE = 2e-5    # ✅ 수정: 2.5e-7 → 2.5e-5
NEW_WARMUP_EPOCHS = 3         # ✅ 수정: 15 → 2
NEW_VAL_INTERVAL = 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./img/val2017')
    parser.add_argument('--resume', type=str, default='./checkpoints/best_model.pth')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📢 Detailed Fast Fine-Tuning Started")
    print("   - Target: Large Angle (±30° ~ ±60°)")
    print("   - Logging: Every 10 batches (Angle & Pixel Error)")
    print("="*70 + "\n")
    
    if not os.path.exists(args.resume):
        print(f"❌ Error: Checkpoint not found at {args.resume}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # [Monkey Patching] 함수 교체 및 설정 덮어쓰기
    # ------------------------------------------------------------------
    # 1. 우리가 만든 상세 로그 함수로 교체
    fine_tune.train_one_epoch = train_one_epoch_detailed
    
    # 2. 설정 덮어쓰기
    fine_tune.CURRICULUM_STAGES = NEW_CURRICULUM_STAGES
    fine_tune.NUM_EPOCHS = NEW_NUM_EPOCHS
    fine_tune.LEARNING_RATE = NEW_LEARNING_RATE
    fine_tune.WARMUP_EPOCHS = NEW_WARMUP_EPOCHS
    fine_tune.VAL_INTERVAL = NEW_VAL_INTERVAL
    fine_tune.BATCH_SIZE = 8
    
    # 3. 학습 시작
    fine_tune.train(args.img_dir, resume_from=args.resume, debug_mode=False)