"""
================================================================================
Fast Fine-Tuning: Single-GPU Version
================================================================================
기존 fast_finetune.py 로직을 유지하되, **단일 GPU 학습**을 지원합니다.

[모델 구조 특성상 DDP forward 래핑이 어려운 이유]
  - CliffordPyramidEmbedder.forward(pyramid_data_list, device): numpy dict → tensor 변환을 
    내부에서 수행하며 device 인자를 받는 비표준 시그니처
  - Phase3Transformer.forward(pyramid_a, pyramid_b): Phase2 tuple list 입력
  → 표준 DDP의 forward hook이 정상 동작하지 않음

[채택 전략: DistributedSampler + Manual Gradient AllReduce]
  1. 각 GPU에 동일 모델을 올려둠 (weight 동기화)
  2. DistributedSampler로 데이터를 GPU별로 겹치지 않게 분배
  3. 각 GPU에서 독립적으로 forward/backward 수행
  4. backward 후 AllReduce로 gradient를 평균화
  5. 동일한 optimizer step → weight가 항상 동기화 상태 유지

[실행 방법]
  python fast_finetune.py --img_dir ./img/val2017
  python fast_finetune.py --img_dir ./img/val2017 --resume ./checkpoints/best_model.pth

[Single-GPU 핵심 포인트]
  - Effective Batch = BATCH_SIZE(per GPU) × ACCUM_STEPS × NUM_GPUS
    = 1 × 8 × 1 = 8 (기본값)
  - 로그 출력, 체크포인트 저장은 rank 0에서만 수행
  - DistributedSampler가 데이터 셔플/분배를 담당
================================================================================
"""

import sys
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import glob
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import json

torch.cuda.empty_cache()

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

import fine_tune as fine_tune


# =============================================================================
# [DDP Helper Functions]
# =============================================================================

def setup_ddp():
    """
    DDP 프로세스 그룹 초기화.
    torchrun이 LOCAL_RANK, RANK, WORLD_SIZE 환경변수를 자동 설정합니다.
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """DDP 프로세스 그룹 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """현재 프로세스가 rank 0인지 확인"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """전체 GPU 수 반환"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """현재 프로세스의 rank 반환"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def print_rank0(*args, **kwargs):
    """rank 0에서만 print 출력 (중복 방지)"""
    if is_main_process():
        print(*args, **kwargs)


def sync_model_params(model, src_rank=0):
    """
    모든 GPU의 모델 파라미터를 src_rank(기본 0)의 값으로 동기화.
    학습 시작 전 / 체크포인트 로드 후에 호출합니다.
    """
    if not dist.is_initialized():
        return
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=src_rank)


def allreduce_gradients(model):
    """
    모든 GPU의 gradient를 평균화 (AllReduce).
    backward() 후, optimizer.step() 전에 호출합니다.
    """
    if not dist.is_initialized():
        return
    world_size = get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def allreduce_gradients_multi(*models):
    """여러 모델의 gradient를 한번에 AllReduce"""
    if not dist.is_initialized():
        return
    for model in models:
        allreduce_gradients(model)


# =============================================================================
# [Custom Training Function] Multi-GPU + Detailed Logging
# =============================================================================

def train_one_epoch_multigpu(
    embedder, transformer, refiner, dataloader, optimizer, criterion,
    device_info, scaler=None, current_rotation_range=None,
    enable_phase4=False, phase4_weight=1.0,
    phase4_backprop_to_phase23=False, **kwargs
):
    """
    Multi-GPU 환경에서의 1 에폭 학습.
    
    기존 fast_finetune의 train_one_epoch_detailed를 기반으로,
    gradient AllReduce를 추가한 버전입니다.
    """
    device = device_info['device']

    embedder.train()
    transformer.train()

    metric_tracker = fine_tune.MetricTracker()
    optimizer.zero_grad()

    rot_str = (f"[{current_rotation_range[0]:.0f}°~{current_rotation_range[1]:.0f}°]"
               if current_rotation_range else "")

    # rank 0에서만 tqdm progress bar 표시
    if is_main_process():
        pbar = tqdm(dataloader, desc=f"Training {rot_str}", leave=False)
    else:
        pbar = dataloader

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

            results_sorted = fine_tune.sort_results_by_level(results)
            finest_res = results_sorted[0] if len(results_sorted) > 0 else results[0]

            pred_W = fine_tune.get_W_AB_from_phase3_result(finest_res)
            if pred_W is None:
                raise ValueError('Phase3 result missing W_AB/W_global/rotor_map')

            cos_raw = pred_W[:, 0, 0]
            sin_raw = pred_W[:, 1, 0]

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

        # --- Optimizer Step (with Gradient AllReduce) ---
        if (batch_idx + 1) % fine_tune.ACCUM_STEPS == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            # ★ 핵심: 모든 GPU의 gradient를 평균화
            allreduce_gradients_multi(embedder, transformer)

            all_params = list(embedder.parameters()) + list(transformer.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # OOM 방지: optimizer step 후 캐시 정리
            torch.cuda.empty_cache()

        # --- Metrics ---
        pred_angle = torch.atan2(sin_t, cos_t)
        metric_tracker.update(pred_W, w_gt, pred_angle, gt_angle,
                              loss.item() * fine_tune.ACCUM_STEPS)

        # rank 0에서만 로그 출력
        if is_main_process() and batch_idx % LOG_STEP == 0:
            metrics = metric_tracker.get_metrics()
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'L': f"{metrics['loss_mean']:.3f}",
                    'Ang': f"{metrics['angle_error_mean']:.1f}°",
                    'Pix': f"{metrics['pixel_error_mean']:.1f}px"
                })

    return metric_tracker.get_metrics()


# =============================================================================
# [Multi-GPU Validation]
# =============================================================================

@torch.no_grad()
def validate_multigpu(embedder, transformer, dataloader, criterion, device_info):
    """Multi-GPU 검증"""
    device = device_info['device']
    embedder.eval()
    transformer.eval()

    metric_tracker = fine_tune.MetricTracker()

    loader = tqdm(dataloader, desc="Validation", leave=False) if is_main_process() else dataloader

    for batch in loader:
        pyramid_a_raw = batch['pyramid_a']
        pyramid_b_raw = batch['pyramid_b']
        w_gt = batch['w_gt'].to(device)
        gt_angle = batch['gt_angle'].to(device)

        phase2_a = embedder(pyramid_a_raw, device)
        phase2_b = embedder(pyramid_b_raw, device)
        results = transformer(phase2_a, phase2_b)

        results_sorted = fine_tune.sort_results_by_level(results)
        finest_res = results_sorted[0] if len(results_sorted) > 0 else results[0]

        pred_W = fine_tune.get_W_AB_from_phase3_result(finest_res)
        if pred_W is None:
            continue

        cos_raw = pred_W[:, 0, 0]
        sin_raw = pred_W[:, 1, 0]
        from losses import normalize_rotor_output
        cos_t, sin_t = normalize_rotor_output(cos_raw, sin_raw)

        loss_val, _ = criterion(
            pred_W, w_gt, cos_t, sin_t, gt_angle,
            phase2_a[0], phase2_b[0]
        )

        pred_angle = torch.atan2(sin_t, cos_t)
        metric_tracker.update(pred_W, w_gt, pred_angle, gt_angle, loss_val.item())

    # 각 GPU의 metric을 AllReduce로 합산하여 평균 (선택 사항, 더 정확한 metric)
    metrics = metric_tracker.get_metrics()

    # loss_mean을 AllReduce로 평균
    if dist.is_initialized():
        loss_tensor = torch.tensor([metrics['loss_mean']], device=device)
        angle_tensor = torch.tensor([metrics['angle_error_mean']], device=device)
        pixel_tensor = torch.tensor([metrics['pixel_error_mean']], device=device)
        count_tensor = torch.tensor([len(metric_tracker.losses)], device=device, dtype=torch.float32)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(angle_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        world_size = get_world_size()
        metrics['loss_mean'] = loss_tensor.item() / world_size
        metrics['angle_error_mean'] = angle_tensor.item() / world_size
        metrics['pixel_error_mean'] = pixel_tensor.item() / world_size

    return metrics


# =============================================================================
# [Main Training Function] Multi-GPU
# =============================================================================

def train_multigpu(img_dir, resume_from=None, debug_mode=False):
    """
    Multi-GPU 학습 메인 함수.
    fine_tune.train()의 전체 로직을 DDP/AllReduce 환경에 맞게 재구현합니다.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    device_info = {
        'type': 'cuda',
        'device': device,
    }

    # GPU 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main_process():
        gpu_name = torch.cuda.get_device_name(local_rank)
        gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
        print(f"✅ GPU 0: {gpu_name} ({gpu_mem:.1f} GB)")
    if world_size > 1:
        dist.barrier()
        if get_rank() == 1:
            gpu_name = torch.cuda.get_device_name(local_rank)
            gpu_mem = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
            print(f"✅ GPU 1: {gpu_name} ({gpu_mem:.1f} GB)")
        dist.barrier()

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------
    limit_samples = fine_tune.MAX_SAMPLES_NUM
    run_epochs = fine_tune.NUM_EPOCHS
    save_name_prefix = "v6_180deg_"

    os.makedirs(fine_tune.CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------
    curriculum = fine_tune.CurriculumScheduler(fine_tune.CURRICULUM_STAGES)

    # -------------------------------------------------------------------------
    # Dataset & DataLoader (DistributedSampler)
    # -------------------------------------------------------------------------
    print_rank0("\n📊 Loading Dataset...")

    all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                           glob.glob(os.path.join(img_dir, "*.jpg")) +
                           glob.glob(os.path.join(img_dir, "*.jpeg")))
    if len(all_img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    if limit_samples is not None and len(all_img_paths) > limit_samples:
        all_img_paths = all_img_paths[:limit_samples]

    # 재현 가능한 train/val 분리
    rng = np.random.RandomState(42)
    rng.shuffle(all_img_paths)
    total_size = len(all_img_paths)
    val_size = max(int(total_size * fine_tune.VAL_SPLIT), 1)
    train_size = total_size - val_size
    train_paths = all_img_paths[:train_size]
    val_paths = all_img_paths[train_size:]

    train_dataset = fine_tune.GeometricRotationDataset(
        img_dir=img_dir, is_train=True, max_samples=None,
        rot_min=fine_tune.ROTATION_MIN, rot_max=fine_tune.ROTATION_MAX,
        curriculum_scheduler=curriculum, img_paths=train_paths
    )
    val_dataset = fine_tune.GeometricRotationDataset(
        img_dir=img_dir, is_train=False, max_samples=None,
        rot_min=fine_tune.ROTATION_MIN, rot_max=fine_tune.ROTATION_MAX,
        curriculum_scheduler=curriculum, img_paths=val_paths
    )

    # ★ DistributedSampler: 각 GPU에 데이터를 겹치지 않게 분배
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=get_rank(), shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=get_rank(), shuffle=False
    )

    num_workers = 2
    train_loader = DataLoader(
        train_dataset, batch_size=fine_tune.BATCH_SIZE,
        shuffle=False,  # sampler가 셔플 담당
        num_workers=num_workers, pin_memory=True,
        collate_fn=fine_tune.collate_fn_geometric, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=fine_tune.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=fine_tune.collate_fn_geometric, sampler=val_sampler
    )

    print_rank0(f"   Total: {total_size}, Train: {train_size}, Val: {val_size}")
    print_rank0(f"   Batch per GPU: {fine_tune.BATCH_SIZE}, GPUs: {world_size}")
    print_rank0(f"   Effective Batch: {fine_tune.BATCH_SIZE * fine_tune.ACCUM_STEPS * world_size}")

    # -------------------------------------------------------------------------
    # Model (각 GPU에 동일 모델)
    # -------------------------------------------------------------------------
    print_rank0("\n🏗️ Building Model...")

    from pipeline.phase2 import CliffordPyramidEmbedder
    from pipeline.phase3 import Phase3Transformer
    from pipeline.phase4 import IterativeRefinementLoop
    from losses import UnifiedGeometricLoss

    embedder = CliffordPyramidEmbedder(hidden_dim=fine_tune.HIDDEN_DIM).to(device)
    transformer = Phase3Transformer(
        feature_dim=fine_tune.FEATURE_DIM, embed_dim=fine_tune.HIDDEN_DIM
    ).to(device)
    refiner = IterativeRefinementLoop(feature_dim=fine_tune.FEATURE_DIM).to(device)

    total_params = (sum(p.numel() for p in embedder.parameters())
                    + sum(p.numel() for p in transformer.parameters())
                    + sum(p.numel() for p in refiner.parameters()))
    print_rank0(f"   Parameters: {total_params:,}")

    # ★ 모든 GPU의 모델 파라미터를 rank 0 기준으로 동기화
    sync_model_params(embedder, src_rank=0)
    sync_model_params(transformer, src_rank=0)
    sync_model_params(refiner, src_rank=0)
    print_rank0("   ✅ Model params synchronized across GPUs")

    # -------------------------------------------------------------------------
    # Optimizer & Scheduler
    # -------------------------------------------------------------------------
    optimizer_params = list(embedder.parameters()) + list(transformer.parameters())
    # Phase4 refiner 파라미터도 포함하려면 아래 주석 해제:
    # optimizer_params += list(refiner.parameters())

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=fine_tune.LEARNING_RATE,
        weight_decay=fine_tune.WEIGHT_DECAY
    )

    scheduler = fine_tune.StageAwareWarmupScheduler(
        optimizer,
        warmup_epochs=fine_tune.WARMUP_EPOCHS,
        total_epochs=run_epochs,
        curriculum_scheduler=curriculum,
        warmup_start_lr=fine_tune.WARMUP_START_LR,
        min_lr=fine_tune.SCHEDULER_ETA_MIN
    )

    criterion = UnifiedGeometricLoss(alpha=1.0, beta=1.5, gamma=0.1).to(device)
    #scaler = torch.amp.GradScaler('cuda')
    scaler = torch.amp.GradScaler('cuda')

    # -------------------------------------------------------------------------
    # Resume (rank 0이 로드 후 broadcast)
    # -------------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from and os.path.exists(resume_from):
        print_rank0(f"\n📥 Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        embedder.load_state_dict(checkpoint['embedder'])
        transformer.load_state_dict(checkpoint['transformer'])
        if 'refiner' in checkpoint:
            refiner.load_state_dict(checkpoint['refiner'])
        else:
            print_rank0("   ⚠️ No 'refiner' in checkpoint. Phase4 starts from random init.")

        start_epoch = 0
        best_val_loss = float('inf')
        scheduler.current_epoch = start_epoch
        scheduler.prev_stage = curriculum.get_stage_info(start_epoch)['stage'] - 1

        # 로드 후 모든 GPU에 broadcast
        sync_model_params(embedder, src_rank=0)
        sync_model_params(transformer, src_rank=0)
        sync_model_params(refiner, src_rank=0)
        print_rank0("   ✅ Checkpoint loaded & synced across GPUs")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print_rank0("\n" + "=" * 70)
    print_rank0("🎯 Starting Multi-GPU Training...")
    print_rank0("=" * 70)

    history = {
        'train_loss': [], 'train_angle': [],
        'val_loss': [], 'val_angle': [],
        'learning_rate': [], 'rotation_range': []
    }

    best_model_path = None

    for epoch in range(start_epoch, run_epochs):
        epoch_start = time.time()

        # ★ DistributedSampler에 epoch 설정 (셔플 재현성)
        train_sampler.set_epoch(epoch)

        # 커리큘럼 업데이트
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        stage_info = curriculum.get_stage_info(epoch)
        current_rot = stage_info['rotation_range']
        current_stage = stage_info['stage']

        if epoch > 0:
            prev_stage_info = curriculum.get_stage_info(epoch - 1)
            if current_stage != prev_stage_info['stage']:
                best_val_loss = float('inf')
                print_rank0(f"   🔄 Stage {current_stage} 시작! best_loss 리셋")

        current_lr = optimizer.param_groups[0]['lr']

        print_rank0(f"\n📌 Epoch {epoch+1}/{run_epochs} | Stage {stage_info['stage']} | "
                    f"Rotation: {current_rot[0]:.0f}°~{current_rot[1]:.0f}° | LR: {current_lr:.2e}")
        print_rank0("-" * 50)

        # --- Train ---
        train_metrics = train_one_epoch_multigpu(
            embedder, transformer, refiner,
            train_loader, optimizer, criterion,
            device_info, scaler, current_rot,
            enable_phase4=False, phase4_weight=1.0,
            phase4_backprop_to_phase23=False,
        )

        # Scheduler Step
        scheduler.step()

        print_rank0(f"   [Train] Loss: {train_metrics['loss_mean']:.4f} | "
                    f"Angle: {train_metrics['angle_error_mean']:.2f}° ± {train_metrics['angle_error_std']:.2f}° | "
                    f"Pixel: {train_metrics['pixel_error_mean']:.2f}px")

        history['train_loss'].append(train_metrics['loss_mean'])
        history['train_angle'].append(train_metrics['angle_error_mean'])
        history['learning_rate'].append(current_lr)
        history['rotation_range'].append(current_rot)

        # --- Validation ---
        if (epoch + 1) % fine_tune.VAL_INTERVAL == 0 or epoch == run_epochs - 1:
            val_metrics = validate_multigpu(
                embedder, transformer, val_loader, criterion, device_info
            )

            current_val_loss = val_metrics['loss_mean']

            print_rank0(f"   [Val]   Loss: {current_val_loss:.4f} | "
                        f"Angle: {val_metrics['angle_error_mean']:.2f}° ± {val_metrics['angle_error_std']:.2f}° | "
                        f"Pixel: {val_metrics['pixel_error_mean']:.2f}px")

            # Best Model 저장 (rank 0에서만)
            if is_main_process() and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                rot_tag = int(abs(current_rot[1]))
                mean = float(val_metrics.get('angle_error_mean', 0.0))
                std = float(val_metrics.get('angle_error_std', 0.0))
                best_fname = f"{save_name_prefix}best_rot{rot_tag}_{mean:.2f}+-{std:.2f}.pth"
                best_path = os.path.join(fine_tune.CHECKPOINT_DIR, best_fname)
                stable_best_path = os.path.join(fine_tune.CHECKPOINT_DIR,
                                                f"{save_name_prefix}best_model.pth")

                ckpt = {
                    'epoch': epoch,
                    'embedder': embedder.state_dict(),
                    'transformer': transformer.state_dict(),
                    'refiner': refiner.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'metrics': val_metrics,
                    'training_config': {
                        'version': 'v6_multigpu',
                        'rotation_range': current_rot,
                        'world_size': world_size,
                    }
                }
                torch.save(ckpt, best_path)
                torch.save(ckpt, stable_best_path)
                best_model_path = best_path
                print(f"   🌟 Best Model Saved! (Loss: {best_val_loss:.4f})")

            history['val_loss'].append(current_val_loss)
            history['val_angle'].append(val_metrics['angle_error_mean'])

        epoch_time = time.time() - epoch_start
        print_rank0(f"   ⏱️ Time: {epoch_time:.1f}s")

        # Last Model Backup (rank 0에서만)
        if is_main_process():
            torch.save({
                'epoch': epoch,
                'embedder': embedder.state_dict(),
                'transformer': transformer.state_dict(),
                'refiner': refiner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'hidden_dim': fine_tune.HIDDEN_DIM,
                'feature_dim': fine_tune.FEATURE_DIM,
                'training_config': {
                    'version': 'v6_multigpu',
                    'rotation_range': current_rot,
                }
            }, os.path.join(fine_tune.CHECKPOINT_DIR, f'{save_name_prefix}last_model.pth'))

        # 모든 프로세스 동기화 (에폭 경계)
        if dist.is_initialized():
            dist.barrier()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if is_main_process():
        print("\n" + "=" * 70)
        print("🎉 Multi-GPU Training Complete!")
        print("=" * 70)

        stable_best_path = os.path.join(fine_tune.CHECKPOINT_DIR,
                                        f'{save_name_prefix}best_model.pth')
        print(f"\n💾 Best checkpoint: {stable_best_path}")

        if best_model_path and os.path.exists(best_model_path):
            best_ckpt = torch.load(best_model_path, weights_only=False)
            final_metrics = best_ckpt.get('metrics', {})
            print(f"\n📊 Best Model:")
            print(f"   Val Loss: {best_val_loss:.4f}")
            print(f"   Angle Error: {final_metrics.get('angle_error_mean', 'N/A'):.2f}°")
            print(f"   Pixel Error: {final_metrics.get('pixel_error_mean', 'N/A'):.2f}px")

        # History 저장
        history_serializable = history.copy()
        history_serializable['rotation_range'] = [list(r) for r in history['rotation_range']]
        with open(os.path.join(fine_tune.CHECKPOINT_DIR,
                               f'{save_name_prefix}history.json'), 'w') as f:
            json.dump(history_serializable, f, indent=2)

    return history


# =============================================================================
# [Hyperparameters] - 기존과 동일
# =============================================================================

NEW_CURRICULUM_STAGES = [
    (0, 50, 10, 25),   
    (50, 100, 20, 32),   
    (50, 150, 28, 40), 
    (150, 300, 32, 48)
]

NEW_NUM_EPOCHS = 300
NEW_LEARNING_RATE = 2e-4
NEW_WARMUP_EPOCHS = 12
NEW_VAL_INTERVAL = 3
NEW_BATCH_SIZE = 1          # per GPU (8GB 환경 OOM 방지)
NEW_ACCUM_STEPS = 16         # Gradient Accumulation (effective batch 유지)


# =============================================================================
# [Entry Point]
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./img/val2017')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # DDP 초기화
    local_rank = setup_ddp()
    world_size = get_world_size()

    print_rank0("\n" + "=" * 70)
    print_rank0("📢 Single-GPU Fast Fine-Tuning")
    print_rank0(f"   - GPUs: {world_size}")
    print_rank0(f"   - Batch Size per GPU: {NEW_BATCH_SIZE}")
    print_rank0(f"   - Accum Steps: {NEW_ACCUM_STEPS}")
    print_rank0(f"   - Effective Batch: {NEW_BATCH_SIZE * NEW_ACCUM_STEPS * world_size}")
    print_rank0(f"   - Learning Rate: {NEW_LEARNING_RATE}")
    print_rank0("=" * 70 + "\n")

    if args.resume and not os.path.exists(args.resume):
        print_rank0(f"❌ Error: Checkpoint not found at {args.resume}")
        cleanup_ddp()
        sys.exit(1)

    # Monkey Patching: fine_tune 모듈 설정 덮어쓰기
    fine_tune.CURRICULUM_STAGES = NEW_CURRICULUM_STAGES
    fine_tune.NUM_EPOCHS = NEW_NUM_EPOCHS
    fine_tune.LEARNING_RATE = NEW_LEARNING_RATE
    fine_tune.WARMUP_EPOCHS = NEW_WARMUP_EPOCHS
    fine_tune.VAL_INTERVAL = NEW_VAL_INTERVAL
    fine_tune.BATCH_SIZE = NEW_BATCH_SIZE
    fine_tune.ACCUM_STEPS = NEW_ACCUM_STEPS

    try:
        train_multigpu(args.img_dir, resume_from=args.resume, debug_mode=False)
    finally:
        cleanup_ddp()