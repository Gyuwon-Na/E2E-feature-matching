#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def list_images(root: Path) -> list[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])


def apply_homography_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts.astype(np.float64), ones], axis=1)
    dst_h = (H @ pts_h.T).T
    dst = dst_h[:, :2] / np.clip(dst_h[:, 2:3], 1e-12, None)
    return dst


def dense_flow_from_homography(H: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing='ij')
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    dst = apply_homography_points(H, pts).reshape(h, w, 2)
    valid = (
        np.isfinite(dst[..., 0])
        & np.isfinite(dst[..., 1])
        & (dst[..., 0] >= 0)
        & (dst[..., 0] <= w - 1)
        & (dst[..., 1] >= 0)
        & (dst[..., 1] <= h - 1)
    )
    return dst.astype(np.float32), valid.astype(bool)


def rotation_homography_about_center(angle_deg: float, scale: float, w: int, h: int) -> np.ndarray:
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale).astype(np.float64)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H


def imread_rgb(path: Path, out_size: Tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to read image: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if out_size is not None:
        # out_size=(width,height) for cv2.resize
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    return img


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def build_pair(img_b: np.ndarray, angle_deg: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_b.shape[:2]
    H_b_to_a = rotation_homography_about_center(angle_deg, scale, w=w, h=h)
    img_a = cv2.warpAffine(
        img_b,
        H_b_to_a[:2, :],
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    H_a_to_b = np.linalg.inv(H_b_to_a)
    return img_a, H_a_to_b


def angle_schedule(num_samples: int, rot_min: float, rot_max: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    if num_samples <= 0:
        return np.empty((0,), dtype=np.float64)
    if mode == 'random':
        return rng.uniform(rot_min, rot_max, size=num_samples)
    if mode == 'linspace':
        if num_samples == 1:
            return np.array([(rot_min + rot_max) / 2.0], dtype=np.float64)
        return np.linspace(rot_min, rot_max, num_samples, dtype=np.float64)
    raise ValueError(f'Unknown angle mode: {mode}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a manifest-based synthetic evaluation set from a flat image folder.')
    parser.add_argument('--input-dir', required=True, help='Directory with single images (used as image1 / reference).')
    parser.add_argument('--output-root', required=True, help='Output directory for images/, arrays/, manifest.jsonl.')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of input images.')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--rot-min', type=float, default=-60.0)
    parser.add_argument('--rot-max', type=float, default=60.0)
    parser.add_argument('--scale-min', type=float, default=1.0)
    parser.add_argument('--scale-max', type=float, default=1.0)
    parser.add_argument('--angle-mode', choices=['random', 'linspace'], default='linspace')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    images_dir = output_root / 'images'
    arrays_dir = output_root / 'arrays'
    manifest_path = output_root / 'manifest.jsonl'

    rng = np.random.default_rng(args.seed)

    image_paths = list_images(input_dir)
    if not image_paths:
        raise RuntimeError(f'No images found in: {input_dir}')
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    output_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    angles = angle_schedule(len(image_paths), args.rot_min, args.rot_max, args.angle_mode, rng)
    scales = rng.uniform(args.scale_min, args.scale_max, size=len(image_paths))

    with manifest_path.open('w', encoding='utf-8') as f:
        for idx, img_path in enumerate(image_paths, start=1):
            sample_id = f'sample_{idx:05d}'
            img_b = imread_rgb(img_path, out_size=(args.width, args.height))
            angle_deg = float(angles[idx - 1])
            scale = float(scales[idx - 1])

            img_a, H_a_to_b = build_pair(img_b, angle_deg=angle_deg, scale=scale)
            flow01, valid_mask = dense_flow_from_homography(H_a_to_b, h=args.height, w=args.width)

            img0_rel = Path('images') / f'{sample_id}_0.png'
            img1_rel = Path('images') / f'{sample_id}_1.png'
            arr_rel = Path('arrays') / sample_id
            arr_dir = output_root / arr_rel
            arr_dir.mkdir(parents=True, exist_ok=True)

            save_rgb(output_root / img0_rel, img_a)
            save_rgb(output_root / img1_rel, img_b)
            np.save(arr_dir / 'H.npy', H_a_to_b.astype(np.float64))
            np.save(arr_dir / 'flow01.npy', flow01)
            np.save(arr_dir / 'valid_mask.npy', valid_mask)

            row = {
                'sample_id': sample_id,
                'image0': str(img0_rel).replace('\\', '/'),
                'image1': str(img1_rel).replace('\\', '/'),
                'gt': {
                    'homography_0to1': str((arr_rel / 'H.npy')).replace('\\', '/'),
                    'flow01': str((arr_rel / 'flow01.npy')).replace('\\', '/'),
                    'valid_mask': str((arr_rel / 'valid_mask.npy')).replace('\\', '/'),
                    'rotation_deg': abs(angle_deg),
                },
                'meta': {
                    'synthetic_from_single_image': True,
                    'source_image': img_path.name,
                    'applied_rotation_deg': angle_deg,
                    'applied_scale': scale,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'[OK] Wrote manifest: {manifest_path}')
    print(f'[OK] Samples: {len(image_paths)}')
    print(f'[OK] Image size: {args.width}x{args.height}')
    print(f'[OK] Rotation range: [{args.rot_min}, {args.rot_max}]')
    print(f'[OK] Scale range: [{args.scale_min}, {args.scale_max}]')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
