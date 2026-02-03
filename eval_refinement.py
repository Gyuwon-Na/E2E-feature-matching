"""
================================================================================
Evaluation: Refinement & Alignment Quality
================================================================================
비교 대상:
- DKM Refiner
- Ours (MPC - Model Predictive Control)
- PMatch

평가 방법:
1. 체커보드 정렬 평가 (Checkerboard Alignment)
   - 정밀 코너 검출 후 서브픽셀 정렬 오차 측정
   
2. 가림 복원 평가 (Occlusion Recovery / Inpainting-style)
   - 이미지 일부를 가리고 올바른 위치를 찾는지 평가

사용법:
    python eval_refinement.py --data_dir ./test_data --output_dir ./results

유지보수:
- Refiner 추가: REFINER_REGISTRY에 래퍼 클래스 등록
- 평가 방법 추가: EVALUATION_REGISTRY에 함수 등록
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
import time
import json
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# [Hyperparameters] Evaluation Configuration
# =============================================================================
DEFAULT_IMG_SIZE = (256, 256)           # [Hyperparameter] 평가 이미지 크기
CHECKERBOARD_SIZE = (7, 7)              # [Hyperparameter] 체커보드 내부 코너 수
SUBPIXEL_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OCCLUSION_RATIOS = [0.1]      # [Hyperparameter] 가림 비율들
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# [Data Classes] 결과 저장용 구조체
# =============================================================================

class EvalType(Enum):
    """평가 유형"""
    CHECKERBOARD = "checkerboard"
    OCCLUSION = "occlusion"


@dataclass
class CheckerboardResult:
    """체커보드 정렬 평가 결과"""
    corner_error_mean: float = 0.0       # 평균 코너 오차 (서브픽셀)
    corner_error_std: float = 0.0        # 코너 오차 표준편차
    corner_error_max: float = 0.0        # 최대 코너 오차
    reprojection_error: float = 0.0      # 재투영 오차
    alignment_success: bool = False      # 정렬 성공 여부
    detected_corners: int = 0            # 검출된 코너 수
    runtime_ms: float = 0.0              # 런타임


@dataclass  
class OcclusionResult:
    """가림 복원 평가 결과"""
    occlusion_ratio: float = 0.0         # 가림 비율
    recovery_accuracy: float = 0.0       # 복원 정확도 (올바른 위치 찾은 비율)
    position_error: float = 0.0          # 위치 오차 (픽셀)
    confidence: float = 0.0              # 신뢰도
    runtime_ms: float = 0.0              # 런타임


@dataclass
class RefinerBenchmark:
    """Refiner별 종합 벤치마크 결과"""
    refiner_name: str
    num_samples: int = 0
    
    # 체커보드 평가 결과
    cb_corner_error_mean: float = 0.0
    cb_corner_error_std: float = 0.0
    cb_reprojection_error_mean: float = 0.0
    cb_success_rate: float = 0.0
    cb_runtime_mean_ms: float = 0.0
    
    # 가림 복원 평가 결과 (비율별)
    occ_recovery_accuracy: Dict[float, float] = field(default_factory=dict)
    occ_position_error: Dict[float, float] = field(default_factory=dict)
    occ_runtime_mean_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# [Refiner Registry] Refiner 래퍼 베이스 클래스 및 레지스트리
# =============================================================================

class BaseRefinerWrapper(ABC):
    """
    [Base Class] 모든 Refiner의 추상 베이스 클래스
    
    Refiner는 초기 변환(coarse)을 받아 정밀 변환(refined)을 반환
    """
    
    @abstractmethod
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        pass
    
    @abstractmethod
    def refine(self, img_a: np.ndarray, img_b: np.ndarray, 
               initial_H: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            img_a: Source 이미지 (H, W, 3) RGB
            img_b: Target 이미지 (H, W, 3) RGB
            initial_H: 초기 변환 행렬 (2x3 또는 3x3)
            
        Returns:
            refined_H: 정밀화된 변환 행렬
            extra_info: 추가 정보 (iterations, confidence 등)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


# -----------------------------------------------------------------------------
# [Ours MPC] Geometric MPC Refiner
# -----------------------------------------------------------------------------
# [eval_refinement.py 수정] OursMPCWrapper 클래스

class OursMPCWrapper(BaseRefinerWrapper):
    """
    [Ours] MPC 기반 정밀 정렬
    
    Architecture.md §4 구현
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        
        from phase1 import MathGeometricPreprocessor
        from phase2 import CliffordPyramidEmbedder
        from phase4 import GeometricMPCRefiner
        
        self.preprocessor = MathGeometricPreprocessor()
        self.embedder = CliffordPyramidEmbedder(hidden_dim=48).to(device)
        self.mpc = GeometricMPCRefiner(device=device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # weights_only=False 옵션 추가 (보안 경고 및 에러 방지)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.embedder.load_state_dict(ckpt['embedder'])
            print(f"[Ours MPC] Loaded checkpoint: {checkpoint_path}")
        
        self.embedder.eval()
    
    @property
    def name(self) -> str:
        return "Ours_MPC"
    
    # -------------------------------------------------------------------------
    # [수정] @torch.no_grad() 제거함! (MPC 최적화를 위해 Gradient 필요)
    # -------------------------------------------------------------------------
    def refine(self, img_a: np.ndarray, img_b: np.ndarray,
               initial_H: np.ndarray) -> Tuple[np.ndarray, Dict]:
        
        # 1. 특징 추출 및 임베딩 (여기는 학습 안 하니까 no_grad로 감쌈)
        with torch.no_grad():
            # Phase 1: 특징 추출
            pyramid_a = self.preprocessor.process_pyramid(img_a, levels=4)
            pyramid_b = self.preprocessor.process_pyramid(img_b, levels=4)
            
            # Phase 2: 임베딩 (레벨 0만 사용)
            def add_batch(data):
                return {k: v[np.newaxis, ...] if isinstance(v, np.ndarray) else v 
                        for k, v in data.items()}
            
            s_a, v_a, b_a = self.embedder.core(add_batch(pyramid_a[0]), self.device)
            s_b, v_b, b_b = self.embedder.core(add_batch(pyramid_b[0]), self.device)
            
            # MPC 데이터 준비
            src_data = {
                'sdf': torch.tensor(pyramid_a[0]['sdf'][np.newaxis, np.newaxis]).float().to(self.device),
                'vector': v_a.mean(dim=1).detach(),
                'rotor': b_a[2].mean(dim=1, keepdim=True).detach()
            }
            tgt_data = {
                'sdf': torch.tensor(pyramid_b[0]['sdf'][np.newaxis, np.newaxis]).float().to(self.device),
                'vector': v_b.mean(dim=1).detach(),
                'rotor': b_b[2].mean(dim=1, keepdim=True).detach()
            }
            
            # Gate 계산
            g_s = torch.sigmoid(torch.mean(torch.abs(s_a), dim=1, keepdim=True))
            g_v = torch.sigmoid(torch.mean(torch.norm(v_a, dim=2), dim=1, keepdim=True))
            g_b = torch.sigmoid(torch.mean(b_a[2], dim=1, keepdim=True))
            gates = (g_s.detach(), g_v.detach(), g_b.detach())

        # 2. MPC 초기화 및 실행 (Gradient 필요 구간)
        # --------------------------------------------------------
        if initial_H.shape[0] == 3:
            initial_H = initial_H[:2]
        
        # 회전 각도 추출
        cos_t, sin_t = initial_H[0, 0], initial_H[1, 0]
        angle = np.arctan2(sin_t, cos_t)
        scale = np.sqrt(cos_t**2 + sin_t**2)
        
        self.mpc.global_filtering_init(mean_rotor=angle, mean_scale=scale)
        
        # [중요] optimize() 내부는 backward()를 호출하므로 no_grad 바깥에 있어야 함
        loss_history = self.mpc.optimize(src_data, tgt_data, gates)
        
        # 결과 추출
        refined_W = self.mpc.W.detach().cpu().numpy()[0]
        
        return refined_W, {
            'iterations': len(loss_history),
            'final_loss': loss_history[-1] if loss_history else 0.0
        }

# -----------------------------------------------------------------------------
# [DKM Refiner] Dense Kernelized Matching Refiner
# -----------------------------------------------------------------------------

class DKMRefinerWrapper(BaseRefinerWrapper):
    """
    [DKM] Dense Kernelized Matching의 Refinement 모듈
    
    DKM은 coarse-to-fine refinement를 내장하고 있음
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        
        try:
            from dkm import DKMv3_outdoor
            self.model = DKMv3_outdoor(device=device)
            self._available = True
            print("[DKM Refiner] Loaded successfully")
        except ImportError:
            print("[DKM Refiner] Not available")
    
    @property
    def name(self) -> str:
        return "DKM_Refiner"
    
    def refine(self, img_a: np.ndarray, img_b: np.ndarray,
               initial_H: np.ndarray) -> Tuple[np.ndarray, Dict]:
        
        if not self._available:
            return initial_H, {'error': 'not_available'}
        
        from PIL import Image
        pil_a = Image.fromarray(img_a)
        pil_b = Image.fromarray(img_b)
        
        # DKM의 전체 파이프라인 실행 (내부적으로 refinement 포함)
        dense_matches, dense_certainty = self.model.match(pil_a, pil_b)
        
        # 고신뢰도 매칭으로 Homography 추정
        mask = dense_certainty > 0.7
        if mask.sum() < 4:
            return initial_H, {'num_matches': 0}
        
        pts_a = dense_matches[mask, :2].cpu().numpy()
        pts_b = dense_matches[mask, 2:].cpu().numpy()
        
        H, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
        if H is None:
            return initial_H, {'error': 'homography_failed'}
        
        return H[:2], {
            'num_matches': mask.sum().item(),
            'mean_certainty': dense_certainty[mask].mean().item()
        }


# -----------------------------------------------------------------------------
# [PMatch] Progressive Matching
# -----------------------------------------------------------------------------

class PMatchRefinerWrapper(BaseRefinerWrapper):
    """
    [PMatch] Progressive Matching의 Refinement
    
    TODO: 공식 구현 필요
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        print("[PMatch Refiner] Placeholder - implement with official code")
    
    @property
    def name(self) -> str:
        return "PMatch"
    
    def refine(self, img_a: np.ndarray, img_b: np.ndarray,
               initial_H: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Placeholder: Identity refinement
        return initial_H, {'error': 'not_implemented'}


# -----------------------------------------------------------------------------
# [Classic Refinement] OpenCV 기반 Baseline
# -----------------------------------------------------------------------------

class ClassicRefinerWrapper(BaseRefinerWrapper):
    """
    [Baseline] OpenCV ECC 기반 정밀 정렬
    
    비교를 위한 전통적 방법 baseline
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.num_iterations = 100
        self.termination_eps = 1e-6
    
    @property
    def name(self) -> str:
        return "Classic_ECC"
    
    def refine(self, img_a: np.ndarray, img_b: np.ndarray,
               initial_H: np.ndarray) -> Tuple[np.ndarray, Dict]:
        
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        
        # ECC criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                    self.num_iterations, self.termination_eps)
        
        # 2x3 → 3x3 변환 (ECC는 3x3 Homography 행렬 필요)
        if initial_H.shape[0] == 2:
            warp_matrix = np.vstack([initial_H, [0, 0, 1]]).astype(np.float32)
        else:
            warp_matrix = initial_H.astype(np.float32)
        
        try:
            # [수정] inputMask=None 을 명시적으로 추가!
            cc, refined_H = cv2.findTransformECC(
                gray_b, gray_a, warp_matrix, 
                motionType=cv2.MOTION_HOMOGRAPHY,
                criteria=criteria,
                inputMask=None,
                gaussFiltSize=5
            )
            return refined_H[:2], {'correlation': cc}
        except cv2.error as e:
            # ECC는 수렴하지 못하면 에러를 뱉으므로, 실패 시 초기값 반환
            return initial_H, {'error': str(e)}

# -----------------------------------------------------------------------------
# [Refiner Registry]
# -----------------------------------------------------------------------------

REFINER_REGISTRY: Dict[str, type] = {
    'ours_mpc': OursMPCWrapper,
    'dkm': DKMRefinerWrapper,
    'pmatch': PMatchRefinerWrapper,
    'classic_ecc': ClassicRefinerWrapper,
}

def register_refiner(name: str, wrapper_class: type):
    """새 Refiner 등록"""
    REFINER_REGISTRY[name.lower()] = wrapper_class
    print(f"Registered refiner: {name}")

def get_refiner(name: str, **kwargs) -> BaseRefinerWrapper:
    """Refiner 인스턴스 생성"""
    if name.lower() not in REFINER_REGISTRY:
        raise ValueError(f"Unknown refiner: {name}. Available: {list(REFINER_REGISTRY.keys())}")
    return REFINER_REGISTRY[name.lower()](**kwargs)


# =============================================================================
# [Evaluation Methods] 평가 방법 구현
# =============================================================================

class CheckerboardEvaluator:
    """
    [평가 방법 1] 체커보드 정렬 평가
    
    체커보드 패턴의 코너를 서브픽셀 정밀도로 검출하여
    정렬 정확도를 측정합니다.
    """
    
    def __init__(self, board_size: Tuple[int, int] = CHECKERBOARD_SIZE):
        self.board_size = board_size
    
    def generate_checkerboard_image(self, img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                                    square_size: int = 30) -> np.ndarray:
        """체커보드 이미지 생성"""
        H, W = img_size
        img = np.ones((H, W), dtype=np.uint8) * 255
        
        rows, cols = self.board_size[1] + 1, self.board_size[0] + 1
        
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    y_start = i * square_size
                    x_start = j * square_size
                    y_end = min(y_start + square_size, H)
                    x_end = min(x_start + square_size, W)
                    img[y_start:y_end, x_start:x_end] = 0
        
        # RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img_rgb
    
    def detect_corners(self, img: np.ndarray) -> Optional[np.ndarray]:
        """서브픽셀 정밀도로 코너 검출"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # 서브픽셀 정밀도 향상
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIXEL_CRITERIA)
            return corners.reshape(-1, 2)
        return None
    
    def evaluate(self, refiner: BaseRefinerWrapper, 
                 num_samples: int = 50,
                 rotation_range: Tuple[float, float] = (-30, 30)) -> List[CheckerboardResult]:
        """체커보드 정렬 평가 실행"""
        results = []
        
        for _ in tqdm(range(num_samples), desc=f"Checkerboard [{refiner.name}]"):
            result = CheckerboardResult()
            
            # 체커보드 이미지 생성
            img_orig = self.generate_checkerboard_image()
            
            # 원본 코너 검출
            corners_orig = self.detect_corners(img_orig)
            if corners_orig is None:
                continue
            
            # 무작위 변환 적용
            H, W = img_orig.shape[:2]
            angle = np.random.uniform(*rotation_range)
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
            img_warped = cv2.warpAffine(img_orig, M, (W, H), borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(128, 128, 128))
            
            # 변환된 이미지 코너 검출
            corners_warped = self.detect_corners(img_warped)
            if corners_warped is None:
                continue
            
            result.detected_corners = len(corners_orig)
            
            # 초기 변환 (노이즈 추가하여 refiner가 할 일 만들기)
            noise = np.random.randn(2, 3) * 0.02
            initial_H = M + noise
            
            # Refinement 실행
            start_time = time.perf_counter()
            refined_H, extra = refiner.refine(img_warped, img_orig, initial_H)
            result.runtime_ms = (time.perf_counter() - start_time) * 1000
            
            # 정밀화된 변환으로 코너 투영
            if refined_H.shape[0] == 2:
                refined_H_full = np.vstack([refined_H, [0, 0, 1]])
            else:
                refined_H_full = refined_H
            
            corners_warped_h = np.hstack([corners_warped, np.ones((len(corners_warped), 1))])
            corners_projected = (refined_H_full @ corners_warped_h.T).T
            corners_projected = corners_projected[:, :2] / corners_projected[:, 2:3]
            
            # 오차 계산
            errors = np.linalg.norm(corners_projected - corners_orig, axis=1)
            result.corner_error_mean = float(errors.mean())
            result.corner_error_std = float(errors.std())
            result.corner_error_max = float(errors.max())
            result.reprojection_error = result.corner_error_mean
            result.alignment_success = result.corner_error_mean < 2.0  # 2픽셀 이내면 성공
            
            results.append(result)
        
        return results


class OcclusionEvaluator:
    """
    [평가 방법 2] 가림 복원 평가 (Inpainting-style)
    
    이미지의 일부를 가리고, refiner가 올바른 대응을 
    찾을 수 있는지 평가합니다.
    """
    
    def __init__(self, occlusion_ratios: List[float] = OCCLUSION_RATIOS):
        self.occlusion_ratios = occlusion_ratios
    
    def apply_occlusion(self, img: np.ndarray, ratio: float,
                        occlusion_type: str = 'random_boxes') -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에 가림 적용
        
        Returns:
            occluded_img: 가려진 이미지
            mask: 가림 마스크 (1 = 가려진 영역)
        """
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        occluded = img.copy()
        
        if occlusion_type == 'random_boxes':
            # 무작위 박스 가림
            num_boxes = max(1, int(ratio * 10))
            total_area = H * W * ratio
            area_per_box = total_area / num_boxes
            
            for _ in range(num_boxes):
                box_area = area_per_box * np.random.uniform(0.5, 1.5)
                box_h = int(np.sqrt(box_area))
                box_w = int(box_area / box_h)
                
                y = np.random.randint(0, max(1, H - box_h))
                x = np.random.randint(0, max(1, W - box_w))
                
                mask[y:y+box_h, x:x+box_w] = 1
                occluded[y:y+box_h, x:x+box_w] = [128, 128, 128]  # Gray fill
        
        elif occlusion_type == 'center':
            # 중앙 사각형 가림
            occ_h = int(H * np.sqrt(ratio))
            occ_w = int(W * np.sqrt(ratio))
            y_start = (H - occ_h) // 2
            x_start = (W - occ_w) // 2
            
            mask[y_start:y_start+occ_h, x_start:x_start+occ_w] = 1
            occluded[y_start:y_start+occ_h, x_start:x_start+occ_w] = [128, 128, 128]
        
        return occluded, mask
    
    def evaluate(self, refiner: BaseRefinerWrapper,
                 img_dir: str,
                 num_samples: int = 50,
                 rotation_range: Tuple[float, float] = (-20, 20)) -> Dict[float, List[OcclusionResult]]:
        """가림 복원 평가 실행"""
        results = {ratio: [] for ratio in self.occlusion_ratios}
        
        # 이미지 로드
        img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
        if not img_paths:
            print(f"No images found in {img_dir}")
            return results
        
        for ratio in self.occlusion_ratios:
            print(f"\nOcclusion ratio: {ratio*100:.0f}%")
            
            for i in tqdm(range(num_samples), desc=f"Occlusion {ratio} [{refiner.name}]"):
                img_path = img_paths[i % len(img_paths)]
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, DEFAULT_IMG_SIZE)
                
                result = OcclusionResult()
                result.occlusion_ratio = ratio
                
                H, W = img.shape[:2]
                
                # 변환 적용
                angle = np.random.uniform(*rotation_range)
                M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
                img_warped = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
                
                # 가림 적용 (Source 이미지에)
                img_occluded, occ_mask = self.apply_occlusion(img_warped, ratio)
                
                # GT 역변환
                M_aug = np.vstack([M, [0, 0, 1]])
                M_inv = np.linalg.inv(M_aug)[:2]
                
                # 노이즈 있는 초기 변환
                initial_H = M_inv + np.random.randn(2, 3) * 0.02
                
                # Refinement 실행
                start_time = time.perf_counter()
                refined_H, extra = refiner.refine(img_occluded, img, initial_H)
                result.runtime_ms = (time.perf_counter() - start_time) * 1000
                
                # 평가: 가려지지 않은 영역에서의 정확도
                if refined_H.shape[0] == 2:
                    refined_H_full = np.vstack([refined_H, [0, 0, 1]])
                else:
                    refined_H_full = refined_H
                
                gt_H_full = np.vstack([M_inv, [0, 0, 1]])
                
                # 샘플 포인트에서 오차 계산
                step = 16
                y_pts, x_pts = np.mgrid[0:H:step, 0:W:step]
                valid_mask = occ_mask[::step, ::step] == 0  # 가려지지 않은 점만
                
                if valid_mask.sum() < 4:
                    continue
                
                points = np.stack([x_pts[valid_mask], y_pts[valid_mask], 
                                  np.ones(valid_mask.sum())], axis=1)
                
                pred_pts = (refined_H_full @ points.T).T
                gt_pts = (gt_H_full @ points.T).T
                
                pred_pts = pred_pts[:, :2] / pred_pts[:, 2:3]
                gt_pts = gt_pts[:, :2] / gt_pts[:, 2:3]
                
                errors = np.linalg.norm(pred_pts - gt_pts, axis=1)
                result.position_error = float(errors.mean())
                result.recovery_accuracy = float((errors < 5.0).mean())  # 5픽셀 이내
                result.confidence = extra.get('confidence', 0.0) if isinstance(extra, dict) else 0.0
                
                results[ratio].append(result)
        
        return results


# =============================================================================
# [Evaluation Registry]
# =============================================================================

EVALUATION_REGISTRY: Dict[str, type] = {
    'checkerboard': CheckerboardEvaluator,
    'occlusion': OcclusionEvaluator,
}

def register_evaluation(name: str, evaluator_class: type):
    """새 평가 방법 등록"""
    EVALUATION_REGISTRY[name.lower()] = evaluator_class
    print(f"Registered evaluation: {name}")


# =============================================================================
# [Main Evaluation Engine]
# =============================================================================

class RefinementEvaluator:
    """
    [Main] Refinement 평가 엔진
    
    Usage:
        evaluator = RefinementEvaluator()
        evaluator.add_refiner('ours_mpc', checkpoint_path='best_model.pth')
        evaluator.add_refiner('dkm')
        results = evaluator.evaluate_all(img_dir='./test_images')
        evaluator.print_results()
    """
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.refiners: Dict[str, BaseRefinerWrapper] = {}
        self.results: Dict[str, RefinerBenchmark] = {}
        
        self.cb_evaluator = CheckerboardEvaluator()
        self.occ_evaluator = OcclusionEvaluator()
    
    def add_refiner(self, name: str, checkpoint_path: Optional[str] = None):
        """Refiner 추가"""
        refiner = get_refiner(name, checkpoint_path=checkpoint_path, device=self.device)
        self.refiners[refiner.name] = refiner
        print(f"Added refiner: {refiner.name}")
    
    def remove_refiner(self, name: str):
        """Refiner 제거"""
        for key in list(self.refiners.keys()):
            if key.lower() == name.lower():
                del self.refiners[key]
                print(f"Removed refiner: {name}")
                return
    
    def evaluate_checkerboard(self, num_samples: int = 50) -> Dict[str, List[CheckerboardResult]]:
        """체커보드 평가 실행"""
        results = {}
        for name, refiner in self.refiners.items():
            results[name] = self.cb_evaluator.evaluate(refiner, num_samples)
        return results
    
    def evaluate_occlusion(self, img_dir: str, num_samples: int = 50) -> Dict[str, Dict[float, List[OcclusionResult]]]:
        """가림 복원 평가 실행"""
        results = {}
        for name, refiner in self.refiners.items():
            results[name] = self.occ_evaluator.evaluate(refiner, img_dir, num_samples)
        return results
    
    def evaluate_all(self, img_dir: str, 
                     cb_samples: int = 50,
                     occ_samples: int = 50) -> Dict[str, RefinerBenchmark]:
        """전체 평가 실행"""
        
        for name, refiner in self.refiners.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print('='*60)
            
            benchmark = RefinerBenchmark(refiner_name=name)
            
            # 1. 체커보드 평가
            print("\n[1] Checkerboard Alignment Evaluation")
            cb_results = self.cb_evaluator.evaluate(refiner, cb_samples)
            
            if cb_results:
                benchmark.num_samples = len(cb_results)
                benchmark.cb_corner_error_mean = np.mean([r.corner_error_mean for r in cb_results])
                benchmark.cb_corner_error_std = np.std([r.corner_error_mean for r in cb_results])
                benchmark.cb_reprojection_error_mean = np.mean([r.reprojection_error for r in cb_results])
                benchmark.cb_success_rate = np.mean([r.alignment_success for r in cb_results])
                benchmark.cb_runtime_mean_ms = np.mean([r.runtime_ms for r in cb_results])
            
            # 2. 가림 복원 평가
            print("\n[2] Occlusion Recovery Evaluation")
            occ_results = self.occ_evaluator.evaluate(refiner, img_dir, occ_samples)
            
            for ratio, results in occ_results.items():
                if results:
                    benchmark.occ_recovery_accuracy[ratio] = np.mean([r.recovery_accuracy for r in results])
                    benchmark.occ_position_error[ratio] = np.mean([r.position_error for r in results])
            
            all_occ_runtimes = []
            for results in occ_results.values():
                all_occ_runtimes.extend([r.runtime_ms for r in results])
            if all_occ_runtimes:
                benchmark.occ_runtime_mean_ms = np.mean(all_occ_runtimes)
            
            self.results[name] = benchmark
        
        return self.results
    
    def print_results(self):
        """결과 테이블 출력"""
        print("\n" + "=" * 120)
        print("REFINEMENT BENCHMARK RESULTS")
        print("=" * 120)
        
        # 체커보드 결과
        print("\n[Checkerboard Alignment]")
        print("-" * 80)
        headers = ['Refiner', 'Corner Err↓', 'Success Rate↑', 'Runtime(ms)↓']
        row_format = "{:<20}" + "{:<20}" * 3
        print(row_format.format(*headers))
        print("-" * 80)
        
        for name, bench in self.results.items():
            row = [
                name,
                f"{bench.cb_corner_error_mean:.3f} ± {bench.cb_corner_error_std:.3f}",
                f"{bench.cb_success_rate*100:.1f}%",
                f"{bench.cb_runtime_mean_ms:.1f}"
            ]
            print(row_format.format(*row))
        
        # 가림 복원 결과
        print("\n[Occlusion Recovery]")
        print("-" * 100)
        
        for ratio in OCCLUSION_RATIOS:
            print(f"\n  Occlusion Ratio: {ratio*100:.0f}%")
            headers = ['Refiner', 'Recovery Acc↑', 'Position Err↓']
            print(f"  {row_format.format(*headers, '')}")
            
            for name, bench in self.results.items():
                acc = bench.occ_recovery_accuracy.get(ratio, 0)
                err = bench.occ_position_error.get(ratio, 0)
                row = [name, f"{acc*100:.1f}%", f"{err:.2f}px"]
                print(f"  {row_format.format(*row, '')}")
        
        print("\n" + "=" * 120)
    
    def save_results(self, path: str):
        """결과 JSON 저장"""
        # Dict를 JSON 직렬화 가능하게 변환
        serializable = {}
        for name, bench in self.results.items():
            d = bench.to_dict()
            # float keys를 str로 변환
            d['occ_recovery_accuracy'] = {str(k): v for k, v in d['occ_recovery_accuracy'].items()}
            d['occ_position_error'] = {str(k): v for k, v in d['occ_position_error'].items()}
            serializable[name] = d
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to: {path}")
    
    def visualize_results(self, save_path: Optional[str] = None):
        """결과 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        names = list(self.results.keys())
        
        # 1. 체커보드 Corner Error
        ax1 = axes[0]
        errors = [self.results[n].cb_corner_error_mean for n in names]
        stds = [self.results[n].cb_corner_error_std for n in names]
        bars = ax1.bar(names, errors, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Corner Error (px)')
        ax1.set_title('Checkerboard Alignment')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 가림 복원 정확도
        ax2 = axes[1]
        x = np.arange(len(OCCLUSION_RATIOS))
        width = 0.8 / len(names)
        
        for i, name in enumerate(names):
            accs = [self.results[name].occ_recovery_accuracy.get(r, 0) * 100 
                   for r in OCCLUSION_RATIOS]
            ax2.bar(x + i*width, accs, width, label=name, alpha=0.8)
        
        ax2.set_ylabel('Recovery Accuracy (%)')
        ax2.set_xlabel('Occlusion Ratio')
        ax2.set_title('Occlusion Recovery')
        ax2.set_xticks(x + width * (len(names)-1) / 2)
        ax2.set_xticklabels([f'{r*100:.0f}%' for r in OCCLUSION_RATIOS])
        ax2.legend()
        
        # 3. 런타임 비교
        ax3 = axes[2]
        cb_runtimes = [self.results[n].cb_runtime_mean_ms for n in names]
        occ_runtimes = [self.results[n].occ_runtime_mean_ms for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        ax3.bar(x - width/2, cb_runtimes, width, label='Checkerboard', color='steelblue', alpha=0.8)
        ax3.bar(x + width/2, occ_runtimes, width, label='Occlusion', color='coral', alpha=0.8)
        ax3.set_ylabel('Runtime (ms)')
        ax3.set_title('Runtime Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


# =============================================================================
# [Main Entry Point]
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Refinement Quality Evaluation')
    parser.add_argument('--data_dir', type=str, default='./val2017',
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./refinement_results',
                        help='Directory to save results')
    parser.add_argument('--ours_checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to our model checkpoint')
    parser.add_argument('--refiners', type=str, nargs='+',
                        default=['ours_mpc', 'classic_ecc'],
                        help='Refiners to evaluate')
    parser.add_argument('--cb_samples', type=int, default=50,
                        help='Number of checkerboard test samples')
    parser.add_argument('--occ_samples', type=int, default=50,
                        help='Number of occlusion test samples')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 평가기 초기화
    evaluator = RefinementEvaluator()
    
    # Refiner 추가
    for refiner_name in args.refiners:
        if refiner_name.lower() == 'ours_mpc':
            evaluator.add_refiner(refiner_name, checkpoint_path=args.ours_checkpoint)
        else:
            evaluator.add_refiner(refiner_name)
    
    # 평가 실행
    evaluator.evaluate_all(
        args.data_dir,
        cb_samples=args.cb_samples,
        occ_samples=args.occ_samples
    )
    
    # 결과 출력 및 저장
    evaluator.print_results()
    evaluator.save_results(os.path.join(args.output_dir, 'refinement_results.json'))
    evaluator.visualize_results(os.path.join(args.output_dir, 'refinement_comparison.png'))
    
    print("\n✅ Refinement evaluation complete!")


if __name__ == "__main__":
    main()