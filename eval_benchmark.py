"""
================================================================================
Evaluation Benchmark: Dense Matching Model Comparison
================================================================================
비교 대상:
- LoFTR, SE(2)-LoFTR, ASPanFormer, DeepMatcher, RoMa, DKM, PMatch, TopicFM, Ours

평가 지표:
- MACE (Mean Average Corner Error)
- SR@10, SR@5 (Success Rate)
- Alignment Error
- Runtime
- Matching Success Rate
- Recall

사용법:
    python eval_benchmark.py --data_dir ./test_data --output_dir ./results

유지보수:
- 모델 추가: MODEL_REGISTRY에 래퍼 클래스 등록
- 지표 추가: METRIC_REGISTRY에 함수 등록
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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# =============================================================================
# [Hyperparameters] Evaluation Configuration
# =============================================================================
DEFAULT_IMG_SIZE = (256, 256)    # [Hyperparameter] 평가 이미지 크기
SR_THRESHOLD_10 = 10.0           # [Hyperparameter] SR@10 임계값 (픽셀)
SR_THRESHOLD_5 = 5.0             # [Hyperparameter] SR@5 임계값 (픽셀)
WARMUP_RUNS = 3                  # [Hyperparameter] 런타임 측정 전 워밍업 횟수
TIMING_RUNS = 10                 # [Hyperparameter] 런타임 측정 반복 횟수
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# [Data Classes] 결과 저장용 구조체
# =============================================================================

@dataclass
class EvalResult:
    """단일 샘플 평가 결과"""
    mace: float = 0.0                    # Mean Average Corner Error
    sr_10: bool = False                  # Success Rate @ 10px
    sr_5: bool = False                   # Success Rate @ 5px
    alignment_error: float = 0.0         # 전체 픽셀 정렬 오차
    runtime_ms: float = 0.0              # 런타임 (밀리초)
    matching_success: bool = False       # 매칭 성공 여부
    recall: float = 0.0                  # 재현율 (Dense에서는 픽셀 기반)
    
    
@dataclass
class ModelBenchmark:
    """모델별 벤치마크 집계 결과"""
    model_name: str
    num_samples: int = 0
    mace_mean: float = 0.0
    mace_std: float = 0.0
    sr_10_rate: float = 0.0              # SR@10 비율 (0~1)
    sr_5_rate: float = 0.0               # SR@5 비율 (0~1)
    alignment_error_mean: float = 0.0
    alignment_error_std: float = 0.0
    runtime_mean_ms: float = 0.0
    runtime_std_ms: float = 0.0
    matching_success_rate: float = 0.0   # 매칭 성공률 (0~1)
    recall_mean: float = 0.0
    recall_std: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# [Model Registry] 모델 래퍼 베이스 클래스 및 레지스트리
# =============================================================================

class BaseModelWrapper(ABC):
    """
    [Base Class] 모든 매칭 모델의 추상 베이스 클래스
    
    새 모델 추가 시 이 클래스를 상속받아 구현:
    1. __init__: 모델 로드
    2. predict: 변환 행렬 예측
    3. name: 모델 이름 반환
    """
    
    @abstractmethod
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        pass
    
    @abstractmethod
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            img_a: Source 이미지 (H, W, 3) RGB
            img_b: Target 이미지 (H, W, 3) RGB
            
        Returns:
            homography: 3x3 또는 2x3 변환 행렬
            extra_info: 추가 정보 (매칭 수, confidence 등)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def warmup(self, img_size: Tuple[int, int] = (256, 256)):
        """GPU 워밍업"""
        dummy_a = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
        dummy_b = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
        for _ in range(WARMUP_RUNS):
            self.predict(dummy_a, dummy_b)


# -----------------------------------------------------------------------------
# [Our Model] Geometric Matching Model Wrapper
# -----------------------------------------------------------------------------

class OursWrapper(BaseModelWrapper):
    """
    [Ours] Clifford Algebra 기반 Geometric Matching 모델
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        
        # Phase 1, 2, 3 로드
        from phase1 import MathGeometricPreprocessor
        from phase2 import CliffordPyramidEmbedder
        from phase3 import Phase3Transformer
        
        self.preprocessor = MathGeometricPreprocessor()
        self.embedder = CliffordPyramidEmbedder(hidden_dim=48).to(device)
        self.transformer = Phase3Transformer(feature_dim=144, embed_dim=48).to(device)
        
        # 체크포인트 로드
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.embedder.load_state_dict(ckpt['embedder'])
            self.transformer.load_state_dict(ckpt['transformer'])
            print(f"[Ours] Loaded checkpoint: {checkpoint_path}")
        
        self.embedder.eval()
        self.transformer.eval()
    
    @property
    def name(self) -> str:
        return "Ours"
    
    @torch.no_grad()
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Phase 1: 피라미드 생성
        pyramid_a = self.preprocessor.process_pyramid(img_a, levels=4)
        pyramid_b = self.preprocessor.process_pyramid(img_b, levels=4)
        
        # Batch 차원 추가를 위한 래핑
        def add_batch_dim(pyramid):
            batched = []
            for level_data in pyramid:
                batched_level = {}
                for key, value in level_data.items():
                    if isinstance(value, np.ndarray):
                        batched_level[key] = value[np.newaxis, ...]
                    else:
                        batched_level[key] = value
                batched.append(batched_level)
            return batched
        
        pyramid_a = add_batch_dim(pyramid_a)
        pyramid_b = add_batch_dim(pyramid_b)
        
        # Phase 2: 임베딩
        phase2_a = self.embedder(pyramid_a, self.device)
        phase2_b = self.embedder(pyramid_b, self.device)
        
        # Phase 3: 트랜스포머
        results = self.transformer(phase2_a, phase2_b)
        
        # 결과 추출
        finest = results[0]
        rotor = finest['rotor_map']
        avg_rotor = rotor.mean(dim=(1, 2))
        
        cos_t, sin_t = avg_rotor[0, 0].item(), avg_rotor[0, 1].item()
        dx, dy = avg_rotor[0, 2].item(), avg_rotor[0, 3].item()
        
        # 정규화
        mag = np.sqrt(cos_t**2 + sin_t**2 + 1e-6)
        cos_t, sin_t = cos_t / mag, sin_t / mag
        
        # 2x3 Affine 행렬 구성
        H = np.array([
            [cos_t, -sin_t, dx],
            [sin_t, cos_t, dy]
        ], dtype=np.float32)
        
        return H, {'confidence': mag}


# -----------------------------------------------------------------------------
# [Placeholder Wrappers] 비교 모델들의 플레이스홀더
# -----------------------------------------------------------------------------

class LoFTRWrapper(BaseModelWrapper):
    """
    [LoFTR] Local Feature Transformer
    
    설치: pip install kornia
    논문: https://arxiv.org/abs/2104.00680
    
    TODO: 실제 구현 시 kornia.feature.LoFTR 사용
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        
        try:
            from kornia.feature import LoFTR as KorniaLoFTR
            self.model = KorniaLoFTR(pretrained='outdoor').to(device).eval()
            self._available = True
            print("[LoFTR] Loaded successfully")
        except ImportError:
            print("[LoFTR] Not available - install kornia: pip install kornia")
    
    @property
    def name(self) -> str:
        return "LoFTR"
    
    @torch.no_grad()
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if not self._available:
            return np.eye(3, dtype=np.float32)[:2], {'error': 'not_available'}
        
        # 그레이스케일 변환
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        
        # 텐서 변환
        tensor_a = torch.from_numpy(gray_a).float()[None, None] / 255.0
        tensor_b = torch.from_numpy(gray_b).float()[None, None] / 255.0
        tensor_a = tensor_a.to(self.device)
        tensor_b = tensor_b.to(self.device)
        
        # 매칭
        input_dict = {'image0': tensor_a, 'image1': tensor_b}
        correspondences = self.model(input_dict)
        
        pts_a = correspondences['keypoints0'].cpu().numpy()
        pts_b = correspondences['keypoints1'].cpu().numpy()
        
        if len(pts_a) < 4:
            return np.eye(3, dtype=np.float32)[:2], {'num_matches': 0}
        
        # Homography 추정
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        
        return H[:2], {'num_matches': len(pts_a), 'inliers': mask.sum() if mask is not None else 0}


class SE2LoFTRWrapper(BaseModelWrapper):
    """
    [SE(2)-LoFTR] Rotation-Equivariant LoFTR
    
    논문: https://arxiv.org/abs/2204.10144
    
    TODO: 공식 구현 필요
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        print("[SE2-LoFTR] Placeholder - implement with official code")
    
    @property
    def name(self) -> str:
        return "SE2-LoFTR"
    
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # Placeholder: Identity 반환
        return np.eye(3, dtype=np.float32)[:2], {'error': 'not_implemented'}


class ASPanFormerWrapper(BaseModelWrapper):
    """
    [ASpanFormer] Adaptive Span Transformer
    
    논문: https://arxiv.org/abs/2208.14201
    GitHub: https://github.com/apple/ml-aspanformer
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        print("[ASpanFormer] Placeholder - implement with official code")
    
    @property
    def name(self) -> str:
        return "ASpanFormer"
    
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return np.eye(3, dtype=np.float32)[:2], {'error': 'not_implemented'}


class DeepMatcherWrapper(BaseModelWrapper):
    """
    [DeepMatcher] Deep Learning based Matcher
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        print("[DeepMatcher] Placeholder - implement with official code")
    
    @property
    def name(self) -> str:
        return "DeepMatcher"
    
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return np.eye(3, dtype=np.float32)[:2], {'error': 'not_implemented'}


class RoMaWrapper(BaseModelWrapper):
    """
    [RoMa] Robust Dense Feature Matching
    
    논문: https://arxiv.org/abs/2305.15404
    GitHub: https://github.com/Parskatt/RoMa
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        
        try:
            # RoMa 설치 확인
            from romatch import roma_outdoor
            self.model = roma_outdoor(device=device)
            self._available = True
            print("[RoMa] Loaded successfully")
        except ImportError:
            print("[RoMa] Not available - install: pip install romatch")
    
    @property
    def name(self) -> str:
        return "RoMa"
    
    @torch.no_grad()
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if not self._available:
            return np.eye(3, dtype=np.float32)[:2], {'error': 'not_available'}
        
        # RoMa expects PIL images or tensors
        from PIL import Image
        pil_a = Image.fromarray(img_a)
        pil_b = Image.fromarray(img_b)
        
        warp, certainty = self.model.match(pil_a, pil_b)
        
        # Dense warp에서 Homography 추정
        H, W = img_a.shape[:2]
        pts_src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        pts_dst = warp[0, [0, W-1, -1, -W], :].cpu().numpy()
        
        H_mat, _ = cv2.findHomography(pts_src, pts_dst)
        if H_mat is None:
            H_mat = np.eye(3, dtype=np.float32)
        
        return H_mat[:2], {'certainty': certainty.mean().item()}


class DKMWrapper(BaseModelWrapper):
    """
    [DKM] Dense Kernelized Matching
    
    논문: https://arxiv.org/abs/2202.00667
    GitHub: https://github.com/Parskatt/DKM
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        
        try:
            from dkm import DKMv3_outdoor
            self.model = DKMv3_outdoor(device=device)
            self._available = True
            print("[DKM] Loaded successfully")
        except ImportError:
            print("[DKM] Not available - install from official repo")
    
    @property
    def name(self) -> str:
        return "DKM"
    
    @torch.no_grad()
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if not self._available:
            return np.eye(3, dtype=np.float32)[:2], {'error': 'not_available'}
        
        from PIL import Image
        pil_a = Image.fromarray(img_a)
        pil_b = Image.fromarray(img_b)
        
        dense_matches, dense_certainty = self.model.match(pil_a, pil_b)
        
        # 신뢰도 높은 매칭만 사용
        mask = dense_certainty > 0.5
        if mask.sum() < 4:
            return np.eye(3, dtype=np.float32)[:2], {'num_matches': 0}
        
        pts_a = dense_matches[mask, :2].cpu().numpy()
        pts_b = dense_matches[mask, 2:].cpu().numpy()
        
        H, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
        if H is None:
            H = np.eye(3, dtype=np.float32)
        
        return H[:2], {'num_matches': mask.sum().item()}


class PMatchWrapper(BaseModelWrapper):
    """
    [PMatch] Progressive Matching
    
    TODO: 공식 구현 필요
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        print("[PMatch] Placeholder - implement with official code")
    
    @property
    def name(self) -> str:
        return "PMatch"
    
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return np.eye(3, dtype=np.float32)[:2], {'error': 'not_implemented'}


class TopicFMWrapper(BaseModelWrapper):
    """
    [TopicFM] Topic-aware Feature Matching
    
    논문: https://arxiv.org/abs/2307.00485
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self._available = False
        print("[TopicFM] Placeholder - implement with official code")
    
    @property
    def name(self) -> str:
        return "TopicFM"
    
    def predict(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[np.ndarray, Dict]:
        return np.eye(3, dtype=np.float32)[:2], {'error': 'not_implemented'}


# -----------------------------------------------------------------------------
# [Model Registry] 모델 등록/관리
# -----------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    'ours': OursWrapper,
    'loftr': LoFTRWrapper,
    'se2_loftr': SE2LoFTRWrapper,
    'aspanformer': ASPanFormerWrapper,
    'deepmatcher': DeepMatcherWrapper,
    'roma': RoMaWrapper,
    'dkm': DKMWrapper,
    'pmatch': PMatchWrapper,
    'topicfm': TopicFMWrapper,
}

def register_model(name: str, wrapper_class: type):
    """
    [Helper] 새 모델 등록
    
    Usage:
        register_model('my_model', MyModelWrapper)
    """
    MODEL_REGISTRY[name.lower()] = wrapper_class
    print(f"Registered model: {name}")

def get_model(name: str, **kwargs) -> BaseModelWrapper:
    """
    [Helper] 모델 인스턴스 생성
    """
    if name.lower() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name.lower()](**kwargs)

def list_models() -> List[str]:
    """등록된 모델 목록"""
    return list(MODEL_REGISTRY.keys())


# =============================================================================
# [Metric Registry] 평가 지표 함수들
# =============================================================================

def compute_mace(pred_H: np.ndarray, gt_H: np.ndarray, 
                 img_size: Tuple[int, int] = (256, 256)) -> float:
    """
    [Metric] MACE - Mean Average Corner Error
    
    이미지 4개 코너를 변환하여 평균 오차 계산 (픽셀 단위)
    """
    H, W = img_size
    corners = np.array([
        [0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]
    ], dtype=np.float32).T  # (3, 4)
    
    # 2x3 → 3x3 변환
    if pred_H.shape[0] == 2:
        pred_H_full = np.vstack([pred_H, [0, 0, 1]])
    else:
        pred_H_full = pred_H
        
    if gt_H.shape[0] == 2:
        gt_H_full = np.vstack([gt_H, [0, 0, 1]])
    else:
        gt_H_full = gt_H
    
    # 변환 적용
    pred_corners = pred_H_full @ corners
    gt_corners = gt_H_full @ corners
    
    # 동차 좌표 정규화
    pred_corners = pred_corners[:2] / (pred_corners[2:3] + 1e-8)
    gt_corners = gt_corners[:2] / (gt_corners[2:3] + 1e-8)
    
    # 평균 오차
    error = np.linalg.norm(pred_corners - gt_corners, axis=0).mean()
    return float(error)


def compute_alignment_error(pred_H: np.ndarray, gt_H: np.ndarray,
                            img_size: Tuple[int, int] = (256, 256)) -> float:
    """
    [Metric] Alignment Error
    
    전체 이미지 그리드에서의 평균 정렬 오차
    """
    H, W = img_size
    
    # 그리드 생성 (샘플링)
    step = 16
    y_coords, x_coords = np.mgrid[0:H:step, 0:W:step]
    points = np.stack([x_coords.flatten(), y_coords.flatten(), 
                       np.ones(x_coords.size)], axis=0)  # (3, N)
    
    # 변환
    if pred_H.shape[0] == 2:
        pred_H = np.vstack([pred_H, [0, 0, 1]])
    if gt_H.shape[0] == 2:
        gt_H = np.vstack([gt_H, [0, 0, 1]])
    
    pred_pts = pred_H @ points
    gt_pts = gt_H @ points
    
    pred_pts = pred_pts[:2] / (pred_pts[2:3] + 1e-8)
    gt_pts = gt_pts[:2] / (gt_pts[2:3] + 1e-8)
    
    error = np.linalg.norm(pred_pts - gt_pts, axis=0).mean()
    return float(error)


def compute_success_rate(mace: float, threshold: float) -> bool:
    """
    [Metric] Success Rate
    
    MACE가 threshold 이하면 성공
    """
    return mace <= threshold


def compute_recall_dense(pred_H: np.ndarray, gt_H: np.ndarray,
                         img_size: Tuple[int, int] = (256, 256),
                         threshold: float = 5.0) -> float:
    """
    [Metric] Recall (Dense Matching용)
    
    정답 변환 후 threshold 이내에 있는 픽셀 비율
    """
    H, W = img_size
    
    # 전체 픽셀 그리드 (서브샘플링)
    step = 8
    y_coords, x_coords = np.mgrid[0:H:step, 0:W:step]
    points = np.stack([x_coords.flatten(), y_coords.flatten(), 
                       np.ones(x_coords.size)], axis=0)
    
    if pred_H.shape[0] == 2:
        pred_H = np.vstack([pred_H, [0, 0, 1]])
    if gt_H.shape[0] == 2:
        gt_H = np.vstack([gt_H, [0, 0, 1]])
    
    pred_pts = pred_H @ points
    gt_pts = gt_H @ points
    
    pred_pts = pred_pts[:2] / (pred_pts[2:3] + 1e-8)
    gt_pts = gt_pts[:2] / (gt_pts[2:3] + 1e-8)
    
    errors = np.linalg.norm(pred_pts - gt_pts, axis=0)
    recall = (errors < threshold).mean()
    
    return float(recall)


METRIC_REGISTRY = {
    'mace': compute_mace,
    'alignment_error': compute_alignment_error,
    'sr_10': lambda pred, gt, size: compute_success_rate(compute_mace(pred, gt, size), SR_THRESHOLD_10),
    'sr_5': lambda pred, gt, size: compute_success_rate(compute_mace(pred, gt, size), SR_THRESHOLD_5),
    'recall': compute_recall_dense,
}


# =============================================================================
# [Evaluation Engine] 평가 실행 클래스
# =============================================================================

class BenchmarkEvaluator:
    """
    [Main] 벤치마크 평가 엔진
    
    Usage:
        evaluator = BenchmarkEvaluator()
        evaluator.add_model('ours', checkpoint_path='best_model.pth')
        evaluator.add_model('loftr')
        results = evaluator.evaluate(test_pairs)
        evaluator.save_results('results.json')
    """
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.models: Dict[str, BaseModelWrapper] = {}
        self.results: Dict[str, ModelBenchmark] = {}
        
    def add_model(self, name: str, checkpoint_path: Optional[str] = None):
        """모델 추가"""
        model = get_model(name, checkpoint_path=checkpoint_path, device=self.device)
        self.models[model.name] = model
        print(f"Added model: {model.name}")
        
    def remove_model(self, name: str):
        """모델 제거"""
        for key in list(self.models.keys()):
            if key.lower() == name.lower():
                del self.models[key]
                print(f"Removed model: {name}")
                return
        print(f"Model not found: {name}")
        
    def list_loaded_models(self) -> List[str]:
        """로드된 모델 목록"""
        return list(self.models.keys())
    
    def evaluate_single(self, model: BaseModelWrapper, 
                        img_a: np.ndarray, img_b: np.ndarray,
                        gt_H: np.ndarray) -> EvalResult:
        """단일 샘플 평가"""
        result = EvalResult()
        img_size = img_a.shape[:2]
        
        # 런타임 측정
        start_time = time.perf_counter()
        try:
            pred_H, extra = model.predict(img_a, img_b)
            result.matching_success = 'error' not in extra
        except Exception as e:
            print(f"[{model.name}] Prediction failed: {e}")
            pred_H = np.eye(3, dtype=np.float32)[:2]
            result.matching_success = False
            
        result.runtime_ms = (time.perf_counter() - start_time) * 1000
        
        # 메트릭 계산
        result.mace = compute_mace(pred_H, gt_H, img_size)
        result.sr_10 = compute_success_rate(result.mace, SR_THRESHOLD_10)
        result.sr_5 = compute_success_rate(result.mace, SR_THRESHOLD_5)
        result.alignment_error = compute_alignment_error(pred_H, gt_H, img_size)
        result.recall = compute_recall_dense(pred_H, gt_H, img_size)
        
        return result
    
    def evaluate(self, test_data: List[Dict], 
                 warmup: bool = True) -> Dict[str, ModelBenchmark]:
        """
        전체 데이터셋 평가
        
        Args:
            test_data: [{'img_a': ndarray, 'img_b': ndarray, 'gt_H': ndarray}, ...]
            warmup: GPU 워밍업 여부
            
        Returns:
            모델별 벤치마크 결과
        """
        # 워밍업
        if warmup:
            print("Warming up models...")
            for model in self.models.values():
                try:
                    model.warmup()
                except:
                    pass
        
        # 평가
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            all_results: List[EvalResult] = []
            
            for sample in tqdm(test_data, desc=model_name):
                result = self.evaluate_single(
                    model, 
                    sample['img_a'], 
                    sample['img_b'], 
                    sample['gt_H']
                )
                all_results.append(result)
            
            # 집계
            benchmark = ModelBenchmark(model_name=model_name)
            benchmark.num_samples = len(all_results)
            
            maces = [r.mace for r in all_results]
            benchmark.mace_mean = np.mean(maces)
            benchmark.mace_std = np.std(maces)
            
            benchmark.sr_10_rate = np.mean([r.sr_10 for r in all_results])
            benchmark.sr_5_rate = np.mean([r.sr_5 for r in all_results])
            
            align_errors = [r.alignment_error for r in all_results]
            benchmark.alignment_error_mean = np.mean(align_errors)
            benchmark.alignment_error_std = np.std(align_errors)
            
            runtimes = [r.runtime_ms for r in all_results]
            benchmark.runtime_mean_ms = np.mean(runtimes)
            benchmark.runtime_std_ms = np.std(runtimes)
            
            benchmark.matching_success_rate = np.mean([r.matching_success for r in all_results])
            
            recalls = [r.recall for r in all_results]
            benchmark.recall_mean = np.mean(recalls)
            benchmark.recall_std = np.std(recalls)
            
            self.results[model_name] = benchmark
        
        return self.results
    
    def print_results(self):
        """결과 테이블 출력"""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)
        
        headers = ['Model', 'MACE↓', 'SR@10↑', 'SR@5↑', 'Align.Err↓', 'Runtime(ms)↓', 'Success↑', 'Recall↑']
        row_format = "{:<15}" + "{:<12}" * 7
        
        print(row_format.format(*headers))
        print("-" * 100)
        
        for name, bench in self.results.items():
            row = [
                name,
                f"{bench.mace_mean:.2f}±{bench.mace_std:.2f}",
                f"{bench.sr_10_rate*100:.1f}%",
                f"{bench.sr_5_rate*100:.1f}%",
                f"{bench.alignment_error_mean:.2f}",
                f"{bench.runtime_mean_ms:.1f}",
                f"{bench.matching_success_rate*100:.1f}%",
                f"{bench.recall_mean*100:.1f}%"
            ]
            print(row_format.format(*row))
        
        print("=" * 100)
    
    def save_results(self, path: str):
        """결과 JSON 저장"""
        data = {name: bench.to_dict() for name, bench in self.results.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to: {path}")
        
    def load_results(self, path: str):
        """결과 JSON 로드"""
        with open(path, 'r') as f:
            data = json.load(f)
        for name, bench_dict in data.items():
            self.results[name] = ModelBenchmark(**bench_dict)
        print(f"Results loaded from: {path}")


# =============================================================================
# [Test Data Generator] 테스트 데이터 생성
# =============================================================================

def generate_synthetic_test_data(img_dir: str, num_samples: int = 100,
                                  rotation_range: Tuple[float, float] = (-45, 45),
                                  scale_range: Tuple[float, float] = (0.8, 1.2)) -> List[Dict]:
    """
    [Helper] 합성 테스트 데이터 생성
    
    이미지에 무작위 변환을 적용하여 (원본, 변환됨, GT) 쌍 생성
    """
    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    
    if len(img_paths) == 0:
        raise ValueError(f"No images found in {img_dir}")
    
    test_data = []
    
    for i in tqdm(range(num_samples), desc="Generating test data"):
        # 이미지 선택
        img_path = img_paths[i % len(img_paths)]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, DEFAULT_IMG_SIZE)
        
        H, W = img.shape[:2]
        
        # 무작위 변환 생성
        angle = np.random.uniform(*rotation_range)
        scale = np.random.uniform(*scale_range)
        
        M = cv2.getRotationMatrix2D((W/2, H/2), angle, scale)
        img_warped = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        
        # GT: 역변환 (warped → original)
        M_aug = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_aug)[:2]
        
        # 정규화 좌표계로 변환
        N = np.array([[2.0/W, 0, -1], [0, 2.0/H, -1], [0, 0, 1]])
        N_inv = np.linalg.inv(N)
        M_inv_aug = np.vstack([M_inv, [0, 0, 1]])
        gt_H_norm = (N @ M_inv_aug @ N_inv)[:2]
        
        test_data.append({
            'img_a': img_warped,  # Source (변환됨)
            'img_b': img,          # Target (원본)
            'gt_H': gt_H_norm,     # GT 변환
            'angle': angle,
            'scale': scale
        })
    
    return test_data


# =============================================================================
# [Main Entry Point]
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dense Matching Model Benchmark')
    parser.add_argument('--data_dir', type=str, default='./val2017',
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--ours_checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to our model checkpoint')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['ours', 'loftr'],
                        help='Models to evaluate')
    parser.add_argument('--rotation_range', type=float, nargs=2, default=[-20, 20],
                        help='Rotation range for synthetic data')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 평가기 초기화
    evaluator = BenchmarkEvaluator()
    
    # 모델 추가
    for model_name in args.models:
        if model_name.lower() == 'ours':
            evaluator.add_model(model_name, checkpoint_path=args.ours_checkpoint)
        else:
            evaluator.add_model(model_name)
    
    # 테스트 데이터 생성
    print(f"\nGenerating {args.num_samples} test samples...")
    test_data = generate_synthetic_test_data(
        args.data_dir, 
        num_samples=args.num_samples,
        rotation_range=tuple(args.rotation_range)
    )
    
    # 평가 실행
    evaluator.evaluate(test_data)
    
    # 결과 출력 및 저장
    evaluator.print_results()
    evaluator.save_results(os.path.join(args.output_dir, 'benchmark_results.json'))
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()