"""Phase 1: geometric raw feature extraction.

This module builds a Gaussian pyramid from an RGB image and extracts raw geometric
features at each level.

Outputs per level
-----------------
- hsi:      [texture, structure_energy, edge_magnitude]
- edf:      edge-distance potential field used as a scalar cue
- gradient: [dx, dy, fx, fy] = gradient vector + texture-flow vector
- bivector: normalized wedge-product candidate v_grad ∧ v_flow
- v_shape:  global summary statistics used by Phase 2
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-6

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)

STRUCTURE_TENSOR_SIGMA = 1.0
STRUCTURE_TENSOR_KSIZE = (5, 5)
STRUCT_ENERGY_GAMMA = 0.4
EDGE_MAG_GAMMA = 0.7

SDF_SKELETON_POWER = 8.0
SDF_FIELD_POWER = 2.0
SDF_FIELD_WEIGHT = 0.4
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 100


class MathGeometricPreprocessor:
    """Extract raw scalar, vector, and bivector-candidate features from an image."""

    def __init__(self, device: str = "cuda") -> None:
        # `device` is kept for backward compatibility with the existing pipeline.
        self.device = device
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE,
        )
        print("Phase 1 Preprocessor Initialized.")

    @staticmethod
    def normalize_minmax(img_data: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1] when the value range is not degenerate."""
        img_min = img_data.min()
        img_max = img_data.max()
        if img_max - img_min < EPS:
            return img_data
        return (img_data - img_min) / (img_max - img_min)

    def get_flow_features(
        self,
        gray_img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute scalar/vector cues derived from gradients and the structure tensor.

        Returns
        -------
        edge_magnitude:
            Direction-independent gradient strength.
        structure_energy:
            Tensor anisotropy measure λ1 - λ2, normalized and gamma-corrected.
        v1_x, v1_y:
            Unit gradient vector.
        v2_x, v2_y:
            Texture-flow (tangent) vector weighted by structure energy.
        bivector_candidate:
            Normalized wedge-product candidate v1 ∧ v2.
        """
        img_float = gray_img.astype(np.float32) / 255.0

        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)

        mag = np.sqrt(gx**2 + gy**2)
        edge_magnitude = np.power(self.normalize_minmax(mag), EDGE_MAG_GAMMA)

        v1_x = gx / (mag + EPS)
        v1_y = gy / (mag + EPS)

        ixx = cv2.GaussianBlur(gx**2, STRUCTURE_TENSOR_KSIZE, STRUCTURE_TENSOR_SIGMA)
        iyy = cv2.GaussianBlur(gy**2, STRUCTURE_TENSOR_KSIZE, STRUCTURE_TENSOR_SIGMA)
        ixy = cv2.GaussianBlur(gx * gy, STRUCTURE_TENSOR_KSIZE, STRUCTURE_TENSOR_SIGMA)

        structure_energy = np.sqrt((ixx - iyy) ** 2 + 4 * ixy**2)
        structure_energy = self.normalize_minmax(structure_energy)
        structure_energy = np.power(structure_energy, STRUCT_ENERGY_GAMMA)

        angle = 0.5 * np.arctan2(2 * ixy, ixx - iyy)
        v2_x = -np.sin(angle) * structure_energy
        v2_y = np.cos(angle) * structure_energy

        bivector_candidate = (v1_x * v2_y) - (v1_y * v2_x)
        bivector_candidate = bivector_candidate / (np.max(np.abs(bivector_candidate)) + EPS)

        return edge_magnitude, structure_energy, v1_x, v1_y, v2_x, v2_y, bivector_candidate

    def get_edge_sdf(self, gray_img: np.ndarray) -> np.ndarray:
        """Return the edge-distance potential field used as the 4th scalar channel.

        Notes
        -----
        The historical name `get_edge_sdf` is preserved for compatibility, but the
        computed quantity is not a signed distance field. It is an unsigned distance-
        transform-based proximity/potential map:

            base = 1 - dist / max(dist)
            out  = max(base^γ_skeleton, weight * base^γ_field)
        """
        edges = cv2.Canny(gray_img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

        base = 1.0 - (dist / (dist.max() + EPS))
        skeleton = np.power(base, SDF_SKELETON_POWER)
        field = np.power(base, SDF_FIELD_POWER)
        return np.maximum(skeleton, field * SDF_FIELD_WEIGHT)

    def _extract_raw_features(self, img_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Extract all raw Phase 1 features from a single RGB image."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        texture = self.clahe.apply(gray).astype(np.float32) / 255.0
        edge_mag, struct_energy, dx, dy, fx, fy, b_candidate = self.get_flow_features(gray)
        sdf_map = self.get_edge_sdf(gray)

        scalar_stack = np.stack([texture, struct_energy, edge_mag], axis=-1)
        vector_field = np.stack([dx, dy, fx, fy], axis=-1)

        v_shape = np.array(
            [
                np.mean(edge_mag),
                np.std(edge_mag),
                np.mean(struct_energy),
                np.std(struct_energy),
                np.mean(texture),
                np.std(texture),
            ],
            dtype=np.float32,
        )

        return {
            "rgb": img_rgb,
            "hsi": scalar_stack,
            "sdf": sdf_map,
            "gradient": vector_field,
            "bivector": b_candidate,
            "v_shape": v_shape,
        }

    def process_pyramid(self, img_rgb: np.ndarray, levels: int = 6) -> list[dict[str, np.ndarray]]:
        """Build a Gaussian pyramid and extract raw features at every level."""
        pyramid_data: list[dict[str, np.ndarray]] = []
        current_img = img_rgb.copy()

        for level_index in range(levels):
            features = self._extract_raw_features(current_img)
            features["level_index"] = level_index
            features["resolution"] = current_img.shape[:2]
            pyramid_data.append(features)

            if level_index < levels - 1:
                current_img = cv2.pyrDown(current_img)

        return pyramid_data


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def vector_to_rgb(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Visualize a vector field with HSV encoding.

    Hue   -> direction
    Value -> magnitude
    """
    magnitude, angle = cv2.cartToPolar(vx, vy)
    hsv = np.zeros((vx.shape[0], vx.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)



def visualize_pyramid_detailed(pyramid_results: list[dict[str, np.ndarray]]) -> None:
    """Visualize the multi-scale Phase 1 outputs."""
    levels = len(pyramid_results)
    cols = 8

    plt.figure(figsize=(24, 3.5 * levels))
    plt.suptitle(
        "Phase 1: Multi-Scale Geometric Pyramid Analysis",
        fontsize=24,
        fontweight="bold",
        y=0.99,
    )

    for i, data in enumerate(pyramid_results):
        h, w = data["resolution"]
        img_rgb = data["rgb"]
        hsi = data["hsi"]
        texture = hsi[:, :, 0]
        struct_energy = hsi[:, :, 1]
        edge_mag = hsi[:, :, 2]
        sdf = data["sdf"]

        vec = data["gradient"]
        v1_x, v1_y = vec[..., 0], vec[..., 1]
        v2_x, v2_y = vec[..., 2], vec[..., 3]

        rgb_v1 = vector_to_rgb(v1_x, v1_y)
        rgb_v2 = vector_to_rgb(v2_x, v2_y)

        importance = (texture * 0.2) + (struct_energy * 0.5) + (edge_mag * 0.3)
        importance = (importance - importance.min()) / (importance.max() - importance.min() + EPS)

        base_idx = i * cols

        plt.subplot(levels, cols, base_idx + 1)
        plt.imshow(img_rgb)
        plt.ylabel(f"Level {i}\n({h}x{w})", fontsize=14, fontweight="bold")
        if i == 0:
            plt.title("1. Original", fontsize=12, fontweight="bold")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(levels, cols, base_idx + 2)
        plt.imshow(texture, cmap="gray")
        if i == 0:
            plt.title("2. Texture (S)", fontsize=12, fontweight="bold")
        plt.axis("off")

        plt.subplot(levels, cols, base_idx + 3)
        plt.imshow(edge_mag, cmap="viridis")
        if i == 0:
            plt.title("3. Edge Mag (S)", fontsize=12, fontweight="bold")
        plt.axis("off")

        plt.subplot(levels, cols, base_idx + 4)
        plt.imshow(struct_energy, cmap="inferno")
        if i == 0:
            plt.title("4. Struct Energy (S)", fontsize=12, fontweight="bold")
        plt.axis("off")

        plt.subplot(levels, cols, base_idx + 5)
        plt.imshow(rgb_v1)
        if i == 0:
            plt.title("5. Gradient Vec (V1)", fontsize=12, fontweight="bold")
        plt.axis("off")

        plt.subplot(levels, cols, base_idx + 6)
        plt.imshow(rgb_v2)
        if i == 0:
            plt.title("6. Flow Vec (V2)", fontsize=12, fontweight="bold")
        plt.axis("off")

        plt.subplot(levels, cols, base_idx + 7)
        plt.imshow(sdf, cmap="coolwarm")
        if i == 0:
            plt.title("7. Distance Potential (S)", fontsize=12, fontweight="bold")
        plt.axis("off")

        plt.subplot(levels, cols, base_idx + 8)
        plt.imshow(img_rgb)
        plt.imshow(importance, cmap="jet", alpha=0.5)
        if i == 0:
            plt.title("8. Attention Map", fontsize=12, fontweight="bold", color="red")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = "./img/val2017/000000569972.jpg"
    img = cv2.imread(img_path)

    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processor = MathGeometricPreprocessor()
        pyramid_results = processor.process_pyramid(img_rgb, levels=6)
        visualize_pyramid_detailed(pyramid_results)
    else:
        print("Image Not Found!")
