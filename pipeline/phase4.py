"""
Phase 4 recurrent refiner (historically called Phase 3.5 in some comments).

This stage starts from Phase 3's accumulated transform and performs a small
number of recurrent residual updates. The design is intentionally conservative:

- internal transform convention: B -> A theta for affine_grid/grid_sample
- initialization: prefer Phase 3 finest accumulated transform, then fall back
  to coarser levels only when the finest result is missing or invalid
- routing: compute the paper-style controller score J_k from S/V/B residuals
  and soft Phase-3 gate priors, then choose both the operating pyramid level
  and one S/V/B branch
- priors: use Phase 3 gate maps through soft controller weights so that no
  stream is completely suppressed by a low prior
- safety: evaluate candidate updates with position, angular, bivector, and
  J_k diagnostics; reject non-improving/divergent proposals, reset hidden
  state after repeated rejections, and decay the step scale before aborting

The module keeps the public class names used by the existing training script.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# =============================================================================
# Phase 4 defaults
# =============================================================================
NUM_ITERATIONS = 4

GRU_HIDDEN_DIM = 16

# Legacy position-only thresholds are kept for external compatibility.
# Internally, Phase 4 now selects the operating level from the paper's
# controller score J_k instead of from e_pos alone.
LEVEL_THRESHOLD_HIGH = 30
LEVEL_THRESHOLD_MID = 10
LEVEL_THRESHOLD_LOW = 5

# Paper Eq. (54): level selection thresholds for the controller score J_k.
LEVEL_SCORE_HIGH = 2.50
LEVEL_SCORE_MID = 1.50
LEVEL_SCORE_LOW = 0.75

# Feature Selection thresholds (Architecture.md / paper §3.4)
FEATURE_POS_THRESHOLD = 10.0      # px
FEATURE_ANGLE_THRESHOLD = 0.1     # cosine-distance angular mismatch
FEATURE_B_THRESHOLD = 0.08        # bivector / rotor residual (heuristic)
FEATURE_ANGLE_FALLBACK_THRESHOLD = 0.06
FEATURE_B_DOMINANCE_MARGIN = 1.15

# Convergence & Target
CONVERGENCE_THRESHOLD = 0.005
TARGET_ERROR_PX = 3.0
# Paper Eq. (63): stop only when position, angular, rotor residuals, and J_k
# are all small. This avoids a position-only early exit.
TARGET_ANGLE_ERROR = 0.08
TARGET_B_ERROR = 0.06
TARGET_CONTROLLER_SCORE = 0.55
TOLERANCE_ALPHA = 0.05

# Paper Eqs. (64)-(65): safety/acceptance thresholds.
POS_REJECT_RATIO = 1.15
ANGLE_REJECT_RATIO = 1.20
B_REJECT_RATIO = 1.15
ACCEPT_ANGLE_IMPROVE_RATIO = 0.90
ACCEPT_RESIDUAL_TOL = 1.05

# Safety Lock
MAX_CONSECUTIVE_REJECTIONS = 3
LR_DECAY_FACTOR = 0.5

# LR(학습률) 개념을 "업데이트 스케일(step_scale)"로 구현
STEP_SCALE_INIT = 1.0


# =============================================================================
# Mini-ConvGRU
# =============================================================================

class MiniConvGRU(nn.Module):
    """
    Architecture.md §3.5.2 - Mini-ConvGRU

    Minimal-GRU 구조:
    - z_k = σ(Conv([h_{k-1}, E_diff]))
    - h̃_k = tanh(Conv([F_selected, E_diff]))
    - h_k = (1-z_k) ⊙ h_{k-1} + z_k ⊙ h̃_k
    - ΔW_k = Head(h_k)  (16→8→4)
    """
    def __init__(self, input_dim, hidden_dim=GRU_HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv_z = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(input_dim * 2, hidden_dim, kernel_size=3, padding=1)

        self.delta_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(hidden_dim // 2, 4, kernel_size=1)  # (cos, sin, dx, dy)
        )

    def init_hidden(self, batch, height, width, device):
        return torch.zeros(batch, self.hidden_dim, height, width, device=device)

    def forward(self, h_prev, e_diff, f_selected):
        z_input = torch.cat([h_prev, e_diff], dim=1)
        z = torch.sigmoid(self.conv_z(z_input))

        h_input = torch.cat([f_selected, e_diff], dim=1)
        h_candidate = torch.tanh(self.conv_h(h_input))

        h_new = (1 - z) * h_prev + z * h_candidate
        delta_w = self.delta_head(h_new)

        return h_new, delta_w


# =============================================================================
# Error diagnostic
# =============================================================================

class ErrorDiagnostic(nn.Module):
    """
    Measure the three residuals used by the selector.

    Notes
    -----
    The Phase 4 input volumes are already grouped as [S | V | B] channel blocks.
    The vector and bivector terms here therefore behave as learned geometric
    descriptors, not as raw 2-D vectors from Phase 1.
    """
    def compute_position_error(self, warped_sdf, target_sdf, valid_mask=None):
        diff = torch.abs(warped_sdf - target_sdf)
        if valid_mask is None:
            error_px = diff.mean(dim=(1, 2, 3)) * 100
        else:
            error_px = (diff * valid_mask).sum(dim=(1, 2, 3)) / (
                valid_mask.sum(dim=(1, 2, 3)) * warped_sdf.shape[1] + 1e-6
            ) * 100
        return error_px

    def compute_angle_error(self, warped_vector, target_vector, valid_mask=None):
        cos_sim = F.cosine_similarity(warped_vector, target_vector, dim=1, eps=1e-6)
        if valid_mask is None:
            mean_cos = cos_sim.mean(dim=(1, 2))
        else:
            m = valid_mask[:, 0]
            mean_cos = (cos_sim * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) + 1e-6)
        return (1.0 - mean_cos).clamp(0, 2)

    def compute_bivector_error(self, warped_bivector, target_bivector, valid_mask=None):
        diff = torch.abs(warped_bivector - target_bivector)
        if valid_mask is None:
            return diff.mean(dim=(1, 2, 3))
        return (diff * valid_mask).sum(dim=(1, 2, 3)) / (
            valid_mask.sum(dim=(1, 2, 3)) * warped_bivector.shape[1] + 1e-6
        )

    def forward(self, warped_features, target_features, valid_mask=None):
        e_pos = self.compute_position_error(warped_features['sdf'], target_features['sdf'], valid_mask)
        e_angle = self.compute_angle_error(warped_features['vector'], target_features['vector'], valid_mask)
        e_b = self.compute_bivector_error(warped_features['bivector'], target_features['bivector'], valid_mask)
        return e_pos, e_angle, e_b


# =============================================================================
# Dual-adaptive routing
# =============================================================================

class DualAdaptiveSelector(nn.Module):
    """Paper-aligned dual-adaptive controller.

    The selector does not choose the already-best-matching stream. It routes the
    next GRU update toward the largest *reliable* residual stream. Reliability is
    injected through the soft gate-prior weight

        w(pi) = 0.5 * clip(pi, 1e-3, 1.0) + 0.5,

    which follows the paper and avoids fully suppressing any S/V/B stream.
    """
    @staticmethod
    def _to_float(value, default=1.0):
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            return float(value.detach().mean().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def soft_prior(cls, value):
        """[Phase4-PAPER-CTRL] Soft prior w(pi), not direct prior multiplication."""
        prior = cls._to_float(value, default=1.0)
        prior = float(np.clip(prior, 1e-3, 1.0))
        return 0.5 * prior + 0.5

    @staticmethod
    def summarize_residuals(e_pos, e_angle, e_b):
        return {
            'pos': e_pos.mean().item(),
            'angle': e_angle.mean().item(),
            'b': e_b.mean().item(),
        }

    def normalized_scores_from_values(self, avg_pos, avg_angle, avg_b, gate_priors=None):
        """Return normalized q_S, q_V, q_B and the soft weights used to form them."""
        if gate_priors is None:
            gate_priors = {'S': 1.0, 'V': 1.0, 'B': 1.0}

        # [Phase4-PAPER-CTRL] Paper uses soft controller weights, so a low gate
        # prior cannot completely suppress an otherwise useful residual stream.
        weight_s = self.soft_prior(gate_priors.get('S', 1.0))
        weight_v = self.soft_prior(gate_priors.get('V', 1.0))
        weight_b = self.soft_prior(gate_priors.get('B', gate_priors.get('R', 1.0)))

        scores = {
            'S': weight_s * (avg_pos / max(FEATURE_POS_THRESHOLD, 1e-6)),
            'V': weight_v * (avg_angle / max(FEATURE_ANGLE_THRESHOLD, 1e-6)),
            'B': weight_b * (avg_b / max(FEATURE_B_THRESHOLD, 1e-6)),
        }
        weights = {'S': weight_s, 'V': weight_v, 'B': weight_b}
        return scores, weights

    def normalized_scores(self, e_pos, e_angle, e_b, gate_priors=None):
        """Return q_S, q_V, q_B using paper-style residual normalization."""
        residuals = self.summarize_residuals(e_pos, e_angle, e_b)
        scores, _ = self.normalized_scores_from_values(
            residuals['pos'], residuals['angle'], residuals['b'], gate_priors
        )
        return scores['S'], scores['V'], scores['B']

    def component_scores(self, e_pos, e_angle, e_b, gate_priors=None):
        residuals = self.summarize_residuals(e_pos, e_angle, e_b)
        scores, weights = self.normalized_scores_from_values(
            residuals['pos'], residuals['angle'], residuals['b'], gate_priors
        )
        return scores, weights, residuals

    def controller_score(self, e_pos, e_angle, e_b, gate_priors=None):
        """Paper Eq. (54): J_k = q_S + q_V + q_B on the finest grid."""
        scores, weights, residuals = self.component_scores(e_pos, e_angle, e_b, gate_priors)
        J = scores['S'] + scores['V'] + scores['B']
        return J, scores, weights, residuals

    @staticmethod
    def select_level_from_score(controller_score):
        """[Phase4-PAPER-CTRL] Eq. (54) level selection from J_k."""
        J_k = float(controller_score)
        if J_k > LEVEL_SCORE_HIGH:
            return 3
        elif J_k > LEVEL_SCORE_MID:
            return 2
        elif J_k > LEVEL_SCORE_LOW:
            return 1
        else:
            return 0

    def select_level(self, e_pos, e_angle=None, e_b=None, gate_priors=None):
        """Choose level from J_k when all residuals are supplied.

        The position-only path is intentionally kept for old/debug callers, but
        the Phase 4 forward loop calls this with e_pos/e_angle/e_b and fine-level
        gate priors so the paper's J_k controller is used.
        """
        if e_angle is None or e_b is None:
            avg_error = e_pos.mean().item()
            if avg_error > LEVEL_THRESHOLD_HIGH:
                return 3
            elif avg_error > LEVEL_THRESHOLD_MID:
                return 2
            elif avg_error > LEVEL_THRESHOLD_LOW:
                return 1
            else:
                return 0

        J, _, _, _ = self.controller_score(e_pos, e_angle, e_b, gate_priors=gate_priors)
        return self.select_level_from_score(J)

    def select_feature(self, e_pos, e_angle, e_b, gate_priors=None):
        """Paper §3.4 stream selection with soft priors and angular fallback.

        The scores select the stream that is both mismatched and reliable:
        - B/Rstr first when q_B is dominant and the B residual is above threshold.
        - S when the normalized position residual dominates V and is above threshold.
        - V when the angular residual is clearly above threshold.
        - V fallback when angular mismatch remains non-negligible, position is
          already acceptable, and B is not clearly dominant.
        - B/Rstr otherwise as the final low-residual corrective stream.
        """
        scores, _, residuals = self.component_scores(e_pos, e_angle, e_b, gate_priors)
        avg_pos = residuals['pos']
        avg_angle = residuals['angle']
        avg_b = residuals['b']

        score_s = scores['S']
        score_v = scores['V']
        score_b = scores['B']

        b_is_dominant = (score_b >= score_s) and (score_b >= score_v)
        if b_is_dominant and (avg_b > FEATURE_B_THRESHOLD):
            return 'B'
        elif (score_s >= score_v) and (avg_pos > FEATURE_POS_THRESHOLD):
            return 'S'
        elif avg_angle > FEATURE_ANGLE_THRESHOLD:
            return 'V'
        elif (
            avg_angle > FEATURE_ANGLE_FALLBACK_THRESHOLD
            and avg_pos < FEATURE_POS_THRESHOLD
            and score_b < FEATURE_B_DOMINANCE_MARGIN * score_v
        ):
            # [Phase4-PAPER-CTRL] Paper fallback: keep using V when angular
            # mismatch remains, S is already within tolerance, and B/Rstr is not
            # clearly dominant.
            return 'V'
        else:
            return 'B'

    def forward(self, e_pos, e_angle, e_b, gate_priors=None, fine_gate_priors=None, level_gate_priors=None):
        """Return level, feature, and J_k.

        `gate_priors` is kept for backward-compatible callers that used the old
        single-prior argument.  The main loop passes `fine_gate_priors` for J_k
        and `level_gate_priors` for stream selection.
        """
        if fine_gate_priors is None:
            fine_gate_priors = gate_priors
        if level_gate_priors is None:
            level_gate_priors = gate_priors

        J_k, _, _, _ = self.controller_score(e_pos, e_angle, e_b, gate_priors=fine_gate_priors)
        selected_level = self.select_level_from_score(J_k)
        selected_feature = self.select_feature(e_pos, e_angle, e_b, gate_priors=level_gate_priors)
        return selected_level, selected_feature, J_k


# =============================================================================
# [Transform Accumulator]
# =============================================================================

class TransformAccumulator:
    """
    Accumulate the global backward warp kept inside Phase 4.

    Convention
    ----------
    `self.W_accum` is always theta_B2A: the matrix passed directly to
    affine_grid/grid_sample to warp source(A) features onto target(B).

    The GRU predicts a residual on the *already warped* feature grid, so the
    update order is:

        W_next = W_curr ∘ Δ_residual

    which corresponds to `W_aug @ delta_aug` in homogeneous coordinates.
    """
    def __init__(self, device):
        self.device = device
        self.W_accum = torch.eye(2, 3, device=device).unsqueeze(0)

    def reset(self, batch_size=1, init_W=None):
        if init_W is not None:
            self.W_accum = init_W.to(self.device)
            if batch_size > 1 and self.W_accum.shape[0] == 1:
                self.W_accum = self.W_accum.repeat(batch_size, 1, 1)
            return

        self.W_accum = torch.eye(2, 3, device=self.device).unsqueeze(0)
        if batch_size > 1:
            self.W_accum = self.W_accum.repeat(batch_size, 1, 1)

    @staticmethod
    def _to_aug(W_2x3):
        B = W_2x3.shape[0]
        bottom = torch.tensor([0, 0, 1], device=W_2x3.device, dtype=W_2x3.dtype).view(1, 1, 3).repeat(B, 1, 1)
        return torch.cat([W_2x3, bottom], dim=1)  # (B,3,3)

    def compose_from_delta_map(self, delta_w_map, step_scale: float = 1.0):
        """
        Convert a dense residual rotor field into a single affine update and
        compose it with the current B->A warp.

        `step_scale` shrinks the proposed update toward identity after repeated
        rejections, which is the Phase 4 analogue of a local learning-rate decay.
        """
        avg_delta = delta_w_map.mean(dim=(2, 3))  # (B,4)
        cos_d, sin_d, dx_d, dy_d = avg_delta[:, 0], avg_delta[:, 1], avg_delta[:, 2], avg_delta[:, 3]

        # --- Step scaling (Identity로 수렴시키는 형태) ---
        # cos' = 1 + s*(cos-1)
        # sin' = s*sin
        # dx'  = s*dx
        # dy'  = s*dy
        s = float(step_scale)
        cos_d = 1.0 + s * (cos_d - 1.0)
        sin_d = s * sin_d
        dx_d = s * dx_d
        dy_d = s * dy_d

        # NLERP 정규화
        norm = torch.sqrt(cos_d**2 + sin_d**2 + 1e-6)
        cos_d = cos_d / norm
        sin_d = sin_d / norm

        B = cos_d.shape[0]
        delta_mat = torch.zeros(B, 2, 3, device=self.device, dtype=delta_w_map.dtype)
        delta_mat[:, 0, 0] = cos_d
        delta_mat[:, 0, 1] = -sin_d
        delta_mat[:, 0, 2] = dx_d
        delta_mat[:, 1, 0] = sin_d
        delta_mat[:, 1, 1] = cos_d
        delta_mat[:, 1, 2] = dy_d

        # 합성
        W_aug = self._to_aug(self.W_accum)
        d_aug = self._to_aug(delta_mat)
        out = torch.bmm(W_aug, d_aug)  # compose: W ∘ ΔW
        return out[:, :2, :]

    def set_current(self, W_2x3):
        self.W_accum = W_2x3

    def get_current(self):
        return self.W_accum


# =============================================================================
# Iterative refinement loop
# =============================================================================

class IterativeRefinementLoop(nn.Module):
    """
    Run a short recurrent refinement loop on top of Phase 3.

    The loop always keeps the transform in B->A(theta) form internally because
    that is the convention required by affine_grid/grid_sample when we warp A
    features onto the B grid.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.error_diagnostic = ErrorDiagnostic()
        self.selector = DualAdaptiveSelector()

        self.chunk_dim = feature_dim // 3

        self.gru_s = MiniConvGRU(input_dim=self.chunk_dim)
        self.gru_v = MiniConvGRU(input_dim=self.chunk_dim)
        self.gru_b = MiniConvGRU(input_dim=self.chunk_dim)

    def extract_feature_by_type(self, features, feature_type):
        s, v, b = torch.chunk(features, 3, dim=1)
        if feature_type == 'S':
            return s
        if feature_type == 'V':
            return v
        return b

    def get_gru_by_type(self, feature_type):
        if feature_type == 'S':
            return self.gru_s
        if feature_type == 'V':
            return self.gru_v
        return self.gru_b

    @staticmethod
    def warp_features(features, W_matrix, target_size, return_mask=False):
        grid = F.affine_grid(
            W_matrix,
            [features.shape[0], features.shape[1], target_size[0], target_size[1]],
            align_corners=True
        )
        warped = F.grid_sample(
            features, grid,
            align_corners=True,
            mode='bilinear',
            padding_mode='zeros'
        )

        if not return_mask:
            return warped

        valid_mask = F.grid_sample(
            torch.ones(features.shape[0], 1, features.shape[2], features.shape[3],
                    device=features.device, dtype=features.dtype),
            grid,
            align_corners=True,
            mode='nearest',
            padding_mode='zeros'
        )
        valid_mask = (valid_mask > 0.999).to(dtype=features.dtype)
        return warped, valid_mask

    def compute_error(self, feat_a, feat_b, W_matrix):
        """
        Evaluate the current B->A warp on a pair of Phase 4 feature volumes.
        """
        B, C, H, W = feat_b.shape
        feat_a_warped, valid_mask = self.warp_features(
            feat_a, W_matrix, (H, W), return_mask=True
        )
        e_diff = torch.abs(feat_a_warped - feat_b) * valid_mask

        warped_dict = {
            'sdf': feat_a_warped[:, :self.chunk_dim, :, :],
            'vector': feat_a_warped[:, self.chunk_dim:2*self.chunk_dim, :, :],
            'bivector': feat_a_warped[:, 2*self.chunk_dim:, :, :],
        }
        target_dict = {
            'sdf': feat_b[:, :self.chunk_dim, :, :],
            'vector': feat_b[:, self.chunk_dim:2*self.chunk_dim, :, :],
            'bivector': feat_b[:, 2*self.chunk_dim:, :, :],
        }

        e_pos, e_angle, e_b = self.error_diagnostic(warped_dict, target_dict, valid_mask)
        return e_pos, e_angle, e_b, e_diff, feat_a_warped

    def check_convergence(self, delta_w):
        """
        Architecture.md 종료 조건 A: ||ΔW - I||_F < ε_conv
        """
        B = delta_w.shape[0]
        avg_delta = delta_w.mean(dim=(2, 3))
        identity = torch.tensor([[1, 0, 0, 0]], device=delta_w.device, dtype=delta_w.dtype).repeat(B, 1)
        diff = avg_delta - identity
        frob = torch.norm(diff, dim=1).mean().item()
        return frob < CONVERGENCE_THRESHOLD

    @staticmethod
    def rotor_map_to_theta(rotor_map):
        """
        rotor_map: (B,H,W,4) -> (B,2,3)
        """
        avg = rotor_map.mean(dim=(1, 2))
        cos_t, sin_t, dx_t, dy_t = avg[:, 0], avg[:, 1], avg[:, 2], avg[:, 3]
        norm = torch.sqrt(cos_t**2 + sin_t**2 + 1e-6)
        cos_t = cos_t / norm
        sin_t = sin_t / norm

        B = rotor_map.shape[0]
        theta = torch.zeros(B, 2, 3, device=rotor_map.device, dtype=rotor_map.dtype)
        theta[:, 0, 0] = cos_t
        theta[:, 0, 1] = -sin_t
        theta[:, 0, 2] = dx_t
        theta[:, 1, 0] = sin_t
        theta[:, 1, 1] = cos_t
        theta[:, 1, 2] = dy_t
        return theta

    @staticmethod
    def invert_affine_2x3(W_2x3):
        B = W_2x3.shape[0]
        device = W_2x3.device
        dtype = W_2x3.dtype
        bottom = torch.tensor([0, 0, 1], device=device, dtype=dtype).view(1, 1, 3).repeat(B, 1, 1)
        W_aug = torch.cat([W_2x3, bottom], dim=1)
        W_inv = torch.inverse(W_aug)
        return W_inv[:, :2, :]

    def get_phase3_result_for_level(self, phase3_results, target_level):
        if phase3_results is None or len(phase3_results) == 0:
            return None

        matched = [res for res in phase3_results if res.get('level', target_level) == target_level]
        if len(matched) > 0:
            return matched[0]

        return min(phase3_results, key=lambda d: abs(d.get('level', target_level) - target_level))

    def reduce_gate_prior(self, gate_map, target_hw=None):
        if gate_map is None:
            return None

        if gate_map.dim() == 3:
            gate_map = gate_map.unsqueeze(1)
        elif gate_map.dim() == 4 and gate_map.shape[1] != 1:
            gate_map = gate_map.mean(dim=1, keepdim=True)

        if (target_hw is not None) and (gate_map.shape[-2:] != target_hw):
            gate_map = F.interpolate(gate_map, size=target_hw, mode='bilinear', align_corners=True)

        return gate_map.mean().item()

    def get_feature_gate_priors(self, phase3_results, level_idx, target_hw=None):
        priors = {'S': 1.0, 'V': 1.0, 'B': 1.0}

        level_res = self.get_phase3_result_for_level(phase3_results, level_idx)
        if level_res is None:
            return priors

        key_groups = {
            'S': ('pure_g_s', 'g_s', 'g_s_task', 'gate_s'),
            'V': ('pure_g_v', 'g_v', 'g_v_task', 'gate_v'),
            'B': ('pure_g_b', 'g_b', 'g_b_task', 'gate_b'),
        }
        tuple_index = {'S': 0, 'V': 1, 'B': 2}

        for feature_type, keys in key_groups.items():
            gate_map = None

            for key in keys:
                if isinstance(level_res, dict) and level_res.get(key, None) is not None:
                    gate_map = level_res[key]
                    break

            if gate_map is None and isinstance(level_res, dict):
                gates = level_res.get('gates', None)
                if isinstance(gates, dict):
                    dict_key = keys[1] if len(keys) > 1 else keys[0]
                    gate_map = gates.get(dict_key, None)
                elif isinstance(gates, (tuple, list)) and len(gates) == 3:
                    gate_map = gates[tuple_index[feature_type]]

            reduced = self.reduce_gate_prior(gate_map, target_hw=target_hw)
            if reduced is not None:
                priors[feature_type] = float(np.clip(reduced, 1e-3, 1.0))

        return priors

    def get_init_transform_from_phase3(self, phase3_results):
        if phase3_results is None or len(phase3_results) == 0:
            return None

        # Start from the finest accumulated transform when it exists.
        # Phase 3 now keeps a more trustworthy fine-level accumulation, so the
        # coarsest level is only used as a fallback.
        results_sorted = sorted(phase3_results, key=lambda d: d.get('level', 0))
        candidate_results = []
        if len(results_sorted) > 0:
            candidate_results.append(results_sorted[0])
        if len(results_sorted) > 1:
            candidate_results.append(results_sorted[-1])

        for res in candidate_results:
            init_W = None

            for key in ('W_B2A', 'W_global'):
                if isinstance(res, dict) and res.get(key, None) is not None:
                    init_W = res[key]
                    break

            if init_W is None and isinstance(res, dict) and res.get('W_AB', None) is not None:
                init_W = self.invert_affine_2x3(res['W_AB'])

            if init_W is None and isinstance(res, dict):
                rotor_map = res.get('delta_rotor_map', None)
                if rotor_map is None:
                    rotor_map = res.get('rotor_map', None)
                if rotor_map is not None:
                    init_W = self.rotor_map_to_theta(rotor_map)

            if init_W is not None and torch.isfinite(init_W).all():
                return init_W.detach()

        return None

    def forward(self, pyramid_features_a, pyramid_features_b, phase3_results, device):
        """
        Args:
            pyramid_features_a/b:
                Per-level Phase 4 input volumes ordered from fine(level 0) to
                coarse(level N-1). In training these come from
                `Phase3Transformer.prepare_phase4_input(...)`.
            phase3_results:
                Phase 3 per-level result dicts. We read the accumulated transform
                and gate maps from here.
        Returns:
            W_final:
                Final B->A(theta) matrix.
            history:
                Per-iteration diagnostics for debugging and ablations.
        """
        B = pyramid_features_a[0].shape[0]
        _, _, H, W = pyramid_features_a[0].shape

        # -------------------------
        # Init transform
        # -------------------------
        init_W = self.get_init_transform_from_phase3(phase3_results)

        accumulator = TransformAccumulator(device)
        accumulator.reset(batch_size=B, init_W=init_W)

        # Hidden state init (Level0 resolution)
        h_s = self.gru_s.init_hidden(B, H, W, device)
        h_v = self.gru_v.init_hidden(B, H, W, device)
        h_b = self.gru_b.init_hidden(B, H, W, device)
        hidden_states = {'S': h_s, 'V': h_v, 'B': h_b}

        # Safety lock state
        consecutive_rejections = 0
        step_scale = STEP_SCALE_INIT

        history = []

        print(f"[Phase 4] Starting Iterative Refinement (max {NUM_ITERATIONS} iterations)")

        for k in range(NUM_ITERATIONS):
            W_curr = accumulator.get_current()

            # 1) Current residual diagnostics on the finest operating grid.
            e_pos, e_angle, e_b, e_diff, feat_a_warped = self.compute_error(
                pyramid_features_a[0], pyramid_features_b[0], W_curr
            )
            avg_pos = e_pos.mean().item()
            avg_angle = e_angle.mean().item()
            avg_b = e_b.mean().item()
            e_curr = avg_pos  # kept for backward-compatible logging / history key names

            # [Phase4-PAPER-CTRL] Paper Eq. (54): J_k uses finest-level gate
            # priors and all three S/V/B residuals instead of using e_pos alone.
            fine_gate_priors = self.get_feature_gate_priors(
                phase3_results,
                level_idx=0,
                target_hw=(H, W),
            )
            j_curr, fine_scores, fine_weights, _ = self.selector.controller_score(
                e_pos, e_angle, e_b, gate_priors=fine_gate_priors
            )

            # 2) Target reached.
            # [Phase4-PAPER-CTRL] Paper Eq. (63): do not stop only because the
            # scalar/position residual is small; angular, rotor-related, and
            # total controller residuals must also be below their tolerances.
            if (
                avg_pos < TARGET_ERROR_PX
                and avg_angle < TARGET_ANGLE_ERROR
                and avg_b < TARGET_B_ERROR
                and j_curr < TARGET_CONTROLLER_SCORE
            ):
                print(
                    f"  [Iter {k+1}] Target reached: "
                    f"pos={avg_pos:.2f}px, angle={avg_angle:.4f}, "
                    f"B={avg_b:.4f}, J={j_curr:.4f} OK"
                )
                break

            # 3) Dual selection.
            # [Phase4-PAPER-CTRL] Select level from J_k, then select the stream
            # from q_S/q_V/q_B computed with the selected level's gate priors.
            selected_level = self.selector.select_level_from_score(j_curr)
            level_idx = min(selected_level, len(pyramid_features_a)-1)

            level_feat_a = pyramid_features_a[level_idx]
            level_feat_b = pyramid_features_b[level_idx]
            _, _, H_lvl, W_lvl = level_feat_b.shape

            gate_priors = self.get_feature_gate_priors(
                phase3_results,
                level_idx=level_idx,
                target_hw=(H_lvl, W_lvl),
            )
            selected_feature = self.selector.select_feature(e_pos, e_angle, e_b, gate_priors=gate_priors)
            score_s, score_v, score_b = self.selector.normalized_scores(
                e_pos, e_angle, e_b, gate_priors=gate_priors
            )

            # 선택된 레벨에서 현재 W로 다시 워핑하여 level-aware residual을 계산
            level_feat_a_warped, valid_mask_level = self.warp_features(
                level_feat_a, W_curr, (H_lvl, W_lvl), return_mask=True
            )
            e_diff_level = torch.abs(level_feat_a_warped - level_feat_b) * valid_mask_level

            # Selected feature + difference
            # - GP가 반영된 B-stream은 현재 정렬 상태(level_feat_a_warped)에서 보는 편이 더 안정적임
            f_selected = self.extract_feature_by_type(level_feat_a_warped, selected_feature) * valid_mask_level
            e_diff_selected = self.extract_feature_by_type(e_diff_level, selected_feature)

            # Resize to Level0 resolution for GRU stability
            if f_selected.shape[-2:] != (H, W):
                f_selected = F.interpolate(f_selected, size=(H, W), mode='bilinear', align_corners=True)
                e_diff_selected = F.interpolate(e_diff_selected, size=(H, W), mode='bilinear', align_corners=True)

            # 4) GRU update proposal
            gru = self.get_gru_by_type(selected_feature)
            h_prev = hidden_states[selected_feature]
            if h_prev.shape[-2:] != (H, W):
                h_prev = F.interpolate(h_prev, size=(H, W), mode='bilinear', align_corners=True)

            h_new, delta_w = gru(h_prev, e_diff_selected, f_selected)
            hidden_states[selected_feature] = h_new

            # 5) Convergence check (ΔW ≈ I)
            if self.check_convergence(delta_w):
                print(f"  [Iter {k+1}] Converged: ΔW ≈ I OK")
                break

            # 6) Candidate update (do NOT commit yet)
            W_candidate = accumulator.compose_from_delta_map(delta_w, step_scale=step_scale)

            # 7) Safety Lock Stage 1: paper-aligned update rejection / acceptance.
            e_pos_next, e_angle_next, e_b_next, _, _ = self.compute_error(
                pyramid_features_a[0], pyramid_features_b[0], W_candidate
            )
            e_next = e_pos_next.mean().item()
            avg_angle_next = e_angle_next.mean().item()
            avg_b_next = e_b_next.mean().item()
            j_next, _, _, _ = self.selector.controller_score(
                e_pos_next, e_angle_next, e_b_next, gate_priors=fine_gate_priors
            )

            # [Phase4-PAPER-CTRL] Paper Eq. (64): hard reject if any residual
            # increases too strongly.  This extends the old position/B-only check
            # with angular residual and uses the paper's 15/20/15% margins.
            pos_diverged = e_next > avg_pos * POS_REJECT_RATIO
            angle_diverged = avg_angle_next > avg_angle * ANGLE_REJECT_RATIO
            b_diverged = avg_b_next > avg_b * B_REJECT_RATIO
            hard_reject = pos_diverged or angle_diverged or b_diverged

            # [Phase4-PAPER-CTRL] Paper Eq. (65): otherwise accept only if the
            # global controller score improves, or if angular mismatch improves
            # by at least 10% without degrading S/B residuals by more than 5%.
            accepted_by_score = j_next < j_curr
            accepted_by_angle = (
                avg_angle_next < ACCEPT_ANGLE_IMPROVE_RATIO * avg_angle
                and e_next <= ACCEPT_RESIDUAL_TOL * avg_pos
                and avg_b_next <= ACCEPT_RESIDUAL_TOL * avg_b
            )
            should_accept = (not hard_reject) and (accepted_by_score or accepted_by_angle)

            if not should_accept:
                consecutive_rejections += 1
                reject_reasons = []
                if pos_diverged:
                    reject_reasons.append('pos')
                if angle_diverged:
                    reject_reasons.append('angle')
                if b_diverged:
                    reject_reasons.append('B')
                if not reject_reasons:
                    reject_reasons.append('no-J/angle-improvement')
                print(
                    f"  [Iter {k+1}] Update Rejected ({'+'.join(reject_reasons)}): "
                    f"pos {e_next:.2f}px vs {avg_pos:.2f}px, "
                    f"angle {avg_angle_next:.4f} vs {avg_angle:.4f}, "
                    f"B {avg_b_next:.4f} vs {avg_b:.4f}, "
                    f"J {j_next:.4f} vs {j_curr:.4f} "
                    f"({consecutive_rejections}/{MAX_CONSECUTIVE_REJECTIONS})"
                )

                # Stage 2: GRU Reset + LR Decay (2회 연속 거부 시)
                if consecutive_rejections >= 2:
                    step_scale *= LR_DECAY_FACTOR
                    # Hidden state reset (전부 0으로)
                    hidden_states['S'] = self.gru_s.init_hidden(B, H, W, device)
                    hidden_states['V'] = self.gru_v.init_hidden(B, H, W, device)
                    hidden_states['B'] = self.gru_b.init_hidden(B, H, W, device)
                    print(f"  [Recovery] GRU reset + step_scale *= {LR_DECAY_FACTOR} -> {step_scale:.4f}")

                # Stage 3: Emergency Exit
                if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                    print(f"  [Emergency] Max rejections reached, aborting.")
                    break

                # Rollback: accumulator는 commit 하지 않았으므로 W_curr 유지
                continue

            # Accept update
            accumulator.set_current(W_candidate)
            consecutive_rejections = 0

            history.append({
                'iteration': k + 1,
                'error_px': e_curr,
                'error_angle': e_angle.mean().item(),
                'error_b': avg_b,
                'controller_score': j_curr,
                'selected_level': level_idx,
                'selected_level_request': selected_level,
                'selected_feature': selected_feature,
                'gate_prior_s': gate_priors.get('S', 1.0),
                'gate_prior_v': gate_priors.get('V', 1.0),
                'gate_prior_b': gate_priors.get('B', 1.0),
                'soft_weight_s': self.selector.soft_prior(gate_priors.get('S', 1.0)),
                'soft_weight_v': self.selector.soft_prior(gate_priors.get('V', 1.0)),
                'soft_weight_b': self.selector.soft_prior(gate_priors.get('B', gate_priors.get('R', 1.0))),
                'score_s': score_s,
                'score_v': score_v,
                'score_b': score_b,
                'step_scale': step_scale,
                'error_next_px': e_next,
                'error_next_angle': avg_angle_next,
                'error_next_b': avg_b_next,
                'controller_score_next': j_next,
                'accepted_by_score': accepted_by_score,
                'accepted_by_angle': accepted_by_angle,
            })

            print(
                f"  [Iter {k+1}] "
                f"Pos={avg_pos:.1f}px → {e_next:.1f}px | "
                f"Ang={avg_angle:.4f} → {avg_angle_next:.4f} | "
                f"B={avg_b:.4f} → {avg_b_next:.4f} | "
                f"J={j_curr:.4f} → {j_next:.4f} | "
                f"Level={selected_level} | Feature={selected_feature} | step_scale={step_scale:.3f}"
            )

        W_final = accumulator.get_current()
        return W_final, history



class Phase4Refiner(nn.Module):
    """Backward-compatible wrapper with the updated Phase 4 naming."""
    def __init__(self, feature_dim):
        super().__init__()
        self.refinement_loop = IterativeRefinementLoop(feature_dim)

    def forward(self, pyramid_features_a, pyramid_features_b, phase3_results=None, device=None):
        if device is None:
            device = pyramid_features_a[0].device
        return self.refinement_loop(pyramid_features_a, pyramid_features_b, phase3_results, device)


# Historical alias kept for older training scripts/checkpoints.
Phase35Refiner = Phase4Refiner
