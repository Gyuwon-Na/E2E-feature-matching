"""
Phase 3: multivector-aware geometric transformer and decoder.

This snapshot keeps the current feature-complete behavior:
- grouped tokenization over S / V / B streams
- rotor-scale-aware encoder attention
- coarse-to-fine decoding with transform-guided warping
- dense residual rotor regression
- high-resolution local-window cross-attention
- confidence-weighted global transform pooling
- dual gate exposure (pooling-side gates + task-side gates)
- downstream Phase 4 adapter for the explicit-GP rotor tuple
- 'pure_g_s' / 'pure_g_v' / 'pure_g_b': less-mixed gate priors for Phase 4
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipeline.phase1 import MathGeometricPreprocessor
from pipeline.phase2 import CliffordPyramidEmbedder, HIDDEN_DIM

FEATURE_DIM = 144                # Transformer 내부 연산 차원 (S/V/B 각 48ch)
NUM_ENCODER_LAYERS = 3           # Encoder 블록 수
NUM_ATTENTION_HEADS = 4          # Multi-Head Attention 헤드 수
SE_REDUCTION = 16                # SE Block 축소 비율

# Chunked-attention guardrails for large spatial maps.
SAFE_N_LIMIT = 2048
SAFE_ELEMENTS = 2**19

# High-resolution policy:
# - level >= HIGH_RES_SKIP_LEVEL: full encoder + full global cross-attention
# - level <  HIGH_RES_SKIP_LEVEL: skip encoder self-attention, but keep
#   cross-attention through a local window on the pre-aligned tokens.
HIGH_RES_SKIP_LEVEL = 2
HIGH_RES_LOCAL_WINDOW_FINE = 7
HIGH_RES_LOCAL_WINDOW_STRUCT = 11

# Rotor-scale attention hyperparameters.
ROTATION_BIAS_SCALE = 0.5
SCALE_INJECTION_LOG_CLIP = 1.50

class Mish(nn.Module):
    """Mish Activation: f(x) = x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    """Channel Attention (Squeeze-and-Excitation)"""
    def __init__(self, channel, reduction=SE_REDUCTION):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Mish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GeometricResBlock(nn.Module):
    """Residual + Dilated Conv + SE Block"""
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=max(1, dim // 3))
        self.norm1 = nn.GroupNorm(max(1, dim // 16), dim)
        self.act = Mish()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(max(1, dim // 16), dim)
        self.se = SEBlock(dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        return out + residual

class GeometricTokenizer(nn.Module):
    """
    Architecture.md §3.1 - 토큰화
    - Input Alignment은 Phase3Transformer.prepare_input()에서 수행
    - Group Conv(groups=3): S/V/B 성분의 물리적 성질이 섞이지 않도록 초기 특징 추출
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.group_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, groups=3)
        self.norm = nn.GroupNorm(3, hidden_dim)

    def forward(self, x):
        return self.norm(self.group_conv(x))

class GeometricCPE(nn.Module):
    """Architecture.md §3.2.2 - Group Convolution 기반 CPE"""
    def __init__(self, dim):
        super().__init__()
        self.pos_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.pos_conv(x)

class IndependentLinear(nn.Module):
    """Architecture.md §3.2.1 - S/V/B 독립 Q/K/V 투영"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // 3
        self.lin_s = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.lin_v = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.lin_b = nn.Linear(self.chunk_dim, self.chunk_dim)

    def forward(self, x):
        # x: (..., C)
        s, v, b = torch.chunk(x, 3, dim=-1)
        return torch.cat([self.lin_s(s), self.lin_v(v), self.lin_b(b)], dim=-1)

class RotorScaleAttention(nn.Module):
    """
    Architecture.md §3.2.3 & §3.2.4 - Rotor-Scale Attention + Injection Fusion

    Path A (Rotation): Unit Rotor 정렬(회전 방향 일치) 정보를 Attention Bias에 추가
    Path B (Scale): Rotor Magnitude 차이를 Attention Bias로 사용 + Value Injection
    """
    def __init__(self, dim, num_heads=NUM_ATTENTION_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = IndependentLinear(dim)
        self.to_k = IndependentLinear(dim)
        self.to_v = IndependentLinear(dim)

        self.proj = nn.Linear(dim, dim)

        # Scale Injection 게이트(0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Mish(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, rotor_tuple):
        """
        Args:
            x: (B, H, W, C)
            rotor_tuple: (unit_cos, unit_sin, rotor_mag) from Phase2
        """
        B, H, W, C = x.shape
        N = H * W

        # 1) Q,K,V
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) Q,K normalize (cosine similarity)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        unit_cos, unit_sin, rotor_mag = rotor_tuple  # (B, hidden_dim, H, W) each

        # Rotor magnitude(Scale) - per-pixel mean
        r_mag = rotor_mag.mean(dim=1, keepdim=True)  # (B,1,H,W)
        r_mag = r_mag.view(B, 1, N, 1)

        # Unit rotor (Rotation direction) - per-pixel mean 후 정규화
        r_cos = unit_cos.mean(dim=1, keepdim=True)  # (B,1,H,W)
        r_sin = unit_sin.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # (cos,sin) 정규화 (NLERP 형태)
        r_norm = torch.sqrt(r_cos**2 + r_sin**2 + 1e-6)
        r_cos = (r_cos / r_norm).view(B, 1, N, 1)
        r_sin = (r_sin / r_norm).view(B, 1, N, 1)

        # Gate Weight: scale injection 신뢰도
        gate_weight = self.gate_net(x).view(B, 1, N, 1)

        # Chunk size 결정
        if N <= SAFE_N_LIMIT:
            CHUNK_SIZE = N
        else:
            CHUNK_SIZE = max(1, SAFE_ELEMENTS // N)

        output_chunks = []

        # Key-side tensor view
        r_mag_k = r_mag.transpose(-2, -1)  # (B,1,1,N)
        r_cos_k = r_cos.transpose(-2, -1)  # (B,1,1,N)
        r_sin_k = r_sin.transpose(-2, -1)  # (B,1,1,N)

        r_mag_v = r_mag.expand(B, self.num_heads, N, 1)

        # NOTE: chunked attention은 내부 루프가 길어져 tqdm 출력이 학습 로그를 오염시킬 수 있습니다.
        # 기본값은 항상 숨김(quiet)이며, 필요할 때만 환경변수로 켤 수 있게 합니다.
        #   SHOW_ATTN_CHUNKS=1  -> tqdm 활성화
        show_attn_chunks = os.getenv("SHOW_ATTN_CHUNKS", "0") == "1"

        pbar = tqdm(
            range(0, N, CHUNK_SIZE),
            desc=f"  [Attn] Chunks (N={N})",
            leave=False,
            disable=(not show_attn_chunks)
        )

        # 이 모듈 내부에서 autocast를 강제로 켜면(=cuda available) dtype이 예기치 않게 fp16으로 고정될 수 있습니다.
        # outer training loop의 autocast 설정을 그대로 따르도록 변경합니다.
        autocast_enabled = torch.is_autocast_enabled()
        device_type = x.device.type

        with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled):
            for i in pbar:
                q_chunk = q[:, :, i:i + CHUNK_SIZE, :]  # (B,heads,M,hd)
                r_mag_q = r_mag[:, :, i:i + CHUNK_SIZE, :]  # (B,1,M,1)
                gate_c = gate_weight[:, :, i:i + CHUNK_SIZE, :]  # (B,1,M,1)

                # ------------------------------
                # Path B: Scale bias (log scale diff)
                # ------------------------------
                scale_diff = torch.abs(
                    torch.log(r_mag_q + 1e-6) - torch.log(r_mag_k + 1e-6)
                )  # (B,1,M,N)
                scale_bias = -scale_diff.to(q.dtype)

                # ------------------------------
                # Path A: Rotation bias (Unit rotor alignment)
                # rot_sim = cos(Δθ) = cos_q*cos_k + sin_q*sin_k
                # ------------------------------
                r_cos_q = r_cos[:, :, i:i + CHUNK_SIZE, :]  # (B,1,M,1)
                r_sin_q = r_sin[:, :, i:i + CHUNK_SIZE, :]
                rot_sim = (r_cos_q * r_cos_k) + (r_sin_q * r_sin_k)  # (B,1,M,N)
                rot_bias = (ROTATION_BIAS_SCALE * rot_sim).to(q.dtype)

                # 총 bias
                attn_mask = scale_bias + rot_bias  # (B,1,M,N)

                # Main attention
                out_chunk = F.scaled_dot_product_attention(
                    q_chunk, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0
                )

                # Scale injection (log-ratio -> clipped exponential gain)
                r_mag_att = F.scaled_dot_product_attention(
                    q_chunk, k, r_mag_v,
                    attn_mask=attn_mask,
                    dropout_p=0.0
                )
                log_ratio = torch.log(r_mag_att + 1e-6) - torch.log(r_mag_q + 1e-6)
                log_ratio = torch.clamp(log_ratio, min=-SCALE_INJECTION_LOG_CLIP, max=SCALE_INJECTION_LOG_CLIP)
                scale_gain = torch.exp(gate_c * log_ratio)
                out_chunk = out_chunk * scale_gain

                output_chunks.append(out_chunk)

                del scale_diff, scale_bias, rot_sim, rot_bias, attn_mask, r_mag_att, log_ratio, scale_gain, out_chunk

        out = torch.cat(output_chunks, dim=2)  # (B,heads,N,hd)
        out = out.transpose(1, 2).reshape(B, H, W, C)

        return self.proj(out)

class GeometricDescriptorGuidance(nn.Module):
    """Architecture.md §3.2.5 - (g_s,g_v,g_b) Gate 기반 모듈레이션"""
    def __init__(self, dim):
        super().__init__()
        self.descriptor_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Mish(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x, return_gates: bool = False):
        # x: (B,H,W,C)
        s, v, b = torch.chunk(x, 3, dim=-1)

        inv_s = torch.mean(torch.abs(s), dim=-1, keepdim=True)
        inv_v = torch.norm(v, dim=-1, keepdim=True)
        inv_b = torch.norm(b, dim=-1, keepdim=True)

        descriptor = torch.cat([inv_s, inv_v, inv_b], dim=-1)  # (B,H,W,3)
        gates = self.descriptor_net(descriptor)                # (B,H,W,3)

        g_s, g_v, g_b = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]

        s_mod = s * g_s
        v_mod = v * g_v
        b_mod = b * g_b

        x_mod = torch.cat([s_mod, v_mod, b_mod], dim=-1)

        if return_gates:
            return x_mod, (g_s, g_v, g_b)
        return x_mod

class GeometricGateHead(nn.Module):
    """Reusable gate head for Phase 3 outputs.

    The same invariant descriptor used in Geometric Descriptor Guidance is
    reapplied to a CHW feature volume and exposed as spatial gate maps
    (g_s, g_v, g_b) in [0, 1].
    """
    def __init__(self):
        super().__init__()
        self.descriptor_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Mish(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, feat_chw: torch.Tensor):
        x = feat_chw.permute(0, 2, 3, 1)  # (B,H,W,C)
        s, v, b = torch.chunk(x, 3, dim=-1)

        inv_s = torch.mean(torch.abs(s), dim=-1, keepdim=True)
        inv_v = torch.norm(v, dim=-1, keepdim=True)
        inv_b = torch.norm(b, dim=-1, keepdim=True)

        descriptor = torch.cat([inv_s, inv_v, inv_b], dim=-1)  # (B,H,W,3)
        gates = self.descriptor_net(descriptor)                # (B,H,W,3)

        g_s = gates[..., 0]
        g_v = gates[..., 1]
        g_b = gates[..., 2]
        return g_s, g_v, g_b

class GeometricEncoderBlock(nn.Module):
    """Architecture.md §3.2 - CPE -> Rotor-Scale Attention -> Guidance -> FFN"""
    def __init__(self, dim):
        super().__init__()
        self.cpe = GeometricCPE(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotorScaleAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.guidance = GeometricDescriptorGuidance(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, rotor_tuple):
        # x: (B,H,W,C)
        x_cpe = self.cpe(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = x_cpe + self.attn(self.norm1(x_cpe), rotor_tuple)
        x_guided = self.guidance(x)
        x = x + self.ffn(self.norm2(x_guided))
        return x

class DenseRotorHead(nn.Module):
    """Architecture.md §3.3.1 - Dense Rotor Regression Head"""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Mish(),
            nn.Linear(in_dim // 2, 4)
        )

    def forward(self, x):
        return self.net(x)

class GeometricCrossAttention(nn.Module):
    """Architecture.md §3.3.2 - Cross-Attention (A query, B key/value)"""
    def __init__(self, dim, num_heads=NUM_ATTENTION_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = IndependentLinear(dim)
        self.to_k = IndependentLinear(dim)
        self.to_v = IndependentLinear(dim)

        self.rotor_head = DenseRotorHead(dim)
        self.proj = IndependentLinear(dim)

    @staticmethod
    def normalize_rotor_field(rotor_map):
        """Normalize the (cos,sin) part of a dense rotor field per pixel."""
        cos_t = rotor_map[..., 0]
        sin_t = rotor_map[..., 1]
        norm = torch.sqrt(cos_t ** 2 + sin_t ** 2 + 1e-6)
        out = rotor_map.clone()
        out[..., 0] = cos_t / norm
        out[..., 1] = sin_t / norm
        return out

    def local_window_attention(self, q, k, v, H, W, window_size):
        """Local-window cross-attention used on the highest resolutions after pre-alignment."""
        B, heads, N, head_dim = q.shape
        radius = window_size // 2
        win2 = window_size * window_size

        k_map = k.transpose(2, 3).reshape(B * heads, head_dim, H, W)
        v_map = v.transpose(2, 3).reshape(B * heads, head_dim, H, W)

        k_unfold = F.unfold(k_map, kernel_size=window_size, padding=radius)
        v_unfold = F.unfold(v_map, kernel_size=window_size, padding=radius)

        k_unfold = k_unfold.view(B, heads, head_dim, win2, N).permute(0, 1, 4, 3, 2)
        v_unfold = v_unfold.view(B, heads, head_dim, win2, N).permute(0, 1, 4, 3, 2)

        logits = (q.unsqueeze(3) * k_unfold).sum(dim=-1) * self.scale
        attn = F.softmax(logits, dim=-1)
        out = (attn.unsqueeze(-1) * v_unfold).sum(dim=3)
        return out

    def forward(self, x_a, x_b, local_window=None):
        # x_a, x_b: (B,H,W,C)
        B, H, W, C = x_a.shape
        N = H * W

        q = self.to_q(x_a.view(B, N, C)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x_b.view(B, N, C)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x_b.view(B, N, C)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Decoder matching도 encoder와 동일하게 cosine-similarity 기반으로 안정화합니다.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if local_window is not None and local_window > 1:
            out = self.local_window_attention(q, k, v, H, W, local_window)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        context = out.transpose(1, 2).reshape(B, H, W, C)

        dense_rotor = self.rotor_head(context)  # (B,H,W,4)
        dense_rotor = self.normalize_rotor_field(dense_rotor)

        return self.proj(context), dense_rotor

class CliffordInterpolation(nn.Module):
    """Architecture.md §3.3.2 - Clifford Interpolation (S/V/B 분리 업샘플 + B 정규화)"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (B,C,H,W), C는 3의 배수
        s, v, b = torch.chunk(x, 3, dim=1)
        s_up = F.interpolate(s, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        v_up = F.interpolate(v, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        b_up = F.interpolate(b, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        b_up = F.normalize(b_up, dim=1)  # NLERP 형태 정규화
        return torch.cat([s_up, v_up, b_up], dim=1)

class RotorMapInterpolator(nn.Module):
    """
    Architecture.md §3.3.2 - Rotor(Bivector) 보간 후 정규화(NLERP)
    delta_rotor_map: (B,H,W,4) where (cos,sin,dx,dy)
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, rotor_map, target_hw=None):
        # rotor_map: (B,H,W,4)
        rotor_chw = rotor_map.permute(0, 3, 1, 2)  # (B,4,H,W)
        up = F.interpolate(rotor_chw, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        if target_hw is not None and up.shape[-2:] != target_hw:
            up = F.interpolate(up, size=target_hw, mode='bilinear', align_corners=True)

        # NLERP: cos/sin 정규화
        cos_t = up[:, 0:1, :, :]
        sin_t = up[:, 1:2, :, :]
        norm = torch.sqrt(cos_t**2 + sin_t**2 + 1e-6)
        cos_t = cos_t / norm
        sin_t = sin_t / norm

        out = torch.cat([cos_t, sin_t, up[:, 2:3, :, :], up[:, 3:4, :, :]], dim=1)
        return out.permute(0, 2, 3, 1)  # (B,H,W,4)

class GeometricSkipConnection(nn.Module):
    """
    Architecture.md §3.3.3 - Geometric Skip-Connection

    1) Rotor Map 기반 Warping
    2) 업샘플링된 문맥(dec_feat) + 정렬된 디테일(enc_feat) 융합
       - Gated Injection: 디테일을 얼마나 주입할지(신뢰도) 결정
    """
    def __init__(self, dim):
        super().__init__()
        self.compress = nn.Conv2d(dim * 2, dim, 1)

        # Predict how much aligned encoder detail to inject.
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Mish(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Refine the fused feature after gated injection.
        self.refine = GeometricResBlock(dim, dilation=2)

    def get_warp_grid(self, rotor_map, B, H, W, device):
        # rotor_map: (B,H,W,4) (cos,sin,dx,dy)
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        cos_t = rotor_map[..., 0].unsqueeze(-1)
        sin_t = rotor_map[..., 1].unsqueeze(-1)
        dx_t = rotor_map[..., 2].unsqueeze(-1)
        dy_t = rotor_map[..., 3].unsqueeze(-1)

        x_new = cos_t * base_grid[..., 0:1] - sin_t * base_grid[..., 1:2] + dx_t
        y_new = sin_t * base_grid[..., 0:1] + cos_t * base_grid[..., 1:2] + dy_t

        return torch.cat([x_new, y_new], dim=-1)

    def forward(self, dec_feat, enc_feat, rotor_map):
        """
        Args:
            dec_feat: (B,C,H,W) - decoder context
            enc_feat: (B,C,H,W) - encoder(Phase2 원본 or Transform-guided warped) feature
            rotor_map: (B,H,W,4) - residual ΔW (warped A -> B)
        """
        B, C, H, W = enc_feat.shape

        # 1) Warping (A -> B)
        grid = self.get_warp_grid(rotor_map, B, H, W, enc_feat.device)
        warped_enc = F.grid_sample(enc_feat, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

        # 2) Concat & Gated Injection
        concat = torch.cat([dec_feat, warped_enc], dim=1)  # (B,2C,H,W)
        gate = self.gate_net(concat)                       # (B,C,H,W)
        injected = torch.cat([dec_feat, warped_enc * gate], dim=1)

        fused = self.compress(injected)
        fused = self.refine(fused)
        return fused

class Phase3Transformer(nn.Module):
    """Phase 3 main wrapper.

    The module receives the Phase 2 pyramid, performs rotor-scale-aware encoding,
    propagates transforms from coarse to fine levels, and exports both backward
    sampling transforms (W_B2A / W_global) and forward evaluation transforms
    (W_AB).
    """
    def __init__(self, feature_dim=FEATURE_DIM, num_layers=NUM_ENCODER_LAYERS, embed_dim=HIDDEN_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # Input Alignment (Phase2 출력의 S/V/B를 Phase3 feature_dim/3로 정렬)
        self.adapt_s = nn.Conv2d(embed_dim, feature_dim // 3, 1)
        self.adapt_v = nn.Conv2d(embed_dim * 2, feature_dim // 3, 1)

        # Main Phase 3 B-stream keeps the legacy unit_sin path.
        # A zero-initialized residual branch lets explicit-GP cos / magnitude
        # cues enter gradually without changing the initial behavior.
        self.adapt_b = nn.Conv2d(embed_dim, feature_dim // 3, 1)
        self.adapt_b_extra = nn.Conv2d(embed_dim * 2, feature_dim // 3, 1)
        with torch.no_grad():
            self.adapt_b_extra.weight.zero_()
            if self.adapt_b_extra.bias is not None:
                self.adapt_b_extra.bias.zero_()

        # Downstream Phase 4 can optionally consume the full explicit-GP tuple
        # (unit_cos, unit_sin, rotor_mag) through a separate adapter. It is
        # initialized to behave like the legacy unit_sin-only path so that old
        # checkpoints remain close to their original feature distribution.
        self.phase4_adapt_b = nn.Conv2d(embed_dim * 3, feature_dim // 3, 1)
        with torch.no_grad():
            self.phase4_adapt_b.weight.zero_()
            self.phase4_adapt_b.weight[:, embed_dim:2 * embed_dim, :, :] = self.adapt_b.weight.detach().clone()
            if self.adapt_b.bias is not None:
                self.phase4_adapt_b.bias.copy_(self.adapt_b.bias.detach())
            else:
                self.phase4_adapt_b.bias.zero_()

        # Stage 1: Tokenization
        self.tokenizer = GeometricTokenizer(in_channels=feature_dim, hidden_dim=feature_dim)

        # Stage 2: Encoder
        self.encoder_layers = nn.ModuleList([GeometricEncoderBlock(feature_dim) for _ in range(num_layers)])

        # Stage 3: Decoder blocks
        self.cross_attn = GeometricCrossAttention(feature_dim)
        self.upsampler = CliffordInterpolation(scale_factor=2)
        self.rotor_upsampler = RotorMapInterpolator(scale_factor=2)
        self.skip_conn = GeometricSkipConnection(feature_dim)

        # Feature refinement + MPC head
        self.refine_net = GeometricResBlock(feature_dim, dilation=1)
        self.head_mpc = nn.Conv2d(feature_dim, 4, 1)  # (Energy, Vx, Vy, Rotation proxy)

        # Two gate exposures:
        #   - gate_head: less-mixed decoder context -> weighting / transform pooling
        #   - task_gate_head: refined feature -> task-side auxiliary exposure
        self.gate_head = GeometricGateHead()
        self.task_gate_head = GeometricGateHead()

    # ---------------------------------------------------------------------
    # Helper: Phase2 Output -> Phase3 Input Volume (S/V/B concat)
    # ---------------------------------------------------------------------
    def prepare_input(self, p2_out):
        """
        p2_out: (S, V, (unit_cos, unit_sin, mag))
          - S: (B,embed_dim,H,W)
          - V: (B,embed_dim,2,H,W)
          - B: tuple of (B,embed_dim,H,W)
        """
        s, v, b = p2_out
        unit_cos, unit_sin, rotor_mag = b

        # V: (B,embed_dim,2,H,W) -> (B,embed_dim*2,H,W)
        v_flat = v.view(v.shape[0], -1, v.shape[-2], v.shape[-1])

        s_feat = self.adapt_s(s)
        v_feat = self.adapt_v(v_flat)

        # Keep the legacy unit_sin path and inject cos / magnitude only through
        # the zero-initialized residual branch.
        rotor_mag_feat = torch.tanh(torch.log1p(torch.clamp(rotor_mag, min=0.0)))
        b_extra = self.adapt_b_extra(torch.cat([unit_cos, rotor_mag_feat], dim=1))
        b_feat = self.adapt_b(unit_sin) + b_extra

        return torch.cat([s_feat, v_feat, b_feat], dim=1)

    def prepare_phase4_input(self, p2_out):
        """Build the downstream Phase 4 input volume.

        Unlike `prepare_input()`, this adapter exposes the full explicit-GP rotor
        tuple. The magnitude branch is compressed by `tanh(log1p(.))` before
        projection.
        """
        s, v, b = p2_out
        unit_cos, unit_sin, rotor_mag = b

        v_flat = v.view(v.shape[0], -1, v.shape[-2], v.shape[-1])
        rotor_mag_feat = torch.tanh(torch.log1p(torch.clamp(rotor_mag, min=0.0)))
        b_stack = torch.cat([unit_cos, unit_sin, rotor_mag_feat], dim=1)

        s_feat = self.adapt_s(s)
        v_feat = self.adapt_v(v_flat)
        b_feat = self.phase4_adapt_b(b_stack)

        return torch.cat([s_feat, v_feat, b_feat], dim=1)

    def load_state_dict(self, state_dict, strict=True):
        """Backward-compatibility shim for older checkpoints."""
        state_dict = dict(state_dict)

        if ('phase4_adapt_b.weight' not in state_dict) and ('adapt_b.weight' in state_dict):
            phase4_w = self.phase4_adapt_b.weight.detach().clone()
            phase4_w.zero_()
            in_ch = self.embed_dim
            phase4_w[:, in_ch:2 * in_ch, :, :] = state_dict['adapt_b.weight']
            state_dict['phase4_adapt_b.weight'] = phase4_w
            if self.phase4_adapt_b.bias is not None:
                if 'adapt_b.bias' in state_dict:
                    state_dict['phase4_adapt_b.bias'] = state_dict['adapt_b.bias']
                else:
                    state_dict['phase4_adapt_b.bias'] = self.phase4_adapt_b.bias.detach().clone()
            if strict:
                strict = False

        if 'adapt_b_extra.weight' not in state_dict:
            state_dict['adapt_b_extra.weight'] = self.adapt_b_extra.weight.detach().clone()
            if self.adapt_b_extra.bias is not None:
                state_dict['adapt_b_extra.bias'] = self.adapt_b_extra.bias.detach().clone()
            if strict:
                strict = False

        if ('task_gate_head.descriptor_net.0.weight' not in state_dict) and ('gate_head.descriptor_net.0.weight' in state_dict):
            for suffix in [
                'descriptor_net.0.weight', 'descriptor_net.0.bias',
                'descriptor_net.2.weight', 'descriptor_net.2.bias'
            ]:
                src_key = f'gate_head.{suffix}'
                dst_key = f'task_gate_head.{suffix}'
                if src_key in state_dict:
                    state_dict[dst_key] = state_dict[src_key]
            if strict:
                strict = False

        return super().load_state_dict(state_dict, strict=strict)

    # ---------------------------------------------------------------------
    # Helper: rotor map 평균 -> 2x3 matrix (Global transform)
    # ---------------------------------------------------------------------
    @staticmethod
    def rotor_map_to_theta(rotor_map, weight_map=None):
        """Pool a dense residual rotor field into a single 2x3 affine theta.

        When `weight_map` is given, pooling becomes confidence-weighted instead
        of uniform averaging.
        """
        if weight_map is None:
            avg = rotor_map.mean(dim=(1, 2))
        else:
            if weight_map.dim() == 4:
                if weight_map.shape[1] == 1:
                    weight = weight_map[:, 0]
                else:
                    weight = weight_map.mean(dim=1)
            else:
                weight = weight_map

            weight = torch.clamp(weight, min=1e-4)
            weight = weight / (weight.sum(dim=(1, 2), keepdim=True) + 1e-6)
            avg = (rotor_map * weight.unsqueeze(-1)).sum(dim=(1, 2))

        cos_t, sin_t, dx_t, dy_t = avg[:, 0], avg[:, 1], avg[:, 2], avg[:, 3]

        # cos/sin 정규화 (NLERP)
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
    def invert_affine_2x3(theta, eps: float = 1e-8):
        """
        Invert 2x3 affine (B,2,3) in normalized coords.
        If theta maps out -> in, inv(theta) maps in -> out.
        """
        # theta = [[a,b,tx],
        #          [c,d,ty]]
        a = theta[:, 0, 0]
        b = theta[:, 0, 1]
        tx = theta[:, 0, 2]
        c = theta[:, 1, 0]
        d = theta[:, 1, 1]
        ty = theta[:, 1, 2]

        det = a * d - b * c
        # avoid divide-by-zero
        det = torch.where(det.abs() < eps, det.sign() * eps, det)

        inv_a =  d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d =  a / det

        inv_tx = -(inv_a * tx + inv_b * ty)
        inv_ty = -(inv_c * tx + inv_d * ty)

        row1 = torch.stack([inv_a, inv_b, inv_tx], dim=1)
        row2 = torch.stack([inv_c, inv_d, inv_ty], dim=1)
        return torch.stack([row1, row2], dim=1)

    @staticmethod
    def compose_theta(delta, prev):
        """
        W_current = ΔW ∘ W_prev
        delta, prev: (B,2,3)
        """
        B = delta.shape[0]
        device = delta.device
        dtype = delta.dtype
        bottom = torch.tensor([0, 0, 1], device=device, dtype=dtype).view(1, 1, 3).repeat(B, 1, 1)

        prev_aug = torch.cat([prev, bottom], dim=1)    # (B,3,3)
        delta_aug = torch.cat([delta, bottom], dim=1)  # (B,3,3)

        out = torch.bmm(prev_aug, delta_aug)
        return out[:, :2, :]  # (B,2,3)

    @staticmethod
    def warp_with_theta(feat, theta, align_corners=True):
        """
        feat: (B,C,H,W)
        theta: (B,2,3)
        """
        grid = F.affine_grid(theta, feat.size(), align_corners=align_corners)
        return F.grid_sample(feat, grid, align_corners=align_corners, mode='bilinear', padding_mode='zeros')

    @staticmethod
    def warp_rotor_tuple(rotor_tuple, theta):
        """
        rotor_tuple: (unit_cos, unit_sin, mag) each (B, C, H, W)
        """
        unit_cos, unit_sin, mag = rotor_tuple
        cos_w = Phase3Transformer.warp_with_theta(unit_cos, theta, align_corners=True)
        sin_w = Phase3Transformer.warp_with_theta(unit_sin, theta, align_corners=True)
        mag_w = Phase3Transformer.warp_with_theta(mag, theta, align_corners=True)

        # unit 재정규화 (NLERP)
        norm = torch.sqrt(cos_w**2 + sin_w**2 + 1e-6)
        cos_w = cos_w / norm
        sin_w = sin_w / norm
        return (cos_w, sin_w, mag_w)

    @staticmethod
    def get_high_res_local_window(level_idx):
        """Window size used by local high-resolution cross-attention."""
        if level_idx == 0:
            return HIGH_RES_LOCAL_WINDOW_FINE
        if level_idx == 1:
            return HIGH_RES_LOCAL_WINDOW_STRUCT
        return None

    @staticmethod
    def build_transform_confidence(mpc_map, gates):
        """Combine MPC energy and less-mixed decoder gates for transform pooling."""
        g_s, g_v, g_b = gates
        gate_mean = (g_s + g_v + g_b) / 3.0
        energy_conf = torch.sigmoid(mpc_map[:, 0, :, :])
        weight = 0.5 * (gate_mean + energy_conf)
        return torch.clamp(weight, min=1e-4)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, pyramid_a, pyramid_b):
        """
        Args:
            pyramid_a, pyramid_b:
              list of Phase2 outputs per level: [(S,V,Btuple), ...]
              level 0: finest, level N-1: coarsest
        Returns:
            results: list of dict (coarse->fine 순서로 append됨)
        """
        results = []

        W_prev = None            # 누적 Global Transform (B,2,3)
        dec_feat_prev = None     # decoder feature volume (coarse level)

        num_levels = len(pyramid_a)

        for level_idx in reversed(range(num_levels)):
            # -----------------------------
            # 0) Prepare features
            # -----------------------------
            raw_a = self.prepare_input(pyramid_a[level_idx])  # (B,C,H,W)
            raw_b = self.prepare_input(pyramid_b[level_idx])  # (B,C,H,W)

            rotor_a = pyramid_a[level_idx][2]
            rotor_b = pyramid_b[level_idx][2]

            B, C, H, W = raw_a.shape

            # -----------------------------
            # 1) Transform-Guided Warping (Architecture.md §3.1.1)
            # -----------------------------
            if W_prev is not None:
                raw_a_warped = self.warp_with_theta(raw_a, W_prev, align_corners=True)
                rotor_a_warped = self.warp_rotor_tuple(rotor_a, W_prev)
            else:
                raw_a_warped = raw_a
                rotor_a_warped = rotor_a

            # -----------------------------
            # 2) Tokenize (GroupConv)
            # -----------------------------
            tok_a = self.tokenizer(raw_a_warped).permute(0, 2, 3, 1)  # (B,H,W,C)
            tok_b = self.tokenizer(raw_b).permute(0, 2, 3, 1)

            # -----------------------------
            # 3) Encoder (self-attention with rotor tuple)
            #    - coarse / structural levels: full encoder
            #    - finest levels: skip encoder self-attention to save memory
            # -----------------------------
            if level_idx >= HIGH_RES_SKIP_LEVEL:
                for layer in self.encoder_layers:
                    tok_a = checkpoint(layer, tok_a, rotor_a_warped, use_reentrant=False)
                    tok_b = checkpoint(layer, tok_b, rotor_b, use_reentrant=False)

            # -----------------------------
            # 4) Cross-Attention -> Residual ΔW (dense rotor map)
            # -----------------------------
            local_window = None if level_idx >= HIGH_RES_SKIP_LEVEL else self.get_high_res_local_window(level_idx)
            ctx, delta_rotor = self.cross_attn(tok_a, tok_b, local_window=local_window)
            ctx_chw = ctx.permute(0, 3, 1, 2)  # (B,C,H,W)

            # -----------------------------
            # 5) Decoder context propagation (coarse->fine)
            # -----------------------------
            if dec_feat_prev is not None:
                dec_up = self.upsampler(dec_feat_prev)
                if dec_up.shape[-2:] != (H, W):
                    dec_up = F.interpolate(dec_up, size=(H, W), mode='bilinear', align_corners=True)
                dec_context = ctx_chw + dec_up
            else:
                dec_context = ctx_chw

            # Less-mixed gates from decoder context: used for confidence pooling.
            g_s, g_v, g_b = self.gate_head(dec_context)

            # -----------------------------
            # 6) Skip-Connection (Warp + Gated Injection + Refinement)
            #    - enc_feat는 "W_prev로 1차 워핑된 raw_a_warped"를 사용
            #    - rotor_map은 "잔차 ΔW" 사용
            # -----------------------------
            fused = self.skip_conn(dec_context, raw_a_warped, delta_rotor)

            # -----------------------------
            # 7) Feature Map 생성 (Refine) + MPC map head
            # -----------------------------
            refined_feature = self.refine_net(fused)     # (B,C,H,W)
            mpc_map = self.head_mpc(refined_feature)     # (B,4,H,W)

            # Task-side gates from the final refined feature.
            g_s_task, g_v_task, g_b_task = self.task_gate_head(refined_feature)

            # -----------------------------
            # 8) Global transform update: W_current = ΔW ∘ W_prev
            #    - delta_theta is estimated by confidence-weighted pooling
            # -----------------------------
            transform_weight = self.build_transform_confidence(mpc_map, (g_s, g_v, g_b))
            delta_theta = self.rotor_map_to_theta(delta_rotor, weight_map=transform_weight)  # (B,2,3)
            if W_prev is None:
                W_prev = delta_theta
            else:
                W_prev = self.compose_theta(delta_theta, W_prev)

            # 다음 레벨로 전달할 decoder feature
            dec_feat_prev = refined_feature
            # W_prev는 Phase3 설계상 theta_B2A (out=B -> in=A) 로 누적되는 값
            theta_B2A = W_prev

            # Loss/Metric용 A->B 로도 같이 제공 (GT w_gt는 A->B)
            W_AB = self.invert_affine_2x3(theta_B2A)

            # 기록
            results.append({
                'level': level_idx,
                'delta_rotor_map': delta_rotor,
                'g_s': g_s.unsqueeze(1),
                'g_v': g_v.unsqueeze(1),
                'g_b': g_b.unsqueeze(1),

                'pure_g_s': g_s.unsqueeze(1),
                'pure_g_v': g_v.unsqueeze(1),
                'pure_g_b': g_b.unsqueeze(1),

                'g_s_task': g_s_task.unsqueeze(1),
                'g_v_task': g_v_task.unsqueeze(1),
                'g_b_task': g_b_task.unsqueeze(1),
                'transform_weight': transform_weight.unsqueeze(1),

                # 명시적으로 방향 표기 (혼동 방지)
                'W_global': theta_B2A,     # (=B->A, out->in)
                'W_B2A': theta_B2A,
                'W_AB': W_AB,             # (=A->B, GT 방향)

                'refined_feature': refined_feature,
                'mpc_map': mpc_map,
            })

        # (중요) 외부 코드/학습 루프가 level=0(최고해상도)을 results[0]로 기대하는 경우가 많아
        # 반환 직전에 level 오름차순(0->coarse)으로 정렬합니다.
        results = sorted(results, key=lambda d: d.get('level', 0))
        return results

def visualize_phase3_results(results):
    """
    Phase 3 결과 시각화:
    - Rotor(ΔW)에서 dx/dy 흐름 색상 표시
    - MPC energy, vector field 표시
    """
    levels = len(results)
    plt.figure(figsize=(15, 4 * levels))
    plt.suptitle("Phase 3: Coarse-to-Fine Geometric Matching Analysis (ArchFull)", fontsize=18, fontweight='bold')

    for idx, res in enumerate(results):
        lvl = res['level']
        rotor = res['delta_rotor_map'].detach().cpu().numpy()[0]
        mpc = res['mpc_map'].detach().cpu().numpy()[0]

        h, w = rotor.shape[:2]

        dx, dy = rotor[..., 2], rotor[..., 3]
        rotor_img = np.zeros((h, w, 3))
        mag, ang = cv2.cartToPolar(dx, dy)
        rotor_img[..., 0] = ang / (2 * np.pi)
        rotor_img[..., 1] = 1.0
        rotor_img[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        import matplotlib.colors as mcolors
        rotor_rgb = mcolors.hsv_to_rgb(rotor_img)

        energy = mpc[0]
        vx, vy = mpc[1], mpc[2]
        vec_img = np.zeros((h, w, 3))
        mag_v, ang_v = cv2.cartToPolar(vx, vy)
        vec_img[..., 0] = ang_v / (2 * np.pi)
        vec_img[..., 1] = 1.0
        vec_img[..., 2] = cv2.normalize(mag_v, None, 0, 1, cv2.NORM_MINMAX)
        vec_rgb = mcolors.hsv_to_rgb(vec_img)

        base = idx * 3

        plt.subplot(levels, 3, base + 1)
        plt.imshow(rotor_rgb)
        plt.title(f"Level {lvl}: Residual Rotor (ΔW)")
        plt.axis('off')

        plt.subplot(levels, 3, base + 2)
        plt.imshow(energy, cmap='inferno')
        plt.title(f"Level {lvl}: MPC Energy")
        plt.axis('off')

        plt.subplot(levels, 3, base + 3)
        plt.imshow(vec_rgb)
        plt.title(f"Level {lvl}: MPC Vector Field")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3 ArchFull Running on: {device}")

    IMG_PATH = "./img/val2017/000000569972.jpg"
    img = cv2.imread(IMG_PATH)

    if img is None:
        print("Image Not Found.")
        sys.exit(0)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Phase 1
    preprocessor = MathGeometricPreprocessor()
    pyramid_raw = preprocessor.process_pyramid(img_rgb, levels=5)

    # Phase 2
    embedder = CliffordPyramidEmbedder(hidden_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        pyramid_a = embedder(pyramid_raw, device)
        pyramid_b = embedder(pyramid_raw, device)

    # Phase 3
    model = Phase3Transformer(feature_dim=FEATURE_DIM, num_layers=NUM_ENCODER_LAYERS, embed_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        results = model(pyramid_a, pyramid_b)

    print(f"Phase 3 Complete. Levels processed: {len(results)}")
    visualize_phase3_results(results)
