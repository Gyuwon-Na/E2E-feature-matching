import os
# [System] OOM 방지를 위한 메모리 단편화 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from phase1 import MathGeometricPreprocessor
from phase2 import CliffordPyramidEmbedder

class Mish(nn.Module):
    """
    [Activation] Softplus보다 Gradient 흐름이 좋고 기하학적 정보 보존에 유리한 Mish
    f(x) = x * tanh(softplus(x))
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    """
    [Lightweight Attention] Channel Attention (Squeeze-and-Excitation)
    중요한 물리적 채널(S, V, B 중 특정 성분)을 강조
    """
    def __init__(self, channel, reduction=16):
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
    """
    [Refinement] Residual + Dilated Conv + SE Block
    넓은 수용장(Context)과 채널 중요도(Attention)를 동시에 잡음
    """
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim//3) # Group Conv 유지
        self.norm1 = nn.GroupNorm(dim//16, dim) # GroupNorm이 회전에 강함
        self.act = Mish()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1) # 일반 Conv로 정보 융합
        self.norm2 = nn.GroupNorm(dim//16, dim)
        self.se = SEBlock(dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out) # Attention
        return out + residual
    
# =============================================================================
# [Stage 1] Tokenization & Alignment
# =============================================================================

class GeometricTokenizer(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        """
        [Stage 1. Tokenization]
        입력된 물리량(S, V, B)을 그룹별로 나누어 Group=3인 Conv 적용.
        S, V, B 그룹 간의 독립성을 유지하며 초기 특징을 추출합니다.
        """
        super().__init__()
        # [Hyperparameter] groups=3: S, V, B 각각 독립적 연산
        self.group_conv = nn.Conv2d(
            in_channels, 
            hidden_dim, 
            kernel_size=3, 
            padding=1, 
            groups=3 
        )
        self.norm = nn.GroupNorm(3, hidden_dim)

    def forward(self, x):
        return self.norm(self.group_conv(x))

# =============================================================================
# [Stage 2] Encoder Components
# =============================================================================

class GeometricCPE(nn.Module):
    def __init__(self, dim):
        """
        [Stage 2.2 위치 정보 주입]
        Group Convolution 기반 CPE를 사용하여 픽셀 단위가 아닌 
        '기하학적 덩어리' 단위로 위치를 파악합니다.
        """
        super().__init__()
        self.pos_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.pos_conv(x)

class IndependentLinear(nn.Module):
    def __init__(self, dim):
        """
        [Stage 2.1 Q, K, V 변환 - Independent Linear]
        각 성분(S, V, B)이 가진 고유한 기하학적 정체성을 유지하기 위해
        채널을 3분할하여 각각 독립된 Linear Layer를 통과시킵니다.
        """
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // 3
        
        # S, V, B 각각을 위한 독립 투영 레이어
        self.lin_s = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.lin_v = nn.Linear(self.chunk_dim, self.chunk_dim)
        self.lin_b = nn.Linear(self.chunk_dim, self.chunk_dim)

    def forward(self, x):
        # x: (B, H, W, C)
        # 채널을 3등분 (S, V, B)
        chunks = torch.chunk(x, 3, dim=-1)
        s, v, b = chunks[0], chunks[1], chunks[2]
        
        s_out = self.lin_s(s)
        v_out = self.lin_v(v)
        b_out = self.lin_b(b)
        
        return torch.cat([s_out, v_out, b_out], dim=-1)

class RotorScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        """
        [Stage 2.3 Rotor-Scale Attention & 2.4 Injection Fusion]
        어텐션을 회전(Path A)과 스케일(Path B) 스트림으로 이원화하여 처리합니다.
        Hidden State뿐만 아니라 Phase 2의 Rotor 정보(Side Input)를 직접 활용합니다.
        하이퍼파라미터
            - num_heads: 헤드 수가 많을수록 다양한 관점(스케일, 회전, 텍스처 등)을 동시에 봄.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
                
        # [Stage 2.1] Q, K, V 변환 (Independent Linear 사용)
        self.to_q = IndependentLinear(dim)
        self.to_k = IndependentLinear(dim)
        self.to_v = IndependentLinear(dim)
        
        self.proj = nn.Linear(dim, dim)

        # 입력(x)을 보고 스케일 정보를 얼마나 반영할지(0~1) 결정
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Mish(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, rotor_tuple):
        """
        Args:
            x: Hidden State (B, H, W, C)
            rotor_tuple: (Unit_Cos, Unit_Sin, Magnitude) from Phase 2
        """
        B, H, W, C = x.shape
        N = H * W
        
        # 1. Hidden State로부터 Q, K, V 생성
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # --- [OOM 방지 및 기하학적 정규화] ---
        # 1. Path A (회전) 점수를 위해 Q, K 정규화
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # --- [Phase 2 정보 추출 (Side Input)] ---
        unit_cos, unit_sin, rotor_mag = rotor_tuple
        r_mag_mean = rotor_mag.mean(dim=1, keepdim=True) 
        r_mag = r_mag_mean.view(B, 1, N, 1)

        # Gated Weight 계산 (B, N, 1) -> (B, 1, N, 1)
        # 각 픽셀마다 "나는 스케일 정보가 필요해/필요없어"를 판단
        gate_weight = self.gate_net(x).view(B, 1, N, 1)

        # -------------------------------------------------------------------------
        # [Adaptive Chunking for Robust Memory Management]
        # 해상도 N이 클수록 Chunk Size를 줄여서, 한 번에 계산하는 행렬 크기를 제한합니다.
        # 목표: Chunk * N 행렬이 약 50MB~100MB를 넘지 않도록 조절
        # -------------------------------------------------------------------------
        
        SAFE_N_LIMIT = 4096 
        
        if N <= SAFE_N_LIMIT:
            # [Fast Path] No Chunking
            CHUNK_SIZE = N
        else:
            # [Safe Path] Adaptive Chunking
            # Attention Matrix size (Chunk * N) 제한
            SAFE_ELEMENTS = 2**20 # 약 100만개 (4MB) 정도로 여유 있게 설정
            CHUNK_SIZE = max(1, SAFE_ELEMENTS // N)
        
        output_chunks = []
        r_mag_v = r_mag.expand(B, self.num_heads, N, 1)
        r_mag_k = r_mag.transpose(-2, -1)
        
        # 루프 진행 상황 시각화 (N이 클 때만 활성화하여 오버헤드 방지)
        pbar = tqdm(range(0, N, CHUNK_SIZE), desc=f"  [Attn] Chunks (N={N})", leave=False, disable=True)

        with torch.amp.autocast('cuda', enabled=True):
            for i in pbar:
                q_chunk = q[:, :, i:i+CHUNK_SIZE, :]
                r_mag_q_chunk = r_mag[:, :, i:i+CHUNK_SIZE, :]
                gate_c = gate_weight[:, :, i:i+CHUNK_SIZE, :]

                scale_diff_chunk = torch.abs(
                    torch.log(r_mag_q_chunk + 1e-6) - 
                    torch.log(r_mag_k + 1e-6)
                )
                attn_mask_chunk = -scale_diff_chunk.to(q.dtype)

                # 1. 메인 어텐션
                out_chunk = F.scaled_dot_product_attention(
                    q_chunk, k, v, 
                    attn_mask=attn_mask_chunk, 
                    dropout_p=0.0
                )

                # 2. Injection용 어텐션
                r_mag_attended_chunk = F.scaled_dot_product_attention(
                    q_chunk, k, r_mag_v,
                    attn_mask=attn_mask_chunk,
                    dropout_p=0.0
                )

                
                injection_factor_chunk = r_mag_attended_chunk / (r_mag_q_chunk + 1e-6)
                # 고정값 0.1 대신 학습된 Gate 사용 (Dynamic Injection)
                # gate_c가 0이면 스케일 무시, 1이면 적극 반영

                out_chunk = out_chunk * (1.0 + gate_c * injection_factor_chunk)
                output_chunks.append(out_chunk)
                
                # [메모리 즉시 해제] Loop 안에서 생성된 중간 텐서 연결 끊기
                del scale_diff_chunk, attn_mask_chunk, r_mag_attended_chunk
            
        out = torch.cat(output_chunks, dim=2)
        out = out.transpose(1, 2).reshape(B, H, W, C)
        return self.proj(out)

class GeometricDescriptorGuidance(nn.Module):
    def __init__(self, dim):
        """
        [Stage 2.5 Geometric Descriptor Guidance]
        각 성분에서 불변량을 뽑아 Gate를 생성하고, S, V, B의 중요도를 동적으로 조절합니다.
        """
        super().__init__()
        self.chunk_dim = dim // 3
        
        # [Stage 2.5.1 Fast Lane & 2.5.2 Descriptor 생성]
        self.descriptor_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Mish(),
            nn.Linear(16, 3), # Output: Gate values for S, V, B
            nn.Sigmoid()      # [Stage 2.5.3 Gate Modulation] 0~1 사이 값
        )

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        
        # 채널 3분할
        s, v, b = torch.chunk(x, 3, dim=-1)
        
        # 1. 불변량 추출
        inv_s = torch.mean(torch.abs(s), dim=-1, keepdim=True)
        inv_v = torch.norm(v, dim=-1, keepdim=True)
        inv_b = torch.norm(b, dim=-1, keepdim=True)
        
        descriptor = torch.cat([inv_s, inv_v, inv_b], dim=-1)
        
        # 2. Gate 생성
        gates = self.descriptor_net(descriptor)
        g_s, g_v, g_b = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        
        # 3. Gate Modulation
        s_mod = s * g_s
        v_mod = v * g_v
        b_mod = b * g_b
        
        return torch.cat([s_mod, v_mod, b_mod], dim=-1)

class GeometricEncoderBlock(nn.Module):
    def __init__(self, dim):
        """
        [Stage 2. 인코더 블록]
        CPE -> Rotor-Scale Attention -> Guidance -> FFN
        """
        super().__init__()
        self.cpe = GeometricCPE(dim) # Position Injection
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
        # 1. 위치 정보 주입
        x_cpe = self.cpe(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # 2. Attention & Injection
        x = x_cpe + self.attn(self.norm1(x_cpe), rotor_tuple)
        
        # 3. Guidance & FFN
        x_guided = self.guidance(x)
        x = x + self.ffn(self.norm2(x_guided))
        
        return x

# =============================================================================
# [Stage 3] Decoder Components
# =============================================================================

class DenseRotorHead(nn.Module):
    def __init__(self, in_dim):
        """
        [Stage 3.1 Cross-Attention Output Head]
        Dense Rotor Regression Head: (Cos, Sin, dx, dy)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Mish(),
            nn.Linear(in_dim // 2, 4) 
        )

    def forward(self, x):
        return self.net(x)

class GeometricCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        """
        [Stage 3.1 Cross-Attention]
        이미지 A(Query)와 B(Key/Value)를 대조하여 Rotor를 추출합니다.
        IndependentLinear를 사용하여 S, V, B 채널 독립성을 유지합니다.
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # [Logic Fix] IndependentLinear 사용
        self.to_q = IndependentLinear(dim)
        self.to_k = IndependentLinear(dim)
        self.to_v = IndependentLinear(dim)
        
        self.rotor_head = DenseRotorHead(dim)
        
        # [Logic Fix] 출력단 독립성 유지
        self.proj = IndependentLinear(dim)

    def forward(self, x_a, x_b):
        B, H, W, C = x_a.shape
        N = H * W
        
        q = self.to_q(x_a.view(B, N, C)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.to_k(x_b.view(B, N, C)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.to_v(x_b.view(B, N, C)).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # [CUDA 최적화] Flash Attention 적용
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Context Vector
        context = out.transpose(1, 2).reshape(B, H, W, C)
        
        # [Dense Rotor Regression]
        dense_rotor = self.rotor_head(context)
        
        return self.proj(context), dense_rotor

class CliffordInterpolation(nn.Module):
    def __init__(self, scale_factor=2):
        """
        [Stage 3.2 Clifford Interpolation]
        - Scalar: 부드러운 선형 보간
        - Vector: 방향성을 유지하며 보간
        - Rotor(Bivector): 보간 후 정규화(NLERP)로 회전 성질 복원
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (B, C, H, W) -> C는 3의 배수 (S, V, B 그룹)
        
        # 1. 성분 분리 (Split)
        s, v, b = torch.chunk(x, 3, dim=1)
        
        # 2. 업샘플링 (Upsampling)
        s_up = F.interpolate(s, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        v_up = F.interpolate(v, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        b_up = F.interpolate(b, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        
        # 3. 기하학적 보정 (Geometric Correction)
        # [Rotor Correction] NLERP: 정규화하여 회전 정보(Unit) 복원
        b_up = F.normalize(b_up, dim=1)
        
        # 4. 재결합 (Concat)
        return torch.cat([s_up, v_up, b_up], dim=1)

class GeometricSkipConnection(nn.Module):
    def __init__(self, dim):
        """
        [Stage 3.3 Geometric Skip-Connection]
        1. 정렬: Rotor Map을 이용해 인코더 피쳐를 Warping
        2. 융합: Concat
        """
        super().__init__()

        # 단순 Conv 대신 Residual + Dilated + SE 적용
        self.fusion = GeometricResBlock(dim, dilation=2) 
        self.compress = nn.Conv2d(dim * 2, dim, 1) # 채널 축소용

    def get_warp_grid(self, rotor_map, B, H, W, device):
        """
        Rotor Map(Cos, Sin, dx, dy)으로부터 Affine Grid 생성
        """
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
        
        grid = torch.cat([x_new, y_new], dim=-1)
        return grid

    def forward(self, dec_feat, enc_feat, rotor_map):
        B, C, H, W = enc_feat.shape
        
        # 1. 정렬 (Warping)
        grid = self.get_warp_grid(rotor_map, B, H, W, enc_feat.device)
        warped_enc = F.grid_sample(enc_feat, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        
        # 2. 융합 (Concat -> Compress -> Powerful Refinement (Res+Dilated+SE))
        concat = torch.cat([dec_feat, warped_enc], dim=1)
        fused = self.compress(concat)
        
        return fused

# =============================================================================
# [Main Wrapper] Phase 3 Transformer
# =============================================================================

class Phase3Transformer(nn.Module):
    def __init__(self, feature_dim=384, num_layers=2):
        """
        [Phase 3 Main Module]
        Phase 2의 피라미드를 입력받아 인코딩 및 디코딩 수행.
        하이퍼파라미터
            - feature_dim: 384 등으로 늘리면 표현력이 좋아짐 (단, 3의 배수 유지 필수)
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Phase 2 Output Adapters
        self.adapt_s = nn.Conv2d(64, feature_dim // 3, 1)
        self.adapt_v_proj = nn.Conv2d(128, feature_dim // 3, 1)
        self.adapt_b = nn.Conv2d(64, feature_dim // 3, 1)
        
        self.tokenizer = GeometricTokenizer(in_channels=feature_dim, hidden_dim=feature_dim)
        
        self.encoder_layers = nn.ModuleList([
            GeometricEncoderBlock(feature_dim) for _ in range(num_layers)
        ])
        
        self.cross_attn = GeometricCrossAttention(feature_dim)
        self.upsampler = CliffordInterpolation()
        self.skip_conn = GeometricSkipConnection(feature_dim)
        
        self.final_net = nn.Sequential(
            GeometricResBlock(feature_dim, dilation=1),
            nn.Conv2d(feature_dim, 4, 1)
        )

    def prepare_input(self, p2_out):
        """
        Phase 2 Output(Tuple)을 Phase 3 Input Tensor로 변환
        """
        s, v, b = p2_out
        v_flat = v.view(v.shape[0], -1, v.shape[-2], v.shape[-1])
        b_mag = b[2] 
        
        s_feat = self.adapt_s(s)
        v_feat = self.adapt_v_proj(v_flat)
        b_feat = self.adapt_b(b_mag)
        
        return torch.cat([s_feat, v_feat, b_feat], dim=1)

    def forward(self, pyramid_a, pyramid_b):
        """
        [Stage 3.4 Feature Map Generation] 
        Coarse(Transformer) to Fine(Skip-Refinement) 루프를 수행합니다.
        속도 향상을 위해 고해상도(Level 0, 1)에서는 무거운 Attention 연산을 생략하고 
        상위 레벨의 문맥을 보간하여 사용.
        """
        results = []
        dec_feat, last_rotor = None, None
        
        # [Visual Fix] tqdm으로 전체 레벨 진행상황 표시
        level_iter = tqdm(reversed(range(len(pyramid_a))), total=len(pyramid_a), desc="Phase 3: Pyramid Levels", disable=True)
        
        for i in level_iter:
            # 현재 어떤 연산 방식이 사용되는지 tqdm 설명에 표시
            mode_str = "(Transformer)" if i >= 2 else "(CNN Refine)"
            level_iter.set_description(f"Phase 3: Level {i} {mode_str}")
            raw_a = self.prepare_input(pyramid_a[i])
            
            # --- [Stage 2.3 속도 최적화: 연산 분기] ---
            if i >= 2:
                # 저해상도: 전역 문맥 파악을 위해 Transformer Encoder/Cross-Attention 실행
                raw_b = self.prepare_input(pyramid_b[i])
                tok_a = self.tokenizer(raw_a).permute(0, 2, 3, 1)
                tok_b = self.tokenizer(raw_b).permute(0, 2, 3, 1)
                
                for layer in self.encoder_layers:
                    # Gradient Checkpointing으로 메모리 절약 유지
                    tok_a = checkpoint(layer, tok_a, pyramid_a[i][2], use_reentrant=False)
                    tok_b = checkpoint(layer, tok_b, pyramid_b[i][2], use_reentrant=False)
                
                # Cross-Attention을 통해 새로운 매칭 정보(Rotor) 추출
                ctx, last_rotor = self.cross_attn(tok_a, tok_b)
                dec_feat_chw = ctx.permute(0, 3, 1, 2)
            else:
                # 고해상도(Level 0, 1): 연산량 폭증을 막기 위해 Attention 생략
                # 대신 상위 레벨의 디코더 특징(dec_feat)을 Clifford Interpolation으로 가져옴 [Stage 3.2]
                dec_feat_up = self.upsampler(dec_feat)
                
                # 타겟 해상도와 미세한 차이가 있을 경우 Bilinear로 최종 조정
                if dec_feat_up.shape[-2:] != raw_a.shape[-2:]:
                    dec_feat_up = F.interpolate(dec_feat_up, size=raw_a.shape[-2:], mode='bilinear', align_corners=True)
                dec_feat_chw = dec_feat_up
                
                # 상위 레벨의 Rotor Map도 현재 해상도에 맞게 보간 (Warping용)
                last_rotor = F.interpolate(
                    last_rotor.permute(0, 3, 1, 2), 
                    size=raw_a.shape[-2:], 
                    mode='bilinear', 
                    align_corners=True
                ).permute(0, 2, 3, 1)

            # --- [Stage 3.3 Geometric Skip-Connection] ---
            # 보간된 문맥(또는 Transformer 결과)과 현재 레벨의 원본(raw_a)을 Warping 후 융합
            fused = self.skip_conn(dec_feat_chw, raw_a, last_rotor)
            
            # --- [Stage 3.4 최종 맵 생성] ---
            # CNN 레이어를 통해 노이즈 제거 및 MPC용 Feature Map 출력
            mpc_map = self.final_net(fused)
            
            # 다음(하위) 레벨을 위한 상태 업데이트
            dec_feat = fused
            
            results.append({
                'level': i,
                'rotor_map': last_rotor,
                'mpc_map': mpc_map
            })
            
        return results

# =============================================================================
# 실행 및 검증 코드 (Visualization)
# =============================================================================

def visualize_phase3_results(results):
    levels = len(results)
    plt.figure(figsize=(15, 4 * levels))
    plt.suptitle("Phase 3: Coarse-to-Fine Geometric Matching Analysis", fontsize=18, fontweight='bold')
    
    for idx, res in enumerate(results):
        lvl = res['level']
        rotor = res['rotor_map'].detach().cpu().numpy()[0]
        mpc = res['mpc_map'].detach().cpu().numpy()[0]
        
        h, w = rotor.shape[:2]
        
        dx, dy = rotor[..., 2], rotor[..., 3]
        rotor_img = np.zeros((h, w, 3))
        mag, ang = cv2.cartToPolar(dx, dy)
        rotor_img[..., 0] = ang / (2*np.pi)
        rotor_img[..., 1] = 1.0
        rotor_img[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
        
        import matplotlib.colors as mcolors
        rotor_rgb = mcolors.hsv_to_rgb(rotor_img)
        
        energy = mpc[0]
        print(f"Level {lvl} Energy Range: Min={energy.min():.4f}, Max={energy.max():.4f}, Mean={energy.mean():.4f}")
        
        vx, vy = mpc[1], mpc[2]
        vec_img = np.zeros((h, w, 3))
        mag_v, ang_v = cv2.cartToPolar(vx, vy)
        vec_img[..., 0] = ang_v / (2*np.pi)
        vec_img[..., 1] = 1.0
        vec_img[..., 2] = cv2.normalize(mag_v, None, 0, 1, cv2.NORM_MINMAX)
        vec_rgb = mcolors.hsv_to_rgb(vec_img)
        
        base = idx * 3
        
        plt.subplot(levels, 3, base + 1)
        plt.imshow(rotor_rgb)
        plt.title(f"Level {lvl}: Predicted Rotor (Flow)")
        plt.axis('off')
        
        plt.subplot(levels, 3, base + 2)
        plt.imshow(energy, cmap='inferno')
        plt.title(f"Level {lvl}: MPC Energy Potential")
        plt.axis('off')
        
        plt.subplot(levels, 3, base + 3)
        plt.imshow(vec_rgb)
        plt.title(f"Level {lvl}: MPC Vector Field")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3 Running on: {device}")
    
    IMG_PATH = "./img/val2017/000000569972.jpg"
    img = cv2.imread(IMG_PATH)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        preprocessor = MathGeometricPreprocessor()
        pyramid_raw = preprocessor.process_pyramid(img_rgb, levels=6)
        print(f"Phase 1 Complete. Levels: {len(pyramid_raw)}")
        
        embedder = CliffordPyramidEmbedder(hidden_dim=64).to(device)
        with torch.no_grad():
            pyramid_a = embedder(pyramid_raw, device)
            pyramid_b = embedder(pyramid_raw, device)
        print(f"Phase 2 Complete.")
        
        model = Phase3Transformer(feature_dim=192).to(device)
        
        with torch.no_grad():
            results = model(pyramid_a, pyramid_b)
            
        print(f"Phase 3 Complete. Processed Levels: {len(results)}")
        
        visualize_phase3_results(results)
    else:
        print("Image Not Found.")