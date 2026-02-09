"""
Phase 4 Fixed for Phase 2 output format
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class GeometricMPCRefiner(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.learning_rate = 0.002
        self.iterations = 200
        self.epsilon = 1e-6
        self.W = nn.Parameter(torch.eye(2, 3).unsqueeze(0).to(device))
        self.best_W = None
        self.best_loss = float('inf')

    def get_affine_grid(self, shape):
        B, C, H, W = shape
        grid = F.affine_grid(self.W, [B, C, H, W], align_corners=False)
        return grid

    def validate_initialization(self, mean_rotor):
        angle_deg = np.degrees(mean_rotor)
        if abs(angle_deg) > 90:
            print(f"⚠️  Warning: Large init angle ({angle_deg:.1f}°)")
            sign = 1 if angle_deg > 0 else -1
            mean_rotor = np.radians(sign * min(abs(angle_deg), 45))
        return mean_rotor

    def global_filtering_init(self, mean_rotor, mean_scale):
        mean_rotor = self.validate_initialization(mean_rotor)
        cos_theta = np.cos(mean_rotor)
        sin_theta = np.sin(mean_rotor)
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], dtype=torch.float32).to(self.device)
        transform_matrix = rotation_matrix * mean_scale
        with torch.no_grad():
            self.W[0, :2, :2] = transform_matrix
            self.W[0, :2, 2] = 0.0
            self.best_W = self.W.detach().clone()
            self.best_loss = float('inf')
        print(f"[Phase 4] Init: Angle={np.degrees(mean_rotor):.2f}°, Scale={mean_scale:.3f}")

    def compute_energy(self, src_dict, tgt_dict, gates):
        grid = self.get_affine_grid(src_dict['sdf'].shape)

        # Warp SDF (먼저!)
        warped_sdf = F.grid_sample(
            src_dict['sdf'],
            grid,
            align_corners=False,
            padding_mode='border'
        )

        # ✅ 임시 binary mask (gradient-safe)
        binary_mask = torch.ones_like(warped_sdf)


        
        # Warp SDF
        warped_sdf = F.grid_sample(src_dict['sdf'], grid, align_corners=False, padding_mode='border')
        
        # Warp Vector - handle [B, C, 2, H, W] format
        src_vec = src_dict['vector']
        tgt_vec = tgt_dict['vector']
        
        if src_vec.dim() == 5:  # [B, C, 2, H, W]
            B, C, _, H, W = src_vec.shape
            # Flatten to [B, C*2, H, W] for grid_sample
            src_vec_flat = src_vec.permute(0, 1, 3, 4, 2).contiguous().view(B, C*2, H, W)
            tgt_vec_flat = tgt_vec.permute(0, 1, 3, 4, 2).contiguous().view(B, C*2, H, W)
        else:
            src_vec_flat = src_vec
            tgt_vec_flat = tgt_vec
        
        warped_vector = F.grid_sample(src_vec_flat, grid, align_corners=False, padding_mode='zeros')
        
        # Warp rotor
        warped_rotor = F.grid_sample(src_dict['rotor'], grid, align_corners=False, padding_mode='zeros')
        
        # Compute energies on full channel data
        # Scalar energy: per-channel difference, then mean
        e_scalar = torch.abs(warped_sdf - tgt_dict['sdf'])  # [B, C, H, W]
        
        # Vector energy: compute on flattened vectors
        # For multi-channel vectors, we need to handle rotation per vector pair
        if warped_vector.shape[1] > 2:
            # Treat each pair as independent and compute similarity
            B, C2, H, W = warped_vector.shape
            n_pairs = C2 // 2
            
            # Reshape to [B, n_pairs, 2, H, W]
            warped_vec_pairs = warped_vector.view(B, n_pairs, 2, H, W)
            tgt_vec_pairs = tgt_vec_flat.view(B, n_pairs, 2, H, W)
            
            # Apply rotation to each warped pair
            rot_matrix = self.W[0, :2, :2]
            vec_flat = warped_vec_pairs.permute(0, 1, 3, 4, 2).reshape(B * n_pairs * H * W, 2)
            vec_rotated = torch.matmul(vec_flat, rot_matrix.t()).view(B, n_pairs, H, W, 2)
            vec_rotated = vec_rotated.permute(0, 1, 4, 2, 3)  # [B, n_pairs, 2, H, W]
            
            # Compute cosine similarity per pair
            tgt_flat = tgt_vec_pairs.view(B, n_pairs, 2, -1)  # [B, n_pairs, 2, H*W]
            rot_flat = vec_rotated.view(B, n_pairs, 2, -1)    # [B, n_pairs, 2, H*W]
            
            # Cosine similarity: dot product / (norm1 * norm2)
            dot_product = (rot_flat * tgt_flat).sum(dim=2)  # [B, n_pairs, H*W]
            norm1 = torch.norm(rot_flat, dim=2) + self.epsilon
            norm2 = torch.norm(tgt_flat, dim=2) + self.epsilon
            cosine_sim = dot_product / (norm1 * norm2)
            
            e_vector = 1.0 - cosine_sim  # [B, n_pairs, H*W]
            e_vector = e_vector.view(B, n_pairs, H, W)  # [B, n_pairs, H, W]
        else:
            # Simple 2-channel case
            rot_matrix = self.W[0, :2, :2]
            vec_permuted = warped_vector.permute(0, 2, 3, 1)
            vec_rotated = torch.einsum('ij,bhwj->bhwi', rot_matrix, vec_permuted)
            vec_rotated = vec_rotated.permute(0, 3, 1, 2)
            
            e_vector = 1.0 - F.cosine_similarity(vec_rotated, tgt_vec_flat, dim=1, eps=self.epsilon)
            e_vector = e_vector.unsqueeze(1)  # [B, 1, H, W]
        
        # Bivector energy
        e_bivector = torch.abs(warped_rotor - tgt_dict['rotor'])  # [B, C, H, W]
        
        # Average across channels to get [B, 1, H, W]
        e_scalar = e_scalar.mean(dim=1, keepdim=True)
        e_vector = e_vector.mean(dim=1, keepdim=True)
        e_bivector = e_bivector.mean(dim=1, keepdim=True)
        
        # Weight and sum
        g_s, g_v, g_b = gates
        total_energy = (g_s * e_scalar) + (g_v * e_vector) + (g_b * e_bivector)
        masked_energy = total_energy * binary_mask
        return masked_energy.sum() / (binary_mask.sum() + self.epsilon)

    def optimize(self, src_dict, tgt_dict, gates, priority_map=None):
        optimizer = optim.Adam([self.W], lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=self.learning_rate * 0.1)
        
        loss_history = []
        patience = 50
        no_improve_count = 0
        
        print(f"\n[Phase 4] MPC Optimization ({self.iterations} iterations)")
        
        for i in range(self.iterations):
            optimizer.zero_grad()
            loss = self.compute_energy(src_dict, tgt_dict, gates)
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_W = self.W.detach().clone()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print(f"  Early stop at iter {i}")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.W], max_norm=0.5)
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
            
            if i % 40 == 0 or i == self.iterations - 1:
                W_np = self.W.detach().cpu().numpy()[0]
                angle = np.degrees(np.arctan2(W_np[1, 0], W_np[0, 0]))
                scale = np.sqrt(W_np[0, 0]**2 + W_np[1, 0]**2)
                print(f"  [{i:3d}] Loss: {loss.item():.6f} | Angle: {angle:.2f}° | Scale: {scale:.3f}")
        
        if self.best_W is not None:
            with torch.no_grad():
                self.W.copy_(self.best_W)
        
        print(f"[Phase 4] Done. Best Loss: {self.best_loss:.6f}")
        return self.W.detach(), loss_history