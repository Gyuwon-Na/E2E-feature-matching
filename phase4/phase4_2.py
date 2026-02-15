"""
================================================================================
Phase 4_2 Improved: Robust Geometric MPC Refiner (Fixed)
================================================================================
수정 사항:
- get_current_W()에서 텐서 shape 불일치 문제 해결
- torch.stack 대신 직접 텐서 구성
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional


class GeometricMPCRefiner(nn.Module):
    """
    Geometric MPC Refiner - Fixed Version
    """
    
    def __init__(self, device='cuda', config: Optional[Dict] = None):
        super().__init__()
        self.device = device
        self.epsilon = 1e-6
        
        # Default configuration
        self.config = {
            'learning_rate': 0.01,
            'iterations': 150,
            'patience': 30,
            'angle_lr_mult': 2.0,
            'scale_range': 0.15,
            'trans_range': 0.1,
            'huber_delta': 0.1,
            'warmup_iters': 10,
            'use_priority': True,
            'verbose': True
        }
        
        if config:
            self.config.update(config)
        
        # Learnable parameters (decomposed) - 모두 1D 텐서로 통일
        self.p_angle = nn.Parameter(torch.zeros(1, device=device))
        self.p_scale = nn.Parameter(torch.zeros(1, device=device))
        self.p_trans = nn.Parameter(torch.zeros(2, device=device))
        
        # Base parameters
        self.base_angle = 0.0
        self.base_scale = 1.0
        
        # Tracking
        self.best_loss = float('inf')
        self.best_state = None
        self.W = None
        
    def reset_parameters(self):
        """Reset learnable parameters to zero"""
        with torch.no_grad():
            self.p_angle.zero_()
            self.p_scale.zero_()
            self.p_trans.zero_()
        self.best_loss = float('inf')
        self.best_state = None
    
    def global_filtering_init(self, mean_rotor: float, mean_scale: float = 1.0):
        """Initialize from Phase 3 predictions"""
        angle_deg = np.degrees(mean_rotor)
        
        if abs(angle_deg) > 90:
            if self.config['verbose']:
                print(f"[Phase 4] Warning: Large init angle ({angle_deg:.1f} deg), clamping to +/-90")
            mean_rotor = np.clip(mean_rotor, -np.pi/2, np.pi/2)
        
        self.base_angle = float(mean_rotor)
        self.base_scale = float(np.clip(mean_scale, 0.5, 2.0))
        
        self.reset_parameters()
        self._update_W()
        
        if self.config['verbose']:
            print(f"[Phase 4] Init: Angle={np.degrees(self.base_angle):.2f} deg, Scale={self.base_scale:.3f}")
    
    def _update_W(self):
        """Update the transformation matrix from current parameters"""
        angle = self.base_angle + self.p_angle.item()
        
        scale_delta = torch.tanh(self.p_scale).item() * self.config['scale_range']
        scale = self.base_scale * (1.0 + scale_delta)
        
        tx = torch.tanh(self.p_trans[0]).item() * self.config['trans_range']
        ty = torch.tanh(self.p_trans[1]).item() * self.config['trans_range']
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        W = torch.zeros(1, 2, 3, device=self.device)
        W[0, 0, 0] = scale * cos_a
        W[0, 0, 1] = -scale * sin_a
        W[0, 0, 2] = tx
        W[0, 1, 0] = scale * sin_a
        W[0, 1, 1] = scale * cos_a
        W[0, 1, 2] = ty
        
        self.W = W
        return W
    
    def get_current_W(self) -> torch.Tensor:
        """
        Get current transformation matrix (differentiable)
        
        Returns:
            W: [1, 2, 3] affine transformation matrix
        """
        # Current angle - p_angle is [1] tensor
        angle = self.base_angle + self.p_angle[0]  # scalar tensor
        
        # Scale with bounded delta
        scale_delta = torch.tanh(self.p_scale[0]) * self.config['scale_range']
        scale = self.base_scale * (1.0 + scale_delta)
        
        # Translation (bounded) - p_trans is [2] tensor
        tx = torch.tanh(self.p_trans[0]) * self.config['trans_range']
        ty = torch.tanh(self.p_trans[1]) * self.config['trans_range']
        
        # Construct W matrix - 직접 텐서 구성 (torch.stack 사용 안함)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # 2x3 행렬 직접 구성
        W = torch.zeros(1, 2, 3, device=self.device, dtype=torch.float32)
        W[0, 0, 0] = scale * cos_a
        W[0, 0, 1] = -scale * sin_a
        W[0, 0, 2] = tx
        W[0, 1, 0] = scale * sin_a
        W[0, 1, 1] = scale * cos_a
        W[0, 1, 2] = ty
        
        return W
    
    def _normalize_features(self, feat_dict: Dict) -> Dict:
        """Normalize input features to consistent format"""
        normalized = {}
        
        # SDF
        sdf = feat_dict['sdf']
        if sdf.dim() == 3:
            sdf = sdf.unsqueeze(0)
        if sdf.shape[1] > 1:
            sdf = sdf.mean(dim=1, keepdim=True)
        normalized['sdf'] = sdf
        
        # Vector
        vec = feat_dict['vector']
        if vec.dim() == 5:  # [B, C, 2, H, W]
            B, C, _, H, W = vec.shape
            vec = vec.mean(dim=1)  # [B, 2, H, W]
        elif vec.dim() == 4 and vec.shape[1] > 2:
            B, C2, H, W = vec.shape
            n_vecs = C2 // 2
            vec = vec.view(B, n_vecs, 2, H, W).mean(dim=1)
        normalized['vector'] = vec
        
        # Rotor
        rotor = feat_dict['rotor']
        if rotor.dim() == 3:
            rotor = rotor.unsqueeze(0)
        if rotor.shape[1] > 1:
            rotor = rotor.mean(dim=1, keepdim=True)
        normalized['rotor'] = rotor
        
        return normalized
    
    def compute_priority_map(self, rotor_variance: torch.Tensor, 
                             feature_magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spatial priority map for weighted optimization"""
        stability = 1.0 / (rotor_variance + 0.01)
        raw_priority = stability * feature_magnitude
        
        B = raw_priority.shape[0]
        priority_map = torch.zeros_like(raw_priority)
        
        for b in range(B):
            flat = raw_priority[b].view(-1)
            p_min = flat.min()
            p_max = torch.quantile(flat, 0.95)
            
            normalized = (raw_priority[b] - p_min) / (p_max - p_min + self.epsilon)
            priority_map[b] = torch.clamp(normalized, 0, 1)
        
        return priority_map
    
    def huber_loss(self, x: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
        """Huber loss - robust to outliers"""
        abs_x = torch.abs(x)
        quadratic = torch.clamp(abs_x, max=delta)
        linear = abs_x - quadratic
        return 0.5 * quadratic ** 2 + delta * linear
    
    def compute_energy(self, src_dict: Dict, tgt_dict: Dict, 
                       gates: Tuple[torch.Tensor, ...],
                       priority_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute alignment energy"""
        # Get current transformation
        W = self.get_current_W()
        
        # Normalize inputs
        src = self._normalize_features(src_dict)
        tgt = self._normalize_features(tgt_dict)
        
        B, _, H, W_size = src['sdf'].shape
        
        # Create affine grid
        grid = F.affine_grid(W, [B, 1, H, W_size], align_corners=False)
        
        # Warp source features
        warped_sdf = F.grid_sample(src['sdf'], grid, align_corners=False, 
                                   padding_mode='border')
        warped_vec = F.grid_sample(src['vector'], grid, align_corners=False,
                                   padding_mode='zeros')
        warped_rotor = F.grid_sample(src['rotor'], grid, align_corners=False,
                                     padding_mode='zeros')
        
        # Valid mask
        ones = torch.ones_like(src['sdf'])
        valid_mask = F.grid_sample(ones, grid, align_corners=False, padding_mode='zeros')
        valid_mask = (valid_mask > 0.5).float()
        
        # Apply rotation to warped vectors
        rot_matrix = W[0, :2, :2]  # [2, 2]
        vec_permuted = warped_vec.permute(0, 2, 3, 1)  # [B, H, W, 2]
        vec_rotated = torch.einsum('ij,bhwj->bhwi', rot_matrix, vec_permuted)
        vec_rotated = vec_rotated.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # Energy terms
        e_scalar = self.huber_loss(warped_sdf - tgt['sdf'], self.config['huber_delta'])
        
        cos_sim = F.cosine_similarity(vec_rotated, tgt['vector'], dim=1, eps=self.epsilon)
        e_vector = (1.0 - cos_sim).unsqueeze(1)
        
        e_bivector = self.huber_loss(warped_rotor - tgt['rotor'], self.config['huber_delta'])
        
        # Apply gate weights
        g_s, g_v, g_b = gates
        
        if g_s.dim() < 4:
            g_s = g_s.view(1, 1, 1, 1)
            g_v = g_v.view(1, 1, 1, 1)
            g_b = g_b.view(1, 1, 1, 1)
        
        # Weighted sum
        total_energy = g_s * e_scalar + g_v * e_vector + g_b * e_bivector
        
        # Apply priority weighting
        if priority_map is not None and self.config['use_priority']:
            if priority_map.shape[-2:] != total_energy.shape[-2:]:
                priority_map = F.interpolate(priority_map, size=total_energy.shape[-2:],
                                            mode='bilinear', align_corners=False)
            total_energy = total_energy * (0.5 + 0.5 * priority_map)
        
        # Masked average
        masked_energy = total_energy * valid_mask
        energy = masked_energy.sum() / (valid_mask.sum() + self.epsilon)
        
        return energy
    
    def optimize(self, src_dict: Dict, tgt_dict: Dict,
                 gates: Tuple[torch.Tensor, ...],
                 priority_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """Run MPC optimization"""
        base_lr = self.config['learning_rate']
        
        optimizer = optim.Adam([
            {'params': [self.p_angle], 'lr': base_lr * self.config['angle_lr_mult']},
            {'params': [self.p_scale], 'lr': base_lr},
            {'params': [self.p_trans], 'lr': base_lr * 0.5}
        ])
        
        n_iters = self.config['iterations']
        warmup = self.config['warmup_iters']
        
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup  # +1 to avoid zero
            else:
                progress = (epoch - warmup) / max(n_iters - warmup, 1)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        loss_history = []
        patience_counter = 0
        
        if self.config['verbose']:
            print(f"\n[Phase 4] MPC Optimization ({n_iters} iterations)")
        
        for i in range(n_iters):
            optimizer.zero_grad()
            
            loss = self.compute_energy(src_dict, tgt_dict, gates, priority_map)
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_state = {
                    'p_angle': self.p_angle.detach().clone(),
                    'p_scale': self.p_scale.detach().clone(),
                    'p_trans': self.p_trans.detach().clone()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                if self.config['verbose']:
                    print(f"  Early stop at iter {i}")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.p_angle, self.p_scale, self.p_trans], 1.0)
            
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            
            if self.config['verbose'] and (i % 30 == 0 or i == n_iters - 1):
                angle_deg, scale, tx, ty = self.get_transform_params()
                print(f"  [{i:3d}] Loss: {loss.item():.6f} | "
                      f"Angle: {angle_deg:.2f} deg | Scale: {scale:.3f}")
        
        # Restore best
        if self.best_state is not None:
            with torch.no_grad():
                self.p_angle.copy_(self.best_state['p_angle'])
                self.p_scale.copy_(self.best_state['p_scale'])
                self.p_trans.copy_(self.best_state['p_trans'])
        
        # Update final W
        self._update_W()
        
        if self.config['verbose']:
            print(f"[Phase 4] Done. Best Loss: {self.best_loss:.6f}")
            angle_deg, scale, tx, ty = self.get_transform_params()
            print(f"[Phase 4] Final: Angle={angle_deg:.2f} deg, Scale={scale:.3f}, "
                  f"Trans=({tx:.4f}, {ty:.4f})")
        
        return self.W.detach(), loss_history
    
    def get_transform_params(self, W: Optional[torch.Tensor] = None) -> Tuple[float, float, float, float]:
        """Extract transformation parameters"""
        if W is not None:
            W_np = W.detach().cpu().numpy()
            if W_np.ndim == 3:
                W_np = W_np[0]
        elif self.W is not None:
            W_np = self.W.detach().cpu().numpy()[0]
        else:
            angle = self.base_angle + self.p_angle.item()
            scale_delta = torch.tanh(self.p_scale).item() * self.config['scale_range']
            scale = self.base_scale * (1.0 + scale_delta)
            tx = torch.tanh(self.p_trans[0]).item() * self.config['trans_range']
            ty = torch.tanh(self.p_trans[1]).item() * self.config['trans_range']
            
            return np.degrees(angle), scale, tx, ty
        
        angle_rad = np.arctan2(W_np[1, 0], W_np[0, 0])
        scale = np.sqrt(W_np[0, 0]**2 + W_np[1, 0]**2)
        tx = W_np[0, 2]
        ty = W_np[1, 2]
        
        return np.degrees(angle_rad), scale, tx, ty


# =============================================================================
# Factory function
# =============================================================================
def create_refiner(device='cuda', **kwargs) -> GeometricMPCRefiner:
    """Factory function to create refiner with custom config"""
    return GeometricMPCRefiner(device=device, config=kwargs)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing GeometricMPCRefiner (Fixed)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    B, C, H, W = 1, 1, 64, 64
    
    src_dict = {
        'sdf': torch.randn(B, C, H, W, device=device),
        'vector': torch.randn(B, 2, H, W, device=device),
        'rotor': torch.randn(B, C, H, W, device=device)
    }
    
    tgt_dict = {
        'sdf': torch.randn(B, C, H, W, device=device),
        'vector': torch.randn(B, 2, H, W, device=device),
        'rotor': torch.randn(B, C, H, W, device=device)
    }
    
    gates = (
        torch.ones(1, device=device) * 0.5,
        torch.ones(1, device=device) * 0.3,
        torch.ones(1, device=device) * 0.2
    )
    
    refiner = GeometricMPCRefiner(device=device, config={'verbose': True, 'iterations': 50})
    
    init_angle = np.radians(30)
    refiner.global_filtering_init(mean_rotor=init_angle, mean_scale=1.0)
    
    W_final, loss_history = refiner.optimize(src_dict, tgt_dict, gates)
    
    angle, scale, tx, ty = refiner.get_transform_params()
    
    print(f"\nFinal Results:")
    print(f"  Angle: {angle:.2f} deg")
    print(f"  Scale: {scale:.3f}")
    print(f"  Translation: ({tx:.4f}, {ty:.4f})")
    print(f"  Loss reduction: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")