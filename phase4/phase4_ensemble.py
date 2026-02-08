"""
================================================================================
Phase 4 Ensemble v3: Error-Aware Sequential Testing
================================================================================
[V2의 문제]
- 항상 V1 먼저 실행 → Early Exit
- V1이 "괜찮은" 결과 내면 V3 Scouting 시도조차 안함
- 하지만 V1 결과가 "괜찮다"는 기준이 절대적이지 않음

[V3 Solution]
1. V1 먼저 실행
2. V1 결과가 "excellent"이면 → Stop (threshold: 0.03)
3. "good"이면 (0.03~0.10) → V3도 시도해서 비교
4. "poor"이면 (>0.10) → V2, V3 모두 시도
================================================================================
"""

import torch
import torch.nn as nn


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from phase4.phase4_1 import HierarchicalMPCRefiner as RefinerV1
from phase4.phase4_2 import HierarchicalMPCRefiner as RefinerV2
from phase4.phase4_3 import HierarchicalMPCRefiner as RefinerV3
import torch.nn.functional as F

class EnsembleMPCRefiner(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        self.refiners = {
            'V1': RefinerV1(device=device),
            'V2': RefinerV2(device=device),
            'V3': RefinerV3(device=device)
        }
    
    def compute_final_energy(self, pyramid_a, pyramid_b, W_matrix):
        safe_l = min(0, len(pyramid_a) - 1)
        
        def pack_feats(p_tuple):
            s, v, b = p_tuple
            return {
                'sdf': s[:, :1].detach(),
                'vector': v.mean(dim=1).detach(),
                'rotor': b[2].mean(dim=1, keepdim=True).detach()
            }
        
        f_a = pack_feats(pyramid_a[safe_l])
        f_b = pack_feats(pyramid_b[safe_l])
        
        B, _, H, W = f_a['sdf'].shape
        grid = F.affine_grid(W_matrix, [B, 1, H, W], align_corners=False)
        
        warped_sdf = F.grid_sample(f_a['sdf'], grid, align_corners=False, padding_mode='border')
        warped_vec = F.grid_sample(f_a['vector'], grid, align_corners=False, padding_mode='zeros')
        warped_rot = F.grid_sample(f_a['rotor'], grid, align_corners=False, padding_mode='zeros')
        
        mask = (F.grid_sample(torch.ones_like(warped_sdf), grid, align_corners=False) > 0.9).float()
        
        rot_mat = W_matrix[:, :2, :2]
        vec_perm = warped_vec.permute(0, 2, 3, 1)
        warped_vec_corr = torch.einsum('bij,bhwj->bhwi', rot_mat, vec_perm).permute(0, 3, 1, 2)
        
        diff_sdf = torch.abs(warped_sdf - f_b['sdf'])
        sim_vec = F.cosine_similarity(warped_vec_corr, f_b['vector'], dim=1)
        diff_vec = (1.0 - sim_vec).unsqueeze(1)
        diff_rot = torch.abs(warped_rot - f_b['rotor'])
        
        loss_map = 1.0 * diff_sdf + 0.5 * diff_vec + 0.2 * diff_rot
        return (loss_map * mask).sum() / (mask.sum() + 1e-6)
    
    def optimize(self, pyramid_a_feats, pyramid_b_feats, W_init):
        print("\n[Ensemble V3] Error-adaptive sequential testing...")
        
        results = {}
        
        # Step 1: Always run V1 first
        print(f"\n  [1/3] Testing V1...")
        W_v1, hist_v1 = self.refiners['V1'].optimize(pyramid_a_feats, pyramid_b_feats, W_init)
        
        with torch.no_grad():
            energy_v1 = self.compute_final_energy(pyramid_a_feats, pyramid_b_feats, W_v1).item()
        
        results['V1'] = {'matrix': W_v1, 'energy': energy_v1, 'history': hist_v1}
        print(f"    V1 Energy: {energy_v1:.6f}")
        
        # Step 2: Decision Tree
        if energy_v1 < 0.03:
            print(f"  ✓ V1 result is excellent (< 0.03), stopping")
            return W_v1, hist_v1
        
        elif energy_v1 < 0.10:
            print(f"  → V1 is good but not perfect, trying V3 (scouting)...")
            W_v3, hist_v3 = self.refiners['V3'].optimize(pyramid_a_feats, pyramid_b_feats, W_init)
            
            with torch.no_grad():
                energy_v3 = self.compute_final_energy(pyramid_a_feats, pyramid_b_feats, W_v3).item()
            
            results['V3'] = {'matrix': W_v3, 'energy': energy_v3, 'history': hist_v3}
            print(f"    V3 Energy: {energy_v3:.6f}")
            
            # Return best
            if energy_v3 < energy_v1:
                print(f"  ✓ V3 wins!")
                return W_v3, hist_v3
            else:
                print(f"  ✓ V1 wins!")
                return W_v1, hist_v1
        
        else:  # Poor result
            print(f"  → V1 struggled, testing V2 and V3...")
            
            # Run V2
            W_v2, hist_v2 = self.refiners['V2'].optimize(pyramid_a_feats, pyramid_b_feats, W_init)
            with torch.no_grad():
                energy_v2 = self.compute_final_energy(pyramid_a_feats, pyramid_b_feats, W_v2).item()
            results['V2'] = {'matrix': W_v2, 'energy': energy_v2, 'history': hist_v2}
            print(f"    V2 Energy: {energy_v2:.6f}")
            
            # Run V3
            W_v3, hist_v3 = self.refiners['V3'].optimize(pyramid_a_feats, pyramid_b_feats, W_init)
            with torch.no_grad():
                energy_v3 = self.compute_final_energy(pyramid_a_feats, pyramid_b_feats, W_v3).item()
            results['V3'] = {'matrix': W_v3, 'energy': energy_v3, 'history': hist_v3}
            print(f"    V3 Energy: {energy_v3:.6f}")
            
            # Return best
            best_name = min(results.keys(), key=lambda k: results[k]['energy'])
            print(f"  ✓ Winner: {best_name} ({results[best_name]['energy']:.6f})")
            return results[best_name]['matrix'], results[best_name]['history']