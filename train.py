import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import random
import os
import glob
from tqdm import tqdm

# ==========================================
# 🏛️ Phase 1. 전처리 (Preprocessing)
# ==========================================
class Phase1Preprocessor:
    def __init__(self, sam_checkpoint, model_type="vit_b", device="cuda"):
        self.device = device
        print(f"Loading SAM model from {sam_checkpoint}...")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        print("✅ Phase 1 Preprocessor Ready.")

    def rgb_to_hsi(self, img_rgb):
        img_rgb = img_rgb.astype(np.float32) / 255.0
        R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        I = np.mean(img_rgb, axis=2)
        min_rgb = np.min(img_rgb, axis=2)
        S = 1.0 - (3.0 * min_rgb / (I * 3.0 + 1e-6))
        S[I == 0] = 0
        num = 0.5 * ((R - G) + (R - B))
        den = np.sqrt((R - G)**2 + (R - B) * (G - B))
        theta = np.arccos(num / (den + 1e-6))
        H = theta.copy()
        H[B > G] = 2 * np.pi - H[B > G]
        H = H / (2 * np.pi)
        return np.stack([H, S, I], axis=-1)

    def get_sdf_and_shape_vector(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        dist_inside = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        dist_outside = cv2.distanceTransform(255 - mask_uint8, cv2.DIST_L2, 5)
        sdf_map = (dist_inside - dist_outside)
        sdf_map = sdf_map / (np.max(np.abs(sdf_map)) + 1e-6)

        y_idxs, x_idxs = np.nonzero(mask)
        if len(x_idxs) > 0:
            coords = np.stack([x_idxs, y_idxs], axis=1).astype(np.float32)
            mean = np.mean(coords, axis=0)
            coords_centered = coords - mean
            U, S, Vt = np.linalg.svd(coords_centered, full_matrices=False)
            components = Vt[:2].flatten()
            eigenvalues = (S ** 2) / (len(coords) - 1)
            total_variance = np.sum(eigenvalues) + 1e-6
            explained_variance = eigenvalues / total_variance
            v_shape = np.concatenate([components, explained_variance])
        else:
            v_shape = np.zeros(6)
        return sdf_map, v_shape

    def process_from_array(self, img_rgb):
        # SAM 마스크 생성
        masks = self.mask_generator.generate(img_rgb)
        if len(masks) == 0:
            h, w = img_rgb.shape[:2]
            mask = np.ones((h, w), dtype=bool)
        else:
            main_mask_info = sorted(masks, key=lambda x: x['area'], reverse=True)[0]
            mask = main_mask_info['segmentation']
        
        hsi_img = self.rgb_to_hsi(img_rgb)
        sdf_map, v_shape = self.get_sdf_and_shape_vector(mask)

        # 저장 용량을 줄이기 위해 float32 -> float16 고려 가능 (여기선 유지)
        return {'hsi': hsi_img, 'sdf': sdf_map, 'v_shape': v_shape}

# ==========================================
# 💎 Phase 2. Clifford Embedding
# ==========================================
class CliffordEmbedding(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.geometry_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1) 
        )
        self.pos_embed_param = nn.Parameter(torch.randn(1, 2, 32, 32) * 0.02)
        self.grouped_proj = nn.Conv2d(8, out_channels, kernel_size=1, groups=4)
        self.mix_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, hsi_img, sdf_map):
        hue = hsi_img[:, 0:1, :, :]        
        saturation = hsi_img[:, 1:2, :, :] 
        scalar_part = torch.cat([hue, saturation], dim=1) 

        intensity = hsi_img[:, 2:3, :, :] 
        geo_input = torch.cat([intensity, sdf_map], dim=1) 
        learned_vector = self.geometry_cnn(geo_input) 
        dx, dy = learned_vector[:, 0:1, :, :], learned_vector[:, 1:2, :, :]
        
        bivector_value = dx * dy 
        bivector_part = torch.cat([bivector_value, sdf_map], dim=1) 

        B, _, H, W = hsi_img.shape
        pos_part = F.interpolate(self.pos_embed_param, size=(H, W), mode='bilinear', align_corners=False) 
        pos_part = pos_part.repeat(B, 1, 1, 1)

        clifford_raw = torch.cat([scalar_part, learned_vector, bivector_part, pos_part], dim=1)
        out = self.grouped_proj(clifford_raw)
        out = self.mix_proj(out)
        return out

# ==========================================
# 👁️‍🗨️ Phase 3. Transformer
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels) 
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)
    def forward(self, x):
        return x + self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))

class DeepGeometricStem(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            ResBlock(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        return self.stem(x)

class GeometricAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sdf_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, num_heads))
        self.to_q, self.to_k, self.to_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv, sdf_q, sdf_k):
        B, N_q, C = x_q.shape
        _, N_k, _ = x_kv.shape
        q = self.to_q(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(x_kv).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(x_kv).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        score_geom = (q @ k.transpose(-2, -1)) * self.scale 
        sdf_diff = (sdf_q.unsqueeze(2) - sdf_k.unsqueeze(1)).abs() 
        score_sdf = self.sdf_mlp(sdf_diff).permute(0, 3, 1, 2) 
        attn = (score_geom - score_sdf).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        return self.proj(out)

class GeometricMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2) 
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act_gate = nn.SiLU()
    def forward(self, x):
        x_L, x_R = self.fc1(x).chunk(2, dim=-1)
        H = (x_L * x_R) + (x_L + x_R)
        return self.fc2(self.act_gate(H) * H)

class GeometricTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = GeometricAttention(dim)
        self.cross_attn = GeometricAttention(dim)
        self.mlp = GeometricMLP(dim, dim * 2)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
    def forward(self, x, source, x_sdf, source_sdf):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), x_sdf, x_sdf)
        x = x + self.cross_attn(self.norm2(x), self.norm2(source), x_sdf, source_sdf)
        x = x + self.mlp(self.norm3(x))
        return x

class DeepGeometricTransformer(nn.Module):
    def __init__(self, in_channels=32, num_layers=2):
        super().__init__()
        self.stem = DeepGeometricStem(in_channels)
        self.layers = nn.ModuleList([GeometricTransformerBlock(in_channels) for _ in range(num_layers)])
        self.output_head = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, img_feat_A, img_feat_B, sdf_A, sdf_B):
        fA, fB = self.stem(img_feat_A), self.stem(img_feat_B)
        sdf_A_s = F.interpolate(sdf_A, size=fA.shape[2:], mode='nearest')
        sdf_B_s = F.interpolate(sdf_B, size=fB.shape[2:], mode='nearest')
        
        B, C, H, W = fA.shape
        fA_flat = fA.flatten(2).transpose(1, 2)
        fB_flat = fB.flatten(2).transpose(1, 2)
        sdf_A_flat = sdf_A_s.flatten(2).transpose(1, 2)
        sdf_B_flat = sdf_B_s.flatten(2).transpose(1, 2)

        for layer in self.layers:
            fA_new = layer(fA_flat, fB_flat, sdf_A_flat, sdf_B_flat)
            fB_new = layer(fB_flat, fA_flat, sdf_B_flat, sdf_A_flat)
            fA_flat, fB_flat = fA_new, fB_new

        fA_out = self.output_head(fA_flat.transpose(1, 2).reshape(B, C, H, W))
        fB_out = self.output_head(fB_flat.transpose(1, 2).reshape(B, C, H, W))
        return fA_out, fB_out

# ==========================================
# 🧩 Phase 4. Refinement (Not used in training)
# ==========================================
class DeformableMesh(nn.Module):
    def __init__(self, height, width, grid_size=32):
        super().__init__()
        self.H, self.W = height, width
        y, x = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size), indexing='ij')
        self.initial_vertices = torch.stack([x, y], dim=-1).reshape(-1, 2)
        self.vertices = nn.Parameter(self.initial_vertices.clone())
        self.faces = self.create_grid_faces(grid_size, grid_size)
        self.update_edge_lengths()
    def create_grid_faces(self, h, w):
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                faces.append([idx, idx + w, idx + 1])
                faces.append([idx + 1, idx + w, idx + w + 1])
        return torch.tensor(faces, dtype=torch.long)
    def compute_edge_lengths(self, vertices, faces):
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        l1, l2, l3 = torch.norm(v1 - v0, dim=1), torch.norm(v2 - v1, dim=1), torch.norm(v0 - v2, dim=1)
        return torch.stack([l1, l2, l3], dim=1)
    def update_edge_lengths(self):
        self.initial_edge_lengths = self.compute_edge_lengths(self.initial_vertices, self.faces)
        if self.vertices.is_cuda:
            self.initial_edge_lengths = self.initial_edge_lengths.to(self.vertices.device)
            self.faces = self.faces.to(self.vertices.device)
            self.initial_vertices = self.initial_vertices.to(self.vertices.device)

class MPCSolver(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_geo=0.5, lambda_topo=100.0, lambda_reg=0.1):
        super().__init__()
        self.lambdas = {'data': lambda_data, 'geo': lambda_geo, 'topo': lambda_topo, 'reg': lambda_reg}
    def sample_features(self, feature_map, vertices):
        grid = vertices.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(feature_map, grid, align_corners=False)
        return sampled.squeeze(2).squeeze(0).transpose(0, 1) 
    def forward(self, mesh, target_feat, source_feat_at_init):
        current_vertices = mesh.vertices
        curr_feat = self.sample_features(target_feat, current_vertices)
        loss_data = 1.0 - F.cosine_similarity(curr_feat, source_feat_at_init, dim=1).mean()
        curr_edge_lengths = mesh.compute_edge_lengths(current_vertices, mesh.faces)
        loss_geo = F.mse_loss(curr_edge_lengths, mesh.initial_edge_lengths.to(current_vertices.device))
        v0, v1, v2 = current_vertices[mesh.faces[:, 0]], current_vertices[mesh.faces[:, 1]], current_vertices[mesh.faces[:, 2]]
        edge1, edge2 = v1 - v0, v2 - v0
        signed_area = 0.5 * (edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
        loss_topo = F.relu(-signed_area).mean()
        delta_v = current_vertices - mesh.initial_vertices.to(current_vertices.device)
        loss_reg = torch.norm(delta_v, dim=1).mean()
        total_loss = (self.lambdas['data'] * loss_data + self.lambdas['geo'] * loss_geo +
                      self.lambdas['topo'] * loss_topo + self.lambdas['reg'] * loss_reg)
        return total_loss, {'data': loss_data.item(), 'geo': loss_geo.item(), 'topo': loss_topo.item(), 'reg': loss_reg.item()}

class AdaptiveTessellator:
    def __init__(self, error_threshold=0.1):
        self.tau = error_threshold

    def subdivision_check(self, mesh, target_feat, source_feat_at_init):
        solver = MPCSolver() 
        curr_feat = solver.sample_features(target_feat, mesh.vertices)
        vertex_errors = 1.0 - F.cosine_similarity(curr_feat, source_feat_at_init, dim=1)
        face_errors = vertex_errors[mesh.faces].mean(dim=1)
        high_error_faces_idx = torch.where(face_errors > self.tau)[0]
        return high_error_faces_idx

    def get_midpoint_idx(self, v1_idx, v2_idx, edge_cache, vertices, initial_vertices, source_features, new_vertices_list, new_initial_vertices_list, new_features_list):
        edge_key = tuple(sorted((v1_idx.item(), v2_idx.item())))
        if edge_key in edge_cache: return edge_cache[edge_key]
        
        p1, p2 = vertices[v1_idx], vertices[v2_idx]
        new_pos = (p1 + p2) / 2.0
        
        init_p1, init_p2 = initial_vertices[v1_idx], initial_vertices[v2_idx]
        new_init_pos = (init_p1 + init_p2) / 2.0
        
        f1, f2 = source_features[v1_idx], source_features[v2_idx]
        new_feat = (f1 + f2) / 2.0
        
        current_len = len(vertices) + len(new_vertices_list)
        new_vertices_list.append(new_pos)
        new_initial_vertices_list.append(new_init_pos)
        new_features_list.append(new_feat) 
        
        edge_cache[edge_key] = current_len
        return current_len

    def subdivide(self, mesh, high_error_faces_idx, source_feat_init):
        if len(high_error_faces_idx) == 0: return mesh, source_feat_init
        
        device = mesh.vertices.device
        
        # CPU로 데이터 내리기 (인덱싱 속도 및 편의성)
        # 주의: 여기서 detach().cpu()를 했으므로 vertices, initial_vertices는 CPU 텐서입니다.
        vertices = mesh.vertices.detach().cpu()
        initial_vertices = mesh.initial_vertices.detach().cpu()
        faces = mesh.faces.detach().cpu()
        source_features = source_feat_init.detach().cpu() 
        
        high_error_set = set(high_error_faces_idx.tolist())
        
        new_faces = []
        new_vertices_list = []
        new_initial_vertices_list = []
        new_features_list = [] 
        edge_cache = {}
        
        for i, face in enumerate(faces):
            if i not in high_error_set:
                new_faces.append(face)
                continue
            
            v0, v1, v2 = face[0], face[1], face[2]
            
            m01 = self.get_midpoint_idx(v0, v1, edge_cache, vertices, initial_vertices, source_features, new_vertices_list, new_initial_vertices_list, new_features_list)
            m12 = self.get_midpoint_idx(v1, v2, edge_cache, vertices, initial_vertices, source_features, new_vertices_list, new_initial_vertices_list, new_features_list)
            m20 = self.get_midpoint_idx(v2, v0, edge_cache, vertices, initial_vertices, source_features, new_vertices_list, new_initial_vertices_list, new_features_list)
            
            # 새 Face들은 나중에 GPU로 한 번에 올릴 것이므로 일단 CPU 텐서로 생성하거나 리스트로 유지
            # 여기서는 편의상 바로 GPU 텐서로 만들되, 리스트에 담아둡니다.
            # (단, faces 리스트가 섞여있으면 나중에 stack할 때 문제되므로 통일 필요)
            # 여기서는 일단 리스트로 유지하고 나중에 변환합니다.
            new_faces.append(torch.tensor([v0, m01, m20]))
            new_faces.append(torch.tensor([v1, m12, m01]))
            new_faces.append(torch.tensor([v2, m20, m12]))
            new_faces.append(torch.tensor([m01, m12, m20]))

        # 데이터 업데이트
        # source_feat_init이 CPU로 내려왔으므로 다시 GPU로 올릴 준비를 해야 합니다.
        updated_source_feat = source_feat_init 

        if len(new_vertices_list) > 0:
            # 새로 생긴 점들을 GPU로 올립니다.
            added_vertices = torch.stack(new_vertices_list).to(device)
            added_initial = torch.stack(new_initial_vertices_list).to(device)
            added_features = torch.stack(new_features_list).to(device) 
            
            # [수정된 부분]
            # 기존 vertices와 initial_vertices는 위에서 .cpu()로 내렸습니다.
            # torch.cat을 하려면 얘네들도 다시 .to(device)로 GPU에 올려야 합니다.
            vertices_gpu = vertices.to(device)
            initial_vertices_gpu = initial_vertices.to(device)
            source_features_gpu = source_features.to(device)

            # 이제 모두 GPU에 있으므로 합칠 수 있습니다.
            new_all_vertices = torch.cat([vertices_gpu, added_vertices], dim=0)
            mesh.vertices = nn.Parameter(new_all_vertices)
            
            # Initial Vertices 업데이트
            mesh.initial_vertices = torch.cat([initial_vertices_gpu, added_initial], dim=0)
            
            # Faces 업데이트 (리스트에 있는 텐서들을 stack 후 GPU로 이동)
            mesh.faces = torch.stack(new_faces).to(device)
            
            # 물리 정보 갱신
            mesh.update_edge_lengths()
            
            # 특징값 업데이트
            updated_source_feat = torch.cat([source_features_gpu, added_features], dim=0)
            
            print(f"   -> Vertices: {len(vertices)} -> {len(mesh.vertices)} (+{len(new_vertices_list)})")
            
        return mesh, updated_source_feat
    
class Phase4Refinement(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.mesh = DeformableMesh(h, w, grid_size=32)
        self.solver = MPCSolver()
        self.tessellator = AdaptiveTessellator(error_threshold=0.1)
    def forward(self, target_feat, source_feat_init, steps=20):
        self.optimize_loop(target_feat, source_feat_init, steps=steps, lr=0.01)
        high_error_faces = self.tessellator.subdivision_check(self.mesh, target_feat, source_feat_init)
        if len(high_error_faces) > 0:
            self.mesh, updated_source_feat = self.tessellator.subdivide(self.mesh, high_error_faces, source_feat_init)
            self.optimize_loop(target_feat, updated_source_feat, steps=10, lr=0.005)
            return self.mesh.vertices, updated_source_feat 
        return self.mesh.vertices, source_feat_init
    def optimize_loop(self, target_feat, source_feat, steps, lr):
        optimizer = torch.optim.Adam([self.mesh.vertices], lr=lr)
        for i in range(steps):
            optimizer.zero_grad()
            loss, details = self.solver(self.mesh, target_feat, source_feat)
            loss.backward()
            optimizer.step()

# ==========================================
# 🛡️ Phase 5. Training (GPU Optimized with DataLoader)
# ==========================================
class HomographyAugmentor:
    """
    [GPU 최적화] Numpy 대신 OpenCV/Tensor를 활용하여 Dataset 내부에서 호출
    """
    def __init__(self, height=128, width=128):
        self.H, self.W = height, width
        self.photo_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    def generate_homography(self):
        # pts1은 그대로
        pts1 = np.float32([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]])
        
        # [🔥 수정됨] -32~32는 너무 빡셉니다. -16~16으로 줄여서 "살살" 비틀게 하세요.
        # 학습이 잘 되면 나중에 늘리면 됩니다.
        pts2 = pts1 + np.random.uniform(-16, 16, pts1.shape).astype(np.float32)
        
        return cv2.getPerspectiveTransform(pts1, pts2)

    def __call__(self, img_rgb, hsi, sdf):
        """
        [핵심 최적화] 이미 계산된 HSI, SDF를 워핑하여 View B 생성
        """
        # View A: Photometric Augmentation on HSI
        hsi_tensor = torch.from_numpy(hsi).permute(2, 0, 1).float()
        hsi_A = self.photo_aug(hsi_tensor)
        sdf_A = torch.from_numpy(sdf).unsqueeze(0).float() # [1, H, W]

        # View B: Geometric Augmentation on HSI & SDF
        H_mat = self.generate_homography()
        
        # Warp HSI
        hsi_B_np = cv2.warpPerspective(hsi, H_mat, (self.W, self.H))
        hsi_B = torch.from_numpy(hsi_B_np).permute(2, 0, 1).float()
        
        # Warp SDF
        sdf_B_np = cv2.warpPerspective(sdf, H_mat, (self.W, self.H))
        sdf_B = torch.from_numpy(sdf_B_np).unsqueeze(0).float()

        # Valid Mask
        ones_mask = np.ones((self.H, self.W), dtype=np.uint8)
        valid_mask_np = cv2.warpPerspective(ones_mask, H_mat, (self.W, self.H))
        valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0).float() # [1, H, W]
        
        # Homography Matrix Tensor
        H_t = torch.from_numpy(H_mat).float()

        return hsi_A, sdf_A, hsi_B, sdf_B, valid_mask, H_t

# [수정] Custom Dataset: Only Loads Cached Data
class COCODataset(Dataset):
    def __init__(self, img_paths, cache_dir="./cache"):
        self.img_paths = img_paths
        self.augmentor = HomographyAugmentor(128, 128)
        self.cache_dir = cache_dir
        # Preprocessor가 여기 없음! (충돌 방지)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        file_name = os.path.basename(img_path).split('.')[0]
        cache_path = os.path.join(self.cache_dir, f"{file_name}.npy")

        try:
            # 무조건 캐시에서 로드 (Pre-caching이 완료되었으므로)
            data = np.load(cache_path, allow_pickle=True).item()
        except Exception:
            # 만약 파일이 없거나 깨졌으면 다른 파일 로드
            return self.__getitem__(random.randint(0, len(self)-1))

        # 3. Augmentation (View A, View B 생성)
        hsi_A, sdf_A, hsi_B, sdf_B, valid_mask, H_t = self.augmentor(None, data['hsi'], data['sdf'])

        return hsi_A, sdf_A, hsi_B, sdf_B, valid_mask, H_t

class FeatureMatchingLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') 

    def forward(self, feat_A, feat_B, H_mat, valid_mask):
        B, C, H, W = feat_A.shape
        fA_flat = feat_A.view(B, C, -1)
        fB_flat = feat_B.view(B, C, -1)
        
        # [🔥 수정됨] L2 Normalization 추가! (이게 없으면 Loss가 5~10까지 튑니다)
        fA_flat = F.normalize(fA_flat, p=2, dim=1)
        fB_flat = F.normalize(fB_flat, p=2, dim=1)
        
        # Similarity Matrix 계산 (이제 값의 범위가 -1 ~ 1로 안정됨)
        sim_matrix = torch.bmm(fA_flat.transpose(1, 2), fB_flat) / self.tau
        
        labels = torch.arange(H * W).to(feat_A.device).unsqueeze(0).repeat(B, 1) 
        
        loss = self.ce_loss(sim_matrix, labels) 
        mask_flat = valid_mask.view(B, -1)
        
        # 유효한 영역의 Loss 평균
        return (loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)

class CliffordNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CliffordEmbedding(out_channels=32)
        self.transformer = DeepGeometricTransformer(in_channels=32, num_layers=2)
    def forward(self, hsi_A, sdf_A, hsi_B, sdf_B):
        emb_A = self.embedding(hsi_A, sdf_A)
        emb_B = self.embedding(hsi_B, sdf_B)
        return self.transformer(emb_A, emb_B, sdf_A, sdf_B)

class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.criterion = FeatureMatchingLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        self.scaler = GradScaler()
        self.device = device

    def train_loop(self, img_paths, epochs=150, patience=10, batch_size=16):
        print(f"🛡️ Start Training for {epochs} epochs with Batch Size {batch_size}...")
        
        # [수정] DataLoader: Preprocessor 제거
        dataset = COCODataset(img_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        best_loss, patience_counter = float('inf'), 0
        
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                hsi_A, sdf_A, hsi_B, sdf_B, valid_mask, H_t = [x.to(self.device, non_blocking=True) for x in batch]
                
                self.optimizer.zero_grad()
                
                with autocast():
                    feat_A_final, feat_B_final = self.model(hsi_A, sdf_A, hsi_B, sdf_B)
                    mask_small = F.interpolate(valid_mask, size=feat_A_final.shape[2:], mode='nearest')
                    loss = self.criterion(feat_A_final, feat_B_final, H_t, mask_small)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"   >>> Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss, patience_counter = avg_loss, 0
                torch.save(self.model.state_dict(), "clifford_model_best.pth")
                print("   💾 Best model saved.")
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print(f"🛑 Early Stopping Triggered at epoch {epoch+1}.")
            #         break
        
        torch.save(self.model.state_dict(), "clifford_model_final.pth")
        print("✅ Final model saved to clifford_model_final.pth")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("\n✅ GPU Optimized Training Pipeline Initialized.")
    SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"⚠️ Error: {SAM_CHECKPOINT} not found.")
        exit()

    img_paths = glob.glob("./img/val2017/*.jpg")
    if not img_paths: img_paths = glob.glob("./img/*.jpg")

    # ========================================================
    # 📉 [수정됨] 데이터셋 크기 제한 (빠른 테스트용)
    # ========================================================
    MAX_IMAGES = 200  # 원하는 장수 설정 (예: 200)
    if len(img_paths) > MAX_IMAGES:
        img_paths = img_paths[:MAX_IMAGES] # 리스트를 앞에서부터 200개만 자름
        print(f"📉 Limiting dataset to first {len(img_paths)} images for fast prototyping.")
    # ========================================================

    print(f"Found {len(img_paths)} training images.")
    
    # [핵심] Pre-caching 실행 (Main Process Only)
    print("⚡ Starting offline preprocessing (caching) to avoid DataLoader CUDA errors...")
    preprocessor = Phase1Preprocessor(sam_checkpoint=SAM_CHECKPOINT, model_type="vit_b")
    os.makedirs("./cache", exist_ok=True)
    
    # 5000장 중 처음 100개만 테스트로 먼저 해보거나, 전체 다 돌리거나 선택
    for img_path in tqdm(img_paths, desc="Pre-caching SAM features"):
        file_name = os.path.basename(img_path).split('.')[0]
        cache_path = os.path.join("./cache", f"{file_name}.npy")
        if not os.path.exists(cache_path):
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data = preprocessor.process_from_array(img)
            np.save(cache_path, data)
    
    # SAM 메모리 해제 (학습을 위해 VRAM 확보)
    del preprocessor
    torch.cuda.empty_cache()
    print("✅ Pre-caching complete! Starting Training...")

    full_model = CliffordNetwork()
    trainer = Trainer(full_model)
    trainer.train_loop(img_paths, epochs=150, patience=10, batch_size=16)
    print("✅ All processes completed.")