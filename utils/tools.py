import os
import random
import time
from datetime import datetime
from matplotlib import cm

import numpy as np
import open3d as o3d

import torch
import torch.nn as nn

from utils.config import Config


def setup_experiment(config: Config):
    # random seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    o3d.utility.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if config.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    # set up experiment
    experiment_path = f"{config.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_path = os.path.join(config.output_path, experiment_path)
    os.makedirs(experiment_path, exist_ok=True)
    config.experiment_path = experiment_path

def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

# borrowed from https://github.com/PRBonn/PIN_SLAM/blob/main/utils/tools.py#L583
def voxel_downsample_torch(points: torch.Tensor, downsample_size: float) -> torch.Tensor:
    _quantization = 1 # 1000

    offset = torch.floor(points.min(dim=0)[0] / downsample_size).long()
    grid = torch.floor(points / downsample_size)
    center = (grid + 0.5) * downsample_size
    dists = ((points - center) ** 2).sum(dim=1) ** 0.5
    dists = (
        dists / dists.max() * (_quantization - 1)
    ).long()

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dists.long() * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce='amin', include_self=False
    )
    idx = idx % offset
    
    return idx

def save_pcd_o3d(points: torch.Tensor, filename: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    o3d.io.write_point_cloud(filename, pcd)

def viz_pcd_o3d(points: torch.Tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])