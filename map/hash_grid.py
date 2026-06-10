import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config
from utils.tools import *


class HashGrid(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        # setting
        self.dtype = config.dtype
        self.idx_dtype = torch.int64
        self.device = config.device

        # feature
        self.feature_dim = config.feature_dim
        self.feature_std = config.feature_std

        # grid
        self.min_resolution = config.min_resolution
        self.num_resolution = config.num_resolution
        self.scale_up = config.scale_up

        self.steps = torch.tensor([
            [0., 0., 0.], [0., 0., 1.], 
            [0., 1., 0.], [0., 1., 1.], 
            [1., 0., 0.], [1., 0., 1.], 
            [1., 1., 0.], [1., 1., 1.]
        ], dtype=self.dtype, device=self.device)

        self.bbox = torch.tensor([
            [ sys.maxsize,  sys.maxsize,  sys.maxsize],  # min_bound
            [-sys.maxsize, -sys.maxsize, -sys.maxsize],  # max_bound
        ], dtype=self.dtype, device=self.device)

        # hash table
        self.buffer_size = config.buffer_size
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=self.idx_dtype, device=self.device
        )

        # parameters
        self.grid_features = nn.ParameterList()
        self.grid_indices = []
        for _ in range(self.num_resolution):
            grid_features = nn.Parameter(
                torch.empty((1, self.feature_dim), dtype=self.dtype, device=self.device)
            )
            grid_indices = torch.full(
                (self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device
            )

            self.grid_features.append(grid_features)
            self.grid_indices.append(grid_indices)

        self.to(self.device)

    def _update_bbox(self, points: torch.Tensor):
        min_bound = torch.min(points, dim=0).values
        max_bound = torch.max(points, dim=0).values

        self.bbox[0] = torch.min(self.bbox[0], min_bound)
        self.bbox[1] = torch.max(self.bbox[1], max_bound)

    def _get_corners(self, points: torch.Tensor, resolution: float) -> torch.Tensor:
        voxel_origin = torch.floor(points / resolution)
        voxel_corners = (voxel_origin[:, None, :] + self.steps[None, :, :]).reshape(-1, 3)
        return voxel_corners
    
    def _get_coeffs(self, points: torch.Tensor, resolution: float) -> torch.Tensor:
        coords = points / resolution
        coords = coords - torch.floor(coords)

        x_d = coords[:, 0]
        _1_x_d = 1 - x_d
        y_d = coords[:, 1]
        _1_y_d = 1 - y_d
        z_d = coords[:, 2]
        _1_z_d = 1 - z_d

        p0 = _1_x_d * _1_y_d * _1_z_d
        p1 = _1_x_d * _1_y_d * z_d
        p2 = _1_x_d * y_d * _1_z_d
        p3 = _1_x_d * y_d * z_d
        p4 = x_d * _1_y_d * _1_z_d
        p5 = x_d * _1_y_d * z_d
        p6 = x_d * y_d * _1_z_d
        p7 = x_d * y_d * z_d

        coeffs = torch.stack((p0, p1, p2, p3, p4, p5, p6, p7), dim=0).T.reshape(-1, 1)

        return coeffs
    
    def update(self, points: torch.Tensor):
        if points.shape[0] == 0:
            return
        
        # update bbox
        self._update_bbox(points)

        # update hash grid
        for level in range(self.num_resolution):
            current_resolution = self.min_resolution * (self.scale_up ** level)

            # get corners
            corners = self._get_corners(points, current_resolution)

            # get unique corners
            offset = corners.min(dim=0).values
            nx, ny, _ = (corners - offset).max(dim=0).values
            flat_corners = corners[:, 0] + corners[:, 1] * (nx + 1) + corners[:, 2] * (nx + 1) * (ny + 1)
            uniques, indices, counts = torch.unique(
                flat_corners, return_inverse=True, return_counts=True
            )
            sums = torch.zeros((uniques.shape[0], 3), dtype=self.dtype, device=self.device)
            sums.scatter_add_(0, indices[:, None].expand(-1, 3), corners)
            unique_corners = sums / counts[:, None]
            
            # get hash keys
            keys = (unique_corners.to(self.primes) * self.primes).sum(dim=-1) % self.buffer_size
            update_mask = (self.grid_indices[level][keys] == -1)
            num_new = update_mask.sum().item()

            # update hash table
            new_keys = keys[update_mask]
            self.grid_indices[level][new_keys] = torch.arange(
                num_new, dtype=self.idx_dtype, device=self.device
            ) + self.grid_features[level].shape[0]

            # update features
            new_features = torch.randn(
                num_new + 1, self.feature_dim, dtype=self.dtype, device=self.device
            ) * self.feature_std
            self.grid_features[level] = nn.Parameter(
                torch.cat([self.grid_features[level][:-1], new_features], dim=0)
            )

    def query_feature(self, points: torch.Tensor, min_corners: int = 8):
        N = points.shape[0]
        total_features = torch.zeros(
            (N, self.feature_dim), dtype=self.dtype, device=self.device
        )
        total_mask = torch.ones(
            (N,), dtype=torch.bool, device=self.device
        )

        for level in range(self.num_resolution):
            current_resolution = self.min_resolution * (self.scale_up ** level)

            # get corners
            corners = self._get_corners(points, current_resolution)

            # get features
            keys = (corners.to(self.primes) * self.primes).sum(dim=-1) % self.buffer_size
            indices = self.grid_indices[level][keys].reshape(-1, 8)
            valid_mask = (indices > -1).sum(dim=1) >= min_corners
            valid_indices = indices[valid_mask].reshape(-1, 1).squeeze(1)
            features = self.grid_features[level][valid_indices]

            # get coeffs
            coeffs = self._get_coeffs(points[valid_mask], current_resolution)
            coeffs[valid_indices == -1] = 0.0
            coeffs = coeffs.reshape(-1, 8)
            coeffs = coeffs / (torch.sum(coeffs, dim=1, keepdim=True) + 1e-15)
            coeffs = coeffs.reshape(-1, 1)

            # total features
            total_features[valid_mask] += (features * coeffs).reshape(-1, 8, self.feature_dim).sum(dim=1)
            # total mask
            total_mask[~valid_mask] = False

        return total_features, total_mask
    
    def query_mask(self, points: torch.Tensor, min_corners: int = 8):
        N = points.shape[0]
        total_mask = torch.ones(
            (N,), dtype=torch.bool, device=self.device
        )
        
        # get corners
        corners = self._get_corners(points, self.min_resolution)

        # get mask
        keys = (corners.to(self.primes) * self.primes).sum(dim=-1) % self.buffer_size
        indices = self.grid_indices[0][keys].reshape(-1, 8)
        valid_mask = (indices > -1).sum(dim=1) >= min_corners

        # total mask
        total_mask[~valid_mask] = False

        return total_mask