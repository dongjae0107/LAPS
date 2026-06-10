import torch

from utils.config import Config


class DataSampler:
    def __init__(self, config: Config):
        self.dtype = config.dtype
        self.device = config.device

        self.max_range = config.max_range
        self.truncation_range = config.truncation_range

        self.front_sample_num = config.front_sample_num
        self.behind_sample_num = config.behind_sample_num
        self.free_sample_num = config.free_sample_num

        self.total_sample_num = (
            1 + self.front_sample_num + self.behind_sample_num + self.free_sample_num
        )

    def sample_along_ray(self, points: torch.Tensor, normals: torch.Tensor, sensor_origin: torch.Tensor):
        N = points.shape[0]

        shifted_points = points - sensor_origin
        dists = torch.linalg.norm(shifted_points, ord=2, dim=1, keepdim=True)

        sensor_origins = sensor_origin.repeat(N, 1)

        # (1) surface samples - measured points
        measured_sample_ratios = torch.ones_like(dists)
        measured_sample_sdf = torch.zeros_like(dists)

        # (2-1) near surface samples - front
        front_samples = torch.rand(N * self.front_sample_num, 1, dtype=self.dtype, device=self.device) * self.truncation_range
        repeated_dists = dists.repeat(1, self.front_sample_num).reshape(-1, 1)
        front_sample_ratios = 1.0 - (front_samples / repeated_dists)
        front_sample_ratios = front_sample_ratios.reshape(N, -1)
        front_sample_sdf = front_samples.reshape(N, -1)

        # (2-2) near surface samples - behind
        behind_samples = -1.0 * torch.rand(N * self.behind_sample_num, 1, dtype=self.dtype, device=self.device) * self.truncation_range
        repeated_dists = dists.repeat(1, self.behind_sample_num).reshape(-1, 1)
        behind_sample_ratios = 1.0 - (behind_samples / repeated_dists)
        behind_sample_ratios = behind_sample_ratios.reshape(N, -1)
        behind_sample_sdf = behind_samples.reshape(N, -1)

        # (3) free samples
        free_samples = torch.rand(N * self.free_sample_num, 1, dtype=self.dtype, device=self.device)
        repeated_dists = dists.repeat(1, self.free_sample_num).reshape(-1, 1)
        free_max_ratio = 1.0 - 2.0 * self.truncation_range / repeated_dists
        free_min_ratio = 0.5
        free_diff_ratio = free_max_ratio - free_min_ratio
        free_sample_ratios = free_min_ratio + (free_samples * free_diff_ratio)
        free_sample_sdf = (1.0 - free_sample_ratios) * repeated_dists
        free_sample_ratios = free_sample_ratios.reshape(N, -1)
        free_sample_sdf = free_sample_sdf.reshape(N, -1)

        # samples
        total_ratios = torch.cat(
            [
                measured_sample_ratios,
                front_sample_ratios,
                behind_sample_ratios,
                free_sample_ratios,
            ],
            dim=1,
        ).reshape(-1, 1)
        sample_points = shifted_points.repeat(1, self.total_sample_num).reshape(-1, 3)
        sample_positions = sensor_origins.repeat(1, self.total_sample_num).reshape(-1, 3)
        samples = sample_positions + (sample_points * total_ratios)

        # labels
        total_sdf = torch.cat(
            [
                measured_sample_sdf,
                front_sample_sdf,
                behind_sample_sdf,
                free_sample_sdf,
            ],
            dim=1,
        ).reshape(-1, 1)
        labels = total_sdf.squeeze()

        # normals
        normals = normals.repeat(1, self.total_sample_num).reshape(-1, 3)

        # weights
        dirs = shifted_points / dists
        dirs = dirs.repeat(1, self.total_sample_num).reshape(-1, 3)
        bias_ang = 1 - torch.abs(torch.sum(dirs * normals, dim=1, keepdim=False))

        dists = dists.repeat(1, self.total_sample_num).reshape(-1)
        sigma_dist = dists / self.max_range

        weights = sigma_dist**2 + bias_ang**2 # MSE
        weights += torch.randn(samples.shape[0], dtype=self.dtype, device=self.device) * 1e-3 # small random perturbation to seperate samples with same MSE

        # Random weights for ablation
        # weights = torch.randn(samples.shape[0], dtype=self.dtype, device=self.device)

        return samples, labels, normals, weights