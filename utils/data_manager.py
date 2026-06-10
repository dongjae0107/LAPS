import copy
import torch

from map.hash_grid import HashGrid
from map.decoder import Decoder

from utils.config import Config
from utils.tools import *


class DataManager:
    def __init__(self, config: Config, hash_grid: HashGrid, decoder: Decoder):
        # setting
        self.dtype = config.dtype
        self.device = config.device
        self.idx_dtype = torch.int64

        self.hash_grid = hash_grid
        self.decoder = decoder

        # pooling
        self.truncation_range = config.truncation_range
        self.max_range = config.max_range + self.truncation_range
        self.active_pooling = config.active_pooling
        self.max_sample_per_voxel = config.max_sample_per_voxel

        # sampling
        self.batch_size = config.batch_size
        self.active_sampling = config.active_sampling

        # logging
        self.log_pool = config.save_log_pool
        if self.log_pool:
            self.log_pool_file = open(config.experiment_path + "/log_pool.txt", "w")
            self.log_pool_file.write("# ID NUM_SAMPLES MEMORY(MB) MAX_SAMPLES MIN_SAMPLES AVG_SAMPLES\n")

        self.init_pool()

    def init_pool(self):
        self.sample_pool = torch.empty((0,3), dtype=self.dtype, device=self.device)
        self.label_pool = torch.empty((0,), dtype=self.dtype, device=self.device)
        self.normal_pool = torch.empty((0,3), dtype=self.dtype, device=self.device)
        self.key_pool = torch.empty((0,), dtype=self.dtype, device=self.device)
        self.weight_pool = torch.empty((0,), dtype=self.dtype, device=self.device)

    def free_pool(self):
        self.init_pool()
        if self.log_pool:
            self.log_pool_file.close()

    @torch.no_grad()
    def update_pool(
        self, 
        frame_id: int, 
        frame_position: torch.Tensor, 
        samples: torch.Tensor, 
        labels: torch.Tensor, 
        normals: torch.Tensor,
        weights: torch.Tensor
    ):
        ## sliding window
        if frame_id > 0:
            keep_mask = self._sliding_window(self.sample_pool, frame_position)
            self.sample_pool = self.sample_pool[keep_mask]
            self.label_pool = self.label_pool[keep_mask]
            self.normal_pool = self.normal_pool[keep_mask]
            self.key_pool = self.key_pool[keep_mask]
            self.weight_pool = self.weight_pool[keep_mask]

        ## check valid samples
        valid_mask = self.hash_grid.query_mask(samples)
        samples = samples[valid_mask]
        labels = labels[valid_mask]
        normals = normals[valid_mask]

        ## compute keys & weights for new samples
        new_keys = self._compute_keys(samples)
        new_weights = weights[valid_mask]

        if frame_id == 0 or not self.active_pooling:
            self.sample_pool = torch.cat((self.sample_pool, samples), dim=0)
            self.label_pool = torch.cat((self.label_pool, labels), dim=0)
            self.normal_pool = torch.cat((self.normal_pool, normals), dim=0)
            self.key_pool = torch.cat((self.key_pool, new_keys), dim=0)
            self.weight_pool = torch.cat((self.weight_pool, new_weights), dim=0)
        else:
            ## --- Active Pooling ---
            all_samples = torch.cat((self.sample_pool, samples), dim=0)
            all_labels = torch.cat((self.label_pool, labels), dim=0)
            all_normals = torch.cat((self.normal_pool, normals), dim=0)
            all_keys = torch.cat((self.key_pool, new_keys), dim=0)
            all_weights = torch.cat((self.weight_pool, new_weights), dim=0)

            # sort by key first, then by weight (ascending)
            sort_keys = all_keys.to(self.idx_dtype) * (10**10) + all_weights
            # sorted_indices = torch.argsort(sort_keys)
            sorted_indices = torch.argsort(sort_keys, descending=True) # descending to keep most uncertain samples
            all_samples = all_samples[sorted_indices]
            all_labels = all_labels[sorted_indices]
            all_normals = all_normals[sorted_indices]
            all_keys = all_keys[sorted_indices]
            all_weights = all_weights[sorted_indices]

            _, counts = torch.unique_consecutive(all_keys, return_counts=True)
            cum = torch.cumsum(counts, dim=0)
            idx = torch.arange(all_keys.shape[0], device=self.device)
            rel = idx - torch.repeat_interleave(cum - counts, counts)
            keep_mask = rel < self.max_sample_per_voxel

            self.sample_pool = all_samples[keep_mask]
            self.label_pool = all_labels[keep_mask]
            self.normal_pool = all_normals[keep_mask]
            self.key_pool = all_keys[keep_mask]
            self.weight_pool = all_weights[keep_mask]

        self._log_pool(frame_id)

    def _sliding_window(self, points: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        dists = torch.linalg.norm(points - origin, ord=2, dim=1, keepdim=False)
        return dists < self.max_range
    
    def _compute_keys(self, samples: torch.Tensor) -> torch.Tensor:
        voxel_origins = torch.floor(samples / self.hash_grid.min_resolution)
        keys = (voxel_origins.to(self.idx_dtype) * self.hash_grid.primes).sum(dim=1) % self.hash_grid.buffer_size
        return keys
    
    def _log_pool(self, frame_id: int):
        if not self.log_pool:
            return
        
        mem = self.sample_pool.element_size() * self.sample_pool.nelement()
        mem_mb = round(mem / (1024 ** 2), 2)

        uniques, counts = torch.unique(self.key_pool, return_counts=True)
        max_samples = counts.max().item()
        min_samples = counts.min().item()
        avg_samples = counts.float().mean().item()

        self.log_pool_file.write(f"{frame_id}, {self.sample_pool.shape[0]}, {mem_mb}, {max_samples}, {min_samples}, {avg_samples}\n")
        self.log_pool_file.flush()

    def get_batch(self, frame_id: int, iter: int):
        ## --- Random Sampling ---
        if frame_id < 5 or not self.active_sampling:
            # Random Sampling
            rand_idx = torch.randint(
                0, self.sample_pool.shape[0], (self.batch_size,), device=self.device
            )
            return self.sample_pool[rand_idx], self.label_pool[rand_idx]
        
        ## --- Active Sampling ---
        if iter == 0:
            voxel_origins = torch.floor(self.sample_pool / self.hash_grid.min_resolution)
            offset = voxel_origins.min(dim=0).values
            shifted_origins = voxel_origins - offset
            nx, ny, _ = shifted_origins.max(dim=0).values + 1
            flat_origins = shifted_origins[:, 0] + shifted_origins[:, 1] * nx + shifted_origins[:, 2] * nx * ny

            uniques, indices, counts = torch.unique(
                flat_origins, return_inverse=True, return_counts=True
            )
            sums = torch.zeros((uniques.shape[0], 3), dtype=self.dtype, device=self.device)
            sums.scatter_add_(0, indices[:, None].expand(-1, 3), voxel_origins)
            voxel_centers = sums / counts[:, None]
            voxel_centers = (voxel_centers + 0.5) * self.hash_grid.min_resolution

            self._voxel_cache = {
                "voxel_centers": voxel_centers,
                "uniques": uniques,
                "indices": indices,
                "counts": counts
            }

            uncertainty = self.get_uncertainty(voxel_centers)
            thresh = 0.98*1e3
            voxel_is_uncertain = (uncertainty > thresh)
            uncertain_mask = voxel_is_uncertain[indices]
            self.u_idx = torch.nonzero(uncertain_mask, as_tuple=False).squeeze(1)
            self.c_idx = torch.nonzero(~uncertain_mask, as_tuple=False).squeeze(1)

        num_uncertain = min(1000, self.u_idx.shape[0])
        num_certain = self.batch_size - num_uncertain

        if self.u_idx.shape[0] > 0:
            rand_idx = torch.randint(0, self.u_idx.shape[0], (num_uncertain,), device=self.device)
            idx_uncertain = self.u_idx[rand_idx]

            idx_certain = torch.randint(0, self.sample_pool.shape[0], (num_certain,), device=self.device)

            total_idx = torch.cat((idx_uncertain, idx_certain), dim=0)
        else:
            rand_idx = torch.randint(0, self.sample_pool.shape[0], (self.batch_size,), device=self.device)
            total_idx = rand_idx

        samples = self.sample_pool[total_idx]
        labels = self.label_pool[total_idx]

        return samples, labels
    
    def get_uncertainty(self, points: torch.Tensor):
        hash_grid_copy = copy.deepcopy(self.hash_grid)
        decoder_copy = copy.deepcopy(self.decoder)

        samples = points.detach()
        perturbs = torch.zeros_like(samples, dtype=self.dtype, device=self.device)
        perturbs.requires_grad = True
        perturbed_samples = samples + perturbs

        perturbed_features, _ = hash_grid_copy.query_feature(perturbed_samples)
        perturbed_sdf = decoder_copy(perturbed_features)
        perturbed_sdf = torch.sum(perturbed_sdf)
        perturbed_sdf.backward()

        grad = perturbs.grad.clone().detach().view(-1, 3)
        perturbs.grad.zero_()

        hessian = torch.sum(grad**2, dim=1)
        uncertainty = 1 / (hessian + 1e-2)

        return uncertainty