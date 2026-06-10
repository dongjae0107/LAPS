from tqdm import tqdm

import torch
import torch.optim as optim

from map.hash_grid import HashGrid
from map.decoder import Decoder
from map.loss import *

from utils.config import Config
from utils.tools import *

from utils.data_loader import DataLoader
from utils.data_sampler import DataSampler
from utils.data_manager import DataManager
from utils.mesher import Mesher

class Mapper:
    def __init__(self, config: Config):
        self.config = config

        self.dtype = config.dtype
        self.device = config.device

        self.hash_grid = HashGrid(config)
        self.decoder = Decoder(config)

        self.dataset = DataLoader(config)
        self.sampler = DataSampler(config)
        self.manager = DataManager(config, self.hash_grid, self.decoder)

        self.mesher = Mesher(config, self.hash_grid, self.decoder)

        self.T_preproc = 0.0
        self.T_ap = 0.0
        self.T_as = 0.0
        self.T_opt = 0.0

    def process_frame(self, frame_id: int, frame_points: torch.Tensor, frame_normals: torch.Tensor, frame_position: torch.Tensor):
        # local window
        dists = torch.linalg.norm(frame_points - frame_position, ord=2, dim=1, keepdim=False)
        mask = (dists > self.config.min_range) & (dists < self.config.max_range)
        frame_points = frame_points[mask]
        frame_normals = frame_normals[mask]

        # voxel downsample
        if self.config.voxel_downsample:
            mask = voxel_downsample_torch(frame_points, self.config.voxel_downsample_size)
            frame_points = frame_points[mask]
            frame_normals = frame_normals[mask]

        # training samples
        samples, labels, normals, weights = self.sampler.sample_along_ray(frame_points, frame_normals, frame_position)

        # update hash grid
        samples_for_update = samples[torch.abs(labels) < self.config.truncation_range]
        self.hash_grid.update(samples_for_update)

        # active pooling
        T_ap_0 = get_time()
        self.manager.update_pool(frame_id, frame_position, samples, labels, normals, weights)
        T_ap_1 = get_time()
        self.T_ap += T_ap_1 - T_ap_0

    def train_frame(self, frame_id: int):
        if frame_id == self.config.freeze_frame:
            freeze_model(self.decoder)

        hash_grid_params = list(self.hash_grid.parameters())
        decoder_params = list(self.decoder.parameters())

        hash_grid_params_opt_dict = {
            'params': hash_grid_params, 
            'lr': self.config.learning_rate
        }
        decoder_params_opt_dict = {
            'params': decoder_params, 
            'lr': self.config.learning_rate
        }

        opt = optim.Adam(
            [hash_grid_params_opt_dict, decoder_params_opt_dict],
            betas = (0.9, 0.99),
            eps = 1e-15
        )

        for iter in range(self.config.iters):
            # active sampling
            T_as_0 = get_time()
            iter_samples, iter_labels = self.manager.get_batch(frame_id, iter)
            T_as_1 = get_time()

            iter_labels = torch.clamp(iter_labels, -self.config.truncation_range, self.config.truncation_range)

            features, mask = self.hash_grid.query_feature(iter_samples)
            preds = self.decoder(features)

            loss = 0.0
            if self.config.loss_type == 'l1':
                sdf_loss = sdf_l1_loss(preds, iter_labels)
            elif self.config.loss_type == 'l2':
                sdf_loss = sdf_l2_loss(preds, iter_labels)
            elif self.config.loss_type == 'huber':
                sdf_loss = sdf_huber_loss(preds, iter_labels)
            else:
                raise NotImplementedError(f"Loss type {self.config.loss_type} not implemented.")
            loss += sdf_loss

            # optimize
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            opt.step()

            self.T_as += T_as_1 - T_as_0

    def mapping(self):
        T_start = get_time()

        for frame_id in tqdm(range(self.dataset.seq_len)):
            T0 = get_time()
            # load frame
            frame_points, frame_normals, frame_position = self.dataset.get_frame(frame_id)
            T1 = get_time()
            # preprocess frame
            self.process_frame(frame_id, frame_points, frame_normals, frame_position)
            T2 = get_time()
            # train frame
            self.train_frame(frame_id)
            T3 = get_time()

            self.T_preproc += T2 - T0
            self.T_opt += T3 - T2
            if self.config.verbose:
                print(f"Frame {frame_id:03d}/{self.dataset.seq_len:03d} | Data loading: {T1 - T0:.3f}s | Preprocessing: {T2 - T1:.3f}s | Training: {T3 - T2:.3f}s")

        T_end = get_time()

        total_time = T_end - T_start
        avg_total = total_time / self.dataset.seq_len

        self.T_preproc = self.T_preproc - self.T_ap
        self.T_opt = self.T_opt - self.T_as

        self.T_preproc /= self.dataset.seq_len
        self.T_ap /= self.dataset.seq_len
        self.T_as /= self.dataset.seq_len
        self.T_opt /= self.dataset.seq_len

        print("\nRuntime summary")
        print("-" * 65)
        print(f"{'Module':<30} {'Avg. Time [s/frame]':>20}")
        print("-" * 65)
        print(f"{'Preprocessing':<30} {self.T_preproc:>20.6f}")
        print(f"{'Active pool update':<30} {self.T_ap:>20.6f}")
        print(f"{'Active sampling':<30} {self.T_as:>20.6f}")
        print(f"{'Training':<30} {self.T_opt:>20.6f}")
        print("-" * 65)
        print(f"{'Total average':<30} {avg_total:>20.6f}")
        print(f"{'Total mapping time':<30} {total_time:>20.2f}")
        print("-" * 65)

        # Free memory
        self.manager.free_pool()

        # Reconstruct mesh
        self.mesher.reconstruct_mesh()