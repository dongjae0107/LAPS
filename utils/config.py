import os
import yaml
import torch


class Config:
    def __init__(self):
        # setting
        self.dtype: torch.dtype = torch.float32
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.random_seed: int = 42

        self.dataset_name: str = ""
        self.scan_path: str = ""
        self.pose_path: str = ""
        self.calib_path: str = ""
        self.output_path: str = ""
        self.experiment_path: str = ""

        self.begin_frame: int = 0
        self.end_frame: int = -1
        self.skip_frame: int = 1

        # preprocess
        self.min_range: float = 0.0
        self.max_range: float = 100.0
        self.voxel_downsample: bool = True
        self.voxel_downsample_size: float = 0.1

        # hash grid
        self.min_resolution: float = 0.3
        self.num_resolution: int = 2
        self.scale_up: float = 1.5

        self.buffer_size: int = int(5e7)

        # decoder
        self.feature_dim: int = 8
        self.feature_std: float = 0.0
        self.hidden_dim: int = 32
        self.num_layers: int = 2
        
        self.freeze_frame: int = 20

        # sampler
        self.truncation_range: float = 0.3
        self.front_sample_num: int = 2
        self.behind_sample_num: int = 2
        self.free_sample_num: int = 2

        # manager
        self.active_pooling: bool = True
        self.max_sample_per_voxel: int = 100
        self.active_sampling: bool = True
        self.save_log_pool: bool = True

        # training
        self.iters: int = 30
        self.batch_size: int = 16384
        self.learning_rate: float = 1e-2
        self.loss_type: str = "l2" # l1, l2, huber

        # output
        self.verbose: bool = True

        self.save_pcd: bool = True
        self.viz_pcd: bool = True

        self.mesh_resolution: float = 0.1
        self.save_mesh: bool = True
        self.viz_mesh: bool = True

    def load_from_file(self, config_path: str):
        config_args = yaml.safe_load(open(os.path.abspath(config_path)))

        self.dtype = config_args["setting"].get("dtype", torch.float32)
        self.device = config_args["setting"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # setting
        if "setting" in config_args:
            self.dataset_name = config_args["setting"].get("dataset_name", "")
            self.scan_path = config_args["setting"].get("scan_path", "")
            self.pose_path = config_args["setting"].get("pose_path", "")
            self.calib_path = config_args["setting"].get("calib_path", "")
            self.output_path = config_args["setting"].get("output_path", "")

            self.begin_frame = config_args["setting"].get("begin_frame", 0)
            self.end_frame = config_args["setting"].get("end_frame", -1)
            self.skip_frame = config_args["setting"].get("skip_frame", 1)

        # preprocess
        if "preprocess" in config_args:
            self.min_range = config_args["preprocess"].get("min_range", 0.0)
            self.max_range = config_args["preprocess"].get("max_range", 100.0)
            self.voxel_downsample = config_args["preprocess"].get("voxel_downsample", True)
            self.voxel_downsample_size = config_args["preprocess"].get("voxel_downsample_size", 0.1)

        # hash grid
        if "hash_grid" in config_args:
            self.min_resolution = config_args["hash_grid"].get("min_resolution", 0.3)
            self.num_resolution = config_args["hash_grid"].get("num_resolution", 2)
            self.scale_up = config_args["hash_grid"].get("scale_up", 1.5)

            self.buffer_size = config_args["hash_grid"].get("buffer_size", int(5e7))

        # decoder
        if "decoder" in config_args:
            self.feature_dim = config_args["decoder"].get("feature_dim", 8)
            self.feature_std = config_args["decoder"].get("feature_std", 0.0)
            self.hidden_dim = config_args["decoder"].get("hidden_dim", 32)
            self.num_layers = config_args["decoder"].get("num_layers", 2)
            self.freeze_frame = config_args["decoder"].get("freeze_frame", 20)

        # sampler
        if "sampler" in config_args:
            self.truncation_range = config_args["sampler"].get("truncation_range", 0.3)
            self.front_sample_num = config_args["sampler"].get("front_sample_num", 2)
            self.behind_sample_num = config_args["sampler"].get("behind_sample_num", 2)
            self.free_sample_num = config_args["sampler"].get("free_sample_num", 2)

        # manager
        if "manager" in config_args:
            self.active_pooling = config_args["manager"].get("active_pooling", True)
            self.max_sample_per_voxel = config_args["manager"].get("max_sample_per_voxel", 100)
            self.active_sampling = config_args["manager"].get("active_sampling", True)
            self.save_log_pool = config_args["manager"].get("save_log_pool", True)

        # training
        if "training" in config_args:
            self.iters = config_args["training"].get("iters", 30)
            self.batch_size = config_args["training"].get("batch_size", 16384)
            self.learning_rate = config_args["training"].get("learning_rate", 1e-2)

        # output
        if "output" in config_args:
            self.verbose = config_args["output"].get("verbose", True)

            self.save_pcd = config_args["output"].get("save_pcd", True)
            self.viz_pcd = config_args["output"].get("viz_pcd", True)

            self.save_mesh = config_args["output"].get("save_mesh", True)
            self.viz_mesh = config_args["output"].get("viz_mesh", True)