import os
import math
import torch
import quaternion
import numpy as np
import open3d as o3d

from utils.config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.device = config.device
        self.dtype = config.dtype

        self.dataset_name = config.dataset_name

        if config.calib_path:
            self.calib = self._load_calib(config.calib_path, self.dataset_name)
        else:
            self.calib = np.eye(4)

        self.poses = self._load_poses(config.pose_path, self.dataset_name)
        assert len(self.poses) > 0, "No poses found."

        self.scan_folder = config.scan_path
        self.scans = os.listdir(self.scan_folder)
        self.scans.sort(key=lambda x: int(x.split('.')[0]))
        assert len(self.scans) > 0, "No scans found."

        # dirty fix for maicity
        if self.dataset_name == "maicity":
            num_scans = len(self.scans)
            self.poses = self.poses[:num_scans]

        assert len(self.scans) == len(self.poses), "Number of scans and poses do not match."

        self.begin_frame = config.begin_frame
        self.end_frame = config.end_frame
        self.skip_frame = config.skip_frame

        self.dataset_len = len(self.scans)
        if self.end_frame == -1:
            self.seq_len = math.ceil((self.dataset_len - self.begin_frame) / self.skip_frame)
        else:
            self.seq_len = math.ceil((min(self.end_frame, self.dataset_len) - self.begin_frame) / self.skip_frame)

    def _load_calib(self, calib_path: str, dataset_name: str) -> np.ndarray:
        calib = np.eye(4)

        with open(calib_path, 'r') as f:
            lines = f.readlines()

            if dataset_name in ["maicity", "ncd"]:
                for line in lines:
                    if line.startswith("Tr:"):
                        values = [float(v) for v in line.strip().split()[1:]]
                        calib[0, 0:4] = values[0:4]
                        calib[1, 0:4] = values[4:8]
                        calib[2, 0:4] = values[8:12]

            elif dataset_name == "spires":
                for i, line in enumerate(lines):
                    values = [float(v) for v in line.split()]
                    calib[i, 0:4] = values[0:4]

        return calib
    
    def _load_poses(self, pose_path: str, dataset_name: str) -> list:
        poses = []

        with open(pose_path, "r") as f:
            lines = f.readlines()

            # KITTI format (SE(3) matrix)
            if dataset_name in ["maicity", "ncd"]:
                for line in lines:
                    values = [float(v) for v in line.strip().split()]

                    T = np.eye(4)
                    T[0, 0:4] = values[0:4]
                    T[1, 0:4] = values[4:8]
                    T[2, 0:4] = values[8:12]

                    Tr = self.calib
                    Tr_inv = np.linalg.inv(Tr)
                    T = np.matmul(Tr_inv, np.matmul(T, Tr))

                    poses.append(T)
                    
            # TUM format (x, y, z, qx, qy, qz, qw)
            elif dataset_name == "spires":
                # Oxford Spires: idx timestamp_s timestamp_ns tx ty tz qx qy qx qw
                for line in lines:
                    values = [float(v) for v in line.strip().split(',')]

                    t = np.array(values[3:6])
                    q = np.quaternion(values[9], values[6], values[7], values[8])
                    R = quaternion.as_rotation_matrix(q)

                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t

                    T = np.matmul(self.calib, T)
                    
                    poses.append(T)

        return poses
    
    def _load_scan(self, scan_path: str) -> torch.Tensor:
        if ".bin" in scan_path:
            points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            points = points[:, :3]
            pcd = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(points)
            )
        elif ".pcd" in scan_path or ".ply" in scan_path:
            pcd = o3d.io.read_point_cloud(scan_path)
            points = np.asarray(pcd.points)

        points = torch.from_numpy(points).to(dtype=self.dtype, device=self.device)

        # estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=20))
        normals = np.asarray(pcd.normals)
        normals = torch.from_numpy(normals).to(dtype=self.dtype, device=self.device)

        return points, normals
    
    def get_frame(self, idx: int):
        frame_id = self.begin_frame + idx * self.skip_frame
        frame_path = os.path.join(self.scan_folder, self.scans[frame_id])

        frame_points, frame_normals = self._load_scan(frame_path)
        frame_pose = torch.tensor(self.poses[frame_id], dtype=self.dtype, device=self.device)

        frame_points_h = torch.cat(
            (frame_points, torch.ones((frame_points.shape[0], 1), dtype=self.dtype, device=self.device)), dim=1
        )
        frame_points = torch.matmul(frame_pose, frame_points_h.T).T[:, :3]

        frame_normals_h = torch.cat(
            (frame_normals, torch.zeros((frame_normals.shape[0], 1), dtype=self.dtype, device=self.device)), dim=1
        )
        frame_normals = torch.matmul(frame_pose, frame_normals_h.T).T[:, :3]

        frame_position = frame_pose[:3, 3]

        return frame_points, frame_normals, frame_position