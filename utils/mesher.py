import os
import math
import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d
import skimage.measure

from utils.config import Config

from map.hash_grid import HashGrid
from map.decoder import Decoder


class Mesher:
    def __init__(self, config: Config, hash_grid: HashGrid, decoder: Decoder):
        self.config = config
        self.dtype = config.dtype
        self.device = config.device

        self.hash_grid = hash_grid
        self.decoder = decoder        

        self.save_pcd = config.save_pcd
        self.viz_pcd = config.viz_pcd

        self.mesh_resolution = self.config.mesh_resolution
        self.save_mesh = config.save_mesh
        self.viz_mesh = config.viz_mesh

    def _get_grid_chunks(self, bbox: np.ndarray):
        min_bound, max_bound = bbox
        grid_len = max_bound - min_bound
        grid_num = (np.ceil(grid_len / self.mesh_resolution) + 1).astype(np.int_)

        chunks = []
        chunk_size = 300
        for i in range(0, grid_num[0], chunk_size):
            for j in range(0, grid_num[1], chunk_size):
                for k in range(0, grid_num[2], chunk_size):
                    chunk = (
                        slice(i, min(i + chunk_size, grid_num[0])),
                        slice(j, min(j + chunk_size, grid_num[1])),
                        slice(k, min(k + chunk_size, grid_num[2]))
                    )
                    chunks.append(chunk)

        return chunks

    def _get_grid_coords(self, bbox: np.ndarray):
        min_bound = bbox[0]
        max_bound = bbox[1]

        grid_len = max_bound - min_bound
        grid_num = (np.ceil(grid_len / self.mesh_resolution) + 1).astype(np.int_)
        grid_origin = min_bound - self.mesh_resolution

        x = torch.arange(grid_num[0], dtype=torch.int16)
        y = torch.arange(grid_num[1], dtype=torch.int16)
        z = torch.arange(grid_num[2], dtype=torch.int16)
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')

        grid_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()))
        grid_coords = torch.transpose(grid_coords, 0, 1).float()
        grid_coords = grid_coords * self.mesh_resolution + torch.tensor(grid_origin, dtype=torch.float32)

        return grid_coords, grid_num, grid_origin

    def _query_grid_coords(self, grid_coords: torch.Tensor):
        N = grid_coords.shape[0]
        num_iters = math.ceil(N / self.config.batch_size)

        sdf = np.zeros(N)
        mask = np.zeros(N)

        with torch.no_grad():
            for iter in tqdm(range(num_iters), disable=True):
                head = iter * self.config.batch_size
                tail = min((iter + 1) * self.config.batch_size, N)

                batch_coords = grid_coords[head:tail].to(self.device)

                batch_features, batch_mask = self.hash_grid.query_feature(batch_coords, min_corners=1)
                batch_sdf = self.decoder(batch_features[batch_mask])
                sdf[head:tail][batch_mask.detach().cpu().numpy()] = batch_sdf.detach().cpu().numpy()

                batch_mask = self.hash_grid.query_mask(batch_coords, min_corners=8)
                mask[head:tail] = batch_mask.detach().cpu().numpy()
        
        return sdf, mask
    
    def _get_pcd_with_sdf(self, grid_coords: torch.Tensor, grid_sdf: np.array, grid_mask: np.array):
        pcd_with_sdf = o3d.t.geometry.PointCloud()
        grid_coords_np = grid_coords.cpu().numpy()[grid_mask > 0]
        grid_sdf_np = grid_sdf.reshape(-1, 1)[grid_mask > 0]

        pcd_with_sdf.point['positions'] = o3d.core.Tensor(grid_coords_np, dtype=o3d.core.float32, device=o3d.core.Device("CPU:0"))
        pcd_with_sdf.point['intensity'] = o3d.core.Tensor(grid_sdf_np, dtype=o3d.core.float32, device=o3d.core.Device("CPU:0")) # save sdf as intensity

        return pcd_with_sdf

    def _marching_cubes(self, sdf: np.array, mask: np.array, origin: np.array):
        verts, faces = np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                sdf, level=0.0, allow_degenerate=False, mask=mask.astype(bool)
            )
        except:
            pass

        verts = origin + verts * self.mesh_resolution

        return verts, faces
    
    def _remove_floaters(self, mesh: o3d.geometry.TriangleMesh, threshold: int = 300):
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
        mesh.remove_triangles_by_mask(triangles_to_remove)

        return mesh
    
    def reconstruct_mesh(self):        
        bbox = self.hash_grid.bbox.cpu().numpy()
        min_bound, max_bound = bbox
        grid_len = max_bound - min_bound
        grid_num = (np.ceil(grid_len / self.mesh_resolution) + 1).astype(np.int_)

        bboxs = []
        total_voxels = np.prod(grid_num)

        # TODO: need better solution for faster meshing when the grid is too large
        if total_voxels > 512 ** 3:
            chunks = self._get_grid_chunks(bbox)
            for chunk in chunks:
                start = np.array([chunk[0].start, chunk[1].start, chunk[2].start])
                stop = np.array([chunk[0].stop, chunk[1].stop, chunk[2].stop])
                chunk_bbox = np.stack([
                    min_bound + start * self.mesh_resolution,
                    min_bound + stop * self.mesh_resolution
                ])
                bboxs.append(chunk_bbox)
        else:
            bboxs.append(bbox)

        mesh = o3d.geometry.TriangleMesh()

        for bbox in tqdm(bboxs, desc="Reconstructing Mesh"):
            grid_coords, grid_num, grid_origin = self._get_grid_coords(bbox)
            grid_sdf, grid_mask = self._query_grid_coords(grid_coords)
            grid_sdf = grid_sdf.reshape(grid_num[0], grid_num[1], grid_num[2])
            grid_mask = grid_mask.reshape(grid_num[0], grid_num[1], grid_num[2])
            
            verts, faces = self._marching_cubes(grid_sdf, grid_mask, grid_origin)

            cur_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(verts),
                triangles=o3d.utility.Vector3iVector(faces)
            )

            mesh += cur_mesh

        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        mesh = self._remove_floaters(mesh, threshold=300)

        if self.save_mesh or self.viz_mesh:
            if self.viz_mesh:
                o3d.visualization.draw_geometries([mesh])
            if self.save_mesh:
                mesh_file = os.path.join(self.config.experiment_path, 'mesh.ply')
                o3d.io.write_triangle_mesh(mesh_file, mesh)

    def recon_mesh_simple(self):        
        bbox = self.hash_grid.bbox.cpu().numpy()
        
        grid_coords, grid_num, grid_origin = self._get_grid_coords(bbox)
        grid_sdf, grid_mask = self._query_grid_coords(grid_coords)

        grid_sdf = grid_sdf.reshape(grid_num[0], grid_num[1], grid_num[2])
        grid_mask = grid_mask.reshape(grid_num[0], grid_num[1], grid_num[2])
        verts, faces = self._marching_cubes(grid_sdf, grid_mask, grid_origin)

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        mesh = self._remove_floaters(mesh, threshold=300)

        if self.save_mesh or self.viz_mesh:
            if self.viz_mesh:
                o3d.visualization.draw_geometries([mesh])
            if self.save_mesh:
                mesh_file = os.path.join(self.config.experiment_path, 'mesh.ply')
                o3d.io.write_triangle_mesh(mesh_file, mesh)