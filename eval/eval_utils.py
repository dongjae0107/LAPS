import tabulate
import numpy as np
import open3d as o3d
from matplotlib import cm

def run_eval_mesh(pred_mesh_path: str, gt_pcd_path: str, downsample_size: float, threshold: float, truncation_precision: float, truncation_recall: float):
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)
    pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=10000000)

    if downsample_size > 0:
        gt_pcd = gt_pcd.voxel_down_sample(voxel_size=downsample_size)
        pred_pcd = pred_pcd.voxel_down_sample(voxel_size=downsample_size)

    gt_points = np.asarray(gt_pcd.points)
    pred_points = np.asarray(pred_pcd.points)

    indices_p, distances_p = get_correspondence(gt_points, pred_points, truncation_precision, ignore_outlier=True)
    indices_r, distances_r = get_correspondence(pred_points, gt_points, truncation_recall, ignore_outlier=False)

    acc = np.mean(distances_p) * 100.0
    comp = np.mean(distances_r) * 100.0
    c_l1 = (acc + comp) / 2.0

    precision = np.mean(distances_p < threshold) * 100.0
    recall = np.mean(distances_r < threshold) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall)

    # print results in a table
    headers = ["Metric", "Acc. (cm)", "Comp. (cm)", "C-L1 (cm)", "Precision (%)", "Recall (%)", "F1 (%)"]
    results = [
        ["Value", f"{acc:.2f}", f"{comp:.2f}", f"{c_l1:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"]
    ]

    print(tabulate.tabulate(results, headers=headers, tablefmt="simple"))


def run_eval_mesh_with_error_map(pred_mesh_path: str, gt_pcd_path: str, downsample_size: float, threshold: float, truncation_precision: float, truncation_recall: float, precision_error_map: bool = True, recall_error_map: bool = True):
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)
    pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=10000000)

    if downsample_size > 0:
        gt_pcd = gt_pcd.voxel_down_sample(voxel_size=downsample_size)
        pred_pcd = pred_pcd.voxel_down_sample(voxel_size=downsample_size)

    gt_points = np.asarray(gt_pcd.points)
    pred_points = np.asarray(pred_pcd.points)

    indices_p, distances_p = get_correspondence(gt_points, pred_points, truncation_precision, ignore_outlier=True)
    indices_r, distances_r = get_correspondence(pred_points, gt_points, truncation_recall, ignore_outlier=False)

    acc = np.mean(distances_p) * 100.0
    comp = np.mean(distances_r) * 100.0
    c_l1 = (acc + comp) / 2.0

    precision = np.mean(distances_p < threshold) * 100.0
    recall = np.mean(distances_r < threshold) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall)

    # print results in a table
    headers = ["Metric", "Acc. (cm)", "Comp. (cm)", "C-L1 (cm)", "Precision (%)", "Recall (%)", "F1 (%)"]
    results = [
        ["Value", f"{acc:.2f}", f"{comp:.2f}", f"{c_l1:.2f}", f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"]
    ]

    print(tabulate.tabulate(results, headers=headers, tablefmt="simple"))

    # (Optinal) Get error map
    if precision_error_map:
        error_pcd = o3d.geometry.PointCloud()
        error_pcd.points = o3d.utility.Vector3dVector(pred_points[indices_p])
        precision_colors = (distances_p / truncation_precision)
        precision_colors = np.clip(precision_colors, 0, 1)
        precision_colors = cm.Reds(precision_colors)[:, :3].astype(np.float64)    
        error_pcd.colors = o3d.utility.Vector3dVector(precision_colors)
        # o3d.visualization.draw_geometries([error_pcd], window_name="Precision Error Map")
        # o3d.io.write_point_cloud("observatory_xo_precision.ply", error_pcd)

    if recall_error_map:
        error_pcd = o3d.geometry.PointCloud()
        error_pcd.points = o3d.utility.Vector3dVector(gt_points)
        recall_colors = (distances_r / truncation_recall)
        recall_colors = np.clip(recall_colors, 0, 1)
        recall_colors = cm.Reds(recall_colors)[:, :3].astype(np.float64)    
        error_pcd.colors = o3d.utility.Vector3dVector(recall_colors)
        # o3d.visualization.draw_geometries([error_pcd], window_name="Recall Error Map")
        # o3d.io.write_point_cloud("observatory_xo_recall.ply", error_pcd)


def get_correspondence(points1: np.ndarray, points2: np.ndarray, radius: float, ignore_outlier: bool = True):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    kdtree1 = o3d.geometry.KDTreeFlann(pcd1)

    indices = []
    distances = []

    for i, point in enumerate(points2):
        _, _, dist = kdtree1.search_knn_vector_3d(point, 1)
        if dist[0] < radius ** 2:
            indices.append(i)
            distances.append(np.sqrt(dist[0]))
        else:
            if not ignore_outlier:
                indices.append(i)
                distances.append(radius)

    indices = np.array(indices, dtype=np.int64)
    distances = np.array(distances, dtype=np.float64)

    return indices, distances
