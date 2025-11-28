import numpy as np
import open3d as o3d
import argparse
import os
import json

from sklearn.neighbors import NearestNeighbors


def add_keypoints_to_pointcloud(keypoints_3d, pointcloud_path):
    """
    将第一帧三维关键点添加到点云中，并替换原始点云文件
    
    参数:
        keypoints_3d: numpy数组，形状为(150, 3)的三维关键点
        pointcloud_path: str，原始点云文件路径
        
    返回:
        None
    """
    # 检查输入形状是否正确
    if keypoints_3d.shape != (150, 3):
        raise ValueError("关键点数组的形状应为(150, 3)")
    
    # 提取第一帧关键点
    first_frame_keypoints = keypoints_3d[0]  # 形状为(3,)
    
    # 读取原始点云及其所有属性
    try:
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        original_points = np.asarray(pcd.points)
        
        # 获取所有属性
        has_normals = pcd.has_normals()
        has_colors = pcd.has_colors()
        
        original_normals = np.asarray(pcd.normals) if has_normals else None
        original_colors = np.asarray(pcd.colors) if has_colors else None
    except Exception as e:
        raise ValueError(f"无法读取点云文件: {e}")
    
    # 将关键点添加到点云中
    # 注意: 这里将(3,)数组reshape为(1, 3)以保持维度一致
    new_points = np.vstack([original_points, first_frame_keypoints.reshape(1, 3)])
    
    # 为新点找到最近邻点的属性
    if original_points.shape[0] > 0:
        nbrs = NearestNeighbors(n_neighbors=1).fit(original_points)
        distances, indices = nbrs.kneighbors(first_frame_keypoints.reshape(1, 3))
        
        # 处理法向量
        if has_normals:
            nearest_normal = original_normals[indices[0][0]]
            new_normals = np.vstack([original_normals, nearest_normal])
        
        # 处理颜色
        if has_colors:
            nearest_color = original_colors[indices[0][0]]
            new_colors = np.vstack([original_colors, nearest_color])
    
    # 创建新的点云对象并设置所有属性
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    
    if has_normals:
        new_pcd.normals = o3d.utility.Vector3dVector(new_normals)
    
    if has_colors:
        new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
    
    # 保存新的点云，替换原始文件
    try:
        o3d.io.write_point_cloud(pointcloud_path, new_pcd, write_ascii=False)
    except Exception as e:
        raise IOError(f"无法写入点云文件: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type=str, required=True, help="人员id")
    parser.add_argument("--position", type=str, help="GT位置")
    opt = parser.parse_args()


    
    op_dir = f"/media/DGST_data/trajectory/{opt.id}-{opt.position}/"
    pointcloud_path = f"/media/DGST_data/Data/{opt.id}/points3D_multipleview.ply"
    with open(os.path.join(op_dir,'_xyz.json'), 'r') as file:
        data_list = json.load(file)

    keypoints_3d = np.array(data_list)

    add_keypoints_to_pointcloud(keypoints_3d, pointcloud_path)
