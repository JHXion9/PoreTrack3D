import os
import sys
import json
import numpy as np
import cv2
from utils_tra.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
import torch
import imageio
from collections import Counter
import trimesh
from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh

def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)

def cut_video(video_path, output_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frame_num):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        out.write(frame)

    cap.release()
    out.release()

def get_k_w2c(datadir, cam_id, timestamp):
    if os.path.exists(os.path.join(datadir, str(timestamp), "psiftproject/sparse/1/images.bin")):
        cameras_extrinsic_file = os.path.join(datadir, str(timestamp), "psiftproject/sparse/1/images.bin")
        cameras_intrinsic_file = os.path.join(datadir, str(timestamp), "psiftproject/sparse/1/cameras.bin")
    else:
        cameras_extrinsic_file = os.path.join(datadir, str(timestamp), "psiftproject/sparse/0/images.bin")
        cameras_intrinsic_file = os.path.join(datadir, str(timestamp), "psiftproject/sparse/0/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    value = f'{cam_id}.png'
    # print(timestamp, value)
    for idx, key in enumerate(cam_extrinsics):
        if cam_extrinsics[key].name == value:
            extr_id = key
            intr_id = cam_extrinsics[key].camera_id

    extr = cam_extrinsics[extr_id]
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    params = cam_intrinsics[intr_id].params
    
    return params, R, T

def get_k_w2c_flame(datadir, cam_id):
    cam1 = '222200042'
    cam2 = '222200044'
    cam3 = '222200046'
    cam4 = '222200040'
    cam5 = '222200036'  #
    cam6 = '222200048'
    cam7 = '220700191'
    cam8 = '222200041'
    cam9 = '222200037'
    cam10 = '222200038'  #
    cam11 = '222200047'
    cam12 = '222200043'
    cam13 = '222200049'
    cam14 = '222200039'
    cam15 = '222200045' #
    cam16 = '221501007'
    cam_identifiers = [cam1, cam2, cam3, cam4, cam5, cam6, cam7, cam8, cam9, cam10, cam11, cam12, cam13, cam14, cam15, cam16]
    with open(datadir, 'r') as file:
        params = json.load(file)
    
    extrinsic = np.array(params['world_2_cam'][cam_identifiers[int(cam_id)-1]])
    # 提取旋转矩阵 R
    R = extrinsic[:3, :3]
    # 提取平移向量 T
    T = extrinsic[:3, 3]
    # 提取相机内参
    intrinsic = np.array(params['intrinsics'])
    param = [intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]]
    return param, R, T

def undistort_point(image_points, params):
    """
    对图像坐标进行Simple Radial去畸变。

    参数：
        image_points: 原始图像坐标 (N, 2)  N为关键点数量
        params: 相机参数 f, cx, cy, k  (其中f可以是单个值，也可以是fx, fy)

    返回：
        去畸变后的图像坐标 (N, 2)
    """
    
    

    fx, fy = params[0], params[0]

    cx, cy = params[1], params[2]
    k = params[3]


    # 1. 归一化图像坐标
    x_n = (image_points[:, 0] - cx) / fx
    y_n = (image_points[:, 1] - cy) / fy

    # 2. 计算径向畸变
    r2 = x_n**2 + y_n**2
    x_d = x_n * (1 + k * r2)
    y_d = y_n * (1 + k * r2)

    # 3. 转换回图像坐标
    x_u = x_d * fx + cx
    y_u = y_d * fy + cy

    return np.stack([x_u, y_u], axis=-1)  # 使用np.stack组合成 (N, 2) 数组

def get_3d_coordinates(mesh_path, params, R, T, keypoints_2d):
    """
    计算相机视角下二维关键点对应的Mesh上的三维坐标。

    Args:
        mesh_path (str): .ply文件的路径。
        params (list): 相机内参 [fx, cx, cy, k]。
        R (np.ndarray): 相机外参旋转矩阵，形状为 (3, 3)。
        T (np.ndarray): 相机外参平移向量，形状为 (3, 1)。
        keypoints_2d (np.ndarray): 二维关键点坐标，形状为 (N, 2)。

    Returns:
        tuple: (valid_keypoints_2d, valid_intersections), 其中：
            - valid_keypoints_2d (np.ndarray):  形状为 (M, 2) 的有效二维关键点数组 (M <= N)。
            - valid_intersections (np.ndarray): 形状为 (M, 3) 的有效三维交点数组。
    """
    keypoints_2d = np.array(keypoints_2d)

    # 1. 加载Mesh
    mesh = trimesh.load_mesh(mesh_path)
    triangles = mesh.vertices[mesh.faces]

    # 2. 构建Embree场景
    scene = rtcs.EmbreeScene()
    mesh_embree = TriangleMesh(scene, triangles.astype(np.float32))

    # 3. 相机参数 (简化：假设 fx=fy)
    f = params[0]
    cx = params[1]
    cy = params[2]
    k = params[3]

    # 4. 去畸变
    undistorted_keypoints = undistort_point(keypoints_2d, [f, cx, cy, k])

    # 5. 构建射线 (考虑外参)
    # 相机中心在世界坐标系下的坐标:
    origin = -R.T @ T  # (3, 1)
    origin = origin.reshape(3)  # 转换为 (3,)，方便后续计算

    # 射线方向 (在相机坐标系下)
    x = (undistorted_keypoints[:, 0] - cx) / f
    y = (undistorted_keypoints[:, 1] - cy) / f
    direction_camera = np.stack([x, y, np.ones_like(x)], axis=-1)  # (N, 3)

    # 将射线方向从相机坐标系转换到世界坐标系
    direction_world = (R.T @ direction_camera.T).T  # (N, 3)

    # 单位化世界坐标系下的方向向量
    direction_world = direction_world / np.linalg.norm(direction_world, axis=1, keepdims=True)

    # 6. 射线求交
    origins = np.tile(origin, (keypoints_2d.shape[0], 1))  # (N, 3)
    res = scene.run(origins.astype(np.float32), direction_world.astype(np.float32), output=1)

    # 7. 获取交点和有效关键点
    valid_intersections_world = np.empty((0, 3), dtype=np.float32)
    valid_keypoints_2d = np.empty((0, 2), dtype=np.float32)
    valid_intersections_camera = np.empty((0, 3), dtype=np.float32)

    valid_intersections_mask = res['geomID'] != -1 # (N,) boolean array
    valid_indices = np.where(valid_intersections_mask)[0]  # 获取有效交点的索引
    
    if valid_indices.size > 0:   # 确保存在有效交点才执行计算
        primIDs = res['primID'][valid_indices]
        u = res['u'][valid_indices]
        v = res['v'][valid_indices]
        w = 1 - u - v

        # 使用有效索引进行批量计算
        valid_intersections_world = (w[:, None] * triangles[primIDs, 0] +
                                         u[:, None] * triangles[primIDs, 1] +
                                         v[:, None] * triangles[primIDs, 2])
        # 转换为相机坐标系下的坐标
        valid_intersections_camera = (R @ valid_intersections_world.T + T.reshape(3, 1)).T

        valid_keypoints_2d = keypoints_2d[valid_indices]

    return valid_keypoints_2d, valid_intersections_world, valid_intersections_camera

def get_3d_coordinates_flame(mesh_path, params, R, T, keypoints_2d):
    """
    计算相机视角下二维关键点对应的Mesh上的三维坐标。

    Args:
        mesh_path (str): .ply文件的路径。
        params (list): 相机内参 [fx, cx, cy, k]。
        R (np.ndarray): 相机外参旋转矩阵，形状为 (3, 3)。
        T (np.ndarray): 相机外参平移向量，形状为 (3, 1)。
        keypoints_2d (np.ndarray): 二维关键点坐标，形状为 (N, 2)。

    Returns:
        tuple: (valid_keypoints_2d, valid_intersections), 其中：
            - valid_keypoints_2d (np.ndarray):  形状为 (M, 2) 的有效二维关键点数组 (M <= N)。
            - valid_intersections (np.ndarray): 形状为 (M, 3) 的有效三维交点数组。
    """
    keypoints_2d = np.array(keypoints_2d)

    # 1. 加载Mesh
    mesh = trimesh.load_mesh(mesh_path)
    triangles = mesh.vertices[mesh.faces]

    # 2. 构建Embree场景
    scene = rtcs.EmbreeScene()
    mesh_embree = TriangleMesh(scene, triangles.astype(np.float32))

    # 3. 相机参数 (简化：假设 fx=fy)
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
   

    # 4. 去畸变
    # undistorted_keypoints = undistort_point(keypoints_2d, [f, cx, cy, k])

    # 5. 构建射线 (考虑外参)
    # 相机中心在世界坐标系下的坐标:
    origin = -R.T @ T  # (3, 1)
    origin = origin.reshape(3)  # 转换为 (3,)，方便后续计算

    # 射线方向 (在相机坐标系下)
    x = (keypoints_2d[:, 0] - cx) / fx
    y = (keypoints_2d[:, 1] - cy) / fy
    direction_camera = np.stack([x, y, np.ones_like(x)], axis=-1)  # (N, 3)

    # 将射线方向从相机坐标系转换到世界坐标系
    direction_world = (R.T @ direction_camera.T).T  # (N, 3)

    # 单位化世界坐标系下的方向向量
    direction_world = direction_world / np.linalg.norm(direction_world, axis=1, keepdims=True)

    # 6. 射线求交
    origins = np.tile(origin, (keypoints_2d.shape[0], 1))  # (N, 3)
    res = scene.run(origins.astype(np.float32), direction_world.astype(np.float32), output=1)

    # 7. 获取交点和有效关键点
    valid_intersections = []
    valid_keypoints_2d = []

    valid_intersections_mask = res['geomID'] != -1 # (N,) boolean array
    valid_indices = np.where(valid_intersections_mask)[0]  # 获取有效交点的索引
    
    if valid_indices.size > 0:   # 确保存在有效交点才执行计算
        primIDs = res['primID'][valid_indices]
        u = res['u'][valid_indices]
        v = res['v'][valid_indices]
        w = 1 - u - v

        # 使用有效索引进行批量计算
        valid_intersections = (w[:, None] * triangles[primIDs, 0] +
                                         u[:, None] * triangles[primIDs, 1] +
                                         v[:, None] * triangles[primIDs, 2])

        valid_keypoints_2d = keypoints_2d[valid_indices]

    return valid_keypoints_2d, valid_intersections

def reproject_points(params, R, t, world_points):
    """
    使用SIMPLE_RADIAL相机模型将世界坐标系下的3D点投影到像素坐标 (优化版本，使用NumPy广播).

    参数：
        world_points (numpy.ndarray): 世界坐标系中的3D点，形状为(N, 3)
        params (list/numpy.ndarray): 相机内参及畸变参数 [焦距f, 主点cx, 主点cy, 畸变系数k]
        R (numpy.ndarray): 旋转矩阵，形状为(3, 3)
        t (numpy.ndarray): 平移向量，形状为(3,)

    返回：
        numpy.ndarray: 投影后的像素坐标，形状为 (N, 2)

    异常：
        ValueError: 当投影点的深度Z为0时抛出
    """
    # 解析相机参数
    f, cx, cy, k = params

    # 将世界坐标转换为相机坐标系
    P_cam = world_points @ R.T + t  # 形状(N, 3)
    X = P_cam[:, 0]
    Y = P_cam[:, 1]
    Z = P_cam[:, 2]

    # 检查深度有效性
    if np.any(np.isclose(Z, 0.0)):
        raise ValueError("投影失败：深度Z不能为零")

    # 归一化平面坐标
    u = X / Z
    v = Y / Z

    # 计算径向畸变
    r_sq = u**2 + v**2
    du = u * k * r_sq
    dv = v * k * r_sq

    # 应用畸变
    u_distorted = u + du
    v_distorted = v + dv

    # 转换为像素坐标
    x = f * u_distorted + cx
    y = f * v_distorted + cy

    points2d = np.stack([x, y], axis=-1)  # 形状 (N, 2)
    return points2d

def reproject_points_pinhole(params, R, t, world_points):
    """
    使用SIMPLE_RADIAL相机模型将世界坐标系下的3D点投影到像素坐标 (优化版本，使用NumPy广播).

    参数：
        world_points (numpy.ndarray): 世界坐标系中的3D点，形状为(N, 3)
        params (list/numpy.ndarray): 相机内参及畸变参数 [焦距f, 主点cx, 主点cy, 畸变系数k]
        R (numpy.ndarray): 旋转矩阵，形状为(3, 3)
        t (numpy.ndarray): 平移向量，形状为(3,)

    返回：
        numpy.ndarray: 投影后的像素坐标，形状为 (N, 2)

    异常：
        ValueError: 当投影点的深度Z为0时抛出
    """
    # 解析相机参数
    fx, fy, cx, cy = params

    # 将世界坐标转换为相机坐标系
    

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    w2c = np.eye(4)  # 创建一个 4x4 的单位矩阵
    w2c[:3, :3] = R  # 填充旋转矩阵 R
    w2c[:3, 3] = t  # 填充平移向量 T

    # 转换为齐次坐标 (Nx4)
    homo_vertices = np.hstack([world_points, np.ones((len(world_points), 1))])
    
    # 世界坐标系 -> 相机坐标系 (矩阵乘法)
    camera_coords = (w2c @ homo_vertices.T).T  # 结果形状 (Nx4)
    
    # 提取有效点 (z>0 表示在相机前方)
    z = camera_coords[:, 2]
    valid_mask = z > 0
    valid_coords = camera_coords[valid_mask, :3]  # 取前三维 (x,y,z)
    
    # 投影到图像平面
    pixel_coords_homo = (K @ valid_coords.T).T  # 内参变换
    points2d = pixel_coords_homo[:, :2] / pixel_coords_homo[:, 2:]  # 齐次除法


    return points2d

def filter_3dpointcloud(pts2d, p3d_path, params, R, T, threshold=5, visual = False):
    """
    Filters 2D keypoints based on reprojection error of nearby 3D points.

    Args:
        pts2d: A numpy array of shape (N, 2) representing 2D keypoints.
        p3d_path: Path to a .ply file containing 3D point cloud data (M, 3).
        K: The camera intrinsic matrix (3, 3).
        w2c: The world-to-camera extrinsic matrix (4, 4).
        threshold: The maximum allowed reprojection error in pixels.

    Returns:
        filtered_pts2d: A numpy array of shape (N', 2) representing the filtered 2D keypoints.
        filtered_p3ds: A numpy array of shape (N', 3) representing the corresponding 3D points, 
                       interpolated if necessary.
    """
    import open3d as o3d
    p3ds = o3d.io.read_point_cloud(p3d_path)
    p3ds = np.asarray(p3ds.points)
    num_pts2d = pts2d.shape[0]
    filtered_pts2d = []
    filtered_p3ds = []

    p3ds_proj = reproject_points(params, R, T, p3ds)
    p3ds_proj = p3ds_proj.reshape(-1, 2)
    

    for i in range(num_pts2d):
        pt2d = pts2d[i]

        # Calculate distances between projected 3D points and the 2D keypoint
        distances = np.linalg.norm(p3ds_proj - pt2d, axis=1)

        # Check if any 3D point has reprojection error less than 1 pixel
        min_error_index = np.argmin(distances)
        min_error = distances[min_error_index]

        if visual:
            print("离最近点云的投影点为：",p3ds_proj[min_error_index])

        if min_error < 1:
            filtered_pts2d.append(pt2d)
            filtered_p3ds.append(p3ds[min_error_index])
            continue

        # Find the indices of the 3 nearest 3D points
        nearest_indices = np.argpartition(distances, 3)[:3]
        # print(distances[nearest_indices])

        # Check reprojection error for the 3 nearest points
        errors = distances[nearest_indices]
        if np.all(errors < threshold):
            filtered_pts2d.append(pt2d)
            # Interpolate the 3D coordinates of the nearest points
            weights = 1 / (errors + 1e-8) # Use inverse distance weighting, adding a small value to avoid division by zero
            weights /= np.sum(weights) # Normalize weights
            interpolated_p3d = np.sum(p3ds[nearest_indices] * weights[:, np.newaxis], axis=0)
            filtered_p3ds.append(interpolated_p3d)
            continue

        

    return np.array(filtered_pts2d), np.array(filtered_p3ds)

def getfromvideo(opt):
    cam1 = 'cam_222200042.mp4'
    cam2 = 'cam_222200044.mp4'
    cam3 = 'cam_222200046.mp4'
    cam4 = 'cam_222200040.mp4'
    cam5 = 'cam_222200036.mp4'  #
    cam6 = 'cam_222200048.mp4'
    cam7 = 'cam_220700191.mp4'
    cam8 = 'cam_222200041.mp4'
    cam9 = 'cam_222200037.mp4'
    cam10 = 'cam_222200038.mp4'  #
    cam11 = 'cam_222200047.mp4'
    cam12 = 'cam_222200043.mp4'
    cam13 = 'cam_222200049.mp4'
    cam14 = 'cam_222200039.mp4'
    cam15 = 'cam_222200045.mp4' #
    cam16 = 'cam_221501007.mp4'
    cam_identifiers = [cam1, cam2, cam3, cam4, cam5, cam6, cam7, cam8, cam9, cam10, cam11, cam12, cam13, cam14, cam15, cam16]
            # 遍历所有摄像头标识符
    for i, cam in enumerate(cam_identifiers):
        if i == opt.base_view-1:
            # print(f'/media/DGST_data/raw_data/pore/{id}/{seq_name}/{cam}')
            cut_video(f'/media/DGST_data/raw_data/eyes/{opt.people_id}/{opt.seq_name}/{cam}', f'/media/Nersemble/video/{opt.people_id}/{opt.seq_name}/cut_{cam}', opt.frame_num)
            video = read_video_from_path(f'/media/Nersemble/video/{opt.people_id}/{opt.seq_name}/cut_{cam}')
            video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
            
    return video

def crop_regions_around_keypoints(img, keypoint, radius=50):
    """
    根据给定的关键点，在其周围切割半径为 radius 的矩形区域。

    参数:
        image_path (str): 图像文件的路径。
        keypoint: 包含关键点坐标(x1, y1)
        radius (int): 切割区域的半径，默认为 50。

    返回:
        cropped_regions (dict): 包含切割出来的区域图像的字典。
    """


    # 切割区域
    x, y = keypoint
    x1 = int(max(0, x - radius))
    y1 = int(max(0, y - radius))
    x2 = int(min(img.shape[1], x + radius))
    y2 = int(min(img.shape[0], y + radius))
      
    # 切割区域
    cropped_region = img[y1:y2, x1:x2]
       

    return cropped_region

def transform_array(numpy_array):
    """
    将形状为 (n, 2) 的 NumPy 数组转换为形状为 (n, 3) 的 PyTorch 张量，并在第一列插入一列 0。

    Args:
        numpy_array: 形状为 (n, 2) 的 NumPy 数组。

    Returns:
        形状为 (n, 3) 的 PyTorch 张量，并移动到 CUDA 设备。如果输入不是numpy数组则返回None
    """
    if not isinstance(numpy_array, np.ndarray):
        print("Error: Input must be a NumPy array.")
        return None
    if numpy_array.shape[1] != 2:
        print("Error: Input array must have shape (n, 2).")
        return None
    # 将 NumPy 数组转换为 PyTorch 张量
    tensor = torch.from_numpy(numpy_array).float()

    # 创建一个形状为 (n, 1) 的全零张量
    zeros = torch.zeros(tensor.shape[0], 1)

    # 将全零张量与原张量在列维度上拼接
    result_tensor = torch.cat((zeros, tensor), dim=1)

    # 将结果移动到 CUDA 设备上
    result_tensor = result_tensor.cuda()

    return result_tensor

def reset_resolution(keypoints_base = None, keypoints = None, a_pts = None, b_pts = None, radius = 50):


    offset_x_base = keypoints_base[0] - radius
    offset_y_base = keypoints_base[1] - radius
    offset_x_dst = keypoints[0] - radius
    offset_y_dst = keypoints[1] - radius

    a_pts_original = []
    b_pts_original = []

    for a, b in zip(a_pts, b_pts):
        a_original = (a[1] + offset_x_base, a[0] + offset_y_base)
        b_original = (b[1] + offset_x_dst, b[0] + offset_y_dst)
        a_pts_original.append(a_original)
        b_pts_original.append(b_original)
        
    return a_pts_original, b_pts_original

def to_resolution(keypoints_base, pts, radius = 50):
    offset_x = keypoints_base[0] - radius
    offset_y = keypoints_base[1] - radius

    pts_original = []

    for pt in pts:
        pt_original = (pt[0] - offset_x, pt[1] - offset_y)
        pts_original.append(pt_original)
        
    return np.array(pts_original)

def create_match_matrix_from_points(keypoints_0, keypoints_1, src, dst):
    """
    根据匹配成功的关键点坐标构建匹配矩阵。

    Args:
        keypoints_0 (np.ndarray): 前一帧的关键点坐标，形状 (m, 2)。
        keypoints_1 (np.ndarray): 后一帧的关键点坐标，形状 (n, 2)。
        src (np.ndarray): 匹配成功的 前一帧关键点坐标，形状 (k, 2)。
        dst (np.ndarray): 匹配成功的 后一帧关键点坐标，形状 (k, 2)。

    Returns:
        np.ndarray: 匹配矩阵，形状 (m, n)。
    """
    m = keypoints_0.shape[0]
    n = keypoints_1.shape[0]
    match_matrix = np.zeros((m, n), dtype=int)

    # 遍历匹配成功的关键点，直接设置匹配矩阵
    for i in range(src.shape[0]):
        src_point = src[i]
        dst_point = dst[i]
        src_dist = np.sqrt(np.sum((keypoints_0 - src_point)**2, axis=1))
        dst_dist = np.sqrt(np.sum((keypoints_1 - dst_point)**2, axis=1))
        min_src_index = np.argmin(src_dist)
        min_dst_index = np.argmin(dst_dist)
        match_matrix[min_src_index, min_dst_index] = 1

    return match_matrix

def save_keypoint_data(all_keypoints, all_p3d, all_match_matrices, output_path='keypoint_data', view = '9'):
    """
    将所有帧的关键点坐标和匹配矩阵保存在单个JSON文件中。

    Args:
        all_keypoints (list): 包含所有帧关键点坐标的列表，每个元素是形状为 (m, 2) 的 numpy 数组。
        all_match_matrices (list): 包含所有帧匹配矩阵的列表，每个元素是形状为 (m, n) 的 numpy 数组。
        match_param (list): 包含所有帧匹配过程中的参数列表。
        output_path (str, optional): 基础保存路径. Defaults to 'keypoint_data'.
    """
    os.makedirs(output_path, exist_ok=True)

    # 保存有效关键点坐标
    if all_keypoints != None:
        if view != 'whole':
            all_keypoints_list = [keypoints.tolist() for keypoints in all_keypoints]
            keypoints_file = os.path.join(output_path, f"all_keypoints_{view}.json")
            with open(keypoints_file, "w") as f:
                json.dump({"keypoints": all_keypoints_list}, f, indent=4)
        else:
            all_keypoints_list = [keypoints.tolist() for keypoints in all_keypoints]
            keypoints_file = os.path.join(output_path, f"whole_keypoints.json")
            with open(keypoints_file, "w") as f:
                json.dump({"keypoints": all_keypoints_list}, f, indent=4)

    # 保存有效关键点对应点云坐标
    if all_p3d != None:
        if view!= 'cam':
            all_p3d_list = [p3d.tolist() for p3d in all_p3d]
            p3ds_file = os.path.join(output_path, f"all_p3ds.json")
            with open(p3ds_file, "w") as f:
                json.dump({"point_clouds": all_p3d_list}, f, indent=4)
        else:
            all_p3d_list = [p3d.tolist() for p3d in all_p3d]
            p3ds_file = os.path.join(output_path, f"all_p3ds_basecam.json")
            with open(p3ds_file, "w") as f:
                json.dump({"point_clouds": all_p3d_list}, f, indent=4)

    # 保存所有匹配矩阵
    if all_match_matrices != None:
        if view != 'whole':
            all_match_matrices_list = [mat.tolist() if isinstance(mat, np.ndarray) else mat for mat in all_match_matrices]
            match_matrix_file = os.path.join(output_path, f"all_match_matrices_{view}.json")
            with open(match_matrix_file, "w") as f:
                json.dump({"match_matrices": all_match_matrices_list}, f, indent=4)
        else:
            all_match_matrices_list = [mat.tolist() if isinstance(mat, np.ndarray) else mat for mat in all_match_matrices]
            match_matrix_file = os.path.join(output_path, f"whole_match_matrices.json")
            with open(match_matrix_file, "w") as f:
                json.dump({"match_matrices": all_match_matrices_list}, f, indent=4)

def load_match_matrices_from_json(json_file):
    """
    从 JSON 文件中加载匹配矩阵，保留三维列表格式。

    Args:
        json_file (str): JSON 文件的路径。

    Returns:
        list: 包含所有帧匹配矩阵的列表，每个元素是一个三维列表。
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    match_matrices_list = data.get("match_matrices", [])
    
    if not match_matrices_list:
        print("Warning: No 'match_matrices' found in the JSON file or it is empty.")
        return []
    
    # 直接返回原始列表，不做numpy转换
    return match_matrices_list

def load_keypoints_from_json(json_file):
    """
    从 JSON 文件中加载关键点位置信息。

    Args:
        json_file (str): JSON 文件的路径。

    Returns:
        list: 包含所有帧关键点位置信息的列表，每个元素是一个numpy数组。
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    keypoints_list = data.get("keypoints", [])
    if not keypoints_list:
        print("Warning: No 'keypoints' found in the JSON file or it is empty.")
        return []
        
    # 将列表中的子列表转成numpy 数组    
    keypoints = [np.array(kps) for kps in keypoints_list]
    
    return keypoints

def load_p3ds_from_json(json_file):
    """
    从 JSON 文件中加载关键点位置信息。

    Args:
        json_file (str): JSON 文件的路径。

    Returns:
        list: 包含所有帧关键点位置信息的列表，每个元素是一个numpy数组。
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    keypoints_list = data.get("point_clouds", [])
    if not keypoints_list:
        print("Warning: No 'Point_clouds' found in the JSON file or it is empty.")
        return []
        
    # 将列表中的子列表转成numpy 数组    
    keypoints = [np.array(kps) for kps in keypoints_list]
    
    return keypoints

def find_longest_match_trajectory(all_match_matrices):
    """
    在匹配矩阵中查找所有关键点中最长的连续匹配轨迹。

    Args:
        all_match_matrices (list): 包含所有帧匹配矩阵的列表，每个元素是一个三维列表。

    Returns:
         tuple: (longest_match_start_frame, longest_match_start_keypoint), 最长匹配轨迹的起始帧和关键点索引
    """
    longest_match_length = 0          # 初始化最长匹配长度为0
    longest_match_start_frame = -1    # 初始化最长匹配轨迹的起始帧为-1
    longest_match_start_keypoint = -1 # 初始化最长匹配轨迹的起始关键点为-1
    num_frames = len(all_match_matrices)+1 # 获取总帧数
    
    # 遍历所有可能的起始帧
    for start_frame_index in range(num_frames-1):
        # 获得当前帧的关键点数量
    
        num_keypoints = np.array(all_match_matrices[start_frame_index]).shape[0]
    
            
        # 遍历当前帧的所有关键点
        for start_keypoint_index in range(num_keypoints):
            current_keypoint = start_keypoint_index   # 初始化当前关键点为起始关键点
            match_length = 0                         # 初始化当前匹配轨迹的长度为0
        
            # 从当前起始帧开始，遍历后续帧，查找匹配轨迹
            for next_frame in range(start_frame_index, num_frames-1):
                # 将下一帧的匹配矩阵转换为 numpy 数组
                match_matrix = np.array(all_match_matrices[next_frame])
                # 如果当前关键点索引大于或等于下一帧的匹配矩阵的行数，则退出内层循环
                if current_keypoint >= match_matrix.shape[0]:
                    break
                # 获取下一帧匹配的关键点索引, 使用np.where 获取匹配矩阵中当前关键点为1的索引，返回的是一个元组，索引值在第一个元素中
                next_keypoint_indices = np.where(match_matrix[current_keypoint] == 1)[0]
                
                # 如果没有找到匹配的关键点，则退出内层循环
                if next_keypoint_indices.size == 0:
                    break
            
                match_length += 1 # 如果找到匹配的关键点，则匹配轨迹长度加1
                current_keypoint = next_keypoint_indices[0] # 更新当前关键点为匹配到的下一个关键点的索引
                # 如果当前匹配轨迹长度大于最长匹配轨迹长度，则更新最长匹配轨迹的长度，起始帧和起始关键点
                if match_length > longest_match_length:
                    longest_match_length = match_length
                    longest_match_start_frame = start_frame_index
                    longest_match_start_keypoint = start_keypoint_index

    return longest_match_start_frame, longest_match_start_keypoint, longest_match_length # 返回具有最长匹配轨迹的起始帧和关键点索引

def complete_keypoint_trajectory(match_matrices_list, keypoints, which_frame, keypoint_idx):
    """
    补全关键点轨迹。

    Args:
        match_matrices_list (list): 匹配矩阵列表。
        keypoints (list): 所有帧的关键点位置列表, 列表的列表。
        which_frame (int): 需要补全的关键点首次出现的帧数。
        keypoint_idx (int): 需要补全的关键点在其首次出现帧中的下标。

    Returns:
        np.ndarray: 补全后的关键点轨迹，形状为 (总帧数, 2)。
        
    """
    total_frames = len(keypoints)
    completed_trajectory = np.full((total_frames, 2), np.nan)  # Initialize with NaN
    color_flag = np.zeros(total_frames)
    completed_trajectory[which_frame] = keypoints[which_frame][keypoint_idx]  # Set the first keypoint

    # 找到该关键点最后一次匹配成功的帧
    last_matched_frame = which_frame
    temp_keypoint_idx = keypoint_idx
    for i in range(which_frame, total_frames -1):
        matrix = match_matrices_list[i]
        
        
        is_matched = False
        if temp_keypoint_idx < len(matrix):
          for j in range(len(matrix[temp_keypoint_idx])):
            if matrix[temp_keypoint_idx][j] == 1:
                is_matched = True
                
                last_matched_frame = i + 1
                temp_keypoint_idx = j
                

                current_keypoint_pos = keypoints[last_matched_frame][j]
                completed_trajectory[last_matched_frame] = current_keypoint_pos

                break
        if not is_matched:
          break
            

    print("last_frame", last_matched_frame)



    # 第一类补全（向前补全）
    current_frame = which_frame
    current_keypoint_pos = keypoints[current_frame][keypoint_idx]

    for prev_frame in range(which_frame - 1, -1, -1):
        match_matrix = match_matrices_list[prev_frame]

        # 检查当前关键点是否在匹配矩阵中匹配
        is_matched = False

        if keypoint_idx < len(match_matrix[0]) and keypoint_idx != -1: # 确保 keypoint_idx 不超出列的范围
          for j in range(len(match_matrix)):
            if match_matrix[j][keypoint_idx] == 1:
                is_matched = True
                # 找到匹配的关键点在前一帧的索引

                keypoint_idx = j
                current_keypoint_pos = keypoints[prev_frame][keypoint_idx]
                completed_trajectory[prev_frame] = current_keypoint_pos
                break

        if is_matched:
            continue # 如果当前关键点已匹配，则跳过平均位移计算

        matched_keypoints_prev = []
        matched_keypoints_curr = []

        for i in range(len(match_matrix)):
            for j in range(len(match_matrix[i])):
                if match_matrix[i][j] == 1:
                    matched_keypoints_prev.append(keypoints[prev_frame][i])
                    matched_keypoints_curr.append(keypoints[prev_frame+1][j])

        # 先使用平均位移计算
        if matched_keypoints_prev:
            # 分别计算 x 和 y 方向的位移
            # displacements_x = np.array(matched_keypoints_curr)[:, 0] - np.array(matched_keypoints_prev)[:, 0]
            # displacements_y = np.array(matched_keypoints_curr)[:, 1] - np.array(matched_keypoints_prev)[:, 1]

            # # 分别计算 x 和 y 方向的平均位移
            # avg_displacement_x = np.mean(displacements_x)
            # avg_displacement_y = np.mean(displacements_y)

            #--------------------------------------------------
            # 计算当前关键点与所有其他关键点的距离
            distances = np.linalg.norm(np.array(matched_keypoints_curr) - current_keypoint_pos, axis=1)

            # 获取距离最近的五个关键点的索引
            k = min(5, len(matched_keypoints_prev))  # 确保不超过总匹配关键点的数量
            nearest_indices = np.argsort(distances)[:k]

            # 使用最近的五个关键点来计算位移
            nearest_keypoints_prev = np.array(matched_keypoints_prev)[nearest_indices]
            nearest_keypoints_curr = np.array(matched_keypoints_curr)[nearest_indices]

            # 分别计算 x 和 y 方向的位移
            displacements_x = nearest_keypoints_curr[:, 0] - nearest_keypoints_prev[:, 0]
            displacements_y = nearest_keypoints_curr[:, 1] - nearest_keypoints_prev[:, 1]

            # 分别计算 x 和 y 方向的平均位移
            avg_displacement_x = np.mean(displacements_x)
            avg_displacement_y = np.mean(displacements_y)
            #--------------------------------------------------

            # 使用各自的平均位移来估计前一帧的 x 和 y 坐标
            estimated_pos_x = current_keypoint_pos[0] - avg_displacement_x
            estimated_pos_y = current_keypoint_pos[1] - avg_displacement_y

            current_keypoint_pos = np.array([estimated_pos_x, estimated_pos_y])
            
            completed_trajectory[prev_frame] = current_keypoint_pos
            color_flag[prev_frame] = 1
            keypoint_idx = -1

        else:
            # 没有匹配的点了，报错
            print(f"向前补全失败, {prev_frame}")
            assert 0
            completed_trajectory[prev_frame] = current_keypoint_pos

        # 尝试查找在1像素范围内是否有关键点
        distances = np.linalg.norm(keypoints[prev_frame] - current_keypoint_pos, axis=1)
        nearest_keypoint_idx = np.argmin(distances)

        if distances[nearest_keypoint_idx] <= 1:
            # 直接将该点视为匹配的关键点，并更新 keypoint_idx

            current_keypoint_pos = keypoints[prev_frame][nearest_keypoint_idx]
            completed_trajectory[prev_frame] = current_keypoint_pos
            keypoint_idx = nearest_keypoint_idx #更新下标


    # 第二类补全（向后补全）
    current_frame = last_matched_frame
    current_keypoint_pos = keypoints[current_frame][temp_keypoint_idx]
    keypoint_idx = temp_keypoint_idx

    for next_frame in range(last_matched_frame + 1, total_frames):
        match_matrix = match_matrices_list[next_frame-1]

        # 检查当前关键点是否在匹配矩阵中匹配
        is_matched = False

        #找到上一帧中和该点匹配的点
        if keypoint_idx < len(match_matrix) and keypoint_idx != -1:
            for i in range(len(match_matrix[keypoint_idx])):
                if match_matrix[keypoint_idx][i] == 1:

                    is_matched = True
                    current_keypoint_pos = keypoints[next_frame][i]
                    completed_trajectory[next_frame] = current_keypoint_pos

                    keypoint_idx = i

                    break
        if is_matched:
            continue

        matched_keypoints_prev = []
        matched_keypoints_curr = []

        for i in range(len(match_matrix)):
            for j in range(len(match_matrix[i])):
                if match_matrix[i][j] == 1:
                    matched_keypoints_prev.append(keypoints[next_frame-1][i])
                    matched_keypoints_curr.append(keypoints[next_frame][j])

        #先进行平均位移计算
        if matched_keypoints_prev:
            # # 分别计算 x 和 y 方向的位移
            # displacements_x = np.array(matched_keypoints_curr)[:, 0] - np.array(matched_keypoints_prev)[:, 0]
            # displacements_y = np.array(matched_keypoints_curr)[:, 1] - np.array(matched_keypoints_prev)[:, 1]

            # # 分别计算 x 和 y 方向的平均位移
            # avg_displacement_x = np.mean(displacements_x)
            # avg_displacement_y = np.mean(displacements_y)
            #--------------------------------------------------
            # 计算当前关键点与所有其他关键点的距离
            distances = np.linalg.norm(np.array(matched_keypoints_curr) - current_keypoint_pos, axis=1)

            # 获取距离最近的五个关键点的索引
            k = min(5, len(matched_keypoints_prev))  # 确保不超过总匹配关键点的数量
            nearest_indices = np.argsort(distances)[:k]

            # 使用最近的五个关键点来计算位移
            nearest_keypoints_prev = np.array(matched_keypoints_prev)[nearest_indices]
            nearest_keypoints_curr = np.array(matched_keypoints_curr)[nearest_indices]

            # 分别计算 x 和 y 方向的位移
            displacements_x = nearest_keypoints_curr[:, 0] - nearest_keypoints_prev[:, 0]
            displacements_y = nearest_keypoints_curr[:, 1] - nearest_keypoints_prev[:, 1]

            # 分别计算 x 和 y 方向的平均位移
            avg_displacement_x = np.mean(displacements_x)
            avg_displacement_y = np.mean(displacements_y)
            #--------------------------------------------------

            # 使用各自的平均位移来估计后一帧的 x 和 y 坐标
            estimated_pos_x = current_keypoint_pos[0] + avg_displacement_x
            estimated_pos_y = current_keypoint_pos[1] + avg_displacement_y

            current_keypoint_pos = np.array([estimated_pos_x, estimated_pos_y])
            completed_trajectory[next_frame] = current_keypoint_pos
            color_flag[next_frame] = 1
            keypoint_idx = -1

        else:
            print(f"向后补全失败, {next_frame}")
            assert 0
            completed_trajectory[next_frame] = current_keypoint_pos

        # 尝试查找在当前帧1像素范围内是否有关键点
        distances = np.linalg.norm(keypoints[next_frame] - current_keypoint_pos, axis=1)
        nearest_keypoint_idx = np.argmin(distances)

        if distances[nearest_keypoint_idx] <= 1:
            # 直接将该点视为匹配的关键点，并更新 keypoint_idx
            current_keypoint_pos = keypoints[next_frame][nearest_keypoint_idx]
            completed_trajectory[next_frame] = current_keypoint_pos
            keypoint_idx = nearest_keypoint_idx




    return completed_trajectory, color_flag

def draw_trajectories(match_matrices, keypoints, tracked_keypoint_coords, tracked_keypoint_color_flags, images, output_file="output.mp4"):
    """
    绘制关键点轨迹并保存为 .mp4 文件。

    Args:
        match_matrices (list): 逐帧匹配矩阵列表。
        keypoints (list): 逐帧关键点坐标列表。
        images (list): 图像列表。
        tracked_keypoint_coords (np.ndarray): 被跟踪关键点逐帧坐标 (总帧数, 2)。
        tracked_keypoint_color_flags (np.ndarray): 被跟踪关键点颜色标志 (总帧数,)。
        output_file (str): 输出视频文件名。
    """

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = images[0].shape
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

    trajectories = {}  # 存储轨迹 {轨迹ID: [(帧ID, x, y), ...]}

    # 首先，遍历所有帧，构建完整的轨迹（不包括被跟踪关键点）
    for frame_idx in range(len(images) - 1):
        match_matrix = match_matrices[frame_idx]
        current_keypoints = keypoints[frame_idx]
        next_keypoints = keypoints[frame_idx + 1]

        for i in range(len(match_matrix)):
            for j in range(len(match_matrix[0])):
                if match_matrix[i][j] == 1:
                    found_trajectory = False
                    for traj_id, traj in trajectories.items():
                        if traj[-1][0] == frame_idx and np.allclose(traj[-1][1:], current_keypoints[i]):
                            trajectories[traj_id].append((frame_idx + 1, next_keypoints[j][0], next_keypoints[j][1]))
                            found_trajectory = True
                            break

                    if not found_trajectory:
                        if not any((frame_idx, *current_keypoints[i]) in traj for traj in trajectories.values()):
                            new_traj_id = len(trajectories) + 1
                            trajectories[new_traj_id] = [(frame_idx, current_keypoints[i][0], current_keypoints[i][1]), (frame_idx + 1, next_keypoints[j][0], next_keypoints[j][1])]

    # 将被跟踪关键点的轨迹添加到 trajectories 中
    tracked_traj_id = len(trajectories) + 1
    trajectories[tracked_traj_id] = [(frame_idx, tracked_keypoint_coords[frame_idx][0], tracked_keypoint_coords[frame_idx][1]) for frame_idx in range(len(tracked_keypoint_coords))]

    # 然后，遍历所有帧，根据轨迹信息绘制轨迹
    for frame_idx in range(len(images)):
        frame = images[frame_idx].copy()

        for traj_id, traj in trajectories.items():
            # 绘制其他匹配成功的轨迹的条件
            if traj_id != tracked_traj_id:
                if len(traj) >= 5:
                    draw_traj = False
                    start_index = 0
                    for i, point in enumerate(traj):
                        if point[0] == frame_idx:
                            draw_traj = True
                            start_index = max(0, i - 4)
                            break

                    if draw_traj:
                        color = (0, 0, 255)  # 默认红色

                        # 检查是否与被跟踪关键点重合
                        for point_idx, point in enumerate(traj):
                            if point_idx < len(tracked_keypoint_coords) and np.allclose(point[1:], tracked_keypoint_coords[point[0]]):
                                if tracked_keypoint_color_flags[point[0]] == 0:
                                    color = (0, 255, 0)
                                elif tracked_keypoint_color_flags[point[0]] == 1:
                                    color = (0, 0, 0)
                                break
                                                               
                        # 绘制轨迹。 现在，每次最多绘制长度为5的轨迹
                        end_index = min(len(traj) - 1, start_index + 4) # 限制最远绘制到 start_index + 4
                        for i in range(start_index, end_index):
                            if traj[i+1][0] <= frame_idx:
                                pt1 = (int(traj[i][1]), int(traj[i][2]))
                                pt2 = (int(traj[i+1][1]), int(traj[i+1][2]))
                                if pt1[0] >= 0 and pt1[0] < width and pt1[1] >= 0 and pt1[1] < height and \
                                   pt2[0] >= 0 and pt2[0] < width and pt2[1] >= 0 and pt2[1] < height:
                                    cv2.line(frame, pt1, pt2, color, 2)
            else:  # 绘制被跟踪关键点的轨迹
                if tracked_keypoint_color_flags[frame_idx] == 0:
                    color = (0, 255, 0)  # 绿色
                elif tracked_keypoint_color_flags[frame_idx] == 1:
                    color = (0, 0, 0)  # 黑色

                # 绘制被跟踪关键点的轨迹, 确保轨迹长度为5
                start_index = max(0, frame_idx - 4) #轨迹长度为5
                for i in range(start_index, frame_idx):
                  pt1 = (int(tracked_keypoint_coords[i][0]), int(tracked_keypoint_coords[i][1]))
                  pt2 = (int(tracked_keypoint_coords[i+1][0]), int(tracked_keypoint_coords[i+1][1]))
                  if pt1[0] >= 0 and pt1[0] < width and pt1[1] >= 0 and pt1[1] < height and \
                      pt2[0] >= 0 and pt2[0] < width and pt2[1] >= 0 and pt2[1] < height:
                        cv2.line(frame, pt1, pt2, color, 2)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_file}")

def complete_match_matrices(folder_path, base_view = '9'):
    """
    从 JSON 文件中加载匹配矩阵并进行补全。

    Args:
        folder_path (str): 包含 JSON 文件的文件夹路径。

    Returns:
        list: 补全后的主匹配矩阵列表 (三维列表)。
    """

    main_matrices = None
    main_keypoints = load_keypoints_from_json(os.path.join(folder_path, f"all_keypoints_{base_view}.json"))
    other_matrices = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue    
        if not filename.startswith("all_match_matrices"):
            continue
        
        filepath = os.path.join(folder_path, filename)
        print(f"Loading {filepath}")
        matrices = load_match_matrices_from_json(filepath)

        if not matrices: # Skip empty matrices
            continue

        if filename.endswith(f"_{base_view}.json"):
            main_matrices = matrices
        else:
            other_matrices.append(matrices)

    if main_matrices is None:
        print("Error: No '_9.json' file found in the folder.")
        return []

    if not other_matrices:
        print("Warning: No other JSON files found for completion. Returning main matrices as is.")
        return main_matrices

    # 1. Shape Check - Check that the number of matrices is the same AND shapes are valid
    num_matrices = len(main_matrices)
    for other in other_matrices:
        if len(other) != num_matrices:
            print(f"Error: Number of matrices mismatch. Main has {num_matrices}, other has {len(other)}.")
            return []

    # Validate Shapes
    for i in range(num_matrices):
        main_shape = np.array(main_matrices[i]).shape
        for other_matrix_list in other_matrices:
            other_shape = np.array(other_matrix_list[i]).shape
            if main_shape != other_shape:
                print(f"Error: Shape mismatch at matrix {i}. Main shape: {main_shape}, Other shape: {other_shape}")
                return []



    completed_matrices = []
    for i in range(len(main_matrices)): # Iterate through the frames
        main_matrix = main_matrices[i].copy() # Important: Create a copy!
        main_matrix_np = np.array(main_matrix) # Convert to numpy array for easier manipulation
        completed_matrix = main_matrix_np.copy() # Important: Create a copy!
        
        rows, cols = completed_matrix.shape

        # --- Calculate average movement distance ---
        distances = []
        if i + 1 < len(main_matrices):  # Ensure we don't go out of bounds
            keypoints1 = main_keypoints[i]
            keypoints2 = main_keypoints[i + 1]
            
            for row_idx in range(rows):
                col_idx = np.argmax(main_matrix_np[row_idx,:])
                if main_matrix_np[row_idx, col_idx] > 0: #Ensure it had a match
                    
                    try:  # Handle potential errors if keypoints are missing
                        x1, y1, _ = keypoints1[row_idx]  # Assuming format [x, y, confidence]
                        x2, y2, _ = keypoints2[col_idx]
                        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        distances.append(distance)
                    except (IndexError, ValueError):  # Be specific with exception handling
                        # print(f"Warning: Keypoint data missing or invalid at row {row_idx}, col {col_idx} - Skipping distance calculation.")
                        continue  # Skip this pair if there's an error

            avg_distance = np.mean(distances) if distances else 0  # Avoid division by zero
        else:
            avg_distance = 0 # Last matrix, cant calculate the average movement, fill with 0
            
        #--- End average distance calculation ---


        for row in range(rows):  # Iterate through the keypoints in the first frame
            if np.any(completed_matrix[row, :]): # Already matched - skip
                continue

            # Collect potential matches from other matrices
            potential_matches = []
            for other_matrix_list in other_matrices:
                other_matrix = other_matrix_list[i]
                other_matrix_np = np.array(other_matrix)

                if other_matrix_np.shape != (rows, cols):
                    print(f"Warning: Shape mismatch for matrix {i} at row {row} - Skipping other matrix from this file.")
                    continue

                if np.any(other_matrix_np[row, :]):  # Found a potential match
                    matched_col = np.argmax(other_matrix_np[row, :])  # Get the column index of the match
                    potential_matches.append(matched_col)

            # Select the most frequent match
            if potential_matches:
                most_common_matches = Counter(potential_matches).most_common()
                chosen_col = None
                for match, count in most_common_matches:
                    if not completed_matrix[:, match].any():  # Is column available?
                        try:
                            x1, y1 = main_keypoints[i][row]
                            x2, y2 = main_keypoints[i + 1][match]
                            current_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                            if abs(current_distance - avg_distance) <= 5:
                                chosen_col = match
                                break
                        except (IndexError, ValueError):  # Be specific with exception handling
                            print(f"Warning: Keypoint data missing or invalid at row {row}, col {match} - Skipping match.")

        

                if chosen_col is not None:
                    completed_matrix[row, chosen_col] = 1

        completed_matrices.append(completed_matrix)  # Convert back to list
    
    return completed_matrices

def full_match_matrices(match_matrices_file1, keypoints_file1, match_matrices_file2, keypoints_file2):
    """
    根据 keypoints_list2 和 match_matrices_list2 补全 match_matrices_list1。
    根据 绑定后的各个视角的匹配矩阵来 补充正脸视角 没有绑定点云 的匹配矩阵

    Args:
        match_matrices_file1 (str): match_matrices_list1 JSON 文件路径。
        keypoints_file1 (str): keypoints_list1 JSON 文件路径。
        match_matrices_file2 (str): match_matrices_list2 JSON 文件路径。
        keypoints_file2 (str): keypoints_list2 JSON 文件路径。

    Returns:
        list[np.ndarray]: 补全后的 match_matrices_list1，元素值为 0 或 1。
    """

    match_matrices_list1 = load_match_matrices_from_json(match_matrices_file1)
    keypoints_list1 = load_keypoints_from_json(keypoints_file1)
    match_matrices_list2 = load_match_matrices_from_json(match_matrices_file2)
    keypoints_list2 = load_keypoints_from_json(keypoints_file2)

    if not (match_matrices_list1 and keypoints_list1 and match_matrices_list2 and keypoints_list2):
        print("Error: One or more input lists are empty.  Returning original list1.")
        return match_matrices_list1 # or [] if you prefer returning an empty list in this case


    completed_matrices = []
    for i in range(len(match_matrices_list1)):  # 遍历每一帧
        match_matrix1 = np.array(match_matrices_list1[i].copy())
        match_matrix2 = np.array(match_matrices_list2[i].copy())
        keypoints1_frame1 = keypoints_list1[i].copy()   # 第 i 帧关键点列表 1
        keypoints1_frame2 = keypoints_list1[i+1].copy() # 第 i+1 帧关键点列表 1
        keypoints2_frame1 = keypoints_list2[i].copy()   # 第 i 帧关键点列表 2
        keypoints2_frame2 = keypoints_list2[i+1].copy() # 第 i+1 帧关键点列表 2
        
        #确保match_matrix1是bool类型的
        match_matrix1 = match_matrix1.astype(bool)
        
        # 找到 match_matrix2 中匹配成功的行的下标 (a, b)
        matched_rows = np.argwhere(match_matrix2 == 1) # 使用 argwhere
        
        # 记录已被设置为1的位置，避免重复设置
        set_positions = set()

        for a, b in matched_rows:  # a 是在 keypoints2_frame1 中的索引， b 是在 keypoints2_frame2 中的索引
            # 在 keypoints2_frame1 中找到第 a 个关键点的位置
            keypoint_A = keypoints2_frame1[a]  # 关键点的位置本身，例如 [x, y]

            # 找到 keypoint_A 在 keypoints1_frame1 中对应的索引 a'
            a_prime = np.where((keypoints1_frame1 == keypoint_A).all(axis=1))[0]

            #验证a_prime是否能找到
            if len(a_prime) == 0:
              continue
            a_prime = a_prime[0] # 取第一个索引，因为我们假设一个关键点只出现一次

            # 如果 match_matrix1 的第 a' 行已经有 1，则跳过
            if np.any(match_matrix1[a_prime]):
                continue

            # 在 keypoints2_frame2 中找到第 b 个关键点的位置
            keypoint_B = keypoints2_frame2[b]

            # 找到 keypoint_B 在 keypoints1_frame2 中对应的索引 b'
            b_prime = np.where((keypoints1_frame2 == keypoint_B).all(axis=1))[0]
            
            #验证b_prime是否能找到
            if len(b_prime) == 0:
              continue
            b_prime = b_prime[0]

            # 如果 match_matrix1 的第 b' 列已经有 1，则跳过
            if np.any(match_matrix1[:, b_prime]):  # 检查列
                continue

            # 检查是否已经设置过这个位置
            if (a_prime, b_prime) in set_positions:
              continue
            
            # 将 match_matrix1 的第 a' 行第 b' 列设置为 1
            match_matrix1[a_prime, b_prime] = True
            set_positions.add((a_prime,b_prime)) # 记录已经设置的位置

        # 将布尔类型的矩阵转换回整数类型 (0 和 1)
        match_matrix1 = match_matrix1.astype(int)
        completed_matrices.append(match_matrix1)

    return completed_matrices

def save_all_trajectories(all_keypoints, all_match_matrices, save_path):
    """
    保存所有轨迹信息，并按轨迹长度从大到小排序。
    将 NumPy 数组转换为 Python 列表。

    Args:
        all_keypoints (list): 包含所有帧关键点位置信息的列表。
        all_match_matrices (list): 包含所有帧匹配矩阵的列表。
        save_path (str): 保存路径

    Returns:
        dict: 包含所有轨迹信息的字典，按长度降序排列。
    """
    trajectories = {}
    trajectory_id_counter = 0
    num_frames = len(all_keypoints)

    # 遍历所有可能的起始帧
    for start_frame_index in range(num_frames - 1):
        num_keypoints = len(all_keypoints[start_frame_index])
        # 遍历当前帧的所有关键点
        for start_keypoint_index in range(num_keypoints):
            current_keypoint = start_keypoint_index
            match_length = 0
            # 将 NumPy 数组转换为 Python 列表
            keypoints_trajectory = [all_keypoints[start_frame_index][start_keypoint_index].tolist()]

            # 从当前起始帧开始，遍历后续帧，查找匹配轨迹
            for next_frame in range(start_frame_index, num_frames - 1):
                match_matrix = np.array(all_match_matrices[next_frame])

                if current_keypoint >= match_matrix.shape[0]:
                    break

                next_keypoint_indices = np.where(match_matrix[current_keypoint] == 1)[0]

                if next_keypoint_indices.size == 0:
                    break

                match_length += 1
                current_keypoint = next_keypoint_indices[0]
                # 将 NumPy 数组转换为 Python 列表
                keypoints_trajectory.append(all_keypoints[next_frame + 1][current_keypoint].tolist())

            # 如果轨迹长度大于0,说明存在轨迹
            if match_length > 0:
                trajectories[trajectory_id_counter] = {
                    "start_frame": start_frame_index,
                    "length": match_length + 1,
                    "keypoints": keypoints_trajectory  # 已经是列表
                }
                trajectory_id_counter += 1


    # 按轨迹长度从大到小排序
    sorted_trajectories = dict(sorted(trajectories.items(), key=lambda item: item[1]['length'], reverse=True))
    with open(os.path.join(save_path,'all_trajectory.json'), 'w') as f:
        json.dump(sorted_trajectories, f, indent=4)
    return trajectory_id_counter