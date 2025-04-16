import subprocess
import os
import time
import scipy.io
import numpy as np
from tools.ply.brownv2_hymodel import HyNet
import torch
import shutil
import cv2
import glob
import open3d as o3d
import torchvision.transforms as T
from tools.GMS_match import GmsMatcher, compute_matches_in_order
import json
import re

def copy_frame(source_root, destination_path, num_frames):
    """Copies the first frame of each camera to the destination path."""
    os.makedirs(destination_path, exist_ok=True)
    i = 0
    dir1 = source_root
    for folder_name in sorted(os.listdir(dir1)):
        if folder_name.startswith("cam"):
            dir2 = os.path.join(dir1, folder_name)
            for file_name in os.listdir(dir2):
                if file_name.startswith(f"frame_{str(num_frames).zfill(5)}"):
                    i += 1
                    src_path = os.path.join(dir2, file_name)
                    if 'mask' in source_root:
                        dst_path = os.path.join(destination_path, f"{i}_mask.png")
                    else:
                        dst_path = os.path.join(destination_path, f"{i}.png")
                    shutil.copyfile(src_path, dst_path)
                    break # Only copy the first matching frame per camera folder
    
def write_matrix(path, matrix):
    with open(path, "wb") as fid:
        shape = np.array(matrix.shape, dtype=np.int32)
        shape.tofile(fid)
        matrix.tofile(fid)

def read_mat_file(file_path):
    # 读取 .mat 文件
    mat_contents = scipy.io.loadmat(file_path)
    
    # 返回读取的内容
    return mat_contents

def save_images_txt(Q_list, t_list, save_pth):
    """
    创建COLMAP格式的images.txt文件
    
    参数:
        Q_list: 四元数列表，每个元素是[q0, q1, q2, q3]
        t_list: 平移向量列表，每个元素是[tx, ty, tz]
        save_pth: 保存路径
    """
    with open(save_pth, 'w') as f:
        # 写入文件头
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(Q_list)}, mean observations per image: 0\n")
        
        # 写入每个图像的数据
        for i, (q, t) in enumerate(zip(Q_list, t_list), start=1):
            # 第一行：图像外参
            f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {i}.png\n")
            # 第二行：空行（因为没有2D点数据）
            f.write("\n")

def save_cameras_txt(fx, fy, cx, cy, save_pth, width=2200, height=3208, camera_model="PINHOLE"):
    """
    创建COLMAP格式的cameras.txt文件
    
    参数:
        fx, fy, cx, cy: 相机内参
        save_pth: 保存路径
        width: 图像宽度(默认1280)
        height: 图像高度(默认720)
        camera_model: 相机模型(默认"PINHOLE")
    """
    with open(save_pth, 'w') as f:
        # 写入文件头
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]\n")
        f.write("# Number of cameras: 16\n")
        for i in range(1,17):
            f.write(f"{i} {camera_model} {width} {height} {fx} {fy} {cx} {cy}\n")

def save_cameras_simple_radial_txt_one(f1, cx, cy, k, save_pth, width=2200, height=3208, camera_model="SIMPLE_RADIAL"):
    """
    创建COLMAP格式的cameras.txt文件
    
    参数:
        fx, fy, cx, cy: 相机内参
        save_pth: 保存路径
        width: 图像宽度(默认1280)
        height: 图像高度(默认720)
        camera_model: 相机模型(默认"PINHOLE")
    """
    with open(save_pth, 'w') as f:
        # 写入文件头
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[f,cx,cy,k]\n")
        f.write("# Number of cameras: 16\n")
        for i in range(1,17):
            f.write(f"{i} {camera_model} {width} {height} {f1} {cx} {cy} {k}\n")

def save_cameras_simple_radial_txt(params, save_pth, width=2200, height=3208, camera_model="PINHOLE"):
    """
    创建COLMAP格式的cameras.txt文件
    
    参数:
        fx, fy, cx, cy: 相机内参
        save_pth: 保存路径
        width: 图像宽度(默认1280)
        height: 图像高度(默认720)
        camera_model: 相机模型(默认"PINHOLE")
    """
    with open(save_pth, 'w') as f:
        # 写入文件头
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[f cx cy k]\n")
        f.write("# Number of cameras: 16\n")
        for i in range(1,17):
            f.write(f"{i} {camera_model} {width} {height} {params[i-1][0]} {params[i-1][1]} {params[i-1][2]} {params[i-1][3]}\n")
       
def save_points3D_txt(save_path):
    """创建一个空的 points3D.txt 文件，符合 COLMAP 格式"""
    with open(save_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0.0\n")

def get_QT_from_colmap(input_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()
    
    Q = []
    T = []
    Camera_id = []
    index = 1
    i = 0
    while i < len(lines):
        # print(i)
        line = lines[i].strip()
        
        if line and not line.startswith('#'):  # 忽略注释行
            # 保留相机信息行
            data = line.split(' ')
            q = [float(part) for part in data[1:5]] 
            t = [float(part) for part in data[5:8]] 
            camera_id = line.split(' ')[8]
            print(i, data[9])
            if data[9] == f'{index}.png':
                Q.append(q)
                T.append(t)
                Camera_id.append(camera_id)
                index += 1


            # 跳过Points2D行，并添加一个空行
            if i + 1 < len(lines):
                i += 1

                continue
            elif i==len(lines)-1 and len(Camera_id) == 16:
                break
            else:
                
                i = 0
            
        else:
            i += 1
            continue
            
    return Q, T, Camera_id

def get_camera_txt_params(camera_ids, cameras_txt_path):
    """
    从 cameras.txt 文件中提取指定 camera_ids 的参数列表。

    Args:
        camera_ids: 一个包含要提取参数的 camera_id 的列表。  (list of str) 注意，camera_ids 现在是字符串列表
        cameras_txt_path: cameras.txt 文件的路径。 (str)

    Returns:
        一个列表，其中包含与 camera_ids 列表中每个 camera_id 对应的参数列表。
        如果某个 camera_id 在文件中找不到，则其对应的参数列表将为 None。
        (list of list of float or None)
    """

    camera_data = {}
    try:
        with open(cameras_txt_path, 'r') as f:
            # Skip the header lines
            next(f)  # Skip "Number of cameras: 16" line
            for line in f:
                parts = line.strip().split()
                camera_id = parts[0]  # camera_id 保持为字符串
                # Ensure at least 7 elements for consistent data structure
                if len(parts) >= 7:  # Checking for 7 parts (ID, Model, Width, Height, params)
                    params = [float(p) for p in parts[4:]]
                    camera_data[camera_id] = params
                else:
                    print(f"Warning: Insufficient data for camera ID {camera_id}. Skipping.")

    except FileNotFoundError:
        print(f"Error: File not found at {cameras_txt_path}")
        return []
    except ValueError:
        print(f"Error: Could not convert data to float in {cameras_txt_path}. Check data format.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

    result = []
    for cam_id in camera_ids:
        if cam_id in camera_data:
            result.append(camera_data[cam_id])
        else:
            result.append(None)  # 或者你可以选择抛出一个异常或记录一个错误
            print(f"Warning: Camera ID {cam_id} not found in cameras.txt")

    return result

def get_camera_params_from_json(json_pth, save_path):
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
    with open(json_pth, 'r') as f:
        data = json.load(f)
    
    os.makedirs(save_path, exist_ok=True)
    Q = []
    T = []
    
    # 遍历所有相机ID
    for cam_name in cam_identifiers:
        for cam_id, extrinsic_matrix in data["world_2_cam"].items():
            # 转换为numpy数组
            if cam_id == cam_name:
                extrinsic = np.array(extrinsic_matrix)
                # 提取旋转和平移
                R = extrinsic[:3, :3]
                t = extrinsic[:3, 3]
                # 计算Q
                q0 = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
                q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
                q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
                q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
                Q.append([q0, q1, q2, q3])
                T.append(t)
    K = data['intrinsics']
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    f = (fx+fy)/2
    images_txt_pth = os.path.join(save_path, 'images.txt')
    cameras_txt_pth = os.path.join(save_path, 'cameras.txt')
    points3D_txt_pth = os.path.join(save_path, 'points3D.txt')
    save_images_txt(Q, T, images_txt_pth)
    save_cameras_simple_radial_txt_one(f, cx, cy, 0, cameras_txt_pth)
    save_points3D_txt(points3D_txt_pth)
    
def get_camera_params_from_exist_colmap(source_pth, save_path):
    os.makedirs(save_path, exist_ok=True)

    images_txt_pth = os.path.join(save_path, 'images.txt')
    cameras_txt_pth = os.path.join(save_path, 'cameras.txt')
    points3D_txt_pth = os.path.join(save_path, 'points3D.txt')

    model_valid_path0 = os.path.join(source_pth, '0')
    model_valid_path1 = os.path.join(source_pth, '1')
    if os.path.exists(model_valid_path1):
        Q, T, Camera_id = get_QT_from_colmap(os.path.join(model_valid_path1,'images.txt'))
        params = get_camera_txt_params(Camera_id, os.path.join(model_valid_path1,'cameras.txt'))
    else:
        Q, T, Camera_id = get_QT_from_colmap(os.path.join(model_valid_path0,'images.txt'))
        params = get_camera_txt_params(Camera_id, os.path.join(model_valid_path0,'cameras.txt'))
    save_cameras_simple_radial_txt(params, cameras_txt_pth, camera_model='SIMPLE_RADIAL')
    save_images_txt(Q, T, images_txt_pth)
    save_points3D_txt(points3D_txt_pth)



def run_psift(dataset_path_call_matlab = '/media/DGST_data/Test_Data', human_number = '018', kpts_number = 10000, which_time = 10):
    get_sift_m='/media/Trajectory3D/tools/get_sift/parameter_eth_get_face_sift.m'

    os.makedirs(f"/media/Trajectory3D/tools/ply/txt/{which_time}", exist_ok=True)
    
    txt_path = f"/media/Trajectory3D/tools/ply/txt/{which_time}/get_sift_parameters.txt"
    env = os.environ.copy()  # 复制当前环境变量
    env["GET_SIFT_PARAMETERS_TXT"] = txt_path

    # 将参数写入 txt 文件
    with open(txt_path, "w") as f:
        f.write(f"dataset_path_call_matlab={dataset_path_call_matlab}\n")
        f.write(f"human_number={human_number}\n")
        f.write(f"kpts_number={kpts_number}\n")
        f.write(f"which_time={which_time}\n")

    print("参数已写入 parameters.txt")

    #调用matlab
    # subprocess.run(["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(get_sift_m)])
    # 调用 matlab (传递正确的 txt_path)
    matlab_command = ["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(get_sift_m)]

    # 使用 subprocess.Popen
    process = subprocess.Popen(matlab_command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

     # 实时显示输出
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(f"进程 {which_time}: MATLAB 输出: {line.decode().strip()}")

    # 获取错误信息
    stderr = process.stderr.read().decode()
    if stderr:
        print(f"进程 {which_time}: MATLAB 错误: {stderr.strip()}")

    # 等待进程完成
    process.wait()

    # 检查返回值
    if process.returncode != 0:
      print(f"进程 {which_time}: MATLAB 脚本执行失败，返回代码: {process.returncode}")

def get_patch(dataset_path_call_matlab = '/media/DGST_data/Test_Data', human_number = '018', which_time = 1):
    # 调用该文件 进行patch的挖取，此处挖取所得的patch大小为  (patch_size/2-1)*2+1 ,可以根据所需patch1大小进行修改patch_size的参数
    patch_size = 64
    patch_folder_path = os.path.join(dataset_path_call_matlab,human_number,f'EMO-1-shout+laugh/{which_time}/psiftproject/images')
    patch_folder_path_pathkp = os.path.join(dataset_path_call_matlab,human_number,f'EMO-1-shout+laugh/{which_time}/psiftproject/keypoints')
    patch_save_path = os.path.join(dataset_path_call_matlab,human_number,f'EMO-1-shout+laugh/{which_time}/psiftproject/descriptors')
    # 确保patch_save_path路径存在
    if not os.path.exists(patch_save_path):
        os.makedirs(patch_save_path)


    txt_path = f"/media/Trajectory3D/tools/ply/get_patch_parameters.txt"


    with open(txt_path, "w") as f:
        f.write(f"patch_size={patch_size}\n")
        f.write(f"patch_folder_path={patch_folder_path}\n")
        f.write(f"patch_folder_path_pathkp={patch_folder_path_pathkp}\n")
        f.write(f"patch_save_path={patch_save_path}\n")
        

    extract_patch_m='/media/Trajectory3D/tools/ply/local-feature-evaluation-master/scripts/parameter_extract_patch.m'
    subprocess.run(["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(extract_patch_m)])

    return patch_folder_path, patch_save_path    

def get_descriptors(gpu_id, patch_folder_path, patch_save_path):
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    net_modelpath = '/media/Trajectory3D/models/net-best.pth'
    model = HyNet().eval()
    model.to(device)
    model.load_state_dict(torch.load(net_modelpath))

    #读入patch数据
    batch_size = 512
    image_names = os.listdir(patch_folder_path)
    for i, image_name in enumerate(image_names):

        patches_path = os.path.join(patch_save_path,
                            image_name + ".bin.patches.mat")
        print(patches_path)
        if not os.path.exists(patches_path):
            continue

        print("Computing features for {} [{}/{}]".format(
                image_name, i + 1, len(image_names)), end="")

        start_time = time.time()

        descriptors_path = os.path.join(patch_save_path,
                                image_name + ".bin")
        # if os.path.exists(descriptors_path):
        #     print(" -> skipping, already exist")
        #     continue
        mat_data = read_mat_file(patches_path)
        dig_patches = np.array(mat_data["patch"])

        if dig_patches.ndim != 3:
            print(" -> skipping, invalid input")
            write_matrix(descriptors_path, np.zeros((0, 512), dtype=np.float32))
            continue
        patches = dig_patches
        transform = T.Resize((32, 32))
        descriptors = []
        for i in range(0, patches.shape[0], batch_size):
            patches_batch = \
                patches[i:min(i + batch_size, patches.shape[0])]
            patches_batch = \
                torch.from_numpy(patches_batch[:, None]).float().to(device)
            patches_batch = transform(patches_batch)
            # descriptors.append(tfeat(patches_batch).detach().cpu().numpy())
            descriptors.append(model(patches_batch).detach().cpu().numpy())
        # ----------
        
        if len(descriptors) == 0:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        else:
            descriptors = np.concatenate(descriptors)

        write_matrix(descriptors_path, descriptors)

        print(" in {:.3f}s".format(time.time() - start_time))

        #生成描述子并保存       

# 穷尽匹配
def get_matches(dataset_path_call_matlab, human_number, which_time):
    txt_path = "/media/Trajectory3D/tools/ply/matching_parameters.txt"
    

    with open(txt_path, "w") as f:
        f.write(f"dataset_root={dataset_path_call_matlab}\n")
        f.write(f"match_human_number={human_number}\n")
        f.write(f"which_time={which_time}\n")
    #创建matching 文件夹
    match_m='/media/Trajectory3D/tools/ply/local-feature-evaluation-master/scripts/parameter_matching_pipeline.m'
    # subprocess.run(["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(match_m)])

    subprocess.run(["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(match_m)])

# GMS匹配
def get_GMS_matches(dataset_path_call_matlab, human_number, which_time):
    img = [cv2.imread(os.path.join(dataset_path_call_matlab, human_number, f'EMO-1-shout+laugh/{which_time}/psiftproject/images', f"{i+1}.png")) for i in range(16)]
    keypoints_pth = [os.path.join(dataset_path_call_matlab, human_number, f'EMO-1-shout+laugh/{which_time}/psiftproject/keypoints', f"{i+1}.png.bin") for i in range(16)]
    descriptors_pth = [os.path.join(dataset_path_call_matlab, human_number, f'EMO-1-shout+laugh/{which_time}/psiftproject/descriptors', f"{i+1}.png.bin") for i in range(16)]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    GMS = GmsMatcher(matcher)

    compute_matches_in_order(GMS, img, keypoints_pth, descriptors_pth)


def get_ply(project_path, human_number, frame):
    if frame == 1:
        get_camera_params_from_json(f"/media/DGST_data/cameras_json_from_nersemble/{human_number}/camera_params.json", os.path.join(project_path, 'created/sparse'))
    # 使用第一帧的相机参数继续重建后续帧
    if frame != 1:
        parent_dir = os.path.join('/', *project_path.split('/')[:-2])
        src_pth = os.path.join(parent_dir,'1/psiftproject/sparse')
        
        get_camera_params_from_exist_colmap(src_pth, os.path.join(project_path, 'created/sparse'))
        
    
    format_db_path = '/media/Trajectory3D/tools/ply/origin_format_database.db'
    reconstruct_dataset_path = project_path
    shutil.copy(format_db_path, os.path.join(reconstruct_dataset_path,'database.db'))
    # 跑 reconstruction
    import subprocess
    reconstruct_py = '/media/Trajectory3D/tools/ply/local-feature-evaluation-master/scripts/parameter_reconstruction_pipeline.py'

    args = ['python', reconstruct_py, f'--dataset_path={reconstruct_dataset_path}', f'--frame={frame}']
#     # 使用 subprocess.run 执行脚本并传递参数
    # subprocess.run(args)
   
    result = subprocess.run(args, capture_output=True, text=True)

    print("matching_stats:", result.stdout)
    if "\"num_inlier_pairs\": 0" in result.stdout:
        print("重建失败，继续重建！！！！")
        loop = True
    else:
        print("重建成功！！！！！")
        loop = False

    return loop

def get_mesh(pcd_path, mesh_output_path):
    pcd=o3d.io.read_point_cloud(pcd_path, format='ply')
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
        o3d.io.write_triangle_mesh(mesh_output_path, mesh)
    