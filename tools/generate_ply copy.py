import subprocess
import os
import time
import scipy.io
import numpy as np
from tools.ply.brownv2_hymodel import HyNet
import torch
import shutil

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
    patch_size = 32
    patch_folder_path = os.path.join(dataset_path_call_matlab,human_number,f'EMO-1-shout+laugh/{which_time}/psiftproject/images')
    patch_folder_path_pathkp = os.path.join(dataset_path_call_matlab,human_number,f'EMO-1-shout+laugh/{which_time}/psiftproject/keypoints')
    patch_save_path = os.path.join(dataset_path_call_matlab,human_number,f'EMO-1-shout+laugh/{which_time}/psiftproject/descriptors')
    # 确保patch_save_path路径存在
    if not os.path.exists(patch_save_path):
        os.makedirs(patch_save_path)


    txt_path = f"/media/Trajectory3D/tools/ply/txt/{which_time}/get_patch_parameters.txt"
    env = os.environ.copy()  # 复制当前环境变量
    env["GET_SIFT_PARAMETERS_TXT"] = txt_path

    with open(txt_path, "w") as f:
        f.write(f"patch_size={patch_size}\n")
        f.write(f"patch_folder_path={patch_folder_path}\n")
        f.write(f"patch_folder_path_pathkp={patch_folder_path_pathkp}\n")
        f.write(f"patch_save_path={patch_save_path}\n")
        

    extract_patch_m='/media/Trajectory3D/tools/ply/local-feature-evaluation-master/scripts/parameter_extract_patch.m'
    # subprocess.run(["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(extract_patch_m)])
     # 调用 matlab (传递正确的 txt_path)
    matlab_command = ["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(extract_patch_m)]

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

    return patch_folder_path, patch_save_path

def get_descriptors(gpu_id, patch_folder_path, patch_save_path):
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    net_modelpath = '/media/Trajectory3D/tools/ply/net-best.pth'
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
        #     print(" -> skipping, already exist")
        #     continue
        mat_data = read_mat_file(patches_path)
        patches31 = np.array(mat_data["patch"])

        if patches31.ndim != 3:
            print(" -> skipping, invalid input")
            write_matrix(descriptors_path, np.zeros((0, 512), dtype=np.float32))
            continue

        patches = np.empty((patches31.shape[0], 32, 32), dtype=np.float32)
        patches[:, :31, :31] = patches31
        patches[:, 31, :31] = patches31[:, 30, :]
        patches[:, :31, 31] = patches31[:, :, 30]
        patches[:, 31, 31] = patches31[:, 30, 30]

        descriptors = []
        for i in range(0, patches.shape[0], batch_size):
            patches_batch = \
                patches[i:min(i + batch_size, patches.shape[0])]
            patches_batch = \
                torch.from_numpy(patches_batch[:, None]).float().to(device)
            # descriptors.append(tfeat(patches_batch).detach().cpu().numpy())
            descriptors.append(model(patches_batch).detach().cpu().numpy())

        if len(descriptors) == 0:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        else:
            descriptors = np.concatenate(descriptors)

        write_matrix(descriptors_path, descriptors)

        print(" in {:.3f}s".format(time.time() - start_time))

        #生成描述子并保存       

def get_matches(dataset_path_call_matlab, human_number, which_time):

    txt_path = f"/media/Trajectory3D/tools/ply/txt/{which_time}/matching_parameters.txt"
    env = os.environ.copy()  # 复制当前环境变量
    env["GET_SIFT_PARAMETERS_TXT"] = txt_path

    with open(txt_path, "w") as f:
        f.write(f"dataset_root={dataset_path_call_matlab}\n")
        f.write(f"match_human_number={human_number}\n")
        f.write(f"which_time={which_time}\n")
    #创建matching 文件夹
    match_m='/media/Trajectory3D/tools/ply/local-feature-evaluation-master/scripts/parameter_matching_pipeline.m'
    # subprocess.run(["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(match_m)])
    
     # 调用 matlab (传递正确的 txt_path)
    matlab_command = ["/usr/local/Matlab/R2020a/bin//matlab", "-nodisplay", "-nosplash", "-r", "run('{}'); exit".format(match_m)]

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


def get_ply(project_path):
    format_db_path = '/media/Trajectory3D/tools/ply/origin_format_database.db'
    reconstruct_dataset_path = project_path
    shutil.copy(format_db_path, os.path.join(reconstruct_dataset_path,'database.db'))
    # 跑 reconstruction
    import subprocess
    reconstruct_py = '/media/Trajectory3D/tools/ply/local-feature-evaluation-master/scripts/parameter_reconstruction_pipeline.py'

    args = ['python', reconstruct_py, f'--dataset_path={reconstruct_dataset_path}']
    # 使用 subprocess.run 执行脚本并传递参数
    subprocess.run(args)