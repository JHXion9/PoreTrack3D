import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import subprocess
import argparse

def copy_mesh2folder(mesh_pth_list, save_pth, max_workers=4):
    os.makedirs(save_pth, exist_ok=True)
    
    def copy_file(idx, mesh_pth):
        shutil.copy(mesh_pth, os.path.join(save_pth, f"mesh_{idx+1}.ply"))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, mesh_pth in enumerate(mesh_pth_list):
            futures.append(executor.submit(copy_file, idx, mesh_pth))
        
        # 使用 tqdm 显示进度
        for future in tqdm(futures, total=len(mesh_pth_list)):
            future.result()  # 确保任务完成

def convert_cameras_file(input_file, output_file):
    """
    修改 cameras.txt 文件：
    1. 将 SIMPLE_RADIAL 替换为 SIMPLE_PINHOLE
    2. 删除最后一个参数 (0.0001)
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('#'):
                # 保留注释行
                f_out.write(line)
            else:
                # 分割行数据
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保是有效的相机数据行
                    camera_id = parts[0]
                    model = "SIMPLE_PINHOLE"  # 替换模型
                    width = parts[2]
                    height = parts[3]
                    params = parts[4:-1]  # 删除最后一个参数 (0.0001)
                    
                    # 重新组合行
                    new_line = f"{camera_id} {model} {width} {height} {' '.join(params)}\n"
                    f_out.write(new_line)
                else:
                    f_out.write(line)  # 其他行保持不变

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type=str, required=True, help="人员id")
    opt = parser.parse_args()

    id = opt.id
    mesh_pth_list = [f'/media/DGST_data/Test_Data/{id}/EMO-1-shout+laugh/{time}/psiftproject/mesh.ply' for time in range(1, 151)]
    save_mesh_pth = f"/media/DGST_data/Data/{id}/mesh"
    
    os.makedirs(save_mesh_pth, exist_ok=True)
    # 复制mesh数据
    if os.path.exists(os.path.join(save_mesh_pth, "mesh_150.ply")):
        pass
    else:
        copy_mesh2folder(mesh_pth_list, save_mesh_pth)
    # 复制ply文件
    ply_pth = f"/media/DGST_data/Test_Data/{id}/EMO-1-shout+laugh/1/psiftproject/dense/0/fused.ply"
    save_pth = f"/media/DGST_data/Data/{id}"
    shutil.copy(ply_pth, os.path.join(save_pth, "points3D_multipleview.ply"))
    # 复制相机内外参
    camera_pth = f"/media/DGST_data/Test_Data/{id}/EMO-1-shout+laugh/1/psiftproject/sparse/1/"
    camera_txt_pth = os.path.join(camera_pth, "cameras.txt")
    images_txt_pth = os.path.join(camera_pth, "images.txt")
    points3D_txt_pth = os.path.join(camera_pth, "points3D.txt")
    save_intrinsic_extrinsic_pth = os.path.join(save_pth, "sparse_")
    temp_pth = './temp/'
    os.makedirs(temp_pth, exist_ok=True)
    os.makedirs(save_intrinsic_extrinsic_pth, exist_ok= True)
    convert_cameras_file(camera_txt_pth, os.path.join(temp_pth, "cameras.txt"))
    shutil.copy(images_txt_pth, os.path.join(temp_pth, "images.txt"))
    shutil.copy(points3D_txt_pth, os.path.join(temp_pth, "points3D.txt"))

    subprocess.call([os.path.join('/media/DenseGSTracking/colmap/build/src/colmap/exe', "colmap"),
                        "model_converter",
                        "--input_path", temp_pth,
                        "--output_path", save_intrinsic_extrinsic_pth,
                        "--output_type", "BIN"])
    

    colmap_temp = '/media/colmap_temp'

    shutil.copytree(camera_pth, os.path.join(colmap_temp, "sparse/0"))
    shutil.copytree(f"/media/DGST_data/Test_Data/{id}/EMO-1-shout+laugh/1/psiftproject/images", os.path.join(colmap_temp, "images"))
    shutil.copy(f"/media/DGST_data/Test_Data/{id}/EMO-1-shout+laugh/1/psiftproject/database.db", os.path.join(colmap_temp, "database.db"))

    del temp_pth


    
