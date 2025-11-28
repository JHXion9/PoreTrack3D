import os
import json
import numpy as np

def merge_xyz_json_files(root_folder_path):
    """
    遍历指定文件夹下的所有子文件夹，将每个子文件夹中的 _xyz.json 文件内容
    整合到一个新的 _xyz.json 文件中，并存放在根文件夹路径下。
    前三个轨迹优先从*_face文件夹中选择。

    Args:
        root_folder_path (str): 包含子文件夹的根文件夹路径。

    Returns:
        bool: 如果成功整合并写入文件则返回 True，否则返回 False。
    """
    if not os.path.isdir(root_folder_path):
        print(f"错误：提供的路径 '{root_folder_path}' 不是一个有效的文件夹。")
        return False

    face_data = []  # 存储来自*_face文件夹的数据
    other_data = []  # 存储来自其他文件夹的数据
    subfolders_processed = 0
    xyz_files_found = 0

    # 1. 遍历根文件夹下的所有项目
    for item_name in sorted(os.listdir(root_folder_path)):
        item_path = os.path.join(root_folder_path, item_name)

        # 2. 检查是否是子文件夹
        if os.path.isdir(item_path):
            subfolders_processed += 1
            xyz_file_path = os.path.join(item_path, "_xyz.json")

            # 3. 检查子文件夹中是否存在 _xyz.json 文件
            if os.path.isfile(xyz_file_path):
                print(f"找到文件: {xyz_file_path}")
                xyz_files_found += 1
                try:
                    with open(xyz_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 根据文件夹名称是否包含_face来分类存储
                        if item_name.endswith('_face') or item_name.startswith('pore'):
                            face_data.append(data)
                            print(f"来自face文件夹的数据: {item_name}")
                        else:
                            other_data.append(data)
                except json.JSONDecodeError:
                    print(f"警告：文件 '{xyz_file_path}' 不是有效的JSON格式，已跳过。")
                except Exception as e:
                    print(f"读取文件 '{xyz_file_path}' 时发生错误: {e}")
            else:
                print(f"在子文件夹 '{item_name}' 中未找到 _xyz.json 文件。")

    # 4. 按要求组织数据：前三个来自face文件夹，其余来自其他文件夹
    all_data_to_merge = []
    
    # 先添加最多3个face数据
    # face_count = min(3, len(face_data))
    # all_data_to_merge.extend(face_data[:face_count])
    all_data_to_merge.extend(face_data)
    all_data_to_merge.extend(other_data)
    # # 如果face数据不足3个，从其他数据中补充
    # if face_count < 3:
    #     needed = 3 - face_count
    #     all_data_to_merge.extend(other_data[:needed])
    #     # 添加剩余的其他数据
    #     all_data_to_merge.extend(other_data[needed:])
    # else:
    #     # 如果face数据超过3个，将剩余的face数据添加到后面
    #     all_data_to_merge.extend(face_data[3:])
    #     # 添加所有其他数据
    #     all_data_to_merge.extend(other_data)

    if not all_data_to_merge:
        if subfolders_processed == 0:
            print(f"在 '{root_folder_path}' 中没有找到任何子文件夹。")
        elif xyz_files_found == 0:
            print("没有找到任何有效的 _xyz.json 文件进行整合。")
        else: # 找到了文件但都无法解析
             print("所有找到的 _xyz.json 文件都无法解析或为空，无法生成合并文件。")
        return False

    # 5. 将整合后的数据写入到根目录下的新 _xyz.json 文件
    output_file_path = os.path.join(root_folder_path, "_xyz.json")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=4 可以让输出的JSON文件更易读
            json.dump(all_data_to_merge, outfile, ensure_ascii=False, indent=4)
        print(f"\n成功！所有数据已整合到: {output_file_path}")
        print(f"共处理了 {subfolders_processed} 个子文件夹，找到了 {xyz_files_found} 个 _xyz.json 文件。")
        print(f"其中包含 {len(face_data)} 个face文件夹的数据，{len(other_data)} 个其他文件夹的数据。")
        print(f"前三个轨迹中有 {min(3, len(face_data))} 个来自face文件夹。")
        

        avg_movement = calculate_average_movement(all_data_to_merge, frames=150)
        print(f"平均每个关键点在150帧内的移动距离: {avg_movement*100:.4f}")
        return True

    except Exception as e:
        print(f"写入整合文件 '{output_file_path}' 时发生错误: {e}")
        return False
    
def calculate_average_movement(xyz_data, frames=150):
    """
    计算关键点在指定帧数内的平均移动距离
    
    Args:
        xyz_data (list): 包含轨迹数据的列表，每个轨迹包含150帧数据
        frames (int): 要统计的帧数，默认150帧
        
    Returns:
        float: 平均每个轨迹的移动距离
    """
    if not xyz_data:
        print("调试：xyz_data为空")
        return 0.0
    
    print(f"调试：总共有 {len(xyz_data)} 个轨迹")
    
    total_movement = 0.0
    valid_trajectories = 0
    
    for idx, trajectory in enumerate(xyz_data):
        if not isinstance(trajectory, list) or len(trajectory) < 2:
            print(f"调试：轨迹 {idx+1} 数据不足")
            continue
            
        # print(f"调试：处理第 {idx+1} 个轨迹，长度: {len(trajectory)}")
        
        # 限制帧数
        max_frames = min(frames, len(trajectory))
        if max_frames < 2:
            continue
        
        # 计算该轨迹所有帧之间的累积移动距离
        trajectory_movement = 0.0
        valid_frames = 0
        
        for frame_idx in range(max_frames - 1):
            current_frame = trajectory[frame_idx]
            next_frame = trajectory[frame_idx + 1]

            current_frame = np.array(current_frame)
            next_frame = np.array(next_frame)

                
            # 计算相邻帧之间的距离
            frame_distance = np.linalg.norm(next_frame - current_frame)
            trajectory_movement += frame_distance
            valid_frames += 1
        
        if valid_frames > 0:
            # print(f"调试：轨迹 {idx+1} 累积移动距离: {trajectory_movement:.6f}, 有效帧数: {valid_frames}")
            total_movement += trajectory_movement
            valid_trajectories += 1
        else:
            print(f"调试：轨迹 {idx+1} 没有有效的帧数据")
    
    print(f"调试：总移动距离: {total_movement}, 有效轨迹数: {valid_trajectories}")
    
    return total_movement / valid_trajectories if valid_trajectories > 0 else 0.0


# --- 使用示例 ---
if __name__ == "__main__":

    """ 
    文件搜索：遍历指定根文件夹下的所有子文件夹，查找名为 _xyz.json 的文件

    数据整合：将找到的所有 _xyz.json 文件中的数据合并到一个统一的数据结构中

    输出保存：将合并后的数据保存为一个新的 _xyz.json 文件，存放在根文件夹路径下
    """

    # IDS = ['031','033','038','056','063','124','196','264']
    IDS = ['196']
    for id in IDS:
        merge_xyz_json_files(f"/media/DGST_data/trajectory/{id}")

    # 测试一个没有子文件夹的路径 (例如当前脚本所在的目录，假设它没有合适的子文件夹)
    # print("\n--- 测试没有子文件夹的路径 ---")
    # merge_xyz_json_files(current_dir) # 这会扫描当前脚本目录