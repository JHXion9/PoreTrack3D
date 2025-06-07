import os
import json

def merge_xyz_json_files(root_folder_path):
    """
    遍历指定文件夹下的所有子文件夹，将每个子文件夹中的 _xyz.json 文件内容
    整合到一个新的 _xyz.json 文件中，并存放在根文件夹路径下。

    Args:
        root_folder_path (str): 包含子文件夹的根文件夹路径。

    Returns:
        bool: 如果成功整合并写入文件则返回 True，否则返回 False。
    """
    if not os.path.isdir(root_folder_path):
        print(f"错误：提供的路径 '{root_folder_path}' 不是一个有效的文件夹。")
        return False

    all_data_to_merge = [] # 用于存储从各个 _xyz.json 文件中读取的数据
    subfolders_processed = 0
    xyz_files_found = 0

    # 1. 遍历根文件夹下的所有项目
    for item_name in os.listdir(root_folder_path):
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
                        # 假设每个 _xyz.json 的内容本身就是一个列表或字典，
                        # 我们将它们收集到一个大列表中。
                        # 如果它们本身不是列表，但你想让它们成为列表中的元素，
                        # 这里的逻辑也是适用的。
                        # 如果你的 _xyz.json 文件内容是一个列表，并且你想合并所有列表，
                        # 则可以使用 all_data_to_merge.extend(data)
                        all_data_to_merge.append(data)
                except json.JSONDecodeError:
                    print(f"警告：文件 '{xyz_file_path}' 不是有效的JSON格式，已跳过。")
                except Exception as e:
                    print(f"读取文件 '{xyz_file_path}' 时发生错误: {e}")
            else:
                print(f"在子文件夹 '{item_name}' 中未找到 _xyz.json 文件。")

    if not all_data_to_merge:
        if subfolders_processed == 0:
            print(f"在 '{root_folder_path}' 中没有找到任何子文件夹。")
        elif xyz_files_found == 0:
            print("没有找到任何有效的 _xyz.json 文件进行整合。")
        else: # 找到了文件但都无法解析
             print("所有找到的 _xyz.json 文件都无法解析或为空，无法生成合并文件。")
        return False

    # 4. 将整合后的数据写入到根目录下的新 _xyz.json 文件
    output_file_path = os.path.join(root_folder_path, "_xyz.json")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # indent=4 可以让输出的JSON文件更易读
            json.dump(all_data_to_merge, outfile, ensure_ascii=False, indent=4)
        print(f"\n成功！所有数据已整合到: {output_file_path}")
        print(f"共处理了 {subfolders_processed} 个子文件夹，找到了 {xyz_files_found} 个 _xyz.json 文件。")
        return True
    except Exception as e:
        print(f"写入整合文件 '{output_file_path}' 时发生错误: {e}")
        return False

# --- 使用示例 ---
if __name__ == "__main__":

    merge_xyz_json_files("/media/DGST_data/trajectory/264")

    # 测试一个没有子文件夹的路径 (例如当前脚本所在的目录，假设它没有合适的子文件夹)
    # print("\n--- 测试没有子文件夹的路径 ---")
    # merge_xyz_json_files(current_dir) # 这会扫描当前脚本目录