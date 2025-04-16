import os
import shutil
from tools.generate_ply import copy_frame, run_psift, get_patch, get_descriptors, get_matches, get_GMS_matches, get_ply, get_mesh
import time
from multiprocessing import Process, Value, Lock
import time
import math

    
# psift 多进程
def process_time_step(name, start_step, end_step, completed_steps, lock):
    """
    执行指定范围的时间步。

    Args:
        name: 进程名称。
        start_step: 起始时间步（包含）。
        end_step: 结束时间步（不包含）。
        completed_steps:  共享的已完成步数计数器 (Value)。
        lock: 锁，用于同步对 completed_steps 的访问。
    """
    for step in range(start_step, end_step):
        print(f'进程: {name}, 时间步: {step + 1}')
        kpts_path = os.path.join(dataset_call_matlab, str(human_number), f'EMO-1-shout+laugh/{step+1}', 'psiftproject', 'keypoints')
        if os.path.exists(kpts_path):
            if '16.png.bin' in os.listdir(kpts_path):
                pass
            else:
                run_psift(dataset_call_matlab, human_number, kpts_number, step+1)
        else:
            run_psift(dataset_call_matlab, human_number, kpts_number, step+1)

        with lock:  # 使用锁来确保对 completed_steps 的原子操作
            completed_steps.value += 1
    

if __name__ == '__main__':  # 确保在 if __name__ == '__main__': 块中运行多进程代码
    start_time = time.time()  # 记录开始时间
    # 你的参数
    human_number = '030'
    dataset_call_matlab = '/media/DGST_data/Test_Data'
    kpts_number = 10000
    times = 1
    gpu_id = 0
    num_processes = 15 # 进程数

    for frame in range(1, times+1):

        human_full_path = os.path.join(dataset_call_matlab, str(human_number), f'EMO-1-shout+laugh/{frame}')
        project_path = os.path.join(human_full_path,"psiftproject")
        project_images_path= os.path.join(project_path,"images")
        project_mask_path= os.path.join(project_path,"mask")

        if not os.path.exists(project_images_path):
            # 移动图片
            source_images_path = os.path.join("/media/DGST_data/Data", human_number)
            source_mask_path = os.path.join("/media/DGST_data/Data", human_number,'mask')

            copy_frame(source_images_path, project_images_path, frame)
            copy_frame(source_mask_path, project_mask_path, frame)

    print("images和mask 移动完毕")


    # 多进程1
    completed_steps = Value('i', 0)  # 'i' 表示整数类型
    lock = Lock()

    steps_per_process = math.ceil(times / num_processes)  # 向上取整，确保所有步骤都被执行
    process_list = []

    start = 0
    for i in range(num_processes):
        end = min(start + steps_per_process, times)  # 确保不超过 total_steps
        p = Process(target=process_time_step, args=(f'Process-{i+1}', start, end, completed_steps, lock))
        p.start()
        process_list.append(p)
        start = end

    for p in process_list:
        p.join()

    with lock: # 读取最终值也需要锁
        print(f'所有任务执行完毕，总共完成 {completed_steps.value} 步')

    # print("psift用时", time.time()-start_time)
    temp_time = time.time()
    # assert 0

    # # 循环跑match处理 
    # for frame in range(1, times+1):
  
    for frame in range(1, times+1):
        loop = True
        Loop_max = 0
        patch_folder_path, patch_save_path = get_patch(dataset_call_matlab, human_number, frame)
        get_descriptors(gpu_id, patch_folder_path, patch_save_path)
        # get_matches(dataset_call_matlab, human_number, frame)
        get_GMS_matches(dataset_call_matlab, human_number, frame)
        assert 0
        human_full_path = os.path.join(dataset_call_matlab, str(human_number), f'EMO-1-shout+laugh/{frame}')
        project_path = os.path.join(human_full_path,"psiftproject")
        
        while loop:
            loop = get_ply(project_path)
            if Loop_max >20:
                print(f"重建超过20次!! 该人物第{frame}帧，重建失败！！！")
                assert 0
            Loop_max+=1
            
        print("稠密点云重建时间", time.time()-temp_time)
        temp_time1 = time.time()
        


        if os.path.exists(project_path):
            if 'mesh.ply' in os.listdir(project_path):
                pass
            else:
                if os.path.exists(os.path.join(project_path,'dense/0/fused.ply')):
                    get_mesh(os.path.join(project_path,'dense/0/fused.ply'), os.path.join(project_path, f'mesh.ply') )
                else:
                    get_mesh(os.path.join(project_path,'dense/1/fused.ply'), os.path.join(project_path, f'mesh.ply') )

        print("mesh重建时间", time.time()-temp_time1)
   
    

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")