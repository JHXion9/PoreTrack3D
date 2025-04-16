import numpy as np
import torch
from utils_tra.cotracker.predictor import CoTrackerPredictor
from utils_tra.PSIFT import Psift
from utils_tra.Hynet.model import HyNet
from utils_tra.utils import *
from utils_tra.eyes_mouse import get_eye_landmarks, get_mouse_landmarks
from configs.option import get_option 

def base_view_tracker(opt, first_frame_base, cam_datadir, mesh_path, tracker_model):
    base_view = opt.base_view
    # 初始化读取图像的路径
    images_base = [cv2.imread(f'/media/DGST_data/Data/{opt.people_id}/cam{str(base_view).zfill(2)}/frame_{str(i).zfill(5)}.png', 0) for i in range(1, opt.frame_num+1)]

    video_base = getfromvideo(opt).cuda()

    # 获取相机参数
    # params_base, R_base, T_base = get_k_w2c(cam_datadir, str(base_view), timestamp = 1)
    params_base, R_base, T_base = get_k_w2c_flame(cam_datadir, str(base_view))

    # 跟踪patch绑定mesh
    _, first_frame_base_p3d = get_3d_coordinates_flame(mesh_path[0], params_base, R_base, T_base, first_frame_base)

    if not first_frame_base_p3d.size > 0:  # 检查NumPy数组是否为空
        print("错误：第一帧定位关键点为空，程序终止。")
        sys.exit(1)  # 使用非零状态码表示错误

    # 计算跟踪patch的粗略位置
    queries_base = transform_array(first_frame_base)
    landmarks_base_1, _ = tracker_model(video_base, queries_base[None])
    landmarks_base = landmarks_base_1.cpu().numpy()[0] #(1, nums_frame, 1, 2)-> (nums_frame, 1, 2)
    landmarks_base = landmarks_base.squeeze(1) # (nums_frame, 1, 2) -> (nums_frame, 2)

    # 切割patch，并在patch里提取关键点和特征描述子
    past_patches = crop_regions_around_keypoints(images_base[0], landmarks_base[0], opt.patch_radius)
    past_psift_base_0 = Psift(past_patches, octave_num=2, sigma1=1.1, upSampling=False)
    past_points, past_features = past_psift_base_0.GetPSIFTFeature(HyNet_model)
    points_reset,_ = reset_resolution(landmarks_base[0], landmarks_base[0], past_points, past_points, opt.patch_radius) # 补偿至切割patch前的坐标

    # 判断 如果是跟踪除了pore以外其他关键点，就需要增加新关键点
    if opt.position == 'eyes':
        print("跟踪眼角")
        image = cv2.imread(f'/media/DGST_data/Data/{opt.people_id}/cam{str(base_view).zfill(2)}/frame_00001.png', 1) 
        eyes_point = get_eye_landmarks(image)
        psift_eye = Psift(images_base[0], octave_num=2, sigma1=1.1, upSampling=False)
        eyes_feature, src_eye = psift_eye.GetFeature_hynet(eyes_point, HyNet_model)
        eyes_point_low = to_resolution(landmarks_base[0], src_eye, opt.patch_radius)


        past_points = np.concatenate((past_points, eyes_point_low), axis=0)
        past_features = np.concatenate((past_features, eyes_feature.reshape(1,-1)), axis=0)
        points_reset = np.concatenate((points_reset, eyes_point), axis=0)
        

    elif opt.position == 'mouse':
        image = cv2.imread(f'/media/DGST_data/Data/{opt.people_id}/cam{str(base_view).zfill(2)}/frame_00001.png', 1) 
        mouse_point = get_mouse_landmarks(image)
        psift_mouse = Psift(images_base[0], octave_num=2, sigma1=1.1, upSampling=False)
        mouse_feature, src = psift_mouse.GetFeature_hynet(mouse_point, HyNet_model)
        mouse_point_low = to_resolution(landmarks_base[0], src, opt.patch_radius)


        past_points = np.concatenate((past_points, mouse_point_low), axis=0)
        past_features = np.concatenate((past_features, mouse_feature.reshape(1,-1)), axis=0)
        points_reset = np.concatenate((points_reset, mouse_point), axis=0)
       

    # elif opt.position == 'mole':
        

    # 第一帧关键点绑定点云筛选
    pts2d, pts3d = get_3d_coordinates_flame(mesh_path[0], params_base, R_base, T_base, points_reset)
    print("第1帧关键点检测","-----","筛选前的关键点数量", len(points_reset), "筛选后的关键点数量", len(pts2d))
    indice = [np.where((points_reset == s).all(axis=1))[0][0] for s in pts2d]# 找通过mesh绑定筛选后 有效关键点的位置
    past_pts_filter = past_points[indice]
    past_features_filter = past_features[indice]


    all_keypoints = [pts2d]
    all_matches = []
    all_p3d = [pts3d]
    # 完整信息
    whole_matches = []
    whole_keypoints = [np.array(points_reset)]

    for time in range(1, opt.frame_num):
        patch_base = crop_regions_around_keypoints(images_base[time], landmarks_base[time], opt.patch_radius)
        psift_base_0 = Psift(patch_base, octave_num=2, sigma1=1.1, upSampling=False)
        points_base_0, feature_base_0  = psift_base_0.GetPSIFTFeature(HyNet_model)
        points_reset,_ = reset_resolution(landmarks_base[time], landmarks_base[time], points_base_0, points_base_0, opt.patch_radius)

        # 不筛选点云   信息保存
        whole_keypoints.append(np.array(points_reset))

        # 绑定点云前 进行匹配
        src, dst = Psift.match(
            [past_features, past_points, past_patches],
            [feature_base_0, points_base_0, patch_base], 
            None, ratio=0.85, RANSAC=True, kmeans=False, dispaly=False, plot_match=False)

        match_matrix_whole = create_match_matrix_from_points(past_points, points_base_0, src, dst)
        whole_matches.append(match_matrix_whole)


        past_points, past_features = points_base_0, feature_base_0

        # 每帧点云绑定筛选
        # params_base, R_base, T_base = get_k_w2c(cam_datadir, str(base_view), timestamp = time+1)
        pts2d, pts3d = get_3d_coordinates_flame(mesh_path[time], params_base, R_base, T_base, points_reset)
        print(f"第{time+1}帧关键点检测","-----","筛选前的关键点数量", len(points_reset), "筛选后的关键点数量", len(pts2d))
        indice = [np.where((points_reset == s).all(axis=1))[0][0] for s in pts2d]
        points_filter = points_base_0[indice]
        features_filter = feature_base_0[indice]
    
        all_keypoints.append(pts2d)
        all_p3d.append(pts3d)
        
        # 绑定点云后 进行匹配
        src_filter, dst_filter = Psift.match(
            [past_features_filter, past_pts_filter, past_patches],
            [features_filter, points_filter, patch_base],
            None, ratio=0.85, RANSAC=True, kmeans=False, dispaly=False, plot_match=False)

        

        match_matrix = create_match_matrix_from_points(past_pts_filter, points_filter, src_filter, dst_filter)
        all_matches.append(match_matrix)

        past_patches = patch_base
        past_pts_filter, past_features_filter = points_filter, features_filter

    # 保存筛选点云后的信息
    save_keypoint_data(all_keypoints, all_p3d, all_matches, output_path=f'./out/{opt.people_id}', view=str(opt.base_view))
    # save_keypoint_data(None, None, all_matches, output_path=f'./out/{opt.people_id}', view='debug')
    # 保存筛选点云前的信息
    save_keypoint_data(whole_keypoints, None, whole_matches, output_path=f'./out/{opt.people_id}', view='whole')

    return first_frame_base_p3d

def other_view_tracker(opt, cam_datadir, first_frame_base_p3d, tracker_model):
    point_clouds = load_p3ds_from_json(f'./out/{opt.people_id}/all_p3ds.json')
    other_view = [ opt.base_view-2, opt.base_view+2]

    for idx, view in enumerate(other_view):
        images_other = [cv2.imread(f'/media/DGST_data/Data/{opt.people_id}/cam{str(view).zfill(2)}/frame_{str(i).zfill(5)}.png', 0) for i in range(1, opt.frame_num+1)]
        video_other = getfromvideo(opt).cuda()

        # params, R, T = get_k_w2c(cam_datadir, str(view), timestamp = 1)
        params, R, T = get_k_w2c_flame(cam_datadir, str(view))
        
        # 确定第一帧patch关键点
        first_frame_other = reproject_points_pinhole(params, R, T, first_frame_base_p3d)
        # 计算跟踪patch的粗略位置
        queries_other = transform_array(first_frame_other)
        landmarks_other_1, _ = tracker_model(video_other, queries_other[None])
        landmarks_other = landmarks_other_1.cpu().numpy()[0] #(1, nums_frame, 1, 2)-> (nums_frame, 1, 2)
        landmarks_other = landmarks_other.squeeze(1) # (nums_frame, 1, 2) -> (nums_frame, 2)
        
        # 确定第一帧该视角 需要匹配的关键点
        points_proj = reproject_points_pinhole(params, R, T, point_clouds[0]).reshape(-1, 2)
        
        past_patch = crop_regions_around_keypoints(images_other[0], landmarks_other[0], opt.patch_radius)
        psift_other = Psift(images_other[0], octave_num=2, sigma1=1.1, upSampling=False)
        past_features, src_opther = psift_other.GetFeature_hynet(points_proj, HyNet_model)
        past_points = to_resolution(landmarks_other[0], src_opther, opt.patch_radius)

        past_proj = points_proj
        all_keypoints = [past_proj]
        all_matches = []
        
        for time in range(1, opt.frame_num):
            # params, R, T = get_k_w2c(cam_datadir, str(view), timestamp=time+1)
            points_proj = reproject_points_pinhole(params, R, T, point_clouds[time]).reshape(-1, 2)

            patch_other = crop_regions_around_keypoints(images_other[time], landmarks_other[time], opt.patch_radius)
            psift_other = Psift(images_other[time], octave_num=2, sigma1=1.1, upSampling=False)
            features_other, src_opther = psift_other.GetFeature_hynet(points_proj, HyNet_model)
            points_other = to_resolution(landmarks_other[time], src_opther, opt.patch_radius)

            src, dst = Psift.match(
                [past_features, past_points, past_patch],
                [features_other, points_other, patch_other],
                None, ratio=0.85, RANSAC=True, kmeans=False, dispaly=False, plot_match=False)
            
            match_matrix = create_match_matrix_from_points(past_points, points_other, src, dst)
            all_matches.append(match_matrix)
            all_keypoints.append(points_proj)
            # 更新模板
            past_features, past_points, past_proj, past_patch = features_other, points_other, points_proj, patch_other
        
        save_keypoint_data(all_keypoints, None, all_matches, output_path=f'./out/{opt.people_id}', view=str(view))

if __name__ == '__main__':
    HyNet_model = HyNet()
    HyNet_model.load_state_dict(torch.load('./utils_tra/modelpath/HyNet_LIB.pth'))
    opt = get_option()

    # 正脸第一帧需要跟踪的patch中心

    first_frame_base = np.array([[1516, 1254.]])

    # 相机参数路径
    cam_datadir = f"/media/VHAP/data/camera_params/{opt.people_id}/camera_params.json"
    # mesh路径
    mesh_path= [f'/media/VHAP/data/{opt.people_id}/output/2025-03-02_09-36-38/eval_30/mesh/frame_{str(frame).zfill(5)}.obj' for frame in range(opt.frame_num)]
    # 保存路径
    output_dir = f"/media/DGST_data/trajectory/{opt.people_id}"

    # patch 使用的CoTracker跟踪
    tracker_model = CoTrackerPredictor(checkpoint='./utils_tra/cotracker/checkpoints/cotracker2.pth')
    if torch.cuda.is_available():
        tracker_model = tracker_model.cuda()

    print("-----------------------------------------------")
    print("---------------进行正脸Psift跟踪----------------")
    print("-----------------------------------------------")
    first_frame_base_p3d = base_view_tracker(opt, first_frame_base, cam_datadir, mesh_path, tracker_model)

    print("-----------------------------------------------")
    print("---------------进行侧脸Psift跟踪----------------")
    print("-----------------------------------------------")
    other_view_tracker(opt, cam_datadir, first_frame_base_p3d, tracker_model)

    # 利用其他视角的匹配矩阵 补充绑定mesh后的正脸的匹配矩阵
    complete_mat = complete_match_matrices(f'./out/{opt.people_id}', base_view = str(opt.base_view))
    save_keypoint_data(None, None, complete_mat, output_path=f'./out/{opt.people_id}', view=f'{str(opt.base_view)}_complete')

    # 绑定mesh后的正脸匹配矩阵 补全 绑定mesh前的正脸匹配矩阵（害怕筛选掉点）
    full_mat = full_match_matrices(f'./out/{opt.people_id}/whole_match_matrices.json', f'./out/{opt.people_id}/whole_keypoints.json',\
                                    f'./out/{opt.people_id}/all_match_matrices_{str(opt.base_view)}_complete.json', f'./out/{opt.people_id}/all_keypoints_{str(opt.base_view)}.json')

    longest_match_start_frame, longest_match_start_keypoint, longest_match_length = find_longest_match_trajectory(full_mat)
    if longest_match_start_frame != -1:
        print(f"Longest match trajectory starts at frame: {longest_match_start_frame}, keypoint index: {longest_match_start_keypoint}, 长度: {longest_match_length}")
    else:
        print("No match trajectory found.")

    keypoints_json = f"./out/{opt.people_id}/whole_keypoints.json"

    # match_matrices_list = load_match_matrices_from_json(match_matrices_json)
    keypoints = load_keypoints_from_json(keypoints_json)

    # 保存所有轨迹信息
    # nums_traj = save_all_trajectories(full_mat, keypoints, f'./out/{opt.people_id}')
    # print(f"总共有{nums_traj}条轨迹")

    which_frame = longest_match_start_frame
    keypoint_idx = longest_match_start_keypoint

    if opt.position == 'pore':
        completed_trajectory, color_flag = complete_keypoint_trajectory(full_mat, keypoints, which_frame, keypoint_idx)
    else:
        completed_trajectory, color_flag = complete_keypoint_trajectory(full_mat, keypoints, 0, len(keypoints[0])-1)

    # 保存完整轨迹
    images_base1 = [cv2.imread(f'/media/DGST_data/Data/{opt.people_id}/cam{str(opt.base_view).zfill(2)}/frame_{str(i).zfill(5)}.png') for i in range(1, opt.frame_num+1)]
    draw_trajectories(full_mat, keypoints, completed_trajectory, color_flag, images_base1, f'./out/{opt.people_id}/output.mp4')