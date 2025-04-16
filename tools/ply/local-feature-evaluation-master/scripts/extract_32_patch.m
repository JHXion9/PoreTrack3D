% 加载 VLFeat 库
VLFEAT_PATH = '/media/dataset_maker_matlab_python/vlfeat-0.9.21';
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));

% 定义文件夹路径
folder_path = '/media/human_face_need_test/brown_v2/use_eth_brownv2/017/EMO-1-shout+laugh/10/images';
folder_path_pathkp = '/media/human_face_need_test/brown_v2/use_eth_brownv2/017/EMO-1-shout+laugh/10/keypoints';
save_path ='/media/human_face_need_test/brown_v2/use_eth_brownv2/017/EMO-1-shout+laugh/10/descriptors'

% 获取文件夹中所有文件的信息
files = dir(fullfile(folder_path, '*.bmp'));  % 假设文件夹中包含 .jpg 文件  

% 创建进度条
h = waitbar(0, '处理进度');
total_files = length(files);

% 遍历所有文件，输出每个文件的完整路径
for i = 1:total_files
    % 更新进度条
    waitbar(i / total_files, h, sprintf('处理进度: %d/%d', i, total_files));
    
    % 获取文件名
    file_name = files(i).name;
    
    % 构建完整路径
    image_path = fullfile(folder_path, file_name);
    output_path = fullfile(save_path, strcat(file_name, '.bin.patches.mat'));
    
    % 检查文件是否存在
    if exist(output_path, 'file')
        disp(['文件已存在: ', output_path]);
        continue;  % 跳过当前文件的处理
    end
    
    disp(output_path);

    % 读取图像
    image = imread(image_path);

    % 如果图像不是灰度图像，则将其转换为灰度图像
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    image = im2single(image);
    
    % 读取关键点描述符
    pathkp = fullfile(folder_path_pathkp, strcat(file_name, '.bin'));
    kp = read_descriptors(pathkp);
    
    % 提取补丁
    patch = extract_patches(image, kp, 15);
    
    % 保存补丁
    save(output_path, 'patch');
end

% 关闭进度条
close(h);