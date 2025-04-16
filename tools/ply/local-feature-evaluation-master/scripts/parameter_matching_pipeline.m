% Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

close all;
clear;
clc;
%DATASET_NAMES = {'human373_64patch_sosplus'};
% txt_path = getenv('GET_SIFT_PARAMETERS_TXT');
file_content = fileread("/media/Trajectory3D/tools/ply/matching_parameters.txt");

% 按行分割
lines = strsplit(file_content, '\n');
%读入获取参数
py_patch_size = '';
folder_path = '';
folder_path_pathkp = '';
save_path = '';

% 解析每一行
for i = 1:length(lines)
    line = lines{i};
    if contains(line, '=')
        % 按等号分割键值对
        parts = strsplit(line, '=');
        key = strtrim(parts{1});   % 去除空格
        value = strtrim(parts{2}); % 去除空格
        
        % 根据键赋值
        switch key
            case 'dataset_root'
                DATASET_ROOT = value;
            case 'match_human_number'
                DATASET_NAMES = {value};
            case 'which_time'
                which_time = value;
            otherwise
                warning('未知参数: %s', key);
        end
    end
end



% 输出参数
fprintf('DATASET_ROOT: %s\n', DATASET_ROOT);
disp(DATASET_NAMES)





%DATASET_NAMES = {'human373'};
for i = 1:length(DATASET_NAMES)
    %% Set the pipeline parameters.
    % TODO: Change this to where your dataset is stored. This directory should
    %       contain an "images" folder and a "database.db" file.
    % 原始路径
    original_path = '/EMO-1-shout+laugh/10/psiftproject';

    % 替换路径中的 '10' 为 which_time 的值
    new_path = strrep(original_path, '10', which_time);
    
    DATASET_PATH = [DATASET_ROOT '/' DATASET_NAMES{i} new_path];
    disp(DATASET_PATH)
    % TODO: Change this to where VLFeat is located.
    VLFEAT_PATH = '/media/Trajectory3D/tools/vlfeat-0.9.21';

    % TODO: Change this to where the COLMAP build directory is located.
    COLMAP_PATH = '/media/DenseGSTracking/colmap/build';

    % Radius of local patches around each keypoint.
    PATCH_RADIUS = 64;

    % Whether to run matching on GPU.
    MATCH_GPU = gpuDeviceCount() > 0;

    % Number of images to match in one block.
    MATCH_BLOCK_SIZE = 50;

    % Maximum distance ratio between first and second best matches.
    MATCH_MAX_DIST_RATIO = 0.95;
    %MATCH_MAX_DIST_RATIO = 0.9;
    % Mnimum number of matches between two images.
    MIN_NUM_MATCHES = 15;

    %% Setup the pipeline environment.

    run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));

    IMAGE_PATH = fullfile(DATASET_PATH, 'images');
    KEYPOINT_PATH = fullfile(DATASET_PATH, 'keypoints');
    DESCRIPTOR_PATH = fullfile(DATASET_PATH, 'descriptors');
    MATCH_PATH = fullfile(DATASET_PATH, 'matches');
    DATABASE_PATH = fullfile(DATASET_PATH, 'database.db');

    %% Create the output directories.

    if ~exist(KEYPOINT_PATH, 'dir')
        mkdir(KEYPOINT_PATH);
    end
    if ~exist(DESCRIPTOR_PATH, 'dir')
        mkdir(DESCRIPTOR_PATH);
    end
    if ~exist(MATCH_PATH, 'dir')
        mkdir(MATCH_PATH);
    end

    %% Extract the image names and paths.
    
    %查看图片路径下的所有图片名字
    image_files = dir(IMAGE_PATH);
    % 图片的数目，减2是减去没有用的其中两个
    num_images = length(image_files) - 2;
    %生成空数组，用于存放图片的文件路径
    image_names = cell(num_images, 1);
    image_paths = cell(num_images, 1);
    keypoint_paths = cell(num_images, 1);
    descriptor_paths = cell(num_images, 1);
    % 生成所有图片对应的文件路径 以及关键点路径 以及描述子路径
    for i = 3:length(image_files)
        image_name = image_files(i).name;
        image_names{i-2} = image_name;
        image_paths{i-2} = fullfile(IMAGE_PATH, image_name);
        keypoint_paths{i-2} = fullfile(KEYPOINT_PATH, [image_name '.bin']);
        descriptor_paths{i-2} = fullfile(DESCRIPTOR_PATH, [image_name '.bin']);
    end

    %% TODO: Compute the keypoints and descriptors.

    %feature_extraction_root_sift;
    % feature_extraction_pca_sift
    % etc.

    %% Match the descriptors.
    %
    %  NOTE: - You must exhaustively match Fountain, Herzjesu, South Building,
    %          Madrid Metropolis, Gendarmenmarkt, and Tower of London.
    %        - You must approximately match Alamo, Roman Forum, Cornell.
      
    %执行穷举匹配
    if num_images < 2000
        exhaustive_matching
    else
        VOCAB_TREE_PATH = fullfile(DATASET_PATH, 'Oxford5k/vocab-tree.bin');
        approximate_matching
    end
end
