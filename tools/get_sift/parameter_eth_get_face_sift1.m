clc
clear
close all


file_content = fileread('/media/Trajectory3D/tools/ply/get_sift_parameters.txt');

% 按行分割
lines = strsplit(file_content, '\n');
%读入获取参数
% 初始化参数
py_path = '';
py_people = '';
no_kpt_py = 0;

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
            case 'dataset_path_call_matlab'
                py_path = value;
            case 'human_number'
                py_people = value;
            case 'kpts_number'
                no_kpt_py = str2double(value); % 转换为数值
            otherwise
                warning('未知参数: %s', key);
        end
    end
end



% 输出参数
fprintf('dataset_path: %s\n', py_path);
fprintf('human_number: %s\n', py_people);
fprintf('kpts_number: %d\n', no_kpt_py);






dist_eyes=1/1.6;
disp(666)
%    ɹؼ   ĸ   ,    ֮      дһ         Ϊeth  ԭ    һЩ ؼ    binҲ  1w3    
no_kpt=no_kpt_py;
%emotions={"disgust","fear","sadness","surprise","netural","happy"};
emotions={"EMO-1-shout+laugh"};
%people={"001","002","003","004","005","006","007","009","008","010","011"};
%     ·  
path = py_path;
pre_path = py_path;
%   ȡ·   µ      ļ   
folders = dir(path);
%   ʼ  һ   ַ              ļ       
folderNames = {};
%      ļ       
for i = 1:length(folders)
    %     Ƿ Ϊ ļ  У        Ʋ    '.'    '..'
    if folders(i).isdir && ~ismember(folders(i).name, {'.', '..'})
        %   ȡ ļ  е     
        folderName = folders(i).name;
        if ~isempty(regexp(folderName, '^\d+$', 'once'))
            % 如果是，则添加到 folderNames 中
            folderNames{end+1} = folderName;
        end
    end
end
% ɾ   ض    ļ       
%indicesToDelete = strcmp(folderNames, '017') | strcmp(folderNames, '018');
%folderNames(indicesToDelete) = [];
%     people    
people = folderNames;

%people ={"017"};
people = {py_people}
for e =1:1
    emotion=emotions{e};
    for p_num=1:length(people)
    people_num=people{p_num}
    
        for cam = 1:16
              
            %֡  ѡ10   Ϊ ˶ Ӧǰ  洢     ݸ ʽ    ʵѡʲô      ν  ֻ  Ҫд  ·      
            for frame=80:80
            
            face_path = sprintf('%s/%s/%s/%d/psiftproject/images/%d.png', pre_path,people_num,emotion,frame,cam);
            image_face = imread(face_path); 
            image_face = rgb2gray(image_face);

            
            
            mask_path = sprintf('%s/%s/%s/%d/psiftproject/mask/%d_mask.png', pre_path,people_num,emotion,frame,cam);
            image_mask = imread(mask_path);  %  ȡmask

            % Ϊ ˼ӿ    ٶȣ   ͼ         
            scale=2
            size(image_face)
            size(image_mask)
            size_p = size(image_face);

            % ü   ͼƬֻ       Ĳ   
            % ú   ͨ      Ŵ   h  w Ͷ Ӧ  ͼƬ  mask    ͼƬ Ŵ   ٽ   mask  ȡ        
            im_crop=func_maskPadding(image_face,image_mask,size_p(2)*scale,size_p(1)*scale); %  ԭͼ Ŀ  ߱ Ҫһ ¡   1024:1224 == 2048:2448

            %% detect pore
            %   ë ף ͨ  pore ı       һ        kptno
            [frames,descriptors] = func_detect_pore_no(im_crop,no_kpt, 1);
            
            %keypoint ļ  Ĵ   
            output_dir = sprintf('%s/%s/%s/%d/psiftproject/keypoints', pre_path,people_num,emotion,frame);
            if exist(output_dir, 'dir') ~= 7
                %     ļ  в    ڣ  򴴽  ļ   
                mkdir(output_dir);
                disp(['Folder created: ', output_dir]);
            else
                disp(['Folder already exists: ', output_dir]);
            end



            px=frames(1,:);
            py=frames(2,:);
            pscale=frames(3,:);
            pori=frames(4,:);
            %ǰ      px  py   ڷŴ        λ ý  м   ó  Ĺؼ   λ ã           Ҫ       Żض Ӧ Ĵ С Դ˱       
            px=round(px);
            py=round(py);
            px = px / 2;
            py = py / 2;
            %    px    py ת  Ϊ single     
            %px = single(px);
            %py = single(py);
            %pscale = single(pscale);
            %pori = single(pori);
            %ɸѡ        λ Ĺؼ  㣬  Ϊ  ͼƬ   й           ĳЩ   λ ò   
%             [mask_y, mask_x] = find(image_mask == 255);
%             px_col = px';
%             py_col = py';
%             valid_indices = ismember([px_col, py_col], [mask_x, mask_y], 'rows');
%             px_col = px_col(valid_indices);
%             py_col = py_col(valid_indices);
%             
%             
% 
%             point = [px_col, py_col];




            point = [px', py',pscale',pori'];
            



            %   ȡͼƬ
            img = imread(face_path); %  滻Ϊ   ͼƬ ļ ·  
            figure;
            %   ʾͼƬ
            imshow(img);
            hold on; %     ͼ    ʾ
            
            %             󣬼      Ѿ            
            % ÿһ еĵ һ    x   ꣬ ڶ     y    
            featurePoints = point; %  滻Ϊʵ ʵ           
            
            %   ͼƬ ϻ         
            plot(featurePoints(:,1), featurePoints(:,2), 'ro'); % 'ro'  ʾ  ɫԲȦ   
            
            %   ѡ            ı  
%             for i = 1:size(featurePoints, 1)
%                 text(featurePoints(i,1), featurePoints(i,2), num2str(i), 'Color', 'yellow', 'FontSize', 12);
%             end
            
            % ȡ      ״̬
            hold off;

            kps_photopath = 'featurePointsImage.png'; %      Ը     Ҫ ޸  ļ   
            saveas(gcf, fullfile(output_dir, kps_photopath));






            % д  bin     
            filename = sprintf('%d.png.bin', cam);
            output_path = fullfile(output_dir, filename);
            write_keypoints(output_path,point)

            end
        end
    end
end


