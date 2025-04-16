clc
clear
close all
dist_eyes=1/1.6;
addpath('/media/dataset_maker_matlab_python/get_sift/imsmooth.mexw64');
% ���ɹؼ���ĸ���,����֮������дһ���������Ϊeth��ԭ����һЩ�ؼ����binҲ��1w3����
no_kpt=8000;
%emotions={"disgust","fear","sadness","surprise","netural","happy"};
emotions={"EMO-1-shout+laugh"};
%people={"001","002","003","004","005","006","007","009","008","010","011"};
% ����·��
path = '/media/human_face_need_test/brown_v2/use_eth_brownv2';
% ��ȡ·���µ������ļ���
folders = dir(path);
% ��ʼ��һ���ַ��������������ļ�������
folderNames = {};
% �����ļ�������
for i = 1:length(folders)
    % ����Ƿ�Ϊ�ļ��У��������Ʋ��� '.' �� '..'
    if folders(i).isdir && ~ismember(folders(i).name, {'.', '..'})
        % ��ȡ�ļ��е�����
        folderName = folders(i).name;
        % ���ӵ��ַ���������
        folderNames{end+1} = folderName;
    end
end
% ɾ���ض����ļ�������
%indicesToDelete = strcmp(folderNames, '017') | strcmp(folderNames, '018');
%folderNames(indicesToDelete) = [];
%  ���people����
people = folderNames;

people ={"017"};
for e =1:1
    emotion=emotions{e};
    for p_num=1:length(people)
    people_num=people{p_num}
    
        for cam = 1:16
                % ����11����ΪĿǰ11��ⲻ������
                if cam == 11
                    continue;
               end
                if cam == 9
                   continue;
                end
               %֡��ѡ10 ��Ϊ�˶�Ӧǰ��洢�����ݸ�ʽ����ʵѡʲô������ν��ֻ��Ҫд��·������
            for frame=10:10
            
            face_path = sprintf('/media/human_face_need_test/brown_v2/use_eth_brownv2/%s/%s/%d/images/%d.bmp', people_num,emotion,frame,cam);
            image_face = imread(face_path); 
            image_face = rgb2gray(image_face);

            
            
            mask_path = sprintf('/media/human_face_need_test/brown_v2/use_eth_brownv2/%s/%s/%d/mask/%d_mask.bmp', people_num,emotion,frame,cam);
            image_mask = imread(mask_path);  %��ȡmask

            % Ϊ�˼ӿ����ٶȣ���ͼ���������
            scale=2
            size(image_face)
            size(image_mask)
            size_p = size(image_face);

            %�ü���ͼƬֻ�������Ĳ���
            %�ú���ͨ������Ŵ���h��w�Ͷ�Ӧ��ͼƬ��mask����ͼƬ�Ŵ���ٽ���mask��ȡ��������
            im_crop=func_maskPadding(image_face,image_mask,size_p(2)*scale,size_p(1)*scale); %��ԭͼ�Ŀ��߱�Ҫһ�¡���1024:1224 == 2048:2448

            %% detect pore
            %���ë�ף�ͨ��pore�ı�������һ��������kptno
            [frames,descriptors] = func_detect_pore_no(im_crop,no_kpt, 1);
            
            %keypoint�ļ��Ĵ���
            output_dir = sprintf('/media/human_face_need_test/brown_v2/use_eth_brownv2/%s/%s/%d/keypoints', people_num,emotion,frame);
            if exist(output_dir, 'dir') ~= 7
                % ����ļ��в����ڣ��򴴽��ļ���
                mkdir(output_dir);
                disp(['Folder created: ', output_dir]);
            else
                disp(['Folder already exists: ', output_dir]);
            end



            px=frames(1,:);
            py=frames(2,:);
            pscale=frames(3,:);
            pori=frames(4,:);
            %ǰ������px��py���ڷŴ��������λ�ý��м���ó��Ĺؼ���λ�ã�����������Ҫ�������Żض�Ӧ�Ĵ�С�Դ˱�������
            px=round(px);
            py=round(py);
            px = px / 2;
            py = py / 2;
            % �� px �� py ת��Ϊ single ����
            %px = single(px);
            %py = single(py);
            %pscale = single(pscale);
            %pori = single(pori);
            %ɸѡ��������λ�Ĺؼ��㣬��Ϊ��ͼƬ���й�����������ĳЩ���λ�ò���
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
            



            % ��ȡͼƬ
            img = imread(face_path); % �滻Ϊ���ͼƬ�ļ�·��
            figure;
            % ��ʾͼƬ
            imshow(img);
            hold on; % ����ͼ����ʾ
            
            % ������������󣬼������Ѿ������������
            % ÿһ�еĵ�һ����x���꣬�ڶ�����y����
            featurePoints = point; % �滻Ϊʵ�ʵ�����������
            
            % ��ͼƬ�ϻ���������
            plot(featurePoints(:,1), featurePoints(:,2), 'ro'); % 'ro'��ʾ��ɫԲȦ���
            
            % ��ѡ������������ı��
%             for i = 1:size(featurePoints, 1)
%                 text(featurePoints(i,1), featurePoints(i,2), num2str(i), 'Color', 'yellow', 'FontSize', 12);
%             end
            
            % ȡ������״̬
            hold off;

            kps_photopath = 'featurePointsImage.png'; % �����Ը�����Ҫ�޸��ļ���
            saveas(gcf, fullfile(output_dir, kps_photopath));






            % д��bin ����
            filename = sprintf('%d.bmp.bin', cam);
            output_path = fullfile(output_dir, filename);
            write_keypoints(output_path,point)

            end
        end
    end
end


