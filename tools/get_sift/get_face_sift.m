clc
clear
close all
dist_eyes=1/1.6;
% 生成关键点的个数
no_kpt=8000;
%emotions={"disgust","fear","sadness","surprise","netural","happy"};
emotions={"EMO-1-shout+laugh"};
%people={"001","002","003","004","005","006","007","009","008","010","011"};
% 定义路径
path = 'E:\framesDataSet';
% 获取路径下的所有文件夹
folders = dir(path);
% 初始化一个字符串数组来保存文件夹名称
folderNames = {};
% 遍历文件夹数组
for i = 1:length(folders)
    % 检查是否为文件夹，并且名称不是 '.' 或 '..'
    if folders(i).isdir && ~ismember(folders(i).name, {'.', '..'})
        % 获取文件夹的名称
        folderName = folders(i).name;
        % 添加到字符串数组中
        folderNames{end+1} = folderName;
    end
end
% 删除特定的文件夹名称
%indicesToDelete = strcmp(folderNames, '017') | strcmp(folderNames, '018');
%folderNames(indicesToDelete) = [];
%  获得people数组
people = folderNames;

%need_only_generate_other_face,you_need_path利用这两个变量控制是否需要单帧输入生成正脸的SIFT
%如果不需要 直接复制need_only_generate_other_face=0后可以按着原本的操作
need_only_generate_other_face = 0
you_need_path = 'E:/dataset/nersemble/030/frame_00001/images-2x-73fps/4.bmp'
output_path1 = 'E:/dataset/nersemble/030/frame_00001/images-2x-73fps/4_front_kpt.txt'
output_path_yml1 = 'E:/dataset/nersemble/030/frame_00001/images-2x-73fps/4_front_kpt.yml'
output_path_yml_desc1 = 'E:/dataset/nersemble/030/frame_00001/images-2x-73fps/4_front_descriptor.yml'
descriptor_output_path1 = 'E:/dataset/nersemble/030/frame_00001/images-2x-73fps/4_front_descriptor.txt'
for e =1:1
emotion=emotions{e};
for p_num=1:length(people)
people_num=people{p_num}
%% load files

% im_face=imread('1_neutral.jpg');
% co_landmark=load('1_neutral.txt');
% mask_face_LU=[co_landmark(1,2)+100 co_landmark(1,1)-600];
% mask_face_RD=[co_landmark(17,2)-100 co_landmark(9,1)];
% im_crop=imcrop(im_face,[mask_face_LU 1200 1800]);
% im_crop=imresize(im_crop,0.5);
% imshow(im_crop);
% hold on
% plot(mask_face_LU(1),mask_face_LU(2),'*');
% hold on
% plot(mask_face_RD(1),mask_face_RD(2),'g*');
%%
%im_crop=rgb2gray(im_crop);:
%for cam = 0: 8

for cam=4:4
for frame=10:10

face_path = sprintf('E:/framesDataSet/%s/%s/%d/%d.bmp', people_num,emotion,frame,cam);
if need_only_generate_other_face ==1
    face_path = you_need_path
end
    

%face_path = sprintf('./fsmview_trainset/1/%s/1.jpg',emotion);
image_face = imread(face_path);  %读取原图像
image_face = rgb2gray(image_face);
%image_face = imrotate(image_face,90);


mask_path = sprintf('E:/framesDataSet/%s/%s/%d/mask/%d_mask.bmp', people_num,emotion,frame,cam);
% mask_path = sprintf('./fsmview_trainset/1/%s_mask/1_mask.jpg',emotion);
image_mask = imread(mask_path);  %读取mask
%image_mask = rgb2gray(image_mask);   
%image_mask = imrotate(image_mask,90);
% 为了加快检测速度，对图像进行缩放
scale=2
size(image_face)
size(image_mask)
size_p = size(image_face);
%裁剪出图片只有人脸的部分
im_crop=func_maskPadding(image_face,image_mask,size_p(2)*scale,size_p(1)*scale); %和原图的宽高比要一致。如1024:1224 == 2048:2448
%im_crop=image_face
%% detect pore
%检测毛孔
[frames,descriptors] = func_detect_pore_no(im_crop,no_kpt, 1);

%%
% face_path=sprintf('./a/A%d.jpg',rank)
%face_path=sprintf('./ucla_sift/1_1.jpg')    
%image_face=imread(face_path);  %读取原图像
% image_face = imrotate(image_face,90);
%image_face=rgb2gray(image_face);
% image_face=imresize(image_face,[3000,2000])
% mask_path=sprintf('./a_mask/A%d_mask.jpg',rank)
%mask_path=sprintf('./ucla_sift/1_1_mask.jpg')
%image_mask=imread(mask_path);  %读取mask
% image_mask=rgb2gray(image_mask);
% image_mask=imresize(image_mask,[3000,2000])
% s_img=size(image_face);
% figure(1);
% imshow(image_face);
% figure(2);
% imshow(image_mask);

%%change the interested area to 1
%im_crop=func_maskPadding(image_face,image_mask,1000,1500); %mask覆盖后的面部区域
% for i=1:s_img(1)
%     for j=1:s_img(2)
%         if image_mask(i,j)==255
%             image_mask(i,j)=1;
%         end
%     end
% end
% im_crop=image_face.*image_mask;
% im_crop=imresize(im_crop,0.5);
% imshow(im_crop);

%% detect pore
%[frames,descriptors] = func_detect_pore_no(im_crop,no_kpt,rank);

%%write point
%output_path = sprintf('../1_1point.txt');
%output_path =  sprintf('./fsmview_trainset/1/%s/fig1_kpt.txt',emotion);

output_path = sprintf('E:/framesDataSet/%s/%s/%d/%d_front_kpt.txt', people_num,emotion,frame,cam);
px=frames(1,:);
py=frames(2,:);
px=round(px);
py=round(py);
point=[px;py];
[m,n]=size(point) %m是数量，n是维度
output_path_yml=sprintf('E:/framesDataSet/%s/%s/%d/%d_front_kpt.yml',people_num,emotion,frame,cam)
output_path_yml_desc=sprintf('E:/framesDataSet/%s/%s/%d/%d_front_descriptor.yml',people_num,emotion,frame,cam)
if need_only_generate_other_face == 1
    output_path = output_path1
    output_path_yml = output_path_yml1
    output_path_yml_desc = output_path_yml_desc1
end
dym_matlab2opencv(point,output_path_yml)
dym_matlab2opencv(descriptors,output_path_yml_desc)
fid=fopen(output_path,'w');
%fid_descr=fopen(output_path_yml_desc,'w');
for i=1:n
fprintf(fid,'%i  ',point(1,i));
fprintf(fid,'%i\n',point(2,i));
end
fclose(fid);
%%write descriptor
% descriptor_output_path=sprintf('./36/%d_descriptor.txt', cam);
descriptor_output_path=sprintf('E:/framesDataSet/%s/%s/%d/%d_front_descriptor.txt',people_num,emotion,frame,cam)
if need_only_generate_other_face == 1
    descriptor_output_path=descriptor_output_path1
end
[m,n]=size(descriptors)
fid=fopen(descriptor_output_path,'w');
for i=1:n
   for j=1:m
      fprintf(fid,'%d  ',descriptors(j,i));
   end
      fprintf(fid,'\n');
end
fclose(fid);
if need_only_generate_other_face ==1
    break
end
end
end
end
end


