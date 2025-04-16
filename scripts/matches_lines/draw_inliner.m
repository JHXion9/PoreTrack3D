% 读取图像

img1 = imread('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/images/1.png');
img2 = imread('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/images/2.png');

% 读取keypoints
key1 = read_keypoints('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/keypoints/1.png.bin');
key2 = read_keypoints('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/keypoints/2.png.bin');

diary('output.txt');
% ... 你的代码 ...

% 读取keypoints
key1 = read_keypoints('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/keypoints/1.png.bin');
key2 = read_keypoints('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/keypoints/2.png.bin');

disp(['key1: ', num2str(size(key1, 1))]); % 显示 key1 的行数
disp('*********************************');
disp('key1:');
disp(key1);       % 直接显示 key1 的内容
disp('*********************************');
diary off;

% 读取matches
matches = read_matches('/media/DGST_data/Test_Data/076/EMO-1-shout+laugh/28/psiftproject/matches/1.png---2.png.bin');

% 从第几个match开始
start  = 1;
% 查看几个matches
showpointnumber = length(matches)-1;
matches = matches(start:start+showpointnumber,:);
matches = matches(1:150,:);
% 画图
% 左下角可以输入索引范围
draw_matches_inliers(img1, img2, key1(:,1:2), key2(:,1:2), matches);