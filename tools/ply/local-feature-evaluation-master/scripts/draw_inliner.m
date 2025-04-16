% 读取图像

img1 = imread('E:\\make_dataset\\human373_64patch\\images\\1.bmp');
img2 = imread('E:\\make_dataset\\human373_64patch\\images\\4.bmp');

% 读取keypoints
key1 = read_keypoints('E:\\make_dataset\\human373_64patch\\keypoints\\1.bmp.bin');
key2 = read_keypoints('E:\\make_dataset\\human373_64patch\\keypoints\\4.bmp.bin');

% 读取matches
matches = read_matches('E:\\make_dataset\\human373_64patch\\matches\\1.bmp---4.bmp.bin');

% 从第几个match开始
start  = 1;
% 查看几个matches
showpointnumber = length(matches)-1;
matches = matches(start:start+showpointnumber,:);
matches = matches(1:150,:);
% 画图
% 左下角可以输入索引范围
draw_matches_inliers(img1, img2, key1(:,1:2), key2(:,1:2), matches);