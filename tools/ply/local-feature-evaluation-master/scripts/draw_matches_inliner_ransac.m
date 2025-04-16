function draw_matches_inliner_ransac(img1, img2, keypoints1, keypoints2, matches)
    % 检查输入图像是否为彩色图像，如果是，则转换为灰度图像
    if size(img1, 3) == 3
        img1 = rgb2gray(img1);
    end
    if size(img2, 3) == 3
        img2 = rgb2gray(img2);
    end

    % 拼接两张图片
    [img1Height, img1Width] = size(img1);
    [img2Height, img2Width] = size(img2);
    imgCombined = [img1, img2];

    % 绘制拼接后的图像
    figure;
    imshow(imgCombined);
    hold on;

    % 提取匹配点的坐标
    matchedPoints1 = keypoints1(matches(:, 1), :); % 第一张图片的匹配点
    matchedPoints2 = keypoints2(matches(:, 2), :); % 第二张图片的匹配点

    % 调整第二张图片的匹配点坐标（x 坐标加上第一张图片的宽度）
    matchedPoints2(:, 1) = matchedPoints2(:, 1) + img1Width;

    % 绘制匹配点
    plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
    plot(matchedPoints2(:, 1), matchedPoints2(:, 2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

    % 绘制匹配点之间的连线（使用蓝色）
    for i = 1:size(matches, 1)
        plot([matchedPoints1(i, 1), matchedPoints2(i, 1)], [matchedPoints1(i, 2), matchedPoints2(i, 2)], 'b', 'LineWidth', 1);
    end

    % 使用 RANSAC 计算内点
    [tform, inlierPoints1, inlierPoints2] = estimateGeometricTransform2D(...
        matchedPoints1, matchedPoints2, 'projective', 'MaxNumTrials', 1000, 'Confidence', 99.9);

    % 提取内点的坐标
    inlierPoints1 = inlierPoints1.Location;
    inlierPoints2 = inlierPoints2.Location;
    inlierPoints2(:, 1) = inlierPoints2(:, 1) + img1Width; % 调整第二张图片的内点坐标

    % 绘制内点
    plot(inlierPoints1(:, 1), inlierPoints1(:, 2), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
    plot(inlierPoints2(:, 1), inlierPoints2(:, 2), 'r*', 'MarkerSize', 10, 'LineWidth', 2);

    % 绘制内点之间的连线（使用绿色）
    for i = 1:size(inlierPoints1, 1)
        plot([inlierPoints1(i, 1), inlierPoints2(i, 1)], [inlierPoints1(i, 2), inlierPoints2(i, 2)], 'g', 'LineWidth', 1);
    end

    hold off;
    title('Matches and Inliers between two images');
end