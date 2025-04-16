% function draw_matches_inliers(img1, img2, keypoints1, keypoints2, matches)
%     % 检查输入图像是否为彩色图像，如果是，则转换为灰度图像
%     if size(img1, 3) == 3
%         img1 = rgb2gray(img1);
%     end
%     if size(img2, 3) == 3
%         img2 = rgb2gray(img2);
%     end
% 
%     % 拼接两张图片
%     [img1Height, img1Width] = size(img1);
%     [img2Height, img2Width] = size(img2);
%     imgCombined = [img1, img2];
% 
%     % 绘制拼接后的图像
%     figure;
%     imshow(imgCombined);
%     hold on;
% 
%     % 提取匹配点的坐标
%     matchedPoints1 = keypoints1(matches(:, 1), :); % 第一张图片的匹配点
%     matchedPoints2 = keypoints2(matches(:, 2), :); % 第二张图片的匹配点
% 
%     % 调整第二张图片的匹配点坐标（x 坐标加上第一张图片的宽度）
%     matchedPoints2(:, 1) = matchedPoints2(:, 1) + img1Width;
% 
%     % 绘制匹配点
%     plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'go', 'MarkerSize', 5, 'LineWidth', 2);
%     plot(matchedPoints2(:, 1), matchedPoints2(:, 2), 'ro', 'MarkerSize', 5, 'LineWidth', 2);
% 
%     % 绘制匹配点之间的连线（使用蓝色）
%     for i = 1:size(matches, 1)
%         plot([matchedPoints1(i, 1), matchedPoints2(i, 1)], [matchedPoints1(i, 2), matchedPoints2(i, 2)], 'b', 'LineWidth', 1);
%     end
% 
%     hold off;
%     title('Matches between two images');
% end


% function draw_matches(img1, img2, keypoints1, keypoints2, matches)
%     % 检查输入图像是否为彩色图像，如果是，则转换为灰度图像
%     if size(img1, 3) == 3
%         img1 = rgb2gray(img1);
%     end
%     if size(img2, 3) == 3
%         img2 = rgb2gray(img2);
%     end
% 
%     % 拼接两张图片
%     [img1Height, img1Width] = size(img1);
%     [img2Height, img2Width] = size(img2);
%     imgCombined = [img1, img2];
% 
%     % 绘制拼接后的图像
%     figure;
%     imshow(imgCombined);
%     hold on;
% 
%     % 提取匹配点的坐标
%     matchedPoints1 = keypoints1(matches(:, 1), :); % 第一张图片的匹配点
%     matchedPoints2 = keypoints2(matches(:, 2), :); % 第二张图片的匹配点
% 
%     % 调整第二张图片的匹配点坐标（x 坐标加上第一张图片的宽度）
%     matchedPoints2(:, 1) = matchedPoints2(:, 1) + img1Width;
% 
%     % 绘制匹配点
%     plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'go', 'MarkerSize', 5, 'LineWidth', 2);
%     plot(matchedPoints2(:, 1), matchedPoints2(:, 2), 'ro', 'MarkerSize', 5, 'LineWidth', 2);
% 
%     % 绘制匹配点之间的连线（使用蓝色）
%     lineHandles = gobjects(size(matches, 1), 1); % 用于存储每条连线的句柄
%     for i = 1:size(matches, 1)
%         lineHandles(i) = plot([matchedPoints1(i, 1), matchedPoints2(i, 1)], ...
%                               [matchedPoints1(i, 2), matchedPoints2(i, 2)], ...
%                               'b', 'LineWidth', 1, 'ButtonDownFcn', @(src, event) highlightLine(src, event, i));
%     end
% 
%     hold off;
%     title('Matches between two images');
% 
%     % 定义高亮连线的回调函数
%     function highlightLine(src, ~, lineIndex)
%         % 取消所有连线的高亮
%         set(lineHandles, 'LineWidth', 1);
%         % 高亮当前点击的连线
%         set(src, 'LineWidth', 1, 'Color', 'yellow');
%         % 显示当前连线的索引
%         fprintf('点击的连线索引: %d\n', lineIndex);
%     end
% end

% function draw_matches_inliers(img1, img2, keypoints1, keypoints2, matches)
%     % 检查输入图像是否为彩色图像，如果是，则转换为灰度图像
%     if size(img1, 3) == 3
%         img1 = rgb2gray(img1);
%     end
%     if size(img2, 3) == 3
%         img2 = rgb2gray(img2);
%     end
% 
%     % 拼接两张图片
%     [img1Height, img1Width] = size(img1);
%     [img2Height, img2Width] = size(img2);
%     imgCombined = [img1, img2];
% 
%     % 绘制拼接后的图像
%     figure;
%     imshow(imgCombined);
%     hold on;
% 
%     % 提取匹配点的坐标
%     matchedPoints1 = keypoints1(matches(:, 1), :); % 第一张图片的匹配点
%     matchedPoints2 = keypoints2(matches(:, 2), :); % 第二张图片的匹配点
% 
%     % 调整第二张图片的匹配点坐标（x 坐标加上第一张图片的宽度）
%     matchedPoints2(:, 1) = matchedPoints2(:, 1) + img1Width;
% 
%     % 绘制匹配点
%     plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
%     plot(matchedPoints2(:, 1), matchedPoints2(:, 2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% 
%     % 绘制匹配点之间的连线（使用蓝色）
%     lineHandles = gobjects(size(matches, 1), 1); % 用于存储每条连线的句柄
%     for i = 1:size(matches, 1)
%         lineHandles(i) = plot([matchedPoints1(i, 1), matchedPoints2(i, 1)], ...
%                               [matchedPoints1(i, 2), matchedPoints2(i, 2)], ...
%                               'b', 'LineWidth', 1, 'ButtonDownFcn', @(src, event) toggleHighlight(src, event, i, lineHandles));
%     end
% 
%     hold off;
%     title('Matches between two images');
% 
%     % 定义切换高亮的回调函数
%     function toggleHighlight(src, ~, lineIndex, lineHandles)
%         persistent highlightedLineIndices; % 记录当前高亮的连线索引
% 
%         % 初始化高亮记录
%         if isempty(highlightedLineIndices)
%             highlightedLineIndices = [];
%         end
% 
%         % 如果当前点击的连线是高亮的，则取消高亮
%         if ismember(lineIndex, highlightedLineIndices)
%             set(src, 'LineWidth', 1, 'Color', 'b'); % 恢复为蓝色
%             highlightedLineIndices = setdiff(highlightedLineIndices, lineIndex); % 从高亮记录中移除
%             fprintf('取消高亮的连线索引: %d\n', lineIndex);
%         else
%             % 高亮当前点击的连线
%             set(src, 'LineWidth', 3, 'Color', 'yellow');
%             highlightedLineIndices = [highlightedLineIndices, lineIndex]; % 添加到高亮记录中
%             fprintf('高亮的连线索引: %d\n', lineIndex);
%         end
%     end
% end

function draw_matches_inliers(img1, img2, keypoints1, keypoints2, matches)
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
    plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'go', 'MarkerSize', 3, 'LineWidth', 2);
    plot(matchedPoints2(:, 1), matchedPoints2(:, 2), 'ro', 'MarkerSize', 3, 'LineWidth', 2);

    % 绘制匹配点之间的连线（使用蓝色）
    lineHandles = gobjects(size(matches, 1), 1); % 用于存储每条连线的句柄
    for i = 1:size(matches, 1)
        lineHandles(i) = plot([matchedPoints1(i, 1), matchedPoints2(i, 1)], ...
                              [matchedPoints1(i, 2), matchedPoints2(i, 2)], ...
                              'b', 'LineWidth', 1, 'Visible', 'on', ...
                              'ButtonDownFcn', @(src, event) toggleHighlight(src, event, i, lineHandles));
    end

    hold off;
    title('Matches between two images');

    % 添加交互式控件（复选框）
    uicontrol('Style', 'text', 'String', '选择显示的匹配对索引:', ...
              'Position', [10, 10, 150, 20]);
    selectedIndices = uicontrol('Style', 'edit', 'String', '1, 2, 3', ...
                                'Position', [170, 10, 100, 20], ...
                                'Callback', @(src, event) updateVisibility(src, event, lineHandles));

    % 定义切换高亮的回调函数
    function toggleHighlight(src, ~, lineIndex, lineHandles)
        persistent highlightedLineIndices; % 记录当前高亮的连线索引

        % 初始化高亮记录
        if isempty(highlightedLineIndices)
            highlightedLineIndices = [];
        end

        % 如果当前点击的连线是高亮的，则取消高亮
        if ismember(lineIndex, highlightedLineIndices)
            set(src, 'LineWidth', 1, 'Color', 'b'); % 恢复为蓝色
            highlightedLineIndices = setdiff(highlightedLineIndices, lineIndex); % 从高亮记录中移除
            fprintf('取消高亮的连线索引: %d\n', lineIndex);
        else
            % 高亮当前点击的连线
            set(src, 'LineWidth', 2, 'Color', 'yellow');
            highlightedLineIndices = [highlightedLineIndices, lineIndex]; % 添加到高亮记录中
            fprintf('高亮的连线索引: %d\n', lineIndex);
        end
    end

    % 定义更新可见性的回调函数
    function updateVisibility(src, ~, lineHandles)
        % 获取用户输入的索引
        inputIndices = str2num(src.String); % 将输入的字符串转换为数字数组

        % 检查输入是否有效
        if isempty(inputIndices) || any(inputIndices < 1) || any(inputIndices > length(lineHandles))
            errordlg('请输入有效的匹配对索引（1 到 %d）', length(lineHandles));
            return;
        end

        % 更新连线的可见性
        for i = 1:length(lineHandles)
            if ismember(i, inputIndices)
                set(lineHandles(i), 'Visible', 'on'); % 显示连线
            else
                set(lineHandles(i), 'Visible', 'off'); % 隐藏连线
            end
        end
    end
end