function draw_and_save_image_inliner(img1, img2, keypoints1, keypoints2, matches)
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
    fig = figure;
    imshow(imgCombined);
    hold on;

    % 提取匹配点的坐标
    matchedPoints1 = keypoints1(matches(:, 1), :); % 第一张图片的匹配点
    matchedPoints2 = keypoints2(matches(:, 2), :); % 第二张图片的匹配点

    % 调整第二张图片的匹配点坐标（x 坐标加上第一张图片的宽度）
    matchedPoints2(:, 1) = matchedPoints2(:, 1) + img1Width;

    % 绘制匹配点（初始状态为隐藏）
    pointHandles1 = plot(matchedPoints1(:, 1), matchedPoints1(:, 2), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'Visible', 'off');
    pointHandles2 = plot(matchedPoints2(:, 1), matchedPoints2(:, 2), 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'Visible', 'off');

    % 绘制匹配点之间的连线（初始状态为隐藏）
    lineHandles = gobjects(size(matches, 1), 1); % 用于存储每条连线的句柄
    for i = 1:size(matches, 1)
        lineHandles(i) = plot([matchedPoints1(i, 1), matchedPoints2(i, 1)], ...
                              [matchedPoints1(i, 2), matchedPoints2(i, 2)], ...
                              'b', 'LineWidth', 1, 'Visible', 'off', ...
                              'ButtonDownFcn', @(src, event) toggleHighlight(src, event, i, lineHandles));
    end

    hold off;
    title('Matches between two images');

    % 添加交互式控件（复选框）
    uicontrol('Style', 'text', 'String', '选择显示的匹配对索引:', ...
              'Position', [10, 10, 150, 20]);
    selectedIndices = uicontrol('Style', 'edit', 'String', '1, 2, 3', ...
                                'Position', [170, 10, 100, 20], ...
                                'Callback', @(src, event) updateVisibility(src, event, lineHandles, pointHandles1, pointHandles2));

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
            set(src, 'LineWidth', 3, 'Color', 'yellow');
            highlightedLineIndices = [highlightedLineIndices, lineIndex]; % 添加到高亮记录中
            fprintf('高亮的连线索引: %d\n', lineIndex);
        end
    end

    % 定义更新可见性的回调函数
    function updateVisibility(src, ~, lineHandles, pointHandles1, pointHandles2)
        % 获取用户输入的索引
        inputIndices = str2num(src.String); % 将输入的字符串转换为数字数组

        % 检查输入是否有效
        if isempty(inputIndices) || any(inputIndices < 1) || any(inputIndices > length(lineHandles))
            errordlg('请输入有效的匹配对索引（1 到 %d）', length(lineHandles));
            return;
        end

        % 隐藏所有连线和匹配点
        set(lineHandles, 'Visible', 'off'); % 隐藏所有连线
        set(pointHandles1, 'Visible', 'off'); % 隐藏第一张图片的匹配点
        set(pointHandles2, 'Visible', 'off'); % 隐藏第二张图片的匹配点

        % 显示用户指定的连线和匹配点
        for i = inputIndices
            set(lineHandles(i), 'Visible', 'on'); % 显示连线
            set(pointHandles1(i), 'Visible', 'on'); % 显示第一张图片的匹配点
            set(pointHandles2(i), 'Visible', 'on'); % 显示第二张图片的匹配点
        end
    end
end