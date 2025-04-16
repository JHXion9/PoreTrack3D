function [frames,pr_new] = func_detect_pore_no_looser_boundaries(I,no_kpttarget,outputno)
% function [frames,descriptors,pore_ratio_list,hrs] = func_detect_pore_no_looser_boundaries(I,no_kpttarget,outputno)

%注意：该版本进行了修改，如果迭代次数到了20次还没结束循环，就允许kpterror在0.15以内，如果到了40次还没结束循环就允许kpterror在0.2以内，
%注意：且微调孔隙率使用更小的更新间隔，在达到了60次还没满足时，就选择当前最接近所要kpttarget的那次结果，并跳出循环。


% 函数定义，输入参数：I-图像，no_kpttarget-目标关键点数量，outputno-输出编号
% 输出参数：frames-关键点框架，descriptors-描述符，pore_ratio_list-孔隙率列表，hrs-Hessian响应

% 初始化孔隙率列表，[最大pr，对应的kpt数量，最小pr，对应的kpt数量，边界标志]
% 边界标志：1表示上边界，3表示下边界，初始值为-1
pore_ratio_list = [0.02 -1 0.00001 100000 -1];

% 计算高斯差分金字塔(DoG)和高斯金字塔
[dogss,gss] = func_dog(I);

% 构造输出文件路径
outputpath = sprintf('Afunc_detect_pore_no%d.png',outputno);

% 初始化参数
max_iterations = 60;  % 最大迭代次数
iterations = 0;
best_frames = [];  % 存储最佳结果
best_kpterror = inf;
best_hrs = [];
best_pr = -1;
kpterror_threshold = 0.1;  % 初始误差阈值
pr_idx = 0; % 孔隙率索引
no_new = -1; % 当前关键点数量
no_old = -2; % 上一次迭代的关键点数量

while iterations < max_iterations
    iterations = iterations + 1;
    pr_idx = pr_idx + 1;

    if pr_idx == 1 % 第一次迭代，使用最大孔隙率
        no_old = pore_ratio_list(2);
        pr_new = pore_ratio_list(1);
    else
        no_old = no_new;
    end

    % 使用当前孔隙率检测关键点
    [frames,oframes_pr, ohrs_pr]=func_detect_dogss_pr(dogss,gss,pr_new);
    


    no_new = size(frames,2);
    % fprintf('The iteration %d, The number of keypoints is %d, the target is %d.\n', iterations, frames, no_kpttarget);
    % 计算关键点数量误差
    kpterror = (no_new - no_kpttarget) / no_kpttarget;

    % 动态调整 kpterror_threshold
    if iterations >= 20 && kpterror_threshold < 0.15
        kpterror_threshold = 0.15;
    elseif iterations >= 40 && kpterror_threshold < 0.2
        kpterror_threshold = 0.2;
    end

    % 记录最接近 no_kpttarget 的结果
    % if abs(no_new - no_kpttarget) < abs(size(best_frames, 2) - no_kpttarget)
    best_frames = oframes_pr;
    best_hrs = ohrs_pr;
    best_pr = pr_new;
    best_kpterror = kpterror; 
    % end

    % 检查是否满足终止条件
    if abs(kpterror) <= kpterror_threshold

        

        break; % 满足条件，跳出循环
    end

    % 更新孔隙率上下界
    if no_new > pore_ratio_list(2) && no_new < pore_ratio_list(4)
        pr_old = pr_new;
        if no_new < no_kpttarget
            pore_ratio_list(1:2) = [pr_new no_new];
            pore_ratio_list(5) = 1;
        else
            pore_ratio_list(3:4) = [pr_new no_new];
            pore_ratio_list(5) = 3;
        end
    end

    % 更新孔隙率（使用更小的更新间隔）
    if abs(kpterror) < 0.3
        pr_new = pr_old + 0.005 * kpterror; % 更小的更新间隔
        pr_new = max(pr_new, pore_ratio_list(3)); % 确保孔隙率不小于最小值
    else
        pr_new = (pore_ratio_list(1) + pore_ratio_list(3)) / 2;
    end
end

% 如果执行了60次都找不到满意的结果，只能使用当前的最接近的结果了
oframes_pr = best_frames;
ohrs_pr = best_hrs;
pr_new = best_pr;

% 写pr_new进txt文件

fprintf('Using the closest result to the target keypoint number.\n');
% fprintf('---------------')
% fprintf('The number of keypoints is %d, the target is %d.\n', size(best_frames, 2), no_kpttarget);


% 绘制尺度颜色图
scale_bin = [0.5*0.4*1.6*2^0.2:0.1:1 1.1:0.2:2.1 2.5:0.5:4.5 5:1:10];
func_plot_scale_color_grayImg(I, frames, scale_bin, dogss, pr_idx, outputpath, outputno);

% 计算关键点描述符和Hessian响应
[frames,descriptors,hrs] = func_psift(gss,oframes_pr,ohrs_pr);


end