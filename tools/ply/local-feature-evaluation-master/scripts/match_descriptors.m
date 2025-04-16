function matches = match_descriptors(descriptors1, descriptors2, max_dist_ratio)
% MATCH_DESCRIPTORS - Exhaustively match two descriptors sets with cross-check.
%   descriptors1:
%       First set of descriptors, where each row is one descriptor.
%   descriptors2:
%       Second set of descriptors, where each row is one descriptor.
%   max_dist_ratio:
%       Maximum distance ratio between first and second best matches.
%
%   matches:
%       The indices of mutually matching descriptors. The matching descriptors
%       can be extracted as descriptors1(matches(:,1),:) and
%       descriptors2(matches(:,2),:).
%
% Copyright 2017: Johannes L. Schönberger <jsch at inf.ethz.ch>

if size(descriptors1, 1) == 0 || size(descriptors2, 1) == 0
    matches = zeros(0, 2, 'uint32');
    return;
end

% Exhaustively compute distances between all descriptors.
% 计算两者之间的距离，行是descriptors1，列是descriptors2
dists = pdist2(descriptors1, descriptors2, 'squaredeuclidean');

% Find the first best matches.
% 创建一个索引数组 idxs1，用于标识第一组描述子中的每个元素
idxs1 = gpuArray(single(1:size(descriptors1, 1)));
%   first_dists12：每个描述子在第二组中的最小距离。 idxs12：每个描述子在第二组中的最佳匹配索引
[first_dists12, idxs12] = min(dists, [], 2);
%  idxs21 对于 descriptor2而言，每列中的最佳匹配索引
[~, idxs21] = min(dists, [], 1);

%  找当前descriptor1  a1在descriptor2中的最佳匹配b1，b1对应的最佳匹配的索引a2，后面用idxs121和idxs12比较
%  就是 看a1和a2 一不一样，如果一样就互为最小
idxs121 = idxs21(idxs12);

%将距离矩阵 dists 中每个描述子的最佳匹配距离设置为最大值，然后再找最小值，就是找第二最小值
% Find the second best matches.
dists(sub2ind(size(dists), idxs1, idxs12')) = single(realmax('single'));
% 找到了descriptor1对descriptor2的第二小距离
second_dists12 = min(dists, [], 2);

% Compute the distance ratios between the first and second best matches.
% 计算descriptor1对descriptor2的最小距离和第二小距离的比例
dist_ratios12 = sqrt(first_dists12) ./ sqrt(second_dists12);

% Enforce the ratio test constraint and mutual nearest neighbors.
% 需要同时满足比例小于等于最大比例限制，而且互为最小值 则ok
%mask = (dist_ratios12(:) <= max_dist_ratio) & (idxs1(:) == idxs121(:));
mask = (dist_ratios12(:) <= max_dist_ratio) ;
idxs1 = idxs1(mask);
idxs2 = idxs12(mask);

% Compose the match matrix.
% 输出的match 是两个图片的 各个点的序号，从一开始，idxs1是满足条件的descriptor1的点的序号
% idxs则是对应的满足条件的 idxs2的序号
matches = uint32(gather([idxs1', idxs2]));

end