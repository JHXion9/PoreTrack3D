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
dists = pdist2(descriptors1, descriptors2, 'squaredeuclidean');

% Find the first best matches.
% 建立最原始的索引
idxs1 = gpuArray(single(1:size(descriptors1, 1)));
% first_dists12每一行的最小值，idxs12最小值对应的索引
[first_dists12, idxs12] = min(dists, [], 2);
% first_dists12每一列的最小值，idxs22最小值对应的索引
[~, idxs21] = min(dists, [], 1);
% 获得 对应列的
idxs121 = idxs21(idxs12);

% Find the second best matches.
dists(sub2ind(size(dists), idxs1, idxs12')) = single(realmax('single'));
second_dists12 = min(dists, [], 2);

% Compute the distance ratios between the first and second best matches.
dist_ratios12 = sqrt(first_dists12) ./ sqrt(second_dists12);

% Enforce the ratio test constraint and mutual nearest neighbors.
mask = (dist_ratios12(:) <= max_dist_ratio) & (idxs1(:) == idxs121(:));
%mask = (dist_ratios12(:) <= max_dist_ratio);
idxs1 = idxs1(mask);
idxs2 = idxs12(mask);

% Compose the match matrix.
matches = uint32(gather([idxs1', idxs2]));
disp(size(matches));
end
