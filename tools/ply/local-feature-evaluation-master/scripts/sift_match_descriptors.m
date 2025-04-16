function matches = sift_match_descriptors(descriptors1, descriptors2, descriptors3, descriptors4, max_dist_ratio)
% MATCH_DESCRIPTORS - Exhaustively match two descriptors sets with cross-check.
%   descriptors1: First set of descriptors, where each row is one descriptor.
%   descriptors2: Second set of descriptors, where each row is one descriptor.
%   descriptors3: Additional descriptors for row-wise sorting.
%   descriptors4: Additional descriptors for column-wise sorting.
%   max_dist_ratio: Maximum distance ratio between first and second best matches.
%
%   matches: The indices of mutually matching descriptors. The matching descriptors
%            can be extracted as descriptors1(matches(:,1),:) and descriptors2(matches(:,2),:).
%
% Copyright 2017: Johannes L. Schönberger <jsch at inf.ethz.ch>

% Check if either descriptor set is empty
if size(descriptors1, 1) == 0 || size(descriptors2, 1) == 0
    matches = zeros(0, 2, 'uint32');
    return;
end

% Move data to GPU
descriptors1 = gpuArray(single(descriptors1));
descriptors2 = gpuArray(single(descriptors2));
descriptors3 = gpuArray(single(descriptors3));
descriptors4 = gpuArray(single(descriptors4));

% Compute squared Euclidean distance between descriptors3 and descriptors4 on GPU
siftdist = pdist2(descriptors3, descriptors4, 'squaredeuclidean');

% Sort rows and columns of siftdist on GPU
[~, rowsortedIndices] = sort(siftdist, 2);
[~, csortedIndices] = sort(siftdist, 1);

% Get the middle column indices for row-wise sorting
[~, sizec] = size(rowsortedIndices);
getcolumn = floor(sizec / 3);
rowpickinde = rowsortedIndices(:, getcolumn:end);

% Get the middle row indices for column-wise sorting
[sizer, ~] = size(csortedIndices);
getr = floor(sizer / 3);
cpickinde = csortedIndices(getr:end, :);

% Compute squared Euclidean distance between descriptors1 and descriptors2 on GPU
distsr = pdist2(descriptors1, descriptors2, 'squaredeuclidean');

% Modify distsr based on row-wise sorted indices
for i = 1:size(distsr, 1)
    distsr(i, rowpickinde(i, :)) = distsr(i, rowpickinde(i, :)) + 100;
end

% Find the first best matches for descriptors1 to descriptors2
[first_dists12, idxs12] = min(distsr, [], 2);

% Modify distsc based on column-wise sorted indices
distsc = pdist2(descriptors1, descriptors2, 'squaredeuclidean');
for i = 1:size(distsc, 2)
    distsc(cpickinde(:, i), i) = distsc(cpickinde(:, i), i) + 100;
end

% Find the first best matches for descriptors2 to descriptors1
[~, idxs21] = min(distsc, [], 1);

% Create an index array for descriptors1 on GPU
idxs1 = gpuArray(single(1:size(descriptors1, 1)));

% Find the second best matches for descriptors1 to descriptors2
distsr(sub2ind(size(distsr), idxs1, idxs12')) = single(realmax('single'));
second_dists12 = min(distsr, [], 2);

% Compute the distance ratios between the first and second best matches
dist_ratios12 = sqrt(first_dists12) ./ sqrt(second_dists12);

% Enforce the ratio test constraint and mutual nearest neighbors
mask = (dist_ratios12(:) <= max_dist_ratio)' & (idxs1(:)' == idxs21(idxs12));
idxs1 = idxs1(mask);
idxs2 = idxs12(mask);

% Compose the match matrix and move results back to CPU
matches = uint32(gather([idxs1, idxs2']));

end