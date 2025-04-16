% Exhaustive matching pipeline.
% num_images 是上一级文件中的变量，其是文件中 图片的个数，MATCH_BLOCK_SIZE则是多少个图片作为一个block 快
%  human的 是等于1
num_blocks = ceil(num_images / MATCH_BLOCK_SIZE);
% 每个block 有多少对 即任选两个视角的图片进行匹配，有多少对
num_pairs_per_block = MATCH_BLOCK_SIZE * (MATCH_BLOCK_SIZE - 1) / 2;
% 用于总的match有多少的
sumprint = 0;
for start_idx1 = 1:MATCH_BLOCK_SIZE:num_images
    % 就是一个block 一个block 进行 这些 的 start_idx1一开始是1，然后 end_idx1是14 
    end_idx1 = min(num_images, start_idx1 + MATCH_BLOCK_SIZE - 1);
    % 这个start_idx2 一开始是1，然后 end_idx2是14 
    for start_idx2 = 1:MATCH_BLOCK_SIZE:num_images
        end_idx2 = min(num_images, start_idx2 + MATCH_BLOCK_SIZE - 1);

        fprintf('Matching block [%d/%d, %d/%d]', ...
                int64(start_idx1 / MATCH_BLOCK_SIZE) + 1, num_blocks, ...
                int64(start_idx2 / MATCH_BLOCK_SIZE) + 1, num_blocks);

        tic;

        % Read the descriptors for current block of images.
        % 创建描述子的map对象，这里的是键值对，前面是keytype被设置为 int32
        descriptors = containers.Map('KeyType', 'int32', ...
                                     'ValueType', 'any');
        siftdescriptors = containers.Map('KeyType', 'int32', ...
                             'ValueType', 'any');

        % 将start_idx1:end_idx1, start_idx2:end_idx2拼接起来，然后遍历每一个 idx，
        % 将描述子将入到descriptors中
        % 这里的 idx 是对应上一级文件中，读取到的各个顺序的图片文件路径 关键点文件路径  描述子文件路径
        for idx = [start_idx1:end_idx1, start_idx2:end_idx2]
            % 如果存在了这一堆对key， 那就不用重复存储
            if descriptors.isKey(idx)
                continue;
            end
            image_descriptors = single(read_descriptors(descriptor_paths{idx}));
            image_siftdescriptors = single(read_descriptors(siftdescriptor_paths{idx}));
            if MATCH_GPU
                descriptors(idx) = gpuArray(image_descriptors);
                siftdescriptors(idx) = gpuArray(image_siftdescriptors);
            else
                descriptors(idx) = image_descriptors;
                siftdescriptors(idx) = image_siftdescriptors;
            end
        end
        % 此时的 descriptors是一个键值对， 对应的idx 存储着对应图片的descriptors


        % Match and write the current block of images.
        for idx1 = start_idx1:end_idx1
            for idx2 = start_idx2:end_idx2
                block_id1 = mod(idx1, MATCH_BLOCK_SIZE);
                block_id2 = mod(idx2, MATCH_BLOCK_SIZE);
                %  令idx1 和 idx2 不相同
                if (idx1 > idx2 && block_id1 <= block_id2) ...
                        || (idx1 < idx2 && block_id1 < block_id2)
                    % Order the indices to avoid duplicate pairs.
                    % 使得 oidx1 必定 小于等于 oidx2
                    if idx1 < idx2
                        oidx1 = idx1;
                        oidx2 = idx2;
                    else
                        oidx1 = idx2;
                        oidx2 = idx1;
                    end
                    
                    % 查看这个匹配对是否存在，如果已经存在就筛除
                    % Check if matches already computed.
                    matches_path = fullfile(...
                        MATCH_PATH, sprintf('%s---%s.bin', ...
                        image_names{oidx1}, image_names{oidx2}));
                    if exist(matches_path, 'file')
                        continue;
                    end

                    % Match the descriptors.
                    % 输出的是一个 数组维度是 match的对数，2，第一维是图片的1的keypoints索引，第二维度则是
                    % 图片2的keypoints索引
                    matches = sift_match_descriptors(descriptors(oidx1), ...
                                                descriptors(oidx2), ...
                                                siftdescriptors(oidx1), ...
                                                siftdescriptors(oidx2), ...
                                                MATCH_MAX_DIST_RATIO);
                    sumprint = sumprint+length(matches);
                    % Write the matches.
                    if size(matches, 1) < MIN_NUM_MATCHES
                        matches = zeros(0, 2, 'uint32');
                    end
                    % 写入矩阵文件
                    write_matches(matches_path, matches);
                end
            end
        end

        fprintf(' in %.3fs\n', toc);
    end
end
fprintf('matches的总数是: %d\n', sumprint)
% Clear the GPU memory.
clear descriptors;
clear matches;
