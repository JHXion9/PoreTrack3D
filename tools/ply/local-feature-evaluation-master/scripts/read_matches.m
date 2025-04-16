function matches = read_matches(path)
% READ_MATCHES - Read the matches from a binary file.
%   path:
%       Path to match file.
%   matches:
%       Integer match indices, where each row represents one
%       match pair between two images.
%
% Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

% 打开文件
fid = fopen(path, 'r');

% 读取文件头（匹配矩阵的维度）
dims = fread(fid, 2, 'int32');
num_matches = dims(1);
num_cols = dims(2);

% 读取匹配数据
matches = fread(fid, [num_cols, num_matches], 'uint32');

% 关闭文件
fclose(fid);

% 将数据转置为行优先格式
matches = matches';

% 从 0-based 索引转换为 1-based 索引
matches = matches + 1;

% 检查读取结果
assert(size(matches, 2) == 2, '匹配矩阵的列数应为 2');

end