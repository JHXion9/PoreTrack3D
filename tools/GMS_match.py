import math
from enum import Enum
import os
import cv2
cv2.ocl.setUseOpenCL(False)

import numpy as np
import struct
THRESHOLD_FACTOR = 6

ROTATION_PATTERNS = [
    [1, 2, 3,
     4, 5, 6,
     7, 8, 9],

    [4, 1, 2,
     7, 5, 3,
     8, 9, 6],

    [7, 4, 1,
     8, 5, 2,
     9, 6, 3],

    [8, 7, 4,
     9, 5, 1,
     6, 3, 2],

    [9, 8, 7,
     6, 5, 4,
     3, 2, 1],

    [6, 9, 8,
     3, 5, 7,
     2, 1, 4],

    [3, 6, 9,
     2, 5, 8,
     1, 4, 7],

    [2, 3, 6,
     1, 5, 9,
     4, 7, 8]]


class Size:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


class GmsMatcher:
    def __init__(self, matcher):
        self.scale_ratios = [1.0, 1.0 / 2, 1.0 / math.sqrt(2.0), math.sqrt(2.0), 2.0]
        # Normalized vectors of 2D points
        self.normalized_points1 = []
        self.normalized_points2 = []
        # Matches - list of pairs representing numbers
        self.matches = []
        self.matches_number = 0
        # Grid Size
        self.grid_size_right = Size(0, 0)
        self.grid_number_right = 0
        # x      : left grid idx
        # y      :  right grid idx
        # value  : how many matches from idx_left to idx_right
        self.motion_statistics = []

        self.number_of_points_per_cell_left = []
        # Inldex  : grid_idx_left
        # Value   : grid_idx_right
        self.cell_pairs = []

        # Every Matches has a cell-pair
        # first  : grid_idx_left
        # second : grid_idx_right
        self.match_pairs = []

        # Inlier Mask for output
        self.inlier_mask = []
        self.grid_neighbor_right = []

        # Grid initialize
        self.grid_size_left = Size(20, 20)
        self.grid_number_left = self.grid_size_left.width * self.grid_size_left.height

        # Initialize the neihbor of left grid
        self.grid_neighbor_left = np.zeros((self.grid_number_left, 9))

        # self.descriptor = descriptor
        self.matcher = matcher
        self.gms_matches = []
        self.keypoints_image1 = []
        self.keypoints_image2 = []

    def empty_matches(self):
        self.normalized_points1 = []
        self.normalized_points2 = []
        self.matches = []
        self.gms_matches = []

    def compute_matches(self, img1, img2, keypoints1_path, keypoints2_path, descriptors1_path, descriptors2_path):
        save_dir = os.path.dirname(keypoints1_path).replace('keypoints', 'matches')
        save_pth = os.path.join(save_dir, os.path.basename(keypoints1_path).split('.')[0] +'.png'+ '---' + os.path.basename(keypoints2_path).split('.')[0] + '.png'+'.bin')
        os.makedirs(save_dir, exist_ok=True)
        # self.keypoints_image1, descriptors_image1 = self.descriptor.detectAndCompute(img1, np.array([]))
        self.keypoints_image1 = self.read_keypoints(keypoints1_path)[:, :2]
        descriptors_image1 = self.read_descriptors(descriptors1_path)
        self.keypoints_image2 = self.read_keypoints(keypoints2_path)[:, :2]
        descriptors_image2 = self.read_descriptors(descriptors2_path)

        # print(descriptors_image1)
        # assert 0

        size1 = Size(img1.shape[1], img1.shape[0])
        size2 = Size(img2.shape[1], img2.shape[0])

        if self.gms_matches:
            self.empty_matches()


        all_matches = self.matcher.match(descriptors_image1, descriptors_image2)
        
        self.normalize_points(self.keypoints_image1, size1, self.normalized_points1)
        self.normalize_points(self.keypoints_image2, size2, self.normalized_points2)
        self.matches_number = len(all_matches)
        self.convert_matches(all_matches, self.matches)
        self.initialize_neighbours(self.grid_neighbor_left, self.grid_size_left)

        mask, num_inliers = self.get_inlier_mask(False, False)
        print('Found', num_inliers, 'matches')
        
        if num_inliers == 0:
            save_matches = np.zeros((0, 2), dtype=np.uint32)
            self.write_matches(save_pth, save_matches)
            return []
        
        else:
            for i in range(len(mask)):
                if mask[i]:
                    self.gms_matches.append(all_matches[i])

            save_matches = []
            for match in self.gms_matches:
                save_matches.append([match.queryIdx, match.trainIdx])

            self.write_matches(save_pth, np.array(save_matches))
            
            return self.gms_matches

    import struct

    def write_matches(self, path, matches):
        """
        Write the matches to a binary file.

        Parameters:
            path (str): Path to the match file.
            matches (numpy.ndarray): Integer match indices, where each row represents one match pair between two images.
        """
        import numpy as np

        # 确保 matches 是整数类型
        assert np.isreal(matches).all() and np.issubdtype(matches.dtype, np.integer)
        assert matches.shape[1] == 2


        # 打开文件，准备写入二进制数据
        with open(path, 'wb') as f:
            # 写入 matches 的大小（行数和列数）
            f.write(struct.pack('i', matches.shape[0]))
            f.write(struct.pack('i', matches.shape[1]))

            # 写入 matches 数据
            matches = matches.flatten().astype(np.uint32)
            f.write(struct.pack('I' * len(matches), *matches))
        print(f"Matches saved to {path}")

    def read_keypoints(self, path):
        """
        READ_KEYPOINTS - Read the keypoints from a binary file.

        Args:
            path: Path to the keypoint file.

        Returns:
            keypoints: The keypoints read from the file.
        """
        with open(path, 'rb') as fid:
            shape = np.fromfile(fid, dtype=np.int32, count=2)
            keypoints = np.fromfile(fid, dtype=np.float32, count=shape[0] * shape[1])
            keypoints = keypoints.reshape((shape[0], shape[1]))  # Transpose to match MATLAB's behavior
        return keypoints

    def read_descriptors(self, path):
        with open(path, 'rb') as fid:
            shape = np.fromfile(fid, dtype=np.int32, count=2)
            descriptors = np.fromfile(fid, dtype=np.float32, count=shape[0] * shape[1])
            descriptors = descriptors.reshape((shape[0], shape[1]))  # Transpose to match MATLAB's behavior

        

        return descriptors
    
    # Normalize Key points to range (0-1)
    def normalize_points(self, kp, size, npts):
        for keypoint in kp:
            npts.append((keypoint[0] / size.width, keypoint[1] / size.height))

    # Convert OpenCV match to list of tuples
    def convert_matches(self, vd_matches, v_matches):
        for match in vd_matches:
            v_matches.append((match.queryIdx, match.trainIdx))

    def initialize_neighbours(self, neighbor, grid_size):
        for i in range(neighbor.shape[0]):
            neighbor[i] = self.get_nb9(i, grid_size)

    def get_nb9(self, idx, grid_size):
        nb9 = [-1 for _ in range(9)]
        idx_x = idx % grid_size.width
        idx_y = idx // grid_size.width

        for yi in range(-1, 2):
            for xi in range(-1, 2):
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi

                if idx_xx < 0 or idx_xx >= grid_size.width or idx_yy < 0 or idx_yy >= grid_size.height:
                    continue
                nb9[xi + 4 + yi * 3] = idx_xx + idx_yy * grid_size.width

        return nb9

    def get_inlier_mask(self, with_scale, with_rotation):
        max_inlier = 0
        self.set_scale(0)

        if not with_scale and not with_rotation:
            max_inlier = self.run(1)
            return self.inlier_mask, max_inlier
        elif with_scale and with_rotation:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                for rotation_type in range(1, 9):
                    num_inlier = self.run(rotation_type)
                    if num_inlier > max_inlier:
                        vb_inliers = self.inlier_mask
                        max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        elif with_rotation and not with_scale:
            vb_inliers = []
            for rotation_type in range(1, 9):
                num_inlier = self.run(rotation_type)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier
        else:
            vb_inliers = []
            for scale in range(5):
                self.set_scale(scale)
                num_inlier = self.run(1)
                if num_inlier > max_inlier:
                    vb_inliers = self.inlier_mask
                    max_inlier = num_inlier

            if vb_inliers != []:
                return vb_inliers, max_inlier
            else:
                return self.inlier_mask, max_inlier

    def set_scale(self, scale):
        self.grid_size_right.width = self.grid_size_left.width * self.scale_ratios[scale]
        self.grid_size_right.height = self.grid_size_left.height * self.scale_ratios[scale]
        self.grid_number_right = self.grid_size_right.width * self.grid_size_right.height

        # Initialize the neighbour of right grid
        self.grid_neighbor_right = np.zeros((int(self.grid_number_right), 9))
        self.initialize_neighbours(self.grid_neighbor_right, self.grid_size_right)

    def run(self, rotation_type):
        self.inlier_mask = [False for _ in range(self.matches_number)]

        # Initialize motion statistics
        self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
        self.match_pairs = [[0, 0] for _ in range(self.matches_number)]

        for GridType in range(1, 5):
            self.motion_statistics = np.zeros((int(self.grid_number_left), int(self.grid_number_right)))
            self.cell_pairs = [-1 for _ in range(self.grid_number_left)]
            self.number_of_points_per_cell_left = [0 for _ in range(self.grid_number_left)]

            self.assign_match_pairs(GridType)
            self.verify_cell_pairs(rotation_type)

            # Mark inliers
            for i in range(self.matches_number):
                if self.cell_pairs[int(self.match_pairs[i][0])] == self.match_pairs[i][1]:
                    self.inlier_mask[i] = True

        return sum(self.inlier_mask)

    def assign_match_pairs(self, grid_type):
        for i in range(self.matches_number):
            lp = self.normalized_points1[self.matches[i][0]]
            rp = self.normalized_points2[self.matches[i][1]]
            lgidx = self.match_pairs[i][0] = self.get_grid_index_left(lp, grid_type)

            if grid_type == 1:
                rgidx = self.match_pairs[i][1] = self.get_grid_index_right(rp)
            else:
                rgidx = self.match_pairs[i][1]

            if lgidx < 0 or rgidx < 0:
                continue
            self.motion_statistics[int(lgidx)][int(rgidx)] += 1
            self.number_of_points_per_cell_left[int(lgidx)] += 1

    def get_grid_index_left(self, pt, type_of_grid):
        x = pt[0] * self.grid_size_left.width
        y = pt[1] * self.grid_size_left.height

        if type_of_grid == 2:
            x += 0.5
        elif type_of_grid == 3:
            y += 0.5
        elif type_of_grid == 4:
            x += 0.5
            y += 0.5

        x = math.floor(x)
        y = math.floor(y)

        if x >= self.grid_size_left.width or y >= self.grid_size_left.height:
            return -1
        return x + y * self.grid_size_left.width

    def get_grid_index_right(self, pt):
        x = int(math.floor(pt[0] * self.grid_size_right.width))
        y = int(math.floor(pt[1] * self.grid_size_right.height))
        return x + y * self.grid_size_right.width

    def verify_cell_pairs(self, rotation_type):
        current_rotation_pattern = ROTATION_PATTERNS[rotation_type - 1]

        for i in range(self.grid_number_left):
            if sum(self.motion_statistics[i]) == 0:
                self.cell_pairs[i] = -1
                continue
            max_number = 0
            for j in range(int(self.grid_number_right)):
                value = self.motion_statistics[i]
                if value[j] > max_number:
                    self.cell_pairs[i] = j
                    max_number = value[j]

            idx_grid_rt = self.cell_pairs[i]
            nb9_lt = self.grid_neighbor_left[i]
            nb9_rt = self.grid_neighbor_right[idx_grid_rt]
            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = nb9_lt[j]
                rr = nb9_rt[current_rotation_pattern[j] - 1]
                if ll == -1 or rr == -1:
                    continue

                score += self.motion_statistics[int(ll), int(rr)]
                thresh += self.number_of_points_per_cell_left[int(ll)]
                numpair += 1

            thresh = THRESHOLD_FACTOR * math.sqrt(thresh/numpair)
            if score < thresh:
                self.cell_pairs[i] = -2

    def draw_matches(self, src1, src2, drawing_type):
        height = (max(src1.shape[0], src2.shape[0]))
        width = (src1.shape[1] + src2.shape[1])
        output = np.zeros((height, width, 3), dtype=np.uint8)
        output[0:src1.shape[0], 0:src1.shape[1]] = src1
        output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    
        if drawing_type == DrawingType.ONLY_LINES:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx]
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx], (src1.shape[1], 0)))
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

        elif drawing_type == DrawingType.LINES_AND_POINTS:
            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))
                cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
                cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

        elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY :
            _1_255 = np.expand_dims( np.array( range( 0, 256 ), dtype='uint8' ), 1 )
            _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

            for i in range(len(self.gms_matches)):
                left = self.keypoints_image1[self.gms_matches[i].queryIdx].pt
                right = tuple(sum(x) for x in zip(self.keypoints_image2[self.gms_matches[i].trainIdx].pt, (src1.shape[1], 0)))

                if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                    colormap_idx = int(left[0] * 256. / src1.shape[1] ) # x-gradient
                if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                    colormap_idx = int(left[1] * 256. / src1.shape[0] ) # y-gradient
                if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                    colormap_idx = int( (left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5) ) # manhattan gradient

                color = tuple( map(int, _colormap[ colormap_idx,0,: ]) )
                cv2.circle(output, tuple(map(int, left)), 1, color, 2)
                cv2.circle(output, tuple(map(int, right)), 1, color, 2)

        output = cv2.resize(output, (int(output.shape[1] * 0.25), int(output.shape[0] * 0.25)))
        cv2.imshow('show', output)
        cv2.waitKey()
        cv2.destroyAllWindows()


def compute_matches_in_order(gms, img, keypoints_pth, descriptors_pth):
    pairs = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16),
    (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
    (5, 6), (5, 7), (5, 8), (5, 9),
    (6, 7), (6, 8), (6, 9),
    (7, 8), (7, 9),
    (8, 9),
    (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16),
    (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16),
    (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 13), (12, 14), (12, 15), (12, 16),
    (13, 2), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (13, 8), (13, 9), (13, 14), (13, 15), (13, 16),
    (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 15), (14, 16),
    (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9), (15, 16),
    (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 9)
    ]

    for i, j in pairs:
        img1, img2 = img[i - 1], img[j - 1]
        keypoints1_pth, keypoints2_pth = keypoints_pth[i - 1], keypoints_pth[j - 1]
        descriptors1_pth, descriptors2_pth = descriptors_pth[i - 1], descriptors_pth[j - 1]
        matches = gms.compute_matches(img1, img2, keypoints1_pth, keypoints2_pth, descriptors1_pth, descriptors2_pth)
        print(f"Matches for {i}-{j}: {len(matches)}")
        # if i == 1 and j == 2:
        #     gms.draw_matches(img1, img2, DrawingType.ONLY_LINES)
    

if __name__ == '__main__':
    # img1 = cv2.imread("./01.jpg")
    # img2 = cv2.imread("./02.jpg")
    img = [cv2.imread(f"./images/{i}.png") for i in range(1,17)]
    keypoints_pth = [f'./keypoints/{i}.png.bin' for i in range(1,17)]
    descriptors_pth = [f'./descriptors/{i}.png.bin' for i in range(1,17)]

    if cv2.__version__.startswith('3'):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher_create(cv2.NORM_L2)
    gms = GmsMatcher( matcher)

    compute_matches_in_order(gms, img, keypoints_pth, descriptors_pth)
    
    # save_matches = []
    # for match in matches:
    #     save_matches.append([match.queryIdx, match.trainIdx])

    # save_matches = np.array(save_matches)
    # print(save_matches.shape)
    # print(matches[5].queryIdx, matches[5].trainIdx)
    # gms.draw_matches(img1, img2, DrawingType.ONLY_LINES)
    # gms.draw_matches(img1, img2, DrawingType.COLOR_CODED_POINTS_XpY)