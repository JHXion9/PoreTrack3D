import cv2
import numpy as np
import torch
import numpy as np
from scipy import misc
from scipy import ndimage
import pylab as pl
import cv2
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

class Psift:
    def __init__(self, im, octave_num, sigma1=0.8,conthreshold=1,upSampling = True):
        """
        初始化 Psift 类

        :param im: 输入图像
        :param octave_num: 金字塔层数
        :param sigma: 初始高斯模糊的标准差
        :param contrast: 中心极值点的阈值
        :param upSampling: 是否对原始图像进行上采样 (默认 True)
        """
        self.octave_num = octave_num  # 金字塔层数
        self.conthreshold = conthreshold  # 对比度阈值
        self.im = im  # 原始图像
        self.sigma = sigma1  # 初始高斯模糊的标准差
        self.octave_im = {}  # 存储图像金字塔
        self.octave_gauss = {}  # 存储高斯金字塔
        self.cov_im = {}  # 存储高斯拉普拉斯 (LoG) 金字塔
        self.center_sigma = {}  # 存储中心极值点对应的高斯金字塔层级的标准差
        self.k = 2.0 ** (1.0 / 6);  # 相邻高斯金字塔层级之间的标准差比率
        if upSampling == True:
            # 如果 upSampling 为 True，则对原始图像进行双三次插值上采样，放大两倍
            self.octave_im[0]=misc.imresize(im,[im.shape[0]*2,im.shape[1]*2],interp='bicubic')/255.0
        else:
            # 否则，直接将原始图像作为金字塔的第一层
            self.octave_im[0] = im /255.0
        # print ('Image`s shape :',im.shape)  # 打印原始图像的形状

    def CreatePyramid(self):
        """
        创建图像的高斯拉普拉斯 (LoG) 金字塔
        """
        sigma = self.sigma/2.0  # 初始标准差除以 2
        for i in range(self.octave_num):
            sigma = sigma * 2.0  # 每一层的标准差是前一层的两倍
            imt = self.octave_im[i]  # 获取当前层的图像
            # 计算当前层的高斯金字塔的各个层级
            blur_a = ndimage.gaussian_filter(imt, sigma=sigma)
            blur_b = ndimage.gaussian_filter(imt, sigma=self.k * sigma)
            blur_c = ndimage.gaussian_filter(imt, sigma=(self.k ** 2) * sigma)
            blur_d = ndimage.gaussian_filter(imt, sigma=(self.k ** 3) * sigma)
            blur_e = ndimage.gaussian_filter(imt, sigma=(self.k ** 4) * sigma)
            blur_f = ndimage.gaussian_filter(imt, sigma=(self.k ** 5) * sigma)
            blur_g = ndimage.gaussian_filter(imt, sigma=(self.k ** 6) * sigma)

            # 计算当前层的 LoG 金字塔
            temp = np.zeros([6, imt.shape[0], imt.shape[1]])
            temp[0] =abs(blur_b - blur_a)
            temp[1] = abs(blur_c - blur_b)
            temp[2] =abs(blur_d - blur_c)
            temp[3] = abs(blur_e - blur_d)
            temp[4] = abs(blur_f - blur_e)
            temp[5] = abs(blur_g - blur_f)

            # 存储当前层的高斯金字塔的一部分层级
            self.octave_gauss[i, 0] = blur_b
            self.octave_gauss[i, 1] = blur_c
            self.octave_gauss[i, 2] = blur_d
            self.octave_gauss[i, 3] = blur_e

            # 构建下一层的图像金字塔，通过对当前层的高斯金字塔的第三层进行下采样实现
            self.octave_im[i + 1] = cv2.resize(blur_c, (int(blur_c.shape[1] / 2), int(blur_c.shape[0] / 2)))
            # 存储当前层的 LoG 金字塔
            self.cov_im[i] = temp
        # print ('Create image`s Pyramid successful')  # 打印金字塔创建成功的消息

    def ScolePoint(self, rmedge=False, specofic_hd=0, curvature=10.0):
        """
        优化后的寻找尺度空间极值点方法。

        :param rmedge: 是否去除边缘响应 (默认 False)
        :param specofic_hd: 极值点过滤的阈值 (默认 0)
        :param curvature: 曲率阈值，用于去除边缘响应 (默认 10.0)
        :return: 极值点列表 (x, y, octave_num, inv_octave_num, lessen)
        """
        # print('Start finding scale-space keypoints...')

        R = (curvature + 1) ** 2 / curvature  # 曲率响应的阈值
        point_list = []  # 存储最终的极值点

        for i in range(self.octave_num):
            for j in range(1, 5):  # 遍历每层的第 2 至 5 张图
                imt = self.cov_im[i][j - 1:j + 2]  # 提取相邻三层
                lessen = self.im.shape[0] / imt.shape[1]  # 缩放比例

                # 遍历图像像素点，去除边缘的 10 个像素
                for x in range(10, imt.shape[1] - 10):
                    for y in range(10, imt.shape[2] - 10):
                        center_value = imt[1, x, y]  # 当前像素的中心值

                        # 判断是否满足极值条件
                        
                        if center_value > specofic_hd and center_value == np.max(imt[:, x - 1:x + 2, y - 1:y + 2]):                           
                            # 如果需要去除边缘响应
                            if rmedge and center_value < self.conthreshold:
                                center_patch = imt[1, x - 1:x + 2, y - 1:y + 2]  # 中心区域 3x3

                                # 计算 Hessian 矩阵的曲率检测值
                                imx = ndimage.sobel(center_patch, axis=1)
                                imy = ndimage.sobel(center_patch, axis=0)

                                Wxx = np.sum(imx ** 2)
                                Wxy = np.sum(imx * imy)
                                Wyy = np.sum(imy ** 2)

                                Wdet = Wxx * Wyy - Wxy ** 2  # 行列式
                                Wtr = Wxx + Wyy  # 迹
                                curvature_ratio = (Wtr ** 2) / (Wdet + 1e-8)

                                if curvature_ratio < R:  # 检查曲率响应
                                    # i：关键点在 图像金字塔层数(octave)，从0开始计数
                                    # j: 关键点所在的 高斯金字塔层数（within the octave），也在 [0, 5]的范围内
                                    # lessen：缩放比例，原始图像的尺寸除以当前 octave 的尺寸，用于将关键点坐标映射回原始图像。实际上应该是self.im.shape[0] / self.cov_im[i].shape[1]
                                    scale = self.sigma * (2**i) * (self.k ** (j + 1))
                                    point_list.append([y, x, i, j - 1, lessen, scale])  
                                   
                            else:
                                scale = self.sigma * (2**i) * (self.k ** (j + 1))
                                point_list.append([y, x, i, j - 1, lessen, scale])
                                

        # print(f'Keypoints found: {len(point_list)}')   
        return np.array(point_list)

    def GetScale(self, point):
        """
        获取关键点的尺度信息。

        :param point: 关键点 [x, y, i, j, lessen, scale]
        :return: 关键点的尺度
        """
        x, y, i, j, lessen, scale = point
        return scale

    def GetFeature_hynet(self, point_list, model):
        # 解包特征点列表
        X, Y= (point_list[:, 0], point_list[:, 1])
        height, width = self.im.shape
        half_patch_size = 64 // 2
        patches = []
        valid_keypoints = []

        for x, y in zip(X, Y):
            x = round(x)
            y = round(y)
            # 检查关键点是否靠近边缘
            if half_patch_size <= x < width - half_patch_size and half_patch_size <= y < height - half_patch_size:
                # 提取 patch
                patch = self.im[y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size]
                patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)
                patches.append(patch)
                valid_keypoints.append([x, y])
        patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1)  # (N, 1, 32, 32)
        
        

        # 使用模型计算特征向量
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            features = model(patches_tensor).cpu().numpy()

        print(f'Keypoints found: {len(valid_keypoints)}')

        return features, np.array(valid_keypoints)
    
    def GetPSIFTFeature(self, model):
        """
        使用合适的参数获取图像的 Pore-SIFT 特征

        :return: 图像矩阵；特征；特征点列表
        """
        self.CreatePyramid()  # 创建图像金字塔
        point_list = self.ScolePoint(rmedge=True, curvature=10, specofic_hd=0)  # 检测尺度空间极值点
        feature , points = self.GetFeature_hynet(point_list=point_list, model = model)  # 提取 Pore-SIFT 特征
        # feature , points = self.GetFeature(point_list=point_list)
        return points, feature # 返回特征、特征点坐标和原始图像
    
    @staticmethod
    def match(featureA,featureB,file,ratio=0.6,RANSAC = False,kmeans= False,dispaly=True,save = False,plot_match=True):
        # 对两组特征进行匹配
        im1 = featureA[2]  # 获取第一幅图像
        im2 = featureB[2]  # 获取第二幅图像
        feature_a=featureA[0]  # 获取第一组特征
        feature_b=featureB[0]  # 获取第二组特征
        a_pts=featureA[1]  # 获取第一组特征点
        b_pts=featureB[1]  # 获取第二组特征点
        # print ('Start match:',feature_a.shape[0],feature_b.shape[0])  # 打印开始匹配的消息，并显示两组特征的数量
        
        #Ratio:
        bf = cv2.BFMatcher()  # Create a Brute-Force Matcher object
        matches = bf.knnMatch(np.float32(feature_a), np.float32(feature_b), k=2)  # Perform KNN matching, finding the 2 best matches for each descriptor

        # FLANN_INDEX_KDTREE = 0  # 定义 FLANN 匹配器的索引类型
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 设置 FLANN 匹配器的索引参数
        # search_params = dict(checks=50)  # or pass empty dictionary  # 设置 FLANN 匹配器的搜索参数
        # flann = cv2.FlannBasedMatcher(index_params, search_params)  # 创建 FLANN 匹配器
        # matches = flann.knnMatch(np.float32(feature_a), np.float32(feature_b), k=2)  # 使用 FLANN 匹配器对两组特征进行匹配，并返回每个特征的两个最佳匹配

        # Apply ratio test
        good = []  # 初始化好的匹配列表
        for m, n in matches:
            # 遍历每一对匹配
            if m.distance <= ratio * n.distance:
                # 如果最佳匹配的距离小于次佳匹配的距离乘以 ratio
                good.append(m)  # 则认为这是一个好的匹配，并将其添加到好的匹配列表中
        # print ('good', len(good))  # 打印好的匹配的数量
        

        #RANSAC
        src_pts = np.float32([ a_pts[m.queryIdx] for m in good ])  # 获取好的匹配中第一组特征点的坐标
        dst_pts = np.float32([ b_pts[m.trainIdx] for m in good ])  # 获取好的匹配中第二组特征点的坐标
        if save == True:
            # 如果需要保存特征点
            Psift.SaveFeaturePoint(im1, src_pts, im2, dst_pts, file + 'favourable_point')  # 保存特征点
        if RANSAC == True:
            # 如果使用 RANSAC 算法
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,mask=8)# eight-point algorithm  # 使用 RANSAC 算法计算单应性矩阵
            matchesMask = mask.ravel()  # 获取 RANSAC 算法的掩码
            src_pts = src_pts[matchesMask>0]  # 筛选出 RANSAC 算法认为是内点的特征点
            dst_pts = dst_pts[matchesMask>0]  # 筛选出 RANSAC 算法认为是内点的特征点
            if dispaly == True:
                # 如果需要显示特征点
                Psift.DisFeaturePoint(im1, src_pts, im2, dst_pts)  # 显示特征点
            if save ==True:
                # 如果需要保存特征点
                Psift.SaveFeaturePoint(im1, src_pts, im2, dst_pts, file + 'RANSAC_point')  # 保存特征点
            print ('RANSAC' ,len(src_pts))  # 打印检测到的特征点数量
        if plot_match==True:
            # 如果需要绘制匹配结果
            Psift.plot_match(im1,im2,src_pts,dst_pts,file,gray=True)  # 绘制匹配结果
        return src_pts, dst_pts  # 返回匹配的特征点坐标

    @staticmethod
    def plot_match(im1, im2, src_point, dst_point, file=None, gray=True):
        """
        显示两幅图像的匹配结果

        Args:
            im1: 第一幅图像 (NumPy 数组)
            im2: 第二幅图像 (NumPy 数组)
            src_point: 第一幅图像上的匹配点坐标 (Nx2 数组，第一列为 x 坐标，第二列为 y 坐标)
            dst_point: 第二幅图像上的匹配点坐标 (Nx2 数组，第一列为 x 坐标，第二列为 y 坐标)
            file: 保存图像的文件名 (如果为 None，则不保存)
            gray: 是否以灰度模式显示图像 (默认为 True)
        """
        plt.close()  # 关闭之前的图形
        x1 = src_point[:, 0]
        y1 = src_point[:, 1]
        x2 = dst_point[:, 0]
        y2 = dst_point[:, 1]

        # 水平拼接图像
        im3 = np.concatenate((im1, im2), axis=1)

        # 创建一个新的图形
        fig, ax = plt.subplots(figsize=(20, 20))

        # 显示拼接后的图像
        if gray:
            ax.imshow(im3, cmap='gray')
        else:
            ax.imshow(im3)

        # 计算第一幅图像的列数
        cols1 = im1.shape[1]

        # 绘制匹配线
        for i in range(x1.shape[0]):
            ax.plot([x1[i], x2[i] + cols1], [y1[i], y2[i]], 'c-', linewidth=1)  # 使用 'c-' 设置青色线，linewidth 设置线宽

        # 设置坐标轴不可见
        ax.axis('off')

        # 保存图像
        
        if file is not None:
            plt.savefig(file + 'mtach_result.png', bbox_inches='tight', pad_inches=0, dpi = 300)  # bbox_inches='tight', pad_inches=0 去除多余空白

        # 显示图形
        plt.show()

    @staticmethod
    def DisFeaturePoint(im1, new_point1, im2, new_point2):
        """
        显示两幅图像上的特征点 (使用 plt 改写)

        :param im1: 第一幅图像
        :param new_point1: 第一幅图像上的特征点坐标 (Nx2 数组，第一列为 x 坐标，第二列为 y 坐标)
        :param im2: 第二幅图像
        :param new_point2: 第二幅图像上的特征点坐标 (Nx2 数组，第一列为 x 坐标，第二列为 y 坐标)
        """

        plt.figure(figsize=(20, 20))  # 创建一个新的图形
        plt.subplot(121)  # 创建一个 1x2 的子图，并选择第一个子图 (左侧)
        plt.imshow(im1)
        plt.gray()
        plt.plot(new_point1[:, 1], new_point1[:, 0], 'r.')  # 在第一幅图像上绘制特征点，注意这里交换了 x 和 y 坐标，因为 plot 函数的第一个参数是 x 坐标，第二个参数是 y 坐标，而 new_point1 的第一列是 x 坐标，第二列是 y 坐标
        plt.axis('off')  # 关闭坐标轴

        plt.subplot(122)  # 选择第二个子图 (右侧)
        plt.imshow(im2)
        plt.gray()
        plt.plot(new_point2[:, 1], new_point2[:, 0], 'r.')  # 在第二幅图像上绘制特征点，同样需要交换 x 和 y 坐标
        plt.axis('off')

        plt.show()
        