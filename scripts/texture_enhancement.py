import cv2
import dlib
import numpy as np
import math
import os
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

def select_n_components(pixels, max_components=10):
    best_n = 1
    best_bic = float('inf')
    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm.fit(pixels)
            bic = gmm.bic(pixels)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception as e:
            print(f"分量数 {n} 训练失败: {e}")
            continue
    print(f"最佳分量数: {best_n} (BIC: {best_bic})")
    return best_n

def line_intersect_box(x0, y0, dx, dy, box_width, box_height):
    t_min = float('inf')
    x_intersect, y_intersect = x0, y0
    if dx != 0:
        t = -x0 / dx
        y = y0 + t * dy
        if 0 <= y <= box_height and t > 0 and t < t_min:
            t_min = t
            x_intersect, y_intersect = 0, y
    if dx != 0:
        t = (box_width - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= box_height and t > 0 and t < t_min:
            t_min = t
            x_intersect, y_intersect = box_width, y
    if dy != 0:
        t = -y0 / dy
        x = x0 + t * dx
        if 0 <= x <= box_width and t > 0 and t < t_min:
            t_min = t
            x_intersect, y_intersect = x, 0
    if dy != 0:
        t = (box_height - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= box_width and t > 0 and t < t_min:
            t_min = t
            x_intersect, y_intersect = x, box_height
    return x_intersect, y_intersect

def texture_enhance(image_pth, output_floder, face_cascade, predictor):
    """
    对输入文件夹中的图像进行纹理增强处理，并将结果保存到输出文件夹。
    
    参数:
    input_floder (str): 输入图像文件夹路径。
    output_floder (str): 输出图像文件夹路径。
    """
    if not os.path.exists(output_floder):
        os.makedirs(output_floder)

    image = cv2.imread(image_pth)

    if image is None:
        raise ValueError("无法加载图像，请检查文件路径！")

    # 获取图像尺寸
    H, W = image.shape[:2]

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 面部检测（Viola-Jones）
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # 假设只处理第一张脸
    if len(faces) == 0:
        print("未检测到面部，程序退出。")
        exit()

    # 获取第一个面部框
    x, y, w, h = faces[0]

    # 将OpenCV的矩形转换为dlib的矩形
    dlib_rect = dlib.rectangle(x, y, x+w, y+h)

    # 使用dlib检测面部特征点
    landmarks = predictor(gray, dlib_rect)

    # 复制图像以绘制关键点和面部框
    image_with_box = image.copy()

    # 绘制所有68个关键点（黑色圆点）并添加标号
    for i in range(68):
        point_x = landmarks.part(i).x
        point_y = landmarks.part(i).y
        cv2.circle(image_with_box, (point_x, point_y), 10, (0, 0, 0), -1)
        cv2.putText(image_with_box, str(i), (point_x - 5, point_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

    # 定义极值特征点
    p_L = (landmarks.part(17).x, landmarks.part(17).y)
    p_R = (landmarks.part(26).x, landmarks.part(26).y)
    p_T = (landmarks.part(27).x, landmarks.part(27).y)
    p_B = (landmarks.part(8).x, landmarks.part(8).y)

    # 计算初始面部框的宽度和高度
    width = abs(p_L[0] - p_R[0])
    height = abs(p_B[1] - p_T[1])

    # 计算扩展后的面部框
    left_margin = int(0.35 * width)
    right_margin = int(0.35 * width)
    top_margin = int(1.1 * height)
    bottom_margin = int(0.1 * height)

    # 计算新面部框的边界
    x_new = max(0, p_L[0] - left_margin)
    y_new = max(0, p_T[1] - top_margin)
    w_new = width + left_margin + right_margin
    h_new = height + top_margin + bottom_margin

    # 确保新面部框不超出图像边界
    x_new = min(x_new, W - w_new)
    y_new = min(y_new, H - h_new)
    w_new = min(w_new, W - x_new)
    h_new = min(h_new, H - y_new)


    # 绘制新面部框（红色）
    cv2.rectangle(image_with_box, (x_new, y_new), (x_new + w_new, y_new + h_new), (0, 0, 255), 5)

    # 计算 p_H（使用原图坐标）
    x_B = p_B[0]
    y_B = p_B[1]
    y_T = p_T[1]
    p_H = (x_B, int(y_T - 0.9 * (y_B - y_T)))

    # 绘制 p_H 点（绿色圆点）
    cv2.circle(image_with_box, p_H, 5, (0, 255, 0), -1)
    cv2.putText(image_with_box, "p_H", (p_H[0] + 10, p_H[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 生成多边形掩码
    vertex_indices = [0, 17, 36, 41, 40, 39, 21, 22, 42, 47, 46, 45, 26, 16, 15, 14, 13, 12, 35, 30, 31, 4, 3, 2, 1, 0]
    contour_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in vertex_indices], dtype=np.int32)
    mask = np.zeros((h_new, w_new), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_points], 255)


    # 腐蚀掩码
    kernel_size = int(201 * min(w_new, h_new) / min(H, W))
    kernel_size = max(3, kernel_size // 2 * 2 + 1)
    blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    _, eroded_mask = cv2.threshold(blurred_mask, 254, 255, cv2.THRESH_BINARY)


    # 应用腐蚀掩码
    face_box_image = image[y_new:y_new+h_new, x_new:x_new+w_new].copy()


    # 拟合面部椭圆
    boundary_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    boundary_points = [(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in boundary_indices]
    x_B = p_B[0] - x_new
    y_B = p_B[1] - y_new
    y_T = p_T[1] - y_new
    p_H_mapped = (x_B, int(y_T - 0.9 * (y_B - y_T)))
    ellipse_points = boundary_points + [p_H_mapped]
    ellipse_points = np.array(ellipse_points, dtype=np.float32)

    if len(ellipse_points) >= 5:
        ellipse = cv2.fitEllipse(ellipse_points)
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse
    else:
        raise ValueError("拟合椭圆失败：点数不足！")

    # 生成颈部矩形
    short_axis_length = major_axis * 0.85 / 2
    short_axis_length = max(10, short_axis_length)
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    short_axis_x1 = center_x + short_axis_length * cos_theta
    short_axis_y1 = center_y + short_axis_length * sin_theta
    short_axis_x2 = center_x - short_axis_length * cos_theta
    short_axis_y2 = center_y - short_axis_length * sin_theta

    short_axis_rad = math.radians(angle + 90)
    cos_short = math.cos(short_axis_rad)
    sin_short = math.sin(short_axis_rad)
    if sin_short < 0:
        short_axis_rad = math.radians(angle + 270)
        cos_short = math.cos(short_axis_rad)
        sin_short = math.sin(short_axis_rad)

    

    neck_x1, neck_y1 = line_intersect_box(short_axis_x1, short_axis_y1, cos_short, sin_short, w_new - 1, h_new - 1)
    neck_x2, neck_y2 = line_intersect_box(short_axis_x2, short_axis_y2, cos_short, sin_short, w_new - 1, h_new - 1)

    neck_rect_points = np.array([
        (short_axis_x1, short_axis_y1),
        (short_axis_x2, short_axis_y2),
        (neck_x2, neck_y2),
        (neck_x1, neck_y1)
    ], dtype=np.int32)


    # 构建非皮肤掩码
    non_skin_mask = np.full((h_new, w_new), 255, dtype=np.uint8)
    cv2.ellipse(non_skin_mask, (int(center_x), int(center_y)), (int(major_axis/2), int(minor_axis/2)), angle, 0, 360, 0, -1)
    cv2.fillPoly(non_skin_mask, [neck_rect_points], 0)

    left_eye_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in range(36, 42)], dtype=np.int32)
    cv2.fillPoly(non_skin_mask, [left_eye_points], 255)

    right_eye_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in range(42, 48)], dtype=np.int32)
    cv2.fillPoly(non_skin_mask, [right_eye_points], 255)

    mouth_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in range(48, 60)], dtype=np.int32)
    cv2.fillPoly(non_skin_mask, [mouth_points], 255)







    # 统计 eroded_mask 和 non_skin_mask 中像素的 RGB 信息，建立 GMM 模型
    skin_pixels = face_box_image[eroded_mask == 255].reshape(-1, 3)
    skin_pixels = skin_pixels.astype(np.float32)

    non_skin_pixels = face_box_image[non_skin_mask == 255].reshape(-1, 3)
    non_skin_pixels = non_skin_pixels.astype(np.float32)

    

    if len(skin_pixels) < 10 or len(non_skin_pixels) < 10:
        print("警告：皮肤或非皮肤区域像素数量不足，无法建立 GMM 模型！")
    else:
        n_components_skin = select_n_components(skin_pixels, max_components=5)
        gmm_skin = GaussianMixture(n_components=n_components_skin, covariance_type='full', random_state=42)
        gmm_skin.fit(skin_pixels)
        
        n_components_non_skin = select_n_components(non_skin_pixels, max_components=5)
        gmm_non_skin = GaussianMixture(n_components=n_components_non_skin, covariance_type='full', random_state=42)
        gmm_non_skin.fit(non_skin_pixels)
        


    # 基于贝叶斯公式计算后验概率，生成连续概率的皮肤区域分割图
    if len(skin_pixels) >= 10 and len(non_skin_pixels) >= 10:
        total_pixels = len(skin_pixels) + len(non_skin_pixels)
        prior_skin = len(skin_pixels) / total_pixels
        prior_non_skin = len(non_skin_pixels) / total_pixels
        
        image_pixels = image.reshape(-1, 3).astype(np.float32)
        
        log_likelihood_skin = gmm_skin.score_samples(image_pixels)
        log_likelihood_non_skin = gmm_non_skin.score_samples(image_pixels)
        
        posterior_skin = np.exp(log_likelihood_skin + np.log(prior_skin))
        posterior_non_skin = np.exp(log_likelihood_non_skin + np.log(prior_non_skin))
        posterior_sum = posterior_skin + posterior_non_skin
        posterior_skin = posterior_skin / (posterior_sum + 1e-10)
        
        # 生成连续概率的皮肤分割图（值在 [0, 1]）
        skin_segmentation = posterior_skin.reshape(H, W)
        # 缩放到 [0, 255] 以保存为图像

        
        # --- 新增：计算 face box mask with features unmasked ---
        # 创建面部框尺寸的掩码，初始为1
        face_box_mask = np.ones((h_new, w_new), dtype=np.float32)
        
        # 定义并去除眼睛和嘴巴区域
        left_eye_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in range(36, 42)], dtype=np.int32)
        right_eye_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in range(42, 48)], dtype=np.int32)
        mouth_points = np.array([(landmarks.part(i).x - x_new, landmarks.part(i).y - y_new) for i in range(48, 60)], dtype=np.int32)
        
        # 在面部框掩码上填充眼睛和嘴巴区域为0
        cv2.fillPoly(face_box_mask, [left_eye_points], 0)
        cv2.fillPoly(face_box_mask, [right_eye_points], 0)
        cv2.fillPoly(face_box_mask, [mouth_points], 0)
        
        # 扩展到原图尺寸，补全区域为0
        expanded_face_box_mask = np.zeros((H, W), dtype=np.float32)
        expanded_face_box_mask[y_new:y_new+h_new, x_new:x_new+w_new] = face_box_mask

        
        # --- 将 skin_segmentation 与 expanded_face_box_mask 相乘，得到中间结果 ---
        intermediate_skin_segmentation = skin_segmentation * expanded_face_box_mask

        
        # --- 应用连通区域算法恢复颈部皮肤区域 ---
        threshold = 0.01
        binary_skin = (skin_segmentation > threshold).astype(np.uint8) * 255


        # 连通区域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_skin, connectivity=8)

        




        # 确定颈部区域底部边缘
        bottom_edge_y = y_new + h_new - 1
        if bottom_edge_y >= H:
            bottom_edge_y = H - 1


        # 查找与底部边缘相连的连通区域
        connected_labels = set()
        for x in range(x_new, min(x_new + w_new, W)):
            if binary_skin[bottom_edge_y, x] == 255:
                label = labels[bottom_edge_y, x]
                if label > 0:
                    connected_labels.add(label)


        # 创建颈部连通区域掩码
        neck_connected_mask = np.zeros_like(skin_segmentation, dtype=np.float32)
        for label in connected_labels:
            neck_connected_mask[labels == label] = 1.0


        # 创建包含眼睛和嘴巴的面部框掩码
        full_face_box_mask = np.ones((h_new, w_new), dtype=np.float32)

        # 扩展到原图尺寸
        expanded_full_face_box_mask = np.zeros((H, W), dtype=np.float32)
        expanded_full_face_box_mask[y_new:y_new+h_new, x_new:x_new+w_new] = full_face_box_mask


        # 排除面部框中的像素
        neck_connected_mask = neck_connected_mask * (1 - expanded_full_face_box_mask)

        
        # --- 生成最终皮肤分割图 ---
        final_skin_segmentation = intermediate_skin_segmentation.copy()
        final_skin_segmentation[neck_connected_mask == 1.0] = skin_segmentation[neck_connected_mask == 1.0]

        final_skin_segmentation_binary=(final_skin_segmentation>0.01)

        
        # 在生成 final_skin_segmentation_binary 后添加高斯模糊
        # 应用高斯模糊到原图
        blur_kernel_size = (51, 51)  # 高斯模糊核大小，需为奇数
        blurred_image = cv2.GaussianBlur(image, blur_kernel_size, 20)

        # 创建模糊后的图像，仅在掩码区域应用模糊
        blurred_skin_image = image.copy()
        blurred_skin_image[final_skin_segmentation_binary == 1] = blurred_image[final_skin_segmentation_binary == 1]


        # --- 新增功能：计算差值并叠加 ---
        diff = image.astype(np.float32) - blurred_skin_image.astype(np.float32)
        scaled_diff = diff * 1.5
        enhanced_diff_image = image.astype(np.float32) + scaled_diff
        enhanced_diff_image = np.clip(enhanced_diff_image, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_pth)), enhanced_diff_image)



if __name__ == "__main__":

    ID = '175'
    image_folder = f'/media/DGST_data/Data/{ID}/cam09/'
    output_dir = f"./out_enhance/{ID}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载预训练模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor("./texture_enhancement/shape_predictor_68_face_landmarks.dat")

    # 检查模型文件是否存在
    if face_cascade.empty():
        raise FileNotFoundError("Haar级联分类器文件未找到！")
    if not predictor:
        raise FileNotFoundError("dlib关键点模型文件未找到！")

    image_files = sorted(os.listdir(image_folder))
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)   
        texture_enhance(image_path, output_dir, face_cascade, predictor)
   
