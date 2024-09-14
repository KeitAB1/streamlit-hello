import os
from datetime import datetime
import cv2
import numpy as np
import pytesseract
from PIL import Image

def ocr_image(image):
    """对传入的图像进行OCR识别"""
    return pytesseract.image_to_string(image, lang='chi_sim')

# 图像预处理函数
def preprocess_image(image):
    """
    对图像进行预处理以提高OCR的识别效果。
    包括灰度转换、双边滤波、阈值处理等。
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用双边滤波，保留边缘并减少噪声
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # 应用自适应阈值，将图像转换为二值图像
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return thresh

#先校正再识别
def process_images_from_folder(folder_path, progress_placeholder, IMAGE_SAVE_DIR):
    """对文件夹中的所有图像进行OCR识别并返回结果，加入图片校正和调整过程"""
    data = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png','.bmp'))]
    total_images = len(image_files)

    if total_images == 0:
        return None, 0  # 如果文件夹中没有图片，返回None

    for idx, file_name in enumerate(image_files):
        image_path = os.path.join(folder_path, file_name)

        # 1. 图像校正
        corrected_image = Corrected_Tilt(image_path)
        # 2. 调整图像
        adjusted_image = corrected_image.adjustImg()
        # 3. 图像预处理
        preprocessed_image = preprocess_image(adjusted_image)
        # 4. 使用校正和预处理后的图像进行OCR识别
        recognized_text = corrected_image.ocr_with_tesseract(preprocessed_image)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 将 numpy 数组转换为 PIL.Image 对象
        adjusted_image = Image.fromarray(adjusted_image)
        # 保存处理过的图片到指定目录
        adjusted_image.save(os.path.join(IMAGE_SAVE_DIR, file_name))  # 保存调整后的图像
        # 保存识别数据
        data.append({"Filename": file_name, "Recognized Text": recognized_text, "Timestamp": timestamp})
        # 更新进度条
        progress_placeholder.progress((idx + 1) / total_images)

    return data, total_images

class Corrected_Tilt:

    def __init__(self,img_path):
        self.img_path = img_path
        self.image = cv2.imread(img_path)

    def detect_skew_angle(self, image):
        """
        检测图像的倾斜角度。

        :param image: 输入的图像（BGR格式）
        :return: 倾斜角度（度）
        """
        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用 Canny 边缘检测算法找到图像中的边缘
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        # 如果检测到直线，则计算所有直线的角度的中位数作为图像的倾斜角度
        print('倾斜度数：',end='')
        if lines is not None:
            angles = []
            for rho, theta in lines[0]:
                angle = theta * 180 / np.pi
                angles.append(angle)

            median_angle = np.median(angles)

            print(median_angle-90)
            return median_angle
        else:
            # 如果没有检测到足够的直线，则返回 0，表示没有明显的倾斜
            return 0

    def rotate_image(self, image, angle):
        """
        旋转图像以校正倾斜。

        :param image: 输入的图像（BGR格式）
        :param angle: 倾斜角度（度）
        :return: 旋转后的图像
        """
        # 获取图像的高度和宽度
        (h, w) = image.shape[:2]

        # 计算旋转中心
        center = (w // 2, h // 2)

        # 创建旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 使用 warpAffine 函数旋转图像
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def ocr_with_tesseract(self, image):
        """
        使用 Tesseract OCR 识别图像中的文本。

        :param image: 输入的图像（BGR格式）
        :return: 识别出的文本
        """
        # 使用 Tesseract OCR 识别图像中的文本
        text = pytesseract.image_to_string(image, lang='chi_sim')

        # 返回识别出的文本（去除多余的空白）
        return text.strip()

    def adjustImg(self):
        angle = self.detect_skew_angle(self.image)  # 检测倾斜角度
        rotated_image = self.rotate_image(self.image, angle-90)   # 旋转图像
        return rotated_image
