import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime
from auxiliary import InterfaceLayout as il
from auxiliary import Tensseract_Test as ts
import pytesseract

IMAGE_SAVE_DIR = 'result/ImageRecognition_Img'
CSV_FILE_DIR = 'result/ImageRecognition_CSV'
CSV_FILE_PATH = CSV_FILE_DIR + '/recognized_results.csv'

# 创建结果保存目录
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)
if not os.path.exists(CSV_FILE_DIR):
    os.makedirs(CSV_FILE_DIR)


def process_uploaded_images(uploaded_files, progress_placeholder):
    """处理上传的图像并返回识别结果"""
    data = []
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        recognized_text = ts.ocr_image(image)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 保存上传的图片到指定目录
        image.save(os.path.join(IMAGE_SAVE_DIR, uploaded_file.name))
        data.append({"Filename": uploaded_file.name, "Recognized Text": recognized_text, "Timestamp": timestamp})

        # 更新进度条
        progress_placeholder.progress((idx + 1) / total_files)
    return data


def append_to_csv(data, file_path):
    """将识别结果追加到CSV文件"""
    df = pd.DataFrame(data)
    # 如果文件存在，追加数据；否则，创建新文件并写入表头
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)  # 追加模式，不写入header
    else:
        df.to_csv(file_path, index=False)  # 如果文件不存在，写入header
        df.to_csv(file_path, mode='a', header=False, index=False)


def clear_csv(file_path):
    """清空CSV文件内容"""
    if os.path.exists(file_path):
        open(file_path, 'w').close()


def is_csv_empty(file_path):
    """检查CSV文件是否为空"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def clear_image_history():
    """清空历史识别图片文件夹"""
    for file_name in os.listdir(IMAGE_SAVE_DIR):
        file_path = os.path.join(IMAGE_SAVE_DIR, file_name)
        os.remove(file_path)


def calculate_text_area_ratio(image):
    """
    使用Tesseract OCR检测图片中文字的占比
    :param image: PIL image
    :return: 文字占图片面积的比例
    """
    # 将PIL图像转换为OpenCV格式
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 使用pytesseract获取图像中文字的bounding box
    h, w, _ = open_cv_image.shape
    boxes = pytesseract.image_to_boxes(open_cv_image)

    total_text_area = 0
    for box in boxes.splitlines():
        b = box.split()
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        text_area = (x2 - x) * (y2 - y)
        total_text_area += text_area

    image_area = w * h
    text_area_ratio = total_text_area / image_area

    return text_area_ratio


def display_image_with_rotation(image_path):
    """
    显示图片，并检测图片中文字占比，决定是否旋转。
    """
    image = Image.open(image_path)

    # 计算文字内容在图像中的面积占比
    text_area_ratio = calculate_text_area_ratio(image)

    # 如果文字占比小于50%，旋转90度
    if text_area_ratio < 0.01 :
        image = image.rotate(90, expand=True)  # 如果满足条件，则旋转90度

    # 显示图片
    st.sidebar.image(image, caption=os.path.basename(image_path), use_column_width=True)


# Streamlit 主程序
def main():
    st.markdown("<h1 style='text-align: center; color: black;'>Image Recognition System</h1>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 选择图像输入方式
    icon_path = "data/introduction_src/icons/ImgRec_选择.png"
    il.display_icon_with_header(icon_path, "Select Image Input Method")
    option = st.selectbox('Please choose', ['Images from project folder', 'Manual upload'])

    if option == 'Images from project folder':
        base_folder_path = 'data/plate_img'
        subfolders = [f for f in os.listdir(base_folder_path) if
                      os.path.isdir(os.path.join(base_folder_path, f)) and f.startswith('Image_src')]

        if subfolders:
            selected_subfolder = st.selectbox('Please select an image folder', subfolders)
            folder_path = os.path.join(base_folder_path, selected_subfolder)

            if st.button('Start Recognition'):
                if os.path.exists(folder_path):
                    progress_placeholder = st.empty()
                    data, total_images = ts.process_images_from_folder(folder_path, progress_placeholder,
                                                                       IMAGE_SAVE_DIR)

                    if total_images == 0:
                        st.warning(f'No images found in folder {selected_subfolder}!')
                    elif data:
                        append_to_csv(data, CSV_FILE_PATH)
                        df = pd.DataFrame(data)
                        st.dataframe(df)  # 实时显示当前处理的图片结果
                        st.success(f'Recognition complete! Results saved to recognized_results.csv (Folder: {selected_subfolder})')
                else:
                    st.error(f'Folder {folder_path} does not exist!')
        else:
            st.error('No eligible folders found!')

    elif option == 'Manual upload':
        uploaded_files = st.file_uploader('Upload image files', type=['jpg', 'png'], accept_multiple_files=True)
        if uploaded_files:
            if st.button('Start Recognition'):
                progress_placeholder = st.empty()
                data = process_uploaded_images(uploaded_files, progress_placeholder)
                if data:
                    append_to_csv(data, CSV_FILE_PATH)
                    df = pd.DataFrame(data)
                    st.dataframe(df)  # 实时显示当前处理的图片结果
                    st.success('Recognition complete! Results saved to recognized_results.csv')

    icon_path = "data/introduction_src/icons/ImgRec_表格.png"
    il.display_icon_with_header(icon_path, "Current Content in CSV File")

    if st.button('Clear CSV File'):
        clear_csv(CSV_FILE_PATH)
        st.success('CSV file content cleared')

    if os.path.exists(CSV_FILE_PATH):
        if is_csv_empty(CSV_FILE_PATH):
            st.write('No recognition data available')
        else:
            df = pd.read_csv(CSV_FILE_PATH)
            st.dataframe(df)

    # 侧边栏展示历史识别图片
    st.sidebar.title("Recognized Image History")
    image_files = os.listdir(IMAGE_SAVE_DIR)
    if image_files:
        selected_image = st.sidebar.selectbox("Select an image to view", image_files)
        image_path = os.path.join(IMAGE_SAVE_DIR, selected_image)
        display_image_with_rotation(image_path)  # 使用新方法显示图片

        if st.sidebar.button('Clear Image History'):
            clear_image_history()
            st.sidebar.success('Image history cleared')
    else:
        st.sidebar.write('No recognized image history available')


if __name__ == '__main__':
    main()
