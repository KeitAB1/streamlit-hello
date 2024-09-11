import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import os
from datetime import datetime

CSV_FILE_PATH = 'recognized_results.csv'
IMAGE_HISTORY_DIR = 'image_history'

# 创建历史图像文件夹
if not os.path.exists(IMAGE_HISTORY_DIR):
    os.makedirs(IMAGE_HISTORY_DIR)

def ocr_image(image):
    """对传入的图像进行OCR识别"""
    return pytesseract.image_to_string(image, lang='chi_sim')

def process_images_from_folder(folder_path):
    """对文件夹中的所有图像进行OCR识别并返回结果"""
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            recognized_text = ocr_image(image)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 保存处理过的图片到历史记录文件夹
            image.save(os.path.join(IMAGE_HISTORY_DIR, file_name))
            data.append({"Filename": file_name, "Recognized Text": recognized_text, "Timestamp": timestamp})
    return data

def process_uploaded_images(uploaded_files):
    """处理上传的图像并返回识别结果"""
    data = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        recognized_text = ocr_image(image)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 保存上传的图片到历史记录文件夹
        image.save(os.path.join(IMAGE_HISTORY_DIR, uploaded_file.name))
        data.append({"Filename": uploaded_file.name, "Recognized Text": recognized_text, "Timestamp": timestamp})
    return data

def append_to_csv(data, file_path):
    """将识别结果追加到CSV文件"""
    df = pd.DataFrame(data)
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)  # 追加模式，不写入header
    else:
        df.to_csv(file_path, index=False)  # 如果文件不存在，写入header

def clear_csv(file_path):
    """清空CSV文件内容"""
    if os.path.exists(file_path):
        open(file_path, 'w').close()

def is_csv_empty(file_path):
    """检查CSV文件是否为空"""
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

def clear_image_history():
    """清空历史识别图片文件夹"""
    for file_name in os.listdir(IMAGE_HISTORY_DIR):
        file_path = os.path.join(IMAGE_HISTORY_DIR, file_name)
        os.remove(file_path)

# Streamlit 主程序
def main():
    st.title('图像识别系统')

    # 选择图像输入方式
    option = st.selectbox('选择图像输入方式', ['项目文件夹中的图片', '手动上传'])

    if option == '项目文件夹中的图片':
        folder_path = 'data/plate_img'
        if st.button('开始识别'):
            if os.path.exists(folder_path):
                data = process_images_from_folder(folder_path)
                if data:
                    append_to_csv(data, CSV_FILE_PATH)
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    st.success('识别完成！结果已保存到 recognized_results.csv')
            else:
                st.error(f'文件夹 {folder_path} 不存在！')

    elif option == '手动上传':
        # 手动上传多张图像
        uploaded_files = st.file_uploader('上传图像文件', type=['jpg', 'png'], accept_multiple_files=True)
        if uploaded_files:
            if st.button('开始识别'):
                data = process_uploaded_images(uploaded_files)
                if data:
                    append_to_csv(data, CSV_FILE_PATH)
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    st.success('识别完成！结果已保存到 recognized_results.csv')

    # 显示清除CSV文件内容的按钮
    if st.button('清空CSV文件内容'):
        clear_csv(CSV_FILE_PATH)
        st.success('CSV文件内容已清空')

    # 显示CSV文件中的当前内容
    if os.path.exists(CSV_FILE_PATH):
        if is_csv_empty(CSV_FILE_PATH):
            st.write('暂无识别数据')
        else:
            st.subheader('CSV文件中的现有内容:')
            df = pd.read_csv(CSV_FILE_PATH)
            st.dataframe(df)

    # 侧边栏展示历史识别图片
    st.sidebar.title("历史识别图片")
    image_files = os.listdir(IMAGE_HISTORY_DIR)
    if image_files:
        selected_image = st.sidebar.selectbox("选择图片查看", image_files)
        image_path = os.path.join(IMAGE_HISTORY_DIR, selected_image)
        st.sidebar.image(image_path, caption=selected_image, use_column_width=True)

        # 添加清空图片历史按钮
        if st.sidebar.button('清空图片历史'):
            clear_image_history()
            st.sidebar.success('图片历史已清空')
    else:
        st.sidebar.write('暂无历史识别图片')

if __name__ == '__main__':
    main()
