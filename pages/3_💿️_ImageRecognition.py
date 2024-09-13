import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import os
from datetime import datetime
from auxiliary import InterfaceLayout as il

CSV_FILE_PATH = 'recognized_results.csv'
IMAGE_SAVE_DIR = 'result/ImageRecognition'

# 创建结果保存目录
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)


def ocr_image(image):
    """对传入的图像进行OCR识别"""
    return pytesseract.image_to_string(image, lang='chi_sim')


def process_images_from_folder(folder_path, progress_placeholder):
    """对文件夹中的所有图像进行OCR识别并返回结果"""
    data = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    total_images = len(image_files)

    if total_images == 0:
        return None, 0  # 如果文件夹中没有图片，返回None

    for idx, file_name in enumerate(image_files):
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)
        recognized_text = ocr_image(image)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 保存处理过的图片到指定目录
        image.save(os.path.join(IMAGE_SAVE_DIR, file_name))
        data.append({"Filename": file_name, "Recognized Text": recognized_text, "Timestamp": timestamp})

        # 更新进度条
        progress_placeholder.progress((idx + 1) / total_images)
    return data, total_images


def process_uploaded_images(uploaded_files, progress_placeholder):
    """处理上传的图像并返回识别结果"""
    data = []
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        recognized_text = ocr_image(image)
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


# Streamlit 主程序
def main():
    # st.title('图像识别系统')
    # 设置页面标题和自定义颜色样式
    st.markdown("<h1 style='text-align: center; color: black;'>图像识别系统</h1>", unsafe_allow_html=True)
    # 添加间距
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 选择图像输入方式
    icon_path = "data/introduction_src/icons/ImgRec_选择.png"
    il.display_icon_with_header(icon_path, "选择图像输入方式")
    option = st.selectbox('请选择', ['项目文件夹中的图片', '手动上传'])

    if option == '项目文件夹中的图片':
        base_folder_path = 'data/plate_img'
        # 获取所有以“Image_src”+序号命名的子文件夹
        subfolders = [f for f in os.listdir(base_folder_path) if
                      os.path.isdir(os.path.join(base_folder_path, f)) and f.startswith('Image_src')]

        if subfolders:
            selected_subfolder = st.selectbox('请选择一个图片集文件夹', subfolders)
            folder_path = os.path.join(base_folder_path, selected_subfolder)

            if st.button('开始识别'):
                if os.path.exists(folder_path):
                    # 创建进度条占位符
                    progress_placeholder = st.empty()

                    # 处理文件夹中的图像
                    data, total_images = process_images_from_folder(folder_path, progress_placeholder)

                    if total_images == 0:
                        st.warning(f'文件夹 {selected_subfolder} 中未查找到图片！')
                    elif data:
                        append_to_csv(data, CSV_FILE_PATH)
                        df = pd.DataFrame(data)
                        st.dataframe(df)  # 实时显示当前处理的图片结果
                        st.success(f'识别完成！结果已保存到 recognized_results.csv（文件夹：{selected_subfolder}）')
                else:
                    st.error(f'文件夹 {folder_path} 不存在！')
        else:
            st.error('未找到符合条件的文件夹！')

    elif option == '手动上传':
        # 手动上传多张图像
        uploaded_files = st.file_uploader('上传图像文件', type=['jpg', 'png'], accept_multiple_files=True)
        if uploaded_files:
            if st.button('开始识别'):
                # 创建进度条占位符
                progress_placeholder = st.empty()

                data = process_uploaded_images(uploaded_files, progress_placeholder)
                if data:
                    append_to_csv(data, CSV_FILE_PATH)
                    df = pd.DataFrame(data)
                    st.dataframe(df)  # 实时显示当前处理的图片结果
                    st.success('识别完成！结果已保存到 recognized_results.csv')

    icon_path = "data/introduction_src/icons/ImgRec_表格.png"
    il.display_icon_with_header(icon_path, "CSV文件中的现有内容")

    if st.button('清空CSV文件内容'):
        clear_csv(CSV_FILE_PATH)
        st.success('CSV文件内容已清空')

    if os.path.exists(CSV_FILE_PATH):
        if is_csv_empty(CSV_FILE_PATH):
            st.write('暂无识别数据')
        else:
            df = pd.read_csv(CSV_FILE_PATH)
            st.dataframe(df)

    # 侧边栏展示历史识别图片
    st.sidebar.title("历史识别图片")
    image_files = os.listdir(IMAGE_SAVE_DIR)
    if image_files:
        selected_image = st.sidebar.selectbox("选择图片查看", image_files)
        image_path = os.path.join(IMAGE_SAVE_DIR, selected_image)
        st.sidebar.image(image_path, caption=selected_image, use_column_width=True)

        # 添加清空图片历史按钮
        if st.sidebar.button('清空图片历史'):
            clear_image_history()
            st.sidebar.success('图片历史已清空')
    else:
        st.sidebar.write('暂无历史识别图片')


if __name__ == '__main__':
    main()
