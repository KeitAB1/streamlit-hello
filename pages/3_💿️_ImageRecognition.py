import streamlit as st
import easyocr
from auxiliary import ImgRec_fun as ir


# 加载模型
@st.cache_resource
def load_ocr_model():
    # 使用 st.spinner 来显示加载提示
    with st.spinner("模型加载中......"):
        reader = easyocr.Reader(['en'], model_storage_directory="data/easyorc_models")
    return reader

# 加载 OCR 模型
reader = load_ocr_model()

# 使用 Img_Rec.py 中的全局实例设置模型
ir.img_rec_instance.set_reader(reader)

# 定义图片和CSV文件保存路径
IMAGE_SAVE_DIR = 'result/ImageRecognition_Img'
CSV_FILE_DIR = 'result/ImageRecognition_CSV'
CSV_FILE_PATH = CSV_FILE_DIR + '/recognized_results.csv'
VIDEO_DIR = 'data/plate_video'  # 保存视频的目录


def main():
    """主函数"""
    st.markdown("<h1 style='text-align: center; color: black;'>Image & Video Recognition System</h1>",
                unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 选择识别模式：图像识别或视频识别
    mode = st.selectbox('Please select recognition mode', ['Image Recognition', 'Video Recognition'])

    if mode == 'Image Recognition':
        ir.img_rec_instance.Image_Recongnotion(IMAGE_SAVE_DIR, CSV_FILE_PATH)

    elif mode == 'Video Recognition':
        ir.img_rec_instance.Video_Recognition(IMAGE_SAVE_DIR,CSV_FILE_PATH)

    ir.Rec_history_image(IMAGE_SAVE_DIR)
    ir.csv_display(CSV_FILE_PATH)


if __name__ == "__main__":
    main()
