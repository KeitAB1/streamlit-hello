import streamlit as st
import easyocr
import cv2
import os
import csv
import pandas as pd
from auxiliary import File_Save as fs
import time

# 通用的文本识别函数
def recognize_text(reader, frame):
    results = reader.readtext(frame)  # 进行文本识别
    recognized_texts = []  # 用于存储识别的文本
    for (bbox, text, prob) in results:
        recognized_texts.append(text)
    return recognized_texts

# 将识别的文本内容保存到 CSV 文件中
def save_to_csv(texts, csv_file):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for text in texts:
            writer.writerow([text])

# 显示 CSV 文件中的内容
def display_csv_content(csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, header=None, names=['识别内容'])
        st.dataframe(df)
    else:
        st.write("文件不存在或没有数据。")

def Video_Recognition():
    st.header("视频文本识别")
    st.write("请选择图像输入方式")

    option = st.selectbox('请选择输入方式', ['项目文件夹中的视频'])
    reader = easyocr.Reader(['en'], model_storage_directory="data/easyorc_models", gpu=False)


    csv_file = 'result/VideoRecognition_CSV/recognized_results.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['识别内容'])

    if option == '项目文件夹中的视频':
        video_folder = 'data/plate_video'
        videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

        if videos:
            selected_video = st.selectbox('请选择视频文件', videos)

            if st.button("开始识别"):
                video_path = os.path.join(video_folder, selected_video)
                cap = cv2.VideoCapture(video_path)

                all_recognized_texts = set()
                last_recognized_time = {}
                recognition_interval = 5  # 设置时间间隔（秒）

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)

                frame_count = 0
                process_every_n_frames = 10  # 每5帧识别一次
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 只处理每N帧
                    if frame_count % process_every_n_frames == 0:
                        recognized_texts = recognize_text(reader, frame)

                        current_time = time.time()
                        for text in recognized_texts:
                            if text not in all_recognized_texts:
                                if text not in last_recognized_time or (current_time - last_recognized_time[text] > recognition_interval):
                                    all_recognized_texts.add(text)
                                    last_recognized_time[text] = current_time

                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)

                cap.release()
                save_to_csv(all_recognized_texts, csv_file)

        else:
            st.write("项目文件夹中没有找到视频文件。")

    st.write("识别的所有内容：")
    if st.button('Clear CSV File'):
        fs.clear_csv(csv_file)
        st.success('CSV file content cleared')
    display_csv_content(csv_file)
    st.write("Current Content in CSV File")

