import streamlit as st
import os
from PIL import Image

# 设置页面标题
st.title("智能仓储系统介绍")

# 第一部分：文字描述
st.header("系统功能介绍")
st.write("""
我们的智能仓储系统包含以下功能：
- 钢板图像识别
- 自动化入库调度
- 自动化出库
- 数据可视化
该系统可以大幅提升仓储管理的效率，实现智能化、自动化的操作流程。
""")

# 第二部分：展示多张图片介绍项目
st.header("项目图片展示")
image_dir = "data/introduction_src"

# 检查是否存在图片
if os.path.exists(image_dir):
    images = [img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    if images:
        for img in images:
            image_path = os.path.join(image_dir, img)
            image = Image.open(image_path)
            st.image(image, caption=f"项目图片: {img}", use_column_width=True)
    else:
        st.write("暂无项目介绍图片")
else:
    st.write("暂无项目介绍图片")

# 第三部分：展示视频介绍项目
st.header("项目视频展示")
video_file = None

# 查找视频文件
if os.path.exists(image_dir):
    videos = [vid for vid in os.listdir(image_dir) if vid.endswith(('mp4', 'avi', 'mov', 'mkv'))]
    if videos:
        video_file = os.path.join(image_dir, videos[0])

# 显示视频
if video_file:
    with open(video_file, 'rb') as video:
        st.video(video.read())
else:
    st.write("暂无项目介绍视频")
