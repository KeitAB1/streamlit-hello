import streamlit as st
import base64
import os
from PIL import Image

# 设置页面标题和自定义颜色样式
st.markdown("<h1 style='text-align: center; color: #ffa365;'>智能仓储系统介绍</h1>", unsafe_allow_html=True)

# 添加英文副标题，颜色为 #ffc36f
st.markdown("<h2 style='text-align: center; color: #ffc36f;'>Intelligent Warehouse System Introduction</h2>", unsafe_allow_html=True)

# 添加间距
st.markdown("<br><br>", unsafe_allow_html=True)

# 将图片转换为 base64 格式
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 自定义一个函数来创建带有图标和标题的组合
def display_icon_with_header(icon_path, header_text):
    if os.path.exists(icon_path):
        # 将图片转换为 Base64 格式
        img_base64 = image_to_base64(icon_path)
        # 使用 HTML 和 CSS 实现图标和标题的对齐
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" width="40" style="margin-right: 10px;">
                <h3 style="margin: 0;">{header_text}</h3>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.header(header_text)

# 第一部分：文字描述，图标放在标题左边
intro_icon_path = "data/introduction_src/icons/intro_介绍.png"
display_icon_with_header(intro_icon_path, "系统功能介绍")

st.write("""
- 钢板图像识别
- 自动化入库调度
- 自动化出库
- 数据可视化
**该系统可以大幅提升仓储管理的效率，实现智能化、自动化的操作流程。**
""")

# 添加间距
st.markdown("<br><br>", unsafe_allow_html=True)

# 第二部分：展示多张图片介绍项目，图标放在标题左边
image_icon_path = "data/introduction_src/icons/intro_图片.png"
display_icon_with_header(image_icon_path, "项目图片展示")

image_dir = "data/introduction_src"

# 初始化 session state 中的图像索引
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

# 检查是否存在图片
if os.path.exists(image_dir):
    images = [img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    if images:
        total_images = len(images)
        current_index = st.session_state.image_index

        current_image_path = os.path.join(image_dir, images[current_index])

        # 显示当前图片
        image = Image.open(current_image_path)
        st.image(image, caption=f"项目图片 {current_index + 1}/{total_images}", use_column_width=True)

        # 创建图片切换按钮放在图片下方
        col_left, col_right = st.columns([1, 1])
        with col_left:
            if st.button("上一张"):
                st.session_state.image_index = (current_index - 1) % total_images
        with col_right:
            if st.button("下一张"):
                st.session_state.image_index = (current_index + 1) % total_images
    else:
        st.write("暂无项目介绍图片")
else:
    st.write("暂无项目介绍图片")

# 添加间距
st.markdown("<br><br>", unsafe_allow_html=True)

# 第三部分：展示视频介绍项目，图标放在标题左边
video_icon_path = "data/introduction_src/icons/Intro_视频.png"
display_icon_with_header(video_icon_path, "项目视频展示")

video_file = None

# 查找视频文件
if os.path.exists(image_dir):
    videos = [vid for vid in os.listdir(image_dir) if vid.endswith(('mp4', 'avi', 'mov', 'mkv'))]
    if videos:
        video_file = os.path.join(image_dir, videos[0])

# 显示视频
if video_file:
    st.video(video_file)
else:
    st.write("暂无项目介绍视频")
