import streamlit as st
import os
import time
from PIL import Image
from auxiliary import InterfaceLayout as il

# 设置页面标题和自定义颜色样式
st.markdown("<h1 style='text-align: center; color: #ffa365;'>智能仓储系统介绍</h1>", unsafe_allow_html=True)

# 添加英文副标题，颜色为 #ffc36f
st.markdown("<h2 style='text-align: center; color: #ffc36f;'>Intelligent Warehouse System Introduction</h2>", unsafe_allow_html=True)

# 添加间距
st.markdown("<br><br>", unsafe_allow_html=True)

# 第一部分：文字描述，图标放在标题左边
intro_icon_path = "data/introduction_src/icons/intro_介绍.png"
il.display_icon_with_header(intro_icon_path, "系统功能介绍")

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
il.display_icon_with_header(image_icon_path, "项目图片展示")

image_dir = "data/introduction_src"

# 初始化 session state 中的图像索引和自动切换状态
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'auto_switch' not in st.session_state:
    st.session_state.auto_switch = True

# 检查是否存在图片
if os.path.exists(image_dir):
    images = [img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    if images:
        total_images = len(images)
        current_index = st.session_state.image_index

        current_image_path = os.path.join(image_dir, images[current_index])

        # 使用st.image来显示图片
        image = Image.open(current_image_path)
        st.image(image, caption=f"项目图片 {current_index + 1}/{total_images}", use_column_width=True)

        # 自动切换逻辑，每5秒自动切换
        if st.session_state.auto_switch:
            # 使用 st_autorefresh，每5秒触发自动切换
            st.experimental_rerun() if time.time() % 5 < 1 else None

        # 创建图片切换按钮放在图片下方
        col_left, col_right = st.columns([1, 1])

        prev_button = col_left.button("上一张")
        next_button = col_right.button("下一张")

        # 使用按钮返回值来控制图像索引的更新
        if prev_button:
            st.session_state.image_index = (current_index - 1) % total_images
            st.session_state.auto_switch = False  # 按按钮后暂停自动切换
            st.experimental_rerun()
        if next_button:
            st.session_state.image_index = (current_index + 1) % total_images
            st.session_state.auto_switch = False  # 按按钮后暂停自动切换
            st.experimental_rerun()

        # JavaScript部分用于暂停自动切换
        st.markdown(
            """
            <script>
            let imgElement = document.querySelector('img');
            let isHovered = false;

            imgElement.addEventListener('mouseover', function() {
                isHovered = true;
                window.parent.postMessage({'type': 'pauseAutoSwitch'}, '*');
            });

            imgElement.addEventListener('mouseout', function() {
                isHovered = false;
                window.parent.postMessage({'type': 'resumeAutoSwitch'}, '*');
            });
            </script>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("暂无项目介绍图片")
else:
    st.write("暂无项目介绍图片")

# 添加间距
st.markdown("<br><br>", unsafe_allow_html=True)

# 第三部分：展示视频介绍项目，图标放在标题左边
video_icon_path = "data/introduction_src/icons/Intro_视频.png"
il.display_icon_with_header(video_icon_path, "项目视频展示")

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
