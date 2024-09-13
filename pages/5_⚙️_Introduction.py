import streamlit as st
import os
from PIL import Image
from auxiliary import InterfaceLayout as il

#图片/视频文件夹地址
image_dir = "data/introduction_src"

# 设置页面标题为黑色，英文副标题为灰色
st.markdown("<h1 style='text-align: center; color: black;'>智能仓储系统介绍</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: gray;'>Intelligent Warehouse System Introduction</h2>", unsafe_allow_html=True)

# 使用 session state 控制显示的部分
if 'section' not in st.session_state:
    st.session_state.section = '功能介绍'

# 侧边栏用于切换部分内容
st.sidebar.title("目录")
if st.sidebar.button("系统功能介绍"):
    st.session_state.section = '功能介绍'
    st.experimental_rerun()

if st.sidebar.button("项目图片展示"):
    st.session_state.section = '图片展示'
    st.experimental_rerun()

if st.sidebar.button("项目视频展示"):
    st.session_state.section = '视频展示'
    st.experimental_rerun()

# 根据 session state 中的值展示不同的部分内容
if st.session_state.section == '功能介绍':
    st.markdown("<h3 style='text-align: left;'>系统功能介绍</h3>", unsafe_allow_html=True)
    intro_icon_path = "data/introduction_src/icons/intro_介绍.png"
    il.display_icon_with_header(intro_icon_path, "系统功能介绍")

    st.write("""
    - 钢板图像识别
    - 自动化入库调度
    - 自动化出库
    - 数据可视化
    **该系统可以大幅提升仓储管理的效率，实现智能化、自动化的操作流程。**
    """)

elif st.session_state.section == '图片展示':
    st.markdown("<h3 style='text-align: left;'>项目图片展示</h3>", unsafe_allow_html=True)
    image_icon_path = "data/introduction_src/icons/intro_图片.png"
    il.display_icon_with_header(image_icon_path, "项目图片展示")

    # 检查是否存在图片
    if os.path.exists(image_dir):
        images = [img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
        if images:
            # 创建一个下拉框供用户选择图片
            selected_image = st.selectbox("选择要显示的图片", images)

            current_image_path = os.path.join(image_dir, selected_image)

            # 使用st.image来显示图片
            image = Image.open(current_image_path)
            st.image(image, caption=f"项目图片：{selected_image}", use_column_width=True)
        else:
            st.write("暂无项目介绍图片")
    else:
        st.write("暂无项目介绍图片")

elif st.session_state.section == '视频展示':
    st.markdown("<h3 style='text-align: left;'>项目视频展示</h3>", unsafe_allow_html=True)
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
