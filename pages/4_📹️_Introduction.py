import streamlit as st
import os
from PIL import Image

# 图片/视频文件夹地址
image_dir = "data/introduction_src"

# 设置页面标题为黑色，英文副标题为灰色
st.markdown("<h1 style='text-align: center; color: black;'>智能仓储系统介绍</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: gray;'>Intelligent Warehouse System Introduction</h2>",
            unsafe_allow_html=True)

# 自定义按钮的 CSS 样式
st.markdown("""
    <style>
    .stButton > button {
        font-size: 18px;
        padding: 10px 20px;
        background-color: white;
        color: black;
        border: 2px solid #cccccc;  /* 边框默认颜色淡灰色 */
        border-radius: 5px;
        transition-duration: 0.4s;
    }
    .stButton > button:hover {
        background-color: #f2f2f2;
        color: #ff6666;  /* 悬停时字体颜色变浅红 */
        border-color: #ff6666;  /* 悬停时边框颜色变浅红 */
    }
    .stButton.selected > button {
        border-color: #ff6666;  /* 选中时的边框颜色浅红 */
        color: #ff6666;  /* 选中时的字体颜色浅红 */
    }
    div[data-testid="stSidebar"] button[data-baseweb="button"]{
        border: 2px solid #cccccc;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# 使用 session state 控制显示的部分
if 'section' not in st.session_state:
    st.session_state.section = '功能介绍'

# 侧边栏用于切换部分内容
st.sidebar.title("目录")


# 创建一个函数来渲染按钮，并根据选中状态动态设置样式
def render_button(label, section_name):
    if st.session_state.section == section_name:
        button_class = "selected"  # 当前页面对应的按钮应用选中样式
    else:
        button_class = ""

    # 渲染按钮，应用动态样式
    if st.sidebar.button(label, key=section_name):
        st.session_state.section = section_name

    # 用 CSS 选中样式实现红色字体和边框
    st.markdown(f"""
        <style>
        div[data-testid="stSidebar"] button[data-baseweb="button"]{{
            border: 2px solid {'#ff6666' if st.session_state.section == section_name else '#cccccc'};
            color: {'#ff6666' if st.session_state.section == section_name else 'black'};
        }}
        </style>
    """, unsafe_allow_html=True)


# 渲染不同的按钮
render_button("系统功能介绍", "功能介绍")
render_button("项目图片展示", "图片展示")
render_button("项目视频展示", "视频展示")

# 根据 session state 中的值展示不同的部分内容
if st.session_state.section == '功能介绍':
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>系统功能介绍</h3>", unsafe_allow_html=True)

    st.write("""
    ### 智能仓储管理系统简介

    智能仓储系统是一种集成多种先进技术的自动化解决方案，旨在优化钢板的仓储管理过程。通过引入图像识别、自动化入库与出库调度以及堆垛优化算法，该系统能够显著提升仓储操作的效率与精准度。

    #### 主要功能包括：
    1. **钢板图像识别与编码**：
       - 系统利用高精度图像识别技术，能够自动检测和识别钢板的尺寸、材质和编码信息。通过摄像头采集的图像，系统可以快速准确地为每块钢板生成唯一的标识码，替代传统的人工记录方式。

    2. **自动化入库与出库调度**：
       - 智能系统根据钢板的尺寸、重量、材质和交货时间，自动规划钢板的入库与出库路线。自动化吊装设备与传送装置确保钢板的高效搬运，减少了人工干预和操作误差。
       - 系统根据订单需求或出库请求，按批次顺序自动安排钢板的出库，极大提高出库效率。

    3. **入库堆垛优化**：
       - 针对入库钢板的体积、重量和尺寸，系统通过优化算法智能规划堆垛位置，确保仓库空间的最大化利用，减少堆垛翻转的次数，降低能耗和堆垛时间。
       - 系统在堆垛过程中，自动计算每块钢板的最佳位置，使得出库时所需的翻垛最少，并且仓库的整体均衡度得到有效提升。

    #### 亮点功能：
    - **高效性**：通过智能调度和自动化设备，入库与出库过程实现无缝衔接，极大减少了人工操作时间。
    - **精确性**：利用图像识别技术，钢板的识别精度高，数据采集速度快，确保仓储数据的实时更新与精准记录。
    - **智能化优化**：仓储空间的利用率、钢板的翻垛次数、出库效率等均通过先进的算法实现最佳优化，大大提升了仓储管理的智能化水平。

    该系统适用于钢铁行业中的大型仓储中心，尤其适合那些需要高效处理大量钢板进出库操作的场景。通过智能化和自动化的技术，仓储管理流程变得更加高效、精准、可靠。

    ### 目标函数:

    #### 1. 最小化翻垛次数与翻转时间：
    此目标函数减少翻垛次数和翻转时间，通过交货时间差与批次不同增加翻垛次数。
    """)

    st.latex(r'''
    F_1 = \omega_m \cdot N_{\text{movements}} + \omega_t \cdot (T_{\text{turnover}} + N_{\text{batch\_turnover}})
    ''')

    st.markdown("""
    - $\\omega_m$：翻垛权重
    - $\\omega_t$：翻转权重
    - $N_{\\text{movements}}$：翻垛次数
    - $T_{\\text{turnover}}$：交货时间差导致的翻转时间总和
    - $N_{\\text{batch\\_turnover}}$：批次不同导致的翻堆次数
    """)

    st.write("""
    #### 2. 最小化出库能耗与时间：
    该目标函数计算钢板从库区出库所需的移动时间及能耗。
    """)

    st.latex(r'''
    F_2 = \sum_{i=1}^{n} T_{\text{move}}(i)
    ''')

    st.markdown("""
    - $T_{\\text{move}}(i)$：第 $i$ 块钢板的移动时间
    - $n$：钢板总数
    """)

    st.write("""
    #### 3. 最大化库存均衡度：
    通过减少每个库区的体积差异，来实现更均衡的库存分布。
    """)

    st.latex(r'''
    F_3 = \frac{1}{m} \sum_{j=1}^{m} \left( V_j - \bar{V} \right)^2
    ''')

    st.markdown("""
    - $m$：库区数量
    - $V_j$：第 $j$ 个库区的体积占用
    - $\\bar{V}$：所有库区的平均体积占用
    """)

    st.write("""
    #### 4. 最大化空间利用率：
    优化每个库区的空间使用率，减少未使用空间。
    """)

    st.latex(r'''
    F_4 = \sum_{i=1}^{m} \alpha_1 \cdot \frac{\max\left( V_{\max}(i) - V_{\text{used}}(i), \epsilon \right)}{V_{\text{used}}(i)}
    ''')

    st.markdown("""
    - $\\alpha_1$：利用率权重
    - $V_{\\max}(i)$：第 $i$ 个库区的最大容量
    - $V_{\\text{used}}(i)$：第 $i$ 个库区已使用的体积
    - $\\epsilon$：防止除零的小值
    """)

elif st.session_state.section == '图片展示':
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>项目图片展示</h3>", unsafe_allow_html=True)

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
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>项目视频展示</h3>", unsafe_allow_html=True)

    # 查找视频文件
    if os.path.exists(image_dir):
        videos = [vid for vid in os.listdir(image_dir) if vid.endswith(('mp4', 'avi', 'mov', 'mkv'))]
        if videos:
            video_file = os.path.join(image_dir, videos[0])

            # 显示视频
            st.video(video_file)
        else:
            st.write("暂无项目介绍视频")
    else:
        st.write("暂无项目介绍视频")
