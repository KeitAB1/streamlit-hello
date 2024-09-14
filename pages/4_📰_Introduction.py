import streamlit as st
import os
from PIL import Image

# 图片/视频文件夹地址
video_dir= "data/introduction_src/videos"
image_dir1 = "data/introduction_src/images01"
image_dir2 = "data/introduction_src/images02"

# 设置页面标题为黑色，英文副标题为灰色
st.markdown("<h1 style='text-align: center; color: black;'>智能仓储系统介绍</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: gray;'>Intelligent Warehouse System Introduction</h2>", unsafe_allow_html=True)

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
render_button("参考文献说明", "参考文献")

# 根据 session state 中的值展示不同的部分内容
if st.session_state.section == '功能介绍':
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>系统功能介绍</h3>", unsafe_allow_html=True)

    st.write("""
    ### 智能仓储管理系统简介

    智能仓储系统结合了多种技术，旨在帮助优化钢板仓储管理流程。该系统整合了图像识别、自动化入库与出库调度以及堆垛优化算法，力求提高仓储操作的效率和准确性。

    #### 主要功能包括：
    1. **钢板图像识别与编码**：
       - 系统利用图像识别技术，能够检测并识别钢板的尺寸、材质和编码信息。通过摄像头采集的图像，系统能够为每块钢板生成唯一的标识码，减少了人工记录带来的错误。

    2. **自动化入库与出库调度**：
       - 系统可以根据钢板的尺寸、重量、材质和交货时间，自动执行钢板的入库与出库操作。自动化设备配合传送装置可以减少部分人工操作，提高工作效率。
       - 系统还可以根据订单需求或出库请求自动进行调度，按批次顺序安排钢板出库。

    3. **入库堆垛优化**：
       - 系统根据钢板的体积、重量和尺寸，采用优化算法规划堆垛位置，尽量优化仓库空间利用率，同时减少堆垛翻转次数，从而降低能耗和堆垛时间。
       - 在堆垛过程中，系统会计算每块钢板的合理位置，以减少出库时的翻垛次数，并努力保持仓库内整体的平衡。

    #### 亮点功能：
    - **效率提升**：智能调度和自动化设备有助于减少部分人工操作时间，并提高入库与出库过程的连续性。
    - **数据精度**：通过图像识别，钢板的识别精度较高，数据采集相对快速，仓储数据得以更加及时更新。
    - **智能优化**：在算法的支持下，仓储空间利用率、翻垛次数以及出库效率得到了优化，提高了管理效率。

    该系统可以适应钢铁行业的大型仓储中心，特别是需要处理大量钢板进出库的场景。系统通过智能化与自动化的方式，有助于提升仓储管理的效率、准确性和可靠性。

    ### 目标函数:

    #### 1. 最小化翻垛次数与倒垛量：
    该目标函数减少翻垛次数和倒垛量，提升仓储管理效率。
    """)

    st.latex(r'''
    F_1 = \omega_m \cdot M + \omega_t \cdot T
    ''')

    st.markdown("""
    - $M$：翻垛次数，表示堆垛过程中需要翻转的钢板数量。
    - $T$：倒垛量，表示堆位中钢板的交货时间差的总和。
    - $\omega_m$ 和 $\omega_t$：控制翻垛次数和倒垛量的权重参数。
    """)

    # 修改标题字号
    st.markdown("<h4 style='text-align: left;'>翻垛次数公式：</h4>", unsafe_allow_html=True)

    st.latex(r'''
    M = \sum_{i=1}^{N} \mathbb{I}(h_i < h_{\text{current}})
    ''')

    st.markdown("""
    - $N$：钢板的数量。
    - $h_i$：当前钢板的厚度。
    - $h_{\text{current}}$：当前堆位的总高度。
    - $\mathbb{I}(h_i < h_{\text{current}})$：一个指示函数，当钢板 $i$ 被放置在当前堆位下方时，它的值为 1，否则为 0。
    """)

    st.markdown("<h4 style='text-align: left;'>倒垛量公式：</h4>", unsafe_allow_html=True)

    st.latex(r'''
    T = \sum_{i=1}^{N} \sum_{j=i+1}^{N} |d_i - d_j|
    ''')

    st.markdown("""
    - $d_i$ 和 $d_j$：第 $i$ 和第 $j$ 块钢板的交货时间差。
    - $N$：钢板的总数。
    """)

    #### 添加目标函数2的公式
    st.markdown("<h4 style='text-align: left;'>2. 最小化出库能耗与时间：</h4>", unsafe_allow_html=True)

    st.write("""
    该目标函数计算钢板从仓库入库和出库时的能耗与时间。
    """)

    st.latex(r'''
    F_2 = \sum_{b \in B}\sum_{i=1}^{N_b}(t_{\text{move}}(i) + t_{\text{pick}}(i) + t_{\text{flip}}(i))
    ''')

    st.markdown("""
    - $B$：钢板批次的集合。
    - $N_b$：属于批次 $b$ 的钢板数量。
    """)

    st.markdown("<h4 style='text-align: left;'>机械臂移动时间公式：</h4>", unsafe_allow_html=True)

    st.latex(r'''
    t_{\text{move}}(i) = \frac{D_{h,in}(x_i, y_i, x_{\text{in}}, y_{\text{in}})}{v_h} + \frac{D_{v,in}(x_i, y_i, x_{\text{in}}, y_{\text{in}})}{v_v} + \frac{D_{h,out}(x_i, y_i, x_{\text{out}}, y_{\text{out}})}{v_h} + \frac{D_{v,out}(x_i, y_i, x_{\text{out}}, y_{\text{out}})}{v_v}
    ''')

    st.markdown("""
    - $D_{h,in}(x_i, y_i, x_{\text{in}}, y_{\text{in}})$：钢板从入库点到堆垛位置的水平距离。
    - $D_{v,in}(x_i, y_i, x_{\text{in}}, y_{\text{in}})$：钢板从入库点到堆垛位置的垂直距离。
    - $D_{h,out}(x_i, y_i, x_{\text{out}}, y_{\text{out}})$：钢板从堆垛位置到出库点的水平距离。
    - $D_{v,out}(x_i, y_i, x_{\text{out}}, y_{\text{out}})$：钢板从堆垛位置到出库点的垂直距离。
    - $v_h$ 和 $v_v$：电磁吊的水平和垂直移动速度。
    """)

    st.markdown("<h4 style='text-align: left;'>取出钢板时间公式：</h4>", unsafe_allow_html=True)

    st.latex(r'''
    t_{\text{pick}}(i) = \frac{h_i}{v_v}
    ''')

    st.markdown("""
    - $h_i$ 是堆垛的高度。
    - $v_v$ 是垂直移动速度。
    """)

    st.markdown("<h4 style='text-align: left;'>翻垛时间公式：</h4>", unsafe_allow_html=True)

    st.latex(r'''
    t_{\text{flip}}(i) = n_{\text{flip}} \cdot t_{\text{flip,per}}
    ''')

    st.markdown("""
    - $n_{\\text{flip}}$ 是需要翻动的钢板数量。
    - $t_{\\text{flip,per}}$ 是每次翻动的时间。
    """)

    # 目标函数3标题
    st.markdown("<h3 style='text-align: left;'>3. 最大化库存均衡度：</h3>", unsafe_allow_html=True)

    st.write("""
    该目标函数通过均衡堆垛体积分布，提升库存管理的均衡度。
    """)

    st.latex(r'''
    F_3 = -\frac{1}{m} \sum_{j=1}^{m} \left( \frac{\sum_{i=1}^{n} \alpha_{ij} \cdot V_{ij}}{V_{\text{total}}/m} - 1 \right)^2
    ''')

    st.markdown("""
    - $m$：堆垛位置的总数。
    - $n$：钢板的总数。
    - $\\alpha_{ij}$：一个指示变量，表示钢板 $i$ 是否放置在堆位 $j$：
    """)

    st.latex(r'''
    \alpha_{ij} = 
    \begin{cases}
    1, & \text{如果钢板 $i$ 放置在堆位 $j$} \\
    0, & \text{否则}
    \end{cases}
    ''')

    st.markdown("""
    - $V_{ij}$：钢板 $i$ 的体积。
    - $V_{\\text{total}}$：所有钢板的总体积。
    - $\\frac{V_{\\text{total}}}{m}$：每个堆位的平均体积。

    该公式的目标是最大化库存的均衡度，确保所有堆位的体积分布均匀，避免某些堆位过载或其他堆位过空。
    """)

    # 目标函数4标题
    st.markdown("<h3 style='text-align: left;'>4. 最大化空间利用率：</h3>", unsafe_allow_html=True)

    st.write("""
    优化每个堆位的空间利用率，确保堆位的合理利用。
    """)

    st.latex(r'''
    F_4 = \alpha_1 \sum_{k=1}^{3} \sum_{i=1}^{M} \sum_{j=1}^{N_p} X_{ij} \cdot \frac{\max(D_{ki} - S_{kj}, \epsilon)}{\sum_{j=1}^{N_p} X_{ij}}
    ''')

    st.markdown("""
    - $\eta$：空间利用率。
    - $\\alpha_1$：控制堆垛利用率的重要性权重。
    - $X_{ij}$：一个决策变量，表示钢板 $j$ 是否放置在堆位 $i$：
    """)

    st.latex(r'''
    X_{ij} = 
    \begin{cases}
    1, & \text{如果钢板 $j$ 放置在堆位 $i$} \\
    0, & \text{否则}
    \end{cases}
    ''')

    st.markdown("""
    - $D_{ki}$：堆位 $i$ 的最大可用空间（长度、宽度和高度）。
    - $S_{kj}$：钢板 $j$ 的尺寸（长度、宽度和厚度）。
    - $\\epsilon$：一个极小值，用于避免分母为零。

    这个公式的目标是最大化空间利用率，即尽可能地利用堆位的可用空间，确保钢板尽量紧密堆叠在可用的堆垛空间中。
    """)



elif st.session_state.section == '图片展示':
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>项目图片展示</h3>", unsafe_allow_html=True)
    # 检查是否存在图片
    if os.path.exists(image_dir1):
        images = [img for img in os.listdir(image_dir1) if img.endswith(('png', 'jpg', 'jpeg'))]
        if images:
            # 创建一个下拉框供用户选择图片
            selected_image = st.selectbox("选择要显示的图片", images)

            current_image_path = os.path.join(image_dir1, selected_image)

            # 使用st.image来显示图片
            image = Image.open(current_image_path)
            st.image(image, caption=f"项目图片：{selected_image}", use_column_width=True)
        else:
            st.write("暂无项目介绍图片")
    else:
        st.write("暂无项目介绍图片")

    st.markdown("<h3 style='text-align: left; font-weight: bold;'>训练图片展示</h3>", unsafe_allow_html=True)

    # 检查是否存在图片
    if os.path.exists(image_dir2):
        images = [img for img in os.listdir(image_dir2) if img.endswith(('png', 'jpg', 'jpeg'))]
        if images:
            # 创建一个下拉框供用户选择图片
            selected_image = st.selectbox("选择要显示的图片", images)

            current_image_path = os.path.join(image_dir2, selected_image)

            # 使用st.image来显示图片
            image = Image.open(current_image_path)
            st.image(image, caption=f"训练图片：{selected_image}", use_column_width=True)
        else:
            st.write("暂无项目训练图片")
    else:
        st.write("暂无项目训练图片")

elif st.session_state.section == '视频展示':
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>项目视频展示</h3>", unsafe_allow_html=True)

    # 查找视频文件
    if os.path.exists(video_dir):
        videos = [vid for vid in os.listdir(video_dir) if vid.endswith(('mp4', 'avi', 'mov', 'mkv'))]
        if videos:
            video_file = os.path.join(video_dir, videos[0])

            # 显示视频
            st.video(video_file)
        else:
            st.write("暂无项目介绍视频")
    else:
        st.write("暂无项目介绍视频")

elif st.session_state.section == '参考文献':
    st.markdown("<h3 style='text-align: left; font-weight: bold;'>参考文献说明</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left; font-weight: bold;'>参考文献如下：</h4>", unsafe_allow_html=True)

    st.markdown("""
    [[1]钟传捷,程文明,杜润,等.基于改进多目标粒子群算法的钢板入库垛位分配研究[J/OL].工程科学与技术,1-18[2024-09-14].](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkm9xhOSzGzSjjngv6yYWb0reabBoZNbLH0zgHFXzYLJdkl8yFNDD_y1FhtzWmvHl8T3SxrLYFz3KlqN3Nv8VHDhfD9Ddf6_zbdBSLb_STaT7QGnOxqBS4PaKPBRr8dagdFT_zjglrLlWl0usm3SpnzujKbYfFggrFs=&uniplatform=NZKPT)
    
    [[2]张琦琪,张涛,刘鹏.精英改进粒子群算法在入库堆垛问题中的应用[J].计算机工程与科学,2015,37(07):1311-1317.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkm9xhOSzGzSjjngv6yYWb0reabBoZNbLH0zgHFXzYLJdkl8yFNDD_y1FhtzWmvHl8T3SxrLYFz3KlqN3Nv8VHDhfD9Ddf6_zbdBSLb_STaT7QGnOxqBS4PaKPBRr8dagdFT_zjglrLlWl0usm3SpnzujKbYfFggrFs=&uniplatform=NZKPT)
    
    [[3]恒正琦.A企业宽厚板厂精整区域中钢板堆垛方法的优化研究[D].上海交通大学,2018.DOI:10.27307/d.cnki.gsjtu.2018.002280.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkkPiSMHrm54E4-Y4Q2XLvAoQDRVZpnqghvP7innHAD5RU4gdAA_jKWPmIC49mBb-YVqlyWiNf7RoewZ8usx3VXWrF2BL1yf-6m5T-PjrkUQi4h2PmIN3gkf8Gh6GdH1O6llihwpok2vMyYu2NK04wsLe3T5q5Nomls=&uniplatform=NZKPT)
    
    [[4]侯俊,张志英.船厂钢板堆场混合存储分配及出入库调度研究[J].哈尔滨工程大学学报,2017,38(11):1786-1793.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkkPjlq5X6c0flOWIkvpUNmTJzh7Quija6asYJeV0ki7VK9j1y3YnyBdje1gft3osu5r3hseJaUGLAUfIb3AB2wCo-UXTInqAYXZjsZyoZ0les5m3hxbvfjsuzYC8vdfC4iKxEP5TXl8m_yfjEcznUp0QN_bUElwe9k=&uniplatform=NZKPT)
    
    [[5]李祥,顾晓波,王炬成,等.基于工业云的钢板堆场数字化管理系统设计与实现[J].造船技术,2023,51(01):80-87.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkny_WOe87edEb4rBNLDPfv195D1BSNmOLEg3W-0wS_ASJd1-xWN-oi3rmFBFPS6_qVT3OYU5MhvosINxvsR_ah5MSnC4ov1hrXs5hgKH8pKJebPYDe7_ABExncoj_p4AJ5rkLM1RK71WGaz3fU3xVZm_fKspBYHJ7g=&uniplatform=NZKPT)
    
    [[6]李祥.基于工业云的钢板堆场数字化管理系统研究[D].江苏科技大学,2022.DOI:10.27171/d.cnki.ghdcc.2022.000095.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkmMqgflDEGfX_A5Yjx4Q-xiYLKn013_N-_fVaL4XLk2ivGt49vFxzo7xqvlKpEm_6SGuThTg1VDCzvgM4kQcxxKn-rUE2a9LTN3l8oA6xcUute_Buq7fX1SzvBdfrUPBLVsonxkLPAmP159R3YT5VIJyhJh50gPF8Q=&uniplatform=NZKPT)
   
    [[7]廖尧.造船厂钢板库布局优化及管理信息系统开发[D].大连理工大学,2010.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkkG2VgDMUEFhb0I-JciM9nqjO5TqhqsEysQhT_aKEjo0mf8PJIiF9NiaEC_rMMA09QrgzJltW7sxOWCr_TIcOGzJnJjfqyYOHNXY0ThG6mh9XUJx6ACIluTWfeXjo4dUYW7crvOSe5FB1fzo--vwncUjxZ19qpe3WE=&uniplatform=NZKPT)
    
    [[8]徐萍.造船厂钢板入库作业优化及钢板管理信息系统开发[D].江苏科技大学,2011.](https://kns.cnki.net/kcms2/article/abstract?v=7gnxONS3vkktDSJZf5E3EIJS0GJ5FguzGyw2fLeCZQ2I4Lup2mHmGPMl9MYeNjzRIXrk48KVSLwImUVIfU7nVoP6YLxeXZC5Ujvh2ennYGP9nssZ6MODWz8CZaYmcgB8oDb42VJNdGMI_2h8q0u7Rx6J3-nwEtlR4GRIaR0pT7E=&uniplatform=NZKPT)
    
    [[9]Wang, D., Tang, O. and Zhang, L. (2023) ‘Inventory stacking with partial information’, International Journal of Production Research, 62(1–2), pp. 586–604. doi: 10.1080/00207543.2023.2219768.](https://doi.org/10.1080/00207543.2023.2219768)
    
    [[10]Tang, L. et al. (2015) ‘Research into container reshuffling and stacking problems in container terminal yards’, IIE Transactions, 47(7), pp. 751–766. doi: 10.1080/0740817X.2014.971201.](https://doi.org/10.1080/0740817X.2014.971201)
    """)

