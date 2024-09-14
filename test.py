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