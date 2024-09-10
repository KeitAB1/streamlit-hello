import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
output_dir = "stack_distribution_plots"
os.makedirs(output_dir, exist_ok=True)

st.title("Steel Plate Stacking Optimization")

# 文件上传功能
uploaded_file = st.file_uploader("Upload your steel plate dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # 读取上传的文件
    df = pd.read_csv(uploaded_file)

    # 显示上传的数据集
    st.write("Uploaded dataset:")
    st.write(df.head())

    # 参数配置（这里假设用户上传了一个与原先结构一致的csv文件）
    plates = df[['Length', 'Width', 'Thickness', 'Material_Code', 'Batch', 'Entry Time', 'Delivery Time']].values
    plate_areas = plates[:, 0] * plates[:, 1]
    num_plates = len(plates)
    batches = df['Batch'].values

    # 库区布局和尺寸
    area_positions = {
        0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        3: [(0, 0), (0, 1), (1, 0), (1, 1)],
        4: [(0, 0), (0, 1), (1, 0), (1, 1)],
        5: [(0, 0), (0, 1), (1, 0), (1, 1)]
    }

    # 库区垛位尺寸
    stack_dimensions = {
        0: [(6000, 3000), (9000, 3000), (9000, 3000), (6000, 3000), (9000, 3000), (9000, 3000), (9000, 4000),
            (15000, 4000)],
        1: [(6000, 3000), (9000, 3000), (9000, 3000), (6000, 3000), (9000, 3000), (9000, 3000), (15000, 4000),
            (9000, 4000)],
        2: [(12000, 3000), (12000, 3000), (12000, 3000), (12000, 3000), (12000, 4000), (12000, 4000)],
        3: [(9000, 5000), (15000, 5000), (9000, 5000), (15000, 5000)],
        4: [(18000, 5000), (6000, 5000), (18000, 5000), (6000, 5000)],
        5: [(12000, 5000), (12000, 5000), (12000, 5000), (12000, 5000)]
    }

    # 执行堆垛优化的代码
    # （简化版，用于示例，假设优化结果直接生成）
    # 初始化每个库区的高度（可以从PSO中得到最终结果）
    heights = np.random.rand(len(area_positions)) * 3000  # 示例数据

    # 保存最终的堆垛结果
    final_stack_distribution_path = os.path.join(output_dir, "final_stack_distribution_plates.csv")
    df.to_csv(final_stack_distribution_path, index=False)

    st.success(f"Optimization complete. Results saved to {final_stack_distribution_path}")

    # 绘制结果的3D图和高度分布条形图
    for area in range(6):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')

        area_data = df[df['Final Area'] == area]
        x = area_data['Final X']
        y = area_data['Final Y']
        z = area_data['Stacking Height']

        ax.scatter(x, y, z, c='r', marker='o')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Stacking Height')
        ax.set_title(f'3D Stack Distribution in Area {area}')

        ax2 = fig.add_subplot(122)
        height_data = heights[area]
        ax2.bar(range(len(height_data)), height_data, color='skyblue')
        ax2.set_title(f'Height Distribution in Area {area}')
        plt.tight_layout()

        # 在Streamlit中显示图像
        st.pyplot(fig)

    st.write("3D plots and height distributions generated and displayed.")
