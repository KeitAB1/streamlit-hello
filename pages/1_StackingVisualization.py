import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 检查是否完成了堆垛优化
if 'optimization_done' not in st.session_state or not st.session_state['optimization_done']:
    st.warning("Please complete stacking optimization in the home page before visualizing results.")
else:
    # 可视化逻辑
    st.title("Stack Visualization")

    # 读取优化后的数据
    data_file_path = r'result/final_stack_distribution/final_stack_distribution_plates.csv'

    # 确保文件存在
    if not os.path.exists(data_file_path):
        st.error(f"File {data_file_path} not found. Please make sure the file exists.")
    else:
        df = pd.read_csv(data_file_path)

        st.write("Loaded optimized dataset:")
        st.write(df.head())

        # 创建一个用于保存图像的文件夹
        output_dir = r'result/stack_distribution_plots'
        os.makedirs(output_dir, exist_ok=True)

        # 示例库区布局
        area_layouts = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 库区 1
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 库区 2
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 库区 3
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 库区 4
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 库区 5
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 库区 6
        }

        # 初始化字典用于保存每个库区垛位的最后 `Stacking Height`
        height_dict = {area: {pos: 0.0 for pos in positions} for area, positions in area_layouts.items()}

        # 遍历每个钢板，更新垛位的堆垛高度（用最后一个钢板的 Stacking Height 更新垛位的 Height）
        for _, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']
            height_dict[area][(x, y)] = stacking_height

        # 为每个库区生成一个3D图和右侧柱状图
        for area in range(6):
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 设置整体图像大小，并放置3D图与柱状图

            # 左侧 3D 图
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

            # 右侧柱状图
            bar_width = 0.2  # 柱状图更窄一些
            positions = area_layouts[area]
            height_data = [height_dict[area][pos] for pos in positions]
            ax2.bar(range(1, len(height_data) + 1), height_data, width=bar_width, color='skyblue')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Stacking Height')
            ax2.set_title(f'Height Distribution in Area {area}')
            ax2.set_xticks(range(1, len(height_data) + 1))
            ax2.set_xticklabels([f'{pos[0]}_{pos[1]}' for pos in positions])

            # 调整3D图与柱状图的距离
            fig.subplots_adjust(wspace=0.5)  # 调整图像之间的距离

            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'stack_distribution_area_{area}.png'))
            st.pyplot(fig)  # 在 Streamlit 中显示图像
            plt.close()

        st.success("3D plots and height distributions generated and displayed successfully.")

        # 生成包含所有库区堆垛高度分布的总图
        fig, ax = plt.subplots(figsize=(12, 6))

        # 将所有库区的垛位高度数据放在一起
        all_height_data = []
        all_positions = []

        # 遍历每个库区
        for area, positions in area_layouts.items():
            for pos in positions:
                height = height_dict[area][pos]  # 取最后的 `Stacking Height`
                all_height_data.append(height)
                all_positions.append(f'Area {area} - {pos[0]}_{pos[1]}')

        # 绘制总图
        ax.bar(range(len(all_height_data)), all_height_data, width=0.5, color='skyblue')

        # 设置图表标签
        ax.set_xlabel('Stack Position')
        ax.set_ylabel('Stacking Height')
        ax.set_title('Final Stack Distribution by Area')

        # 设置 x 轴刻度和标签
        ax.set_xticks(range(len(all_positions)))
        ax.set_xticklabels(all_positions, rotation=90)

        # 保存总图
        output_image_path = os.path.join(output_dir, 'final_stack_distribution_all_areas.png')
        plt.tight_layout()
        plt.savefig(output_image_path)
        plt.close()

        st.success(f"Final stack distribution image saved to {output_image_path}")
        st.image(output_image_path)
