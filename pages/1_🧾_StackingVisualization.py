import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

# 检查是否完成了堆垛优化
if 'optimization_done' not in st.session_state or not st.session_state['optimization_done']:
    st.warning("Please complete stacking optimization in the home page before visualizing results.")
else:
    # 可视化逻辑
    st.title("Stack Visualization")

    # 在侧边栏中添加优化算法的选择
    algorithm_choice = st.sidebar.selectbox("Select Optimization Algorithm", ["PSO", "GA", "SA"])

    # 根据选择的算法读取对应的文件
    if algorithm_choice == "PSO":
        data_file_path = r'result/final_stack_distribution/final_stack_distribution_plates_pso.csv'
    elif algorithm_choice == "GA":
        data_file_path = r'result/final_stack_distribution/final_stack_distribution_plates_ga.csv'
    elif algorithm_choice == "SA":
        data_file_path = r'result/final_stack_distribution/final_stack_distribution_plates_sa.csv'

    # 确保文件存在
    if not os.path.exists(data_file_path):
        st.error(
            f"File for {algorithm_choice} optimization not found. Please return to the home page to complete the stacking optimization using the {algorithm_choice} algorithm.")
    else:
        df = pd.read_csv(data_file_path)

        st.write(f"Loaded optimized dataset for {algorithm_choice}:")
        st.write(df.head())

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

        # 提供库区选择的下拉框
        selected_area = st.selectbox("Select Area to visualize", list(area_layouts.keys()))

        # 为选定的库区生成3D图
        area_data = df[df['Final Area'] == selected_area]

        # 生成3D散点图
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=area_data['Final X'],
            y=area_data['Final Y'],
            z=area_data['Stacking Height'],
            mode='markers',
            marker=dict(size=5, color=area_data['Stacking Height'], colorscale='Viridis', opacity=0.8)
        )])

        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Stacking Height'
            ),
            title=f'3D Stack Distribution in Area {selected_area}',
            margin=dict(l=0, r=0, b=0, t=40)
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # 提供图表类型的选择
        chart_type = st.selectbox(f"Select chart type for Area {selected_area}", ["Combo", "Bar", "Line", "Area"])

        # 获取垛位的高度数据
        positions = area_layouts[selected_area]
        height_data = [height_dict[selected_area][pos] for pos in positions]
        position_labels = [f'{pos[0]}_{pos[1]}' for pos in positions]

        # 根据选择的图表类型生成图表
        if chart_type == "Bar":
            fig_bar = go.Figure([go.Bar(x=position_labels, y=height_data, width=0.3)])
            fig_bar.update_layout(
                title=f'Height Distribution in Area {selected_area} (Bar Chart)',
                xaxis_title='Position',
                yaxis_title='Stacking Height'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        elif chart_type == "Line":
            fig_line = go.Figure([go.Scatter(x=position_labels, y=height_data, mode='lines')])
            fig_line.update_layout(
                title=f'Height Distribution in Area {selected_area} (Line Chart)',
                xaxis_title='Position',
                yaxis_title='Stacking Height'
            )
            st.plotly_chart(fig_line, use_container_width=True)

        elif chart_type == "Area":
            fig_area = go.Figure([go.Scatter(x=position_labels, y=height_data, fill='tozeroy')])
            fig_area.update_layout(
                title=f'Height Distribution in Area {selected_area} (Area Chart)',
                xaxis_title='Position',
                yaxis_title='Stacking Height'
            )
            st.plotly_chart(fig_area, use_container_width=True)

        else:  # Combo 图，包含柱状图和线条图
            fig_combo = go.Figure()
            fig_combo.add_trace(go.Bar(x=position_labels, y=height_data, width=0.3, name='Bar'))
            fig_combo.add_trace(go.Scatter(x=position_labels, y=height_data, mode='lines+markers', name='Line'))
            fig_combo.update_layout(
                title=f'Height Distribution in Area {selected_area} (Combo Chart)',
                xaxis_title='Position',
                yaxis_title='Stacking Height'
            )
            st.plotly_chart(fig_combo, use_container_width=True)

        st.success(f"3D plot and height distribution for Area {selected_area} generated and displayed successfully.")

        # 生成包含所有库区堆垛高度分布的总图
        st.subheader("Final Stack Distribution by Area")
        all_height_data = []
        all_positions = []

        # 遍历每个库区
        for area, positions in area_layouts.items():
            for pos in positions:
                height = height_dict[area][pos]  # 取最后的 `Stacking Height`
                all_height_data.append(height)
                all_positions.append(f'Area {area} - {pos[0]}_{pos[1]}')

        # 选择总图的图表类型
        total_chart_type = st.selectbox("Select total chart type", ["Bar", "Line", "Area", "Combo"])

        # 绘制总图
        if total_chart_type == "Bar":
            fig_total = go.Figure([go.Bar(x=all_positions, y=all_height_data, width=0.3)])
        elif total_chart_type == "Line":
            fig_total = go.Figure([go.Scatter(x=all_positions, y=all_height_data, mode='lines')])
        elif total_chart_type == "Area":
            fig_total = go.Figure([go.Scatter(x=all_positions, y=all_height_data, fill='tozeroy')])
        else:  # Combo 图
            fig_total = go.Figure()
            fig_total.add_trace(go.Bar(x=all_positions, y=all_height_data, width=0.3, name='Bar'))
            fig_total.add_trace(go.Scatter(x=all_positions, y=all_height_data, mode='lines+markers', name='Line'))

        fig_total.update_layout(
            title=f'Final Stack Distribution by Area ({total_chart_type} Chart)',
            xaxis_title='Stack Position',
            yaxis_title='Stacking Height',
            xaxis=dict(tickangle=-45)
        )

        st.plotly_chart(fig_total, use_container_width=True)
