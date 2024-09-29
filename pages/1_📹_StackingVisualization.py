import streamlit as st
import pandas as pd
import os
import plotly.graph_objs as go  # 确保导入 plotly.graph_objs 并定义 go



# Streamlit 页面配置
st.set_page_config(page_title="Stacking Visualization", page_icon="📹")

# 页面标题
st.title("Stack Visualization")

# 选择使用系统固定的数据还是训练后的收敛曲线数据
data_source_choice = st.sidebar.radio(
    "Choose data source",
    ("Use training data", "Use fixed system data")
)

# 定义数据集文件夹路径
data_dir = "data"  # 你的数据集存放的目录

# 获取可用的 CSV 数据集，并移除 .csv 后缀
available_datasets = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.csv')]

# 在侧边栏中显示数据集选择框
selected_dataset = st.sidebar.selectbox("Select Dataset", available_datasets)

# 显示选定的数据集的名称
st.write(f"Selected dataset: {selected_dataset}")

# 然后在侧边栏中选择优化算法
algorithm_choice = st.sidebar.selectbox("Select Optimization Algorithm", [
    "SA (Simulated Annealing)",
    "GA (Genetic Algorithm)",
    "PSO (Particle Swarm Optimization)",
    "PSO + SA (Hybrid Optimization)",
    "ACO (Ant Colony Optimization)",
    "DE (Differential Evolution)",
    "CoEA (Co-Evolutionary Algorithm)",
    "EDA (Estimation of Distribution Algorithm)",
    "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)",
    "NSGA-II (Non-dominated Sorting Genetic Algorithm II)"
])

# 根据用户的选择决定路径
if data_source_choice == "Use fixed system data":
    output_dir_base = f'result/Fix_final_stack_distribution/{selected_dataset}/final_stack_distribution_plates'
else:
    output_dir_base = f'result/final_stack_distribution/{selected_dataset}/final_stack_distribution_plates'

# 文件映射
file_mapping = {
    "PSO (Particle Swarm Optimization)": 'final_stack_distribution_plates_pso.csv',
    "GA (Genetic Algorithm)": 'final_stack_distribution_plates_ga.csv',
    "SA (Simulated Annealing)": 'final_stack_distribution_plates_sa.csv',
    "PSO + SA (Hybrid Optimization)": 'final_stack_distribution_plates_psosa.csv',
    "ACO (Ant Colony Optimization)": 'final_stack_distribution_plates_aco.csv',
    "DE (Differential Evolution)": 'final_stack_distribution_plates_de.csv',
    "CoEA (Co-Evolutionary Algorithm)": 'final_stack_distribution_plates_coea.csv',
    "EDA (Estimation of Distribution Algorithm)": 'final_stack_distribution_plates_eda.csv',
    "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)": 'final_stack_distribution_plates_moead.csv',
    "NSGA-II (Non-dominated Sorting Genetic Algorithm II)": 'final_stack_distribution_plates_nsga2.csv'
}

# 根据算法选择生成的文件名
data_file_path = os.path.join(output_dir_base, file_mapping[algorithm_choice])

# 确保文件存在
if not os.path.exists(data_file_path):
    if data_source_choice == "Use training convergence data":
        st.error(
            f"File for {algorithm_choice} optimization not found in the selected dataset ({selected_dataset}). "
            f"Please return to the home page to complete the stacking optimization using the {algorithm_choice} algorithm."
        )
    else:
        st.error(
            f"Fixed system data for {algorithm_choice} optimization not found in the selected dataset ({selected_dataset})."
        )
    # 直接返回，阻止后续代码执行
    st.stop()

# 如果文件存在，读取并展示结果
try:
    df = pd.read_csv(data_file_path)
    st.write(f"Loaded optimized dataset for {algorithm_choice}:")
    st.write(df.head())
except Exception as e:
    st.error(f"Error loading the file: {e}")
    # 直接返回，阻止后续代码执行
    st.stop()





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
chart_type = st.selectbox(f"Select chart type for Area {selected_area}", ["Combo", "Bar", "Line", "Area", "Scatter", "Pie"])

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

elif chart_type == "Scatter":
    fig_scatter = go.Figure([go.Scatter(x=position_labels, y=height_data, mode='markers')])
    fig_scatter.update_layout(
        title=f'Height Distribution in Area {selected_area} (Scatter Plot)',
        xaxis_title='Position',
        yaxis_title='Stacking Height'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif chart_type == "Pie":
    fig_pie = go.Figure([go.Pie(labels=position_labels, values=height_data)])
    fig_pie.update_layout(
        title=f'Stacking Height Distribution in Area {selected_area} (Pie Chart)'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

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
total_chart_type = st.selectbox("Select total chart type", ["Bar", "Line", "Area", "Combo", "Scatter", "Pie"])

# 绘制总图
if total_chart_type == "Bar":
    fig_total = go.Figure([go.Bar(x=all_positions, y=all_height_data, width=0.3)])
elif total_chart_type == "Line":
    fig_total = go.Figure([go.Scatter(x=all_positions, y=all_height_data, mode='lines')])
elif total_chart_type == "Area":
    fig_total = go.Figure([go.Scatter(x=all_positions, y=all_height_data, fill='tozeroy')])
elif total_chart_type == "Scatter":
    fig_total = go.Figure([go.Scatter(x=all_positions, y=all_height_data, mode='markers')])
elif total_chart_type == "Pie":
    fig_total = go.Figure([go.Pie(labels=all_positions, values=all_height_data)])
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


st.subheader("Download Steel Plate Statistics")


# 提供表格下载功能
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# 创建下载按钮
csv = convert_df_to_csv(df)
st.download_button(
    label="Download Stacking of steel plates data as CSV",
    data=csv,
    file_name=f'{algorithm_choice}_stack_distribution.csv',
    mime='text/csv',
)

# 确保下载按钮出现在表格的下方
st.write(df.head())
