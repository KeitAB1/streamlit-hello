import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from utils import calculate_statistics  # 引入统计函数

# Streamlit 页面配置
st.set_page_config(page_title="AlgorithmsComparison.py", page_icon="🔗")

# 设置训练数据和固定数据的文件夹路径
convergence_data_dir = "result/ConvergenceData"
fixed_convergence_data_dir = "result/FixConvergenceData"
performance_data_dir = "result/comparison_performance"
fixed_performance_data_dir = "result/fixed_comparison_performance"
history_data_dir = "result/History_ConvergenceData"  # 新增历史数据路径

# 用户界面 - 左侧栏选择数据源
st.sidebar.title("Convergence Curve and Performance Comparison")

data_source = st.sidebar.radio("Select Data Source", ["Trained Data", "Fixed Data", "History Data"])  # 增加History Data选项

# 根据选择的数据源，设置数据路径
if data_source == "Trained Data":
    available_datasets_dir = convergence_data_dir
    performance_data_dir = "result/comparison_performance"
elif data_source == "Fixed Data":
    available_datasets_dir = fixed_convergence_data_dir
    performance_data_dir = "result/fixed_comparison_performance"
else:
    available_datasets_dir = history_data_dir  # 历史数据路径

# 获取不同数据集的文件夹
available_datasets = [d for d in os.listdir(available_datasets_dir) if
                      os.path.isdir(os.path.join(available_datasets_dir, d))]

# 确保有数据集可选
if not available_datasets:
    st.error("No datasets found. Please check your data directories.")
else:
    # 用户界面 - 左侧栏选择数据集
    selected_dataset = st.sidebar.selectbox("Select Dataset", available_datasets)

    # 如果选择历史数据，列出可用算法历史文件
    if data_source == "History Data":
        # 获取所选数据集下的所有算法的历史 CSV 文件
        dataset_dir = os.path.join(history_data_dir, selected_dataset)
        available_algorithms = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

        # 默认选择 "SA" 算法，如果存在
        default_algorithm = "SA" if "SA" in available_algorithms else available_algorithms[0]
        selected_algorithm = st.sidebar.selectbox("Select Algorithm", available_algorithms, index=available_algorithms.index(default_algorithm))

        algorithm_history_dir = os.path.join(dataset_dir, selected_algorithm)
        available_history_files = [f for f in os.listdir(algorithm_history_dir) if f.endswith('.csv')]

        if not available_history_files:
            st.error("No history data found for this algorithm.")
        else:
            # 默认选择第一个历史数据文件
            default_files = [available_history_files[0]] if available_history_files else []

            # 用户界面 - 左侧栏多选历史数据文件
            selected_history_files = st.sidebar.multiselect("Select History Data Files", available_history_files,
                                                            default=default_files)

            if selected_history_files:
                st.subheader(f"Convergence History: {selected_algorithm}")

                # 读取每个历史文件并获取最大轮数
                max_iters_in_files = []
                for history_file in selected_history_files:
                    history_file_path = os.path.join(algorithm_history_dir, history_file)
                    try:
                        history_df = pd.read_csv(history_file_path)
                        max_iters_in_files.append(history_df['Iteration'].max())
                    except Exception as e:
                        st.error(f"Error loading history data from {history_file}: {e}")

                # 如果成功读取了历史数据，添加滑动条来选择显示的最大轮数
                if max_iters_in_files:
                    max_iterations = min(max_iters_in_files)  # 选择最小的最大轮数，以避免越界
                    max_iter_display = st.sidebar.slider("Max Iterations to Display", min_value=1, max_value=max_iterations, value=10)

                    fig = go.Figure()

                    # 提取文件名中的日期和时间戳作为图例
                    def extract_timestamp_from_filename(file_name):
                        parts = file_name.split('_')
                        if len(parts) >= 5:
                            date_part = parts[-2]  # YYYYMMDD
                            time_part = parts[-1].replace('.csv', '')  # HHMMSS
                            return f"{date_part}_{time_part}"
                        return file_name

                    # 读取每个历史文件并绘制收敛曲线
                    all_history_data = []
                    for history_file in selected_history_files:
                        history_file_path = os.path.join(algorithm_history_dir, history_file)

                        try:
                            history_df = pd.read_csv(history_file_path)
                            # 只显示用户选择的最大轮数数据
                            truncated_df = history_df[history_df['Iteration'] <= max_iter_display]
                            all_history_data.append(truncated_df['Best Score'].values)

                            # 提取文件名末尾的时间戳作为图例
                            timestamp = extract_timestamp_from_filename(history_file)

                            # 绘制每个历史文件的收敛曲线，使用时间戳作为图例名称
                            fig.add_trace(go.Scatter(x=truncated_df['Iteration'], y=truncated_df['Best Score'],
                                                     mode='lines+markers', name=f"{timestamp}"))

                        except Exception as e:
                            st.error(f"Error loading history data from {history_file}: {e}")

                    # 配置图表样式
                    fig.update_layout(
                        title=f"Convergence Curves - {selected_algorithm}",
                        xaxis_title="Iterations",
                        yaxis_title="Best Score"
                    )
                    st.plotly_chart(fig)

                    # 调用函数计算统计信息
                    if all_history_data:
                        statistics_df = calculate_statistics(all_history_data, selected_history_files)

                        # 显示统计信息表格
                        st.subheader("Statistics Comparison")
                        st.dataframe(statistics_df)

    else:
        # 获取所选数据集下的所有 CSV 文件
        dataset_dir = os.path.join(available_datasets_dir, selected_dataset)
        available_algorithms = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]


        # 从文件名中提取算法名称
        def extract_algorithm_name(file_name):
            base_name = os.path.basename(file_name)
            if "psosa" in base_name:
                return "PSO_SA"
            elif "ga" in base_name:
                return "GA"
            elif "pso" in base_name:
                return "PSO"
            elif "sa" in base_name:
                return "SA"
            elif "aco" in base_name:
                return "ACO"
            elif "de" in base_name:
                return "DE"
            elif "coea" in base_name:
                return "CoEA"
            elif "eda" in base_name:
                return "EDA"
            elif "nsga2" in base_name:
                return "NSGA-II"
            else:
                return "unknown"


        # 获取算法名称列表
        algorithm_names = [extract_algorithm_name(f) for f in available_algorithms]


        # 创建函数来检查文件是否包含期望的列，并处理错误
        def load_convergence_data(file_path, algorithm_name):
            try:
                df = pd.read_csv(file_path)

                # 处理不同算法使用的列名
                if 'Iteration' in df.columns:
                    iteration_column = 'Iteration'
                elif 'Generation' in df.columns:
                    iteration_column = 'Generation'
                else:
                    st.warning(
                        f"The file {os.path.basename(file_path)} does not contain the required 'Iteration' or 'Generation' columns.")
                    return None, None

                if 'Best Score' not in df.columns:
                    st.warning(
                        f"The file {os.path.basename(file_path)} does not contain the required 'Best Score' column.")
                    return None, None

                return df, iteration_column
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
                return None, None


        # 检查每个算法的收敛数据
        convergence_data = {}
        iteration_columns = {}  # 记录每个算法对应的轮数列名
        for file_name in available_algorithms:
            algorithm_name = extract_algorithm_name(file_name)
            file_path = os.path.join(dataset_dir, file_name)
            df, iteration_column = load_convergence_data(file_path, algorithm_name)
            if df is not None:
                convergence_data[algorithm_name] = df
                iteration_columns[algorithm_name] = iteration_column

        # 如果没有找到有效的收敛数据，显示错误
        if not convergence_data:
            st.error("No valid convergence data found for this dataset. Please check your CSV files.")
        else:
            # 获取不同算法的最大轮数
            max_iterations = {algo: df[iteration_columns[algo]].max() for algo, df in convergence_data.items()}

            # 确保所有算法的 max_iterations 中没有 None 或空值
            valid_max_iterations = [v for v in max_iterations.values() if v is not None and pd.notna(v)]

            if not valid_max_iterations:
                st.error("No valid iteration data found in the convergence files.")
            else:
                # 用户选择要比较的算法，默认只选中 SA
                selected_algorithms = st.sidebar.multiselect("Select Algorithms for Comparison", algorithm_names,
                                                             default=["SA"])

                # 用户选择最大显示轮数，默认显示 10 轮
                if selected_algorithms:
                    max_iter_display = st.sidebar.slider("Max Iterations to Display", min_value=1,
                                                         max_value=max(valid_max_iterations), value=10)

                    st.title("Algorithms Comparison")
                    # 绘制收敛曲线

                    st.subheader("Convergence Curves Comparison")
                    st.write(f"For Dataset: {selected_dataset}")

                    # 使用 Plotly 绘制收敛曲线
                    fig = go.Figure()

                    final_best_scores = {}

                    for algo in selected_algorithms:
                        iterations = convergence_data[algo][iteration_columns[algo]][:max_iter_display]
                        best_scores = convergence_data[algo]['Best Score'][:max_iter_display]

                        # 检查是否有足够的轮数数据
                        if not best_scores.empty:
                            fig.add_trace(go.Scatter(x=iterations, y=best_scores, mode='lines+markers', name=f"{algo}"))

                            # 记录最终的 Best Score，确保数据非空
                            final_best_scores[algo] = best_scores.iloc[-1]
                        else:
                            final_best_scores[algo] = "No data"

                    fig.update_layout(
                        title="Convergence Curves",
                        xaxis_title="Iterations / Generations",
                        yaxis_title="Best Score",
                        legend_title="Algorithms"
                    )

                    # 显示 Plotly 图表
                    st.plotly_chart(fig)

                    # 显示最终对比的 Best Score
                    st.write("Final Best Scores Comparison")

                    # 创建一个 DataFrame 显示最终 Best Score
                    final_scores_df = pd.DataFrame({
                        'Algorithm': list(final_best_scores.keys()),
                        'Final Best Score': list(final_best_scores.values())
                    })

                    # 显示 DataFrame
                    st.table(final_scores_df)












                # **Performance Comparison Section**
                st.subheader("Performance Comparison")
                performance_metrics = []

                for algo in selected_algorithms:
                    # 特殊处理 PSO_SA 文件名
                    algo_lower = "psosa" if algo == "PSO_SA" else algo.lower()

                    # 获取算法对应的性能数据文件
                    performance_file_path = os.path.join(performance_data_dir, selected_dataset, f"comparison_performance_{algo_lower}.csv")
                    if os.path.exists(performance_file_path):
                        performance_df = pd.read_csv(performance_file_path)
                        # 将 inf 和 -inf 转换为 NaN
                        performance_df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

                        performance_metrics.append({
                            'Algorithm': algo,
                            'Iterations': max_iterations[algo],
                            'Best Score': performance_df['Best Score'].values[0],
                            'Worst Score': performance_df['Worst Score'].values[0],
                            'Best Improvement': performance_df['Best Improvement'].values[0],
                            'Average Improvement': performance_df['Average Improvement'].values[0],
                            'Time (s)': performance_df['Time (s)'].values[0],
                            'Convergence Rate': performance_df['Convergence Rate'].values[0],
                            'Relative Error': performance_df['Relative Error'].values[0],
                            'Convergence Speed (Stable Iterations)': performance_df['Convergence Speed (Stable Iterations)'].values[0]
                        })
                    else:
                        st.warning(f"Performance data for {algo} not found.")

                # 将性能指标展示成表格
                if performance_metrics:
                    performance_df = pd.DataFrame(performance_metrics)
                    st.table(performance_df)

                    # 用户选择图表类型
                    chart_type = st.selectbox("Select Chart Type", ["Combined (Bar + Line)", "Line Chart", "Bar Chart", "Area Chart"])

                    # 用户选择要展示的性能指标
                    metrics_to_visualize = st.selectbox("Select Performance Metric", [
                        'Best Score', 'Worst Score', 'Best Improvement', 'Average Improvement', 'Time (s)'
                    ])

                    # 替换 inf 和 NaN 数据为 0
                    performance_df.replace([float('inf'), float('-inf'), pd.NA], 0, inplace=True)

                    # 提示用户哪些算法的哪些指标为 inf
                    for algo in selected_algorithms:
                        for metric in ['Best Score', 'Worst Score', 'Best Improvement', 'Average Improvement', 'Time (s)']:
                            # 检查查询结果是否为空，防止越界访问
                            filtered_data = performance_df.loc[performance_df['Algorithm'] == algo, metric]
                            if filtered_data.empty:
                                st.warning(f"No data available for {metric} in {algo}.")
                            elif filtered_data.values[0] == 0:
                                st.warning(f"{algo} has invalid values (like inf or NaN) for {metric}, replaced with 0 for visualization.")

                    import plotly.colors


                    # 动态调整宽度，根据算法数量设置合理的宽度
                    def get_bar_width(num_algorithms):
                        if num_algorithms <= 3:
                            return 0.1  # 少量算法时较宽
                        elif num_algorithms <= 6:
                            return 0.3  # 中等数量算法时适中宽度
                        else:
                            return 0.2  # 大量算法时较窄


                    # 使用 Plotly 绘制性能比较图表
                    if chart_type == "Bar Chart":
                        # 获取算法数量并动态设置柱体宽度
                        bar_width = get_bar_width(len(selected_algorithms))
                        fig = go.Figure(go.Bar(
                            x=performance_df['Algorithm'],
                            y=performance_df[metrics_to_visualize],
                            width=[bar_width] * len(performance_df),
                            marker_color='lightblue'  # 设置柱体颜色为浅蓝色
                        ))
                        fig.update_layout(title=f"{metrics_to_visualize} Comparison - Bar Chart")
                    elif chart_type == "Line Chart":
                        fig = px.line(
                            performance_df,
                            x="Algorithm",
                            y=metrics_to_visualize,
                            markers=True,
                            title=f"{metrics_to_visualize} Comparison - Line Chart"
                        )
                    elif chart_type == "Combined (Bar + Line)":
                        bar_width = get_bar_width(len(selected_algorithms))  # 动态调整宽度
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=performance_df['Algorithm'],
                            y=performance_df[metrics_to_visualize],
                            name='Bar',
                            width=[bar_width] * len(performance_df),
                            marker_color='lightblue'  # 设置柱体颜色为浅蓝色
                        ))
                        fig.add_trace(go.Scatter(
                            x=performance_df['Algorithm'],
                            y=performance_df[metrics_to_visualize],
                            mode='lines+markers',
                            name='Line'
                        ))
                        fig.update_layout(title=f"{metrics_to_visualize} Comparison - Combined Chart")
                    elif chart_type == "Area Chart":
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=performance_df['Algorithm'],
                            y=performance_df[metrics_to_visualize],
                            fill='tozeroy',
                            mode='lines+markers',
                            name='Area'
                        ))
                        fig.update_layout(title=f"{metrics_to_visualize} Comparison - Area Chart")

                    # 显示 Plotly 图表
                    st.plotly_chart(fig)




                else:
                    st.warning("No performance metrics found for the selected algorithms.")

