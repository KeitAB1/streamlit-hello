import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging  # 日志模块

import numpy as np
import pandas as pd

def extract_timestamp_from_filename(file_name):
    """
    从文件名中提取日期和时间戳作为图例。
    假设文件名格式为 'convergence_data_algorithm_datasetname_YYYYMMDD_HHMMSS.csv'
    """
    parts = file_name.split('_')
    if len(parts) >= 5:
        date_part = parts[-2]  # YYYYMMDD
        time_part = parts[-1].replace('.csv', '')  # HHMMSS
        return f"{date_part}_{time_part}"
    return file_name

def calculate_statistics(data, file_names):
    """
    计算给定数据集的统计信息，包括均值、标准差、最大值、最小值、范围和最终 Best Score。
    :param data: 收敛曲线数据（列表的列表，每个子列表对应一个历史数据文件的 Best Score 列）
    :param file_names: 历史数据文件名列表，用于结果表格中
    :return: 包含统计信息的 pandas DataFrame
    """
    statistics = []

    for i, scores in enumerate(data):
        mean_value = np.mean(scores)
        std_dev = np.std(scores)
        max_value = np.max(scores)
        min_value = np.min(scores)
        value_range = max_value - min_value
        best_score = scores[-1]  # 取最后一个 Best Score

        # 提取文件名中的时间戳
        timestamp = extract_timestamp_from_filename(file_names[i])

        statistics.append({
            'Timestamp': timestamp,
            'Mean': f"{mean_value:.2e}",  # 将均值以科学计数法表示
            'Std Dev': f"{std_dev:.2e}",  # 将标准差以科学计数法表示
            'Max': f"{max_value:.2e}",    # 将最大值以科学计数法表示
            'Min': f"{min_value:.2e}",    # 将最小值以科学计数法表示
            'Range': f"{value_range:.2e}", # 将范围以科学计数法表示
            'Best Score': f"{best_score:.2e}"  # 最后一个 Best Score 以科学计数法表示
        })

    return pd.DataFrame(statistics)




def get_selected_history_data(history_data_dir, selected_file):
    """
    从选定的历史文件中读取数据。
    :param history_data_dir: 存储历史数据的目录
    :param selected_file: 用户选择的历史数据文件
    :return: DataFrame, 读取的历史数据
    """
    if selected_file:
        file_path = os.path.join(history_data_dir, selected_file)
        return pd.read_csv(file_path)
    return None



def load_history_data(history_data_dir):
    """
    从指定目录中加载历史数据文件，并提供用户选择。
    :param history_data_dir: 存储历史数据的目录
    :return: 返回用户选择的历史数据文件内容（DataFrame），或者 None 如果没有文件
    """
    if not os.path.exists(history_data_dir):
        os.makedirs(history_data_dir)

    history_files = [f for f in os.listdir(history_data_dir) if f.endswith('.csv')]
    if not history_files:
        return None, None

    # 返回历史文件列表和选择框
    return history_files

def save_convergence_history(convergence_data, algorithm, dataset_name, history_data_dir):
    """
    保存当前的收敛数据到历史文件中，文件名包含时间戳。
    :param convergence_data: 当前优化的收敛数据
    :param algorithm: 使用的优化算法名称
    :param dataset_name: 数据集名称
    :param history_data_dir: 存储历史数据的目录
    """
    os.makedirs(history_data_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file_name = f"convergence_data_{algorithm}_{dataset_name}_{current_time}.csv"
    history_file_path = os.path.join(history_data_dir, history_file_name)

    # 保存收敛数据到 CSV 文件
    convergence_df = pd.DataFrame(convergence_data, columns=['Iteration', 'Best Score'])
    convergence_df.to_csv(history_file_path, index=False)

    return history_file_path

# 保存当前收敛数据并调用保存历史数据函数
def save_convergence_plot(convergence_data, current_iteration, best_score, algorithm_name, dataset_name, history_file_path=None):
    """
    保存收敛数据到结果目录，并调用历史数据保存函数将数据追加到指定的历史文件。
    :param convergence_data: 当前优化的收敛数据
    :param current_iteration: 当前迭代次数
    :param best_score: 当前最佳得分
    :param algorithm_name: 使用的算法名称
    :param dataset_name: 数据集名称
    :param history_file_path: 历史文件的路径（同一优化期间只生成一次）
    """
    # 1. 保存收敛数据到 result 文件夹
    dataset_folder = dataset_name.split('.')[0]  # 移除文件扩展名
    convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
    os.makedirs(convergence_data_dir, exist_ok=True)

    # 保存收敛数据到 result 文件夹
    convergence_data_df = pd.DataFrame(convergence_data, columns=['Iteration', 'Best Score'])
    convergence_data_path = os.path.join(convergence_data_dir, f'convergence_data_{algorithm_name}.csv')
    convergence_data_df.to_csv(convergence_data_path, index=False)





# 通用的保存优化性能指标函数
def save_performance_metrics(best_score, worst_score, best_improvement, total_improvement,
                             iterations, time_elapsed, convergence_data, stable_iterations,
                             dataset_name, algorithm_name):
    # 计算优化的各项指标
    best_score = best_score
    iterations = len(convergence_data)
    convergence_rate_value = (convergence_data[-1][1] - convergence_data[0][1]) / iterations
    relative_error_value = abs(best_score) / (abs(best_score) + 1e-6)
    fitness_evaluations = None  # SA 没有适应度函数评估次数
    average_improvement = total_improvement / iterations if iterations > 0 else 0

    # 创建字典存储所有指标
    metrics = {
        'Best Score': best_score,
        'Worst Score': worst_score,
        'Best Improvement': best_improvement,
        'Average Improvement': average_improvement,
        'Iterations': iterations,
        'Time (s)': time_elapsed,
        'Convergence Rate': convergence_rate_value,
        'Relative Error': relative_error_value,
        'Convergence Speed (Stable Iterations)': stable_iterations
    }

    # 保存性能指标
    dataset_folder = f"result/comparison_performance/{dataset_name.split('.')[0]}"
    os.makedirs(dataset_folder, exist_ok=True)
    file_name = f"comparison_performance_{algorithm_name}.csv"
    file_path = os.path.join(dataset_folder, file_name)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(file_path, index=False)
    # print(f"Performance metrics saved to {file_path}")


# 保存并可视化优化结果
def save_and_visualize_results(optimizer, df, area_positions, output_dir_base, algorithm_name):
    # 根据算法名字区分最优解的引用方式
    if algorithm_name == "ga":
        final_positions_with_batch = optimizer.best_individual
    else:
        final_positions_with_batch = optimizer.best_position

    final_x, final_y = [], []

    for i, position in enumerate(final_positions_with_batch):
        area = position
        # 获取每个位置的x、y坐标
        x, y = area_positions[area][i % len(area_positions[area])]
        final_x.append(x)
        final_y.append(y)

    # 更新df并保存结果
    df['Final Area'] = final_positions_with_batch
    df['Final X'] = final_x
    df['Final Y'] = final_y

    # 保存到 final_stack_distribution 文件夹
    final_stack_dir = os.path.join(output_dir_base, 'final_stack_distribution')
    os.makedirs(final_stack_dir, exist_ok=True)
    output_file_plates_with_batch = os.path.join(final_stack_dir, f'final_stack_distribution_{algorithm_name}.csv')
    df.to_csv(output_file_plates_with_batch, index=False)

    st.success(f"{algorithm_name} optimization completed! You can now visualize the results.")

    heights_dict = {}
    df['Stacking Start Height'] = 0.0
    df['Stacking Height'] = 0.0

    # 更新垛位高度信息
    for i in range(len(df)):
        area = df.loc[i, 'Final Area']
        x = df.loc[i, 'Final X']
        y = df.loc[i, 'Final Y']
        key = (area, x, y)
        current_height = heights_dict.get(key, 0.0)
        df.loc[i, 'Stacking Start Height'] = current_height
        df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
        heights_dict[key] = df.loc[i, 'Stacking Height']

    # 保存钢板位置分布的最终结果到 final_stack_distribution_plates 文件夹
    final_stack_plates_dir = os.path.join(output_dir_base, 'final_stack_distribution_plates')
    os.makedirs(final_stack_plates_dir, exist_ok=True)
    output_file_plates_with_batch = os.path.join(final_stack_plates_dir,
                                                 f'final_stack_distribution_plates_{algorithm_name}.csv')
    df.to_csv(output_file_plates_with_batch, index=False)

    return output_file_plates_with_batch


# 生成垛位分布的统计数据
def generate_stacking_distribution_statistics(df, area_positions, output_dir_base, algorithm_name):
    height_dict = {}
    plate_count_dict = {}

    # 初始化每个区域的高度和钢板计数
    for area in area_positions.keys():
        for pos in area_positions[area]:
            height_dict[(area, pos[0], pos[1])] = 0.0
            plate_count_dict[(area, pos[0], pos[1])] = 0

    # 检查给定位置是否有效
    def is_valid_position(area, x, y):
        return (area in area_positions) and ((int(x), int(y)) in area_positions[area])

    # 遍历数据框，更新高度和钢板计数
    for index, row in df.iterrows():
        area = row['Final Area']
        x = row['Final X']
        y = row['Final Y']
        stacking_height = row['Stacking Height']

        x = int(x)
        y = int(y)

        if is_valid_position(area, x, y):
            height_dict[(area, x, y)] = stacking_height
            plate_count_dict[(area, x, y)] += 1

    # 生成每个区域的统计结果
    results = []

    for area, positions in area_positions.items():
        total_plates = 0
        heights = []

        for pos in positions:
            height = height_dict[(area, pos[0], pos[1])]
            heights.append(height)
            total_plates += plate_count_dict[(area, pos[0], pos[1])]

        average_height = np.mean(heights)

        result_entry = {'Area': area, 'Total Plates': total_plates, 'Average Height': average_height}
        for i, pos in enumerate(positions):
            result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

        results.append(result_entry)

    result_df = pd.DataFrame(results)

    st.write("### Stacking Distribution Statistics Table")
    st.dataframe(result_df)

    # 保存最终的统计数据到 final_stack_distribution_height 文件夹
    final_stack_height_dir = os.path.join(output_dir_base, 'final_stack_distribution_height')
    os.makedirs(final_stack_height_dir, exist_ok=True)
    final_stack_distribution_path = os.path.join(final_stack_height_dir,
                                                 f'final_stack_distribution_height_{algorithm_name}.csv')
    result_df.to_csv(final_stack_distribution_path, index=False)

    return final_stack_distribution_path


# 添加下载按钮
def add_download_button(file_path, algorithm_name):

    # 显示文件的前5行
    st.write("### Final Stack Distribution Plates")

    with open(file_path, 'rb') as file:
        st.download_button(
            label=f"Download All SteelPlate Data",
            data=file,
            file_name=f'final_stack_distribution_plates_{algorithm_name}.csv',
            mime='text/csv'
        )

    df_plates_with_batch = pd.read_csv(file_path)
    st.dataframe(df_plates_with_batch.head(5))  # 只显示前5行


# 根据用户选择的算法执行优化
def run_optimization(optimizer_class, params, df, area_positions, output_dir_base, algorithm_name):
    optimizer = optimizer_class(**params)
    optimizer.optimize()

    # 保存和展示优化结果
    output_file_plates_with_batch = save_and_visualize_results(optimizer, df, area_positions, output_dir_base,
                                                               algorithm_name)

    # 生成垛位分布统计
    generate_stacking_distribution_statistics(df, area_positions, output_dir_base, algorithm_name)

    # 添加下载按钮
    add_download_button(output_file_plates_with_batch, algorithm_name)
