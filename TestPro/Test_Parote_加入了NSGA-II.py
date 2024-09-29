import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from deap import base, creator, tools


from auxiliary import InterfaceLayout as il

# from optimization_functions import  minimize_stack_movements_and_turnover, minimize_outbound_energy_time_with_batch, \
#     maximize_inventory_balance_v2, maximize_space_utilization_v3
#
# from optimizers.pso_optimizer import PSO_with_Batch
# from optimizers.ga_optimizer import GA_with_Batch
# from optimizers.sa_optimizer import SA_with_Batch

# 创建适应度类和个体类
# 由于在NSGA-II中需要最小化两个目标，所以使用weights=(-1.0, -1.0)
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
output_dir = "../stack_distribution_plots/final_stack_distribution"
convergence_dir = "../result/ConvergenceData"
data_dir = "../test/steel_data"  # 统一数据集目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(convergence_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# 定义保存路径为 test_data.csv
test_data_path = os.path.join(data_dir, "test_data.csv")

st.title("Steel Plate Stacking Optimization")
# # 显示图标和标题
# icon_path = "data/introduction_src/icons/home_标题.png"
# il.display_icon_with_header(icon_path, "Steel Plate Stacking Optimization", font_size='45px')

# 获取 data 文件夹下的所有 CSV 文件
system_data_dir = "../data"  # 系统数据集目录
available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]

# 选择数据集的方式
data_choice = st.selectbox("Choose dataset", ("Use system dataset", "Upload your own dataset"))

# 初始化 df 为 None
df = None
dataset_name = None  # 确保 dataset_name 变量存在

# 如果用户选择上传数据集
if data_choice == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload your steel plate dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name.split('.')[0]  # 获取上传文件的文件名（不含扩展名）
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset:")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset to proceed.")

# 如果用户选择使用系统自带的数据集
elif data_choice == "Use system dataset":
    # 列出可用的数据集供用户选择，并去掉 .csv 后缀
    selected_dataset = st.selectbox("Select a system dataset", [""] + available_datasets)
    if selected_dataset and selected_dataset != "":
        dataset_name = selected_dataset  # 获取数据集文件名（不含扩展名）
        system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")  # 重新加上 .csv 后缀
        df = pd.read_csv(system_dataset_path)
        st.write(f"Using system dataset: {selected_dataset}")
        st.write(df.head())
    else:
        st.warning("Please select a system dataset to proceed.")



# # 单目标优化和多目标优化算法分开模块
# optimization_mode = st.radio("Select Optimization Mode", ("Single-Objective", "Multi-Objective"))

# # 单目标优化算法选择
# if optimization_mode == "Single-Objective":
# 算法选择放在侧边栏

with st.sidebar:
    algorithms = ["SA (Simulated Annealing)", "GA (Genetic Algorithm)", "PSO (Particle Swarm Optimization)",
                  "PSO + SA (Hybrid Optimization)", "ACO (Ant Colony Optimization)", "DE (Differential Evolution)",
                  "CoEA (Co-Evolutionary Algorithm)", "EDA (Estimation of Distribution Algorithm)", "NSGA-II (Non-Dominated Sorting Genetic Algorithm II)"]
    selected_algorithm = st.selectbox("Select Optimization Algorithm", algorithms)

    # 根据选择的算法动态显示相关参数设置
    if selected_algorithm == "PSO (Particle Swarm Optimization)":
        st.subheader("PSO Parameters")
        max_iter = st.number_input("Max Iterations", 1, 1000, 1)
        w = st.slider("Inertia Weight (w)", 0.0, 1.0, 0.5)
        c1 = st.slider("Cognitive Coefficient (c1)", 0.0, 4.0, 2.0)
        c2 = st.slider("Social Coefficient (c2)", 0.0, 4.0, 2.0)
        st.write(f"Note: c1 + c2 should equal 4.0. Current sum: {c1 + c2}")
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "GA (Genetic Algorithm)":
        st.subheader("GA Parameters")
        generations = st.number_input("Generations", 1, 1000, 3)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.05)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        population_size = st.slider("Population Size", 100, 500, 100)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "SA (Simulated Annealing)":
        st.subheader("SA Parameters")
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "PSO + SA (Hybrid Optimization)":
        st.subheader("PSO + SA Parameters")

        # PSO Parameters
        st.write("### PSO Parameters")
        max_iter_pso = st.number_input("PSO Max Iterations", 1, 1000, 1)
        w = st.slider("PSO Inertia Weight (w)", 0.0, 1.0, 0.5)
        c1 = st.slider("PSO Cognitive Coefficient (c1)", 0.0, 4.0, 2.0)
        c2 = st.slider("PSO Social Coefficient (c2)", 0.0, 4.0, 2.0)
        st.write(f"Note: c1 + c2 should equal 4.0. Current sum: {c1 + c2}")

        # SA Parameters
        st.write("### SA Parameters")
        initial_temperature = st.number_input("Initial Temperature (SA)", value=1000.0)
        cooling_rate = st.slider("SA Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("SA Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("SA Max Iterations", 1, 1000, 100)

        # Common weights for both PSO and SA
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "ACO (Ant Colony Optimization)":
        st.subheader("ACO Parameters")
        max_iter = st.number_input("Max Iterations", 1, 1000, 5)  # 默认迭代轮数为5
        num_ants = st.slider("Number of Ants", 10, 100, 50)
        alpha = st.slider("Pheromone Importance (α)", 0.0, 5.0, 1.0)
        beta = st.slider("Heuristic Importance (β)", 0.0, 5.0, 2.0)
        evaporation_rate = st.slider("Evaporation Rate", 0.0, 1.0, 0.5)
        q = st.number_input("Pheromone Constant (Q)", 1.0, 1000.0, 100.0)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "DE (Differential Evolution)":
        st.subheader("DE Parameters")
        max_iter = st.number_input("Max Iterations", 1, 1000, 100)
        F = st.slider("Mutation Factor (F)", 0.0, 2.0, 0.5)
        CR = st.slider("Crossover Rate (CR)", 0.0, 1.0, 0.9)
        pop_size = st.slider("Population Size", 10, 200, 50)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "CoEA (Co-Evolutionary Algorithm)":
        st.subheader("CoEA Parameters")
        # CoEA-specific parameters
        generations = st.number_input("Generations", 1, 1000, 100)
        population_size = st.slider("Population Size", 10, 500, 100)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        num_subpopulations = st.slider("Number of Subpopulations", 2, 10, 2)
        exchange_rate = st.slider("Exchange Rate Between Subpopulations", 0.0, 1.0, 0.1)

        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    elif selected_algorithm == "EDA (Estimation of Distribution Algorithm)":
        st.subheader("EDA Parameters")
        pop_size = st.slider("Population Size", 10, 200, 50)
        max_iter = st.number_input("Max Iterations", 1, 1000, 100)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)
    elif selected_algorithm == "NSGA-II (Non-Dominated Sorting Genetic Algorithm II)":
        st.subheader("NSGA-II Parameters")

        # 参数设置部分
        population_size = st.number_input("Population Size", value=100, min_value=10, max_value=1000, step=10)
        num_generations = st.number_input("Number of Generations", value=100, min_value=10, max_value=1000, step=10)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, value=0.9)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, value=0.1)

        # 优化目标的权重设置部分
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Weight for Minimize Stack Movements and Turnover)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Weight for Minimize Outbound Energy and Time)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Weight for Maximize Inventory Balance)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Weight for Maximize Space Utilization)", value=1.0)

# # 多目标优化算法选择
# elif optimization_mode == "Multi-Objective":
#     with st.sidebar:
#         multi_objective_algorithms = ["NSGA-II (Non-Dominated Sorting Genetic Algorithm II)",
#                                       "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)",
#                                       "MOPSO (Multi-Objective Particle Swarm Optimization)"]
#         selected_multi_algo = st.selectbox("Select Multi-Objective Optimization Algorithm", multi_objective_algorithms)
#
#         # 显示不同算法的相关参数
#         if selected_multi_algo == "NSGA-II (Non-Dominated Sorting Genetic Algorithm II)":
#             st.subheader("NSGA-II Parameters")
#             generations_nsga = st.number_input("Generations", 1, 1000, 200)
#             population_size_nsga = st.number_input("Population Size", 50, 500, 100)
#             crossover_rate_nsga = st.slider("Crossover Rate", 0.0, 1.0, 0.9)
#             mutation_rate_nsga = st.slider("Mutation Rate", 0.0, 1.0, 0.05)
#
#         elif selected_multi_algo == "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)":
#             st.subheader("MOEA/D Parameters")
#             generations_moead = st.number_input("Generations", 1, 1000, 200)
#             population_size_moead = st.number_input("Population Size", 50, 500, 100)
#             neighbor_size_moead = st.number_input("Neighborhood Size", 1, 50, 10)
#             crossover_rate_moead = st.slider("Crossover Rate", 0.0, 1.0, 0.9)
#             mutation_rate_moead = st.slider("Mutation Rate", 0.0, 1.0, 0.05)
#
#         elif selected_multi_algo == "MOPSO (Multi-Objective Particle Swarm Optimization)":
#             st.subheader("MOPSO Parameters")
#             max_iter_mopso = st.number_input("Max Iterations", 1, 1000, 200)
#             population_size_mopso = st.number_input("Population Size", 50, 500, 100)
#             inertia_weight_mopso = st.slider("Inertia Weight", 0.0, 1.0, 0.5)
#             cognitive_coeff_mopso = st.slider("Cognitive Coefficient", 0.0, 4.0, 2.0)
#             social_coeff_mopso = st.slider("Social Coefficient", 0.0, 4.0, 2.0)

# 动态显示收敛曲线的占位符
convergence_plot_placeholder = st.empty()



# 如果 df 已经加载，进行堆垛优化分析
if df is not None:

    # 根据用户选择的数据集名称动态创建保存路径
    output_dir_base = f"result/final_stack_distribution/{dataset_name}"
    os.makedirs(os.path.join(output_dir_base, 'final_stack_distribution_height'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_base, 'final_stack_distribution_plates'), exist_ok=True)

    # 参数配置（假设数据集结构一致）
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

    # 每个库区最大堆垛容量，垛位限制高为3000mm
    Dki = np.array([
        np.sum([dim[0] * dim[1] * 3000 for dim in stack_dimensions[0]]),  # 库区 1
        np.sum([dim[0] * dim[1] * 3000 for dim in stack_dimensions[1]]),  # 库区 2
        np.sum([dim[0] * dim[1] * 3000 for dim in stack_dimensions[2]]),  # 库区 3
        np.sum([dim[0] * dim[1] * 3000 for dim in stack_dimensions[3]]),  # 库区 4
        np.sum([dim[0] * dim[1] * 3000 for dim in stack_dimensions[4]]),  # 库区 5
        np.sum([dim[0] * dim[1] * 3000 for dim in stack_dimensions[5]])  # 库区 6
    ])

    # 初始化每个库区中的垛位高度
    heights = np.zeros(len(Dki))

    # 电磁吊的速度，单位：米/分钟 -> 毫米/秒
    horizontal_speed = 72 * 1000 / 60  # 水平速度：72m/min，转换为 mm/s
    vertical_speed = 15 * 1000 / 60  # 垂直速度：15m/min，转换为 mm/s
    stack_flip_time_per_plate = 10  # 每次翻垛需要10秒

    # # 传送带位置参数
    # conveyor_position_x = 2000  # 距离库区1-3的传送带水平距离
    # conveyor_position_y = 14000  # 距离库区4-6的传送带水平距离

    # 新增：入库口和出库口的坐标
    inbound_point = (41500, 3000)  # 入库口坐标
    outbound_point = (41500, 38000)  # 出库口坐标

    #  将交货时间从字符串转换为数值
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values




    # 目标函数1：最小化翻垛次数
    def minimize_stack_movements_and_turnover(particle_positions, heights, plates, delivery_times, batches,
                                              weight_movement=1.0, weight_turnover=1.0):
        num_movements = 0
        total_turnover = 0
        batch_turnover = 0

        for plate_idx, position in enumerate(particle_positions):
            area = position
            plate_height = plates[plate_idx, 2]  # 厚度即为钢板的高度
            current_height = heights[area]

            # 判断是否需要翻垛（按高度限制）
            if current_height + plate_height > 3000:
                num_movements += 1  # 超过限制高度需要翻垛

            heights[area] += plate_height  # 更新当前垛位高度

        for i in range(len(particle_positions)):
            for j in range(i + 1, len(particle_positions)):
                # 考虑钢板的交货时间差来计算翻转次数
                time_diff = abs(delivery_times[i] - delivery_times[j])
                total_turnover += time_diff

                # 如果属于不同批次，增加翻堆次数
                if batches[i] != batches[j]:
                    batch_turnover += 1

        combined_score = weight_movement * num_movements + weight_turnover * (total_turnover + batch_turnover)
        return combined_score


    # 目标函数2：最小化出库能耗与时间
    def minimize_outbound_energy_time_with_batch(particle_positions, plates, heights):
        total_time_energy = 0

        # 按批次将钢板索引分配
        sorted_batches = sorted(set(batches), key=lambda x: int(x[1:]))
        plate_indices_by_batch = {batch: [] for batch in sorted_batches}

        # 将钢板按批次分配
        for plate_idx, plate in enumerate(plates):
            batch = plate[4]  # 批次信息在第5列（索引4）
            plate_indices_by_batch[batch].append(plate_idx)

        # 计算水平和垂直距离的通用函数
        def calculate_distance(x1, y1, x2, y2):
            horizontal_distance = abs(x2 - x1)
            vertical_distance = abs(y2 - y1)
            return horizontal_distance, vertical_distance

        # 取出钢板的时间，基于堆垛高度
        def calculate_pick_time(plate_height):
            return plate_height / vertical_speed

        # 翻垛时间的计算，计算翻垛层数
        def calculate_flip_time(plate_idx, particle_positions, heights):
            # 查找当前堆垛位置上方的钢板数量
            area = particle_positions[plate_idx]
            current_height = heights[area]
            plate_height = plates[plate_idx, 2]  # 当前钢板的厚度

            # 如果当前堆垛的高度超过了当前钢板的高度，计算翻垛数量
            if current_height > plate_height:
                n_flip = int(current_height // plate_height)
                t_flip_per = 10  # 每次翻垛时间，单位为秒
                return n_flip * t_flip_per
            else:
                return 0

        # 入库操作：计算每块钢板的入库时间和能耗
        for plate_idx, position in enumerate(particle_positions):
            area = position  # 获取钢板所在的库区
            plate_height = plates[plate_idx, 2]  # 获取钢板厚度

            # 获取钢板的坐标（根据area和layout的位置关系）
            x, y = area_positions[area][plate_idx % len(area_positions[area])]

            # 计算入库时间和能耗
            inbound_horizontal_dist, inbound_vertical_dist = calculate_distance(x, y, inbound_point[0],
                                                                                inbound_point[1])
            inbound_time = (inbound_horizontal_dist / horizontal_speed) + (inbound_vertical_dist / vertical_speed)

            # 更新堆垛高度
            heights[area] += plate_height

            # 记录入库时间和能耗
            total_time_energy += inbound_time

        # 出库操作：按批次顺序出库
        for batch in sorted_batches:
            for plate_idx in plate_indices_by_batch[batch]:
                position = particle_positions[plate_idx]
                area = position  # 获取钢板所在库区
                plate_height = plates[plate_idx, 2]  # 获取钢板厚度

                # 获取钢板的坐标（根据area和layout的位置关系）
                x, y = area_positions[area][plate_idx % len(area_positions[area])]

                # 计算出库时间和能耗
                outbound_horizontal_dist, outbound_vertical_dist = calculate_distance(x, y, outbound_point[0],
                                                                                      outbound_point[1])
                outbound_time = (outbound_horizontal_dist / horizontal_speed) + (
                            outbound_vertical_dist / vertical_speed)

                # 计算取出时间
                pick_time = calculate_pick_time(plate_height)

                # 计算翻垛时间
                flip_time = calculate_flip_time(plate_idx, particle_positions, heights)

                # 更新堆垛高度
                heights[area] -= plate_height

                # 记录出库总时间，包括移动、取出和翻垛时间
                total_time_energy += (outbound_time + pick_time + flip_time)

        return total_time_energy


    # 目标函数3：最大化库存均衡度
    def maximize_inventory_balance_v2(particle_positions, plates):
        total_variance = 0
        total_volume = np.sum(plates[:, 0] * plates[:, 1] * plates[:, 2])
        num_positions = len(Dki)
        mean_volume_per_position = total_volume / num_positions
        area_volumes = np.zeros(num_positions)

        # 计算每个库区的体积占用
        for plate_idx, position in enumerate(particle_positions):
            plate_volume = plates[plate_idx][0] * plates[plate_idx][1] * plates[plate_idx][2]
            area_volumes[position] += plate_volume

        # 计算均衡度的方差，通过减小方差使各个库区的体积更均衡
        for j in range(num_positions):
            total_variance += (area_volumes[j] - mean_volume_per_position) ** 2

        return total_variance / num_positions  # 方差越小，均衡度越好


    # 目标函数4：空间利用率最大化
    def maximize_space_utilization_v3(particle_positions, plates, Dki, alpha_1=1.0, epsilon=1e-6):
        total_space_utilization = 0
        for i in range(len(Dki)):
            used_volume = 0
            max_volume = Dki[i]

            for j in range(len(plates)):
                if particle_positions[j] == i:
                    plate_volume = plates[j][0] * plates[j][1] * plates[j][2]
                    used_volume += plate_volume

            if used_volume > 0:
                utilization = alpha_1 * max((max_volume - used_volume), epsilon) / used_volume
                total_space_utilization += utilization
            else:
                total_space_utilization += 0

        return total_space_utilization


    # 定义粒子类
    class Particle:
        def __init__(self, num_positions):
            self.position = np.random.randint(0, num_positions, size=num_plates)  # 随机初始化位置
            self.velocity = np.zeros(num_plates)
            self.best_position = self.position.copy()
            self.best_score = np.inf

        def update_velocity(self, gbest_position, w, c1, c2):
            r1 = np.random.rand(num_plates)
            r2 = np.random.rand(num_plates)
            cognitive = c1 * r1 * (self.best_position - self.position)
            social = c2 * r2 * (gbest_position - self.position)
            self.velocity = w * self.velocity + cognitive + social

        def update_position(self, num_positions):
            self.position = np.clip(self.position + self.velocity, 0, num_positions - 1).astype(int)

    class PSO_with_Batch:
        def __init__(self, num_particles, num_positions, w, c1, c2, max_iter, lambda_1, lambda_2, lambda_3, lambda_4,
                     dataset_name):
            self.num_particles = num_particles
            self.num_positions = num_positions
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.max_iter = max_iter
            self.particles = [Particle(num_positions) for _ in range(self.num_particles)]
            self.gbest_position = None
            self.gbest_score = np.inf
            self.worst_score = -np.inf  # Initialize the worst score
            self.best_improvement = 0  # Initialize the best improvement
            self.total_improvement = 0  # For calculating average improvement
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.convergence_data = []  # 初始化收敛数据
            self.dataset_name = dataset_name  # 增加数据集名称属性
            self.stable_iterations = 0  # 稳定的迭代次数
            self.prev_best_score = np.inf  # 用于跟踪上一次迭代的最佳得分
            self.start_time = None  # 记录算法开始的时间

        def optimize(self):
            global heights
            self.start_time = time.time()  # 记录开始时间
            for iteration in range(self.max_iter):
                improvement_flag = False  # 标志是否有改进
                for particle in self.particles:
                    if self.gbest_position is None:
                        self.gbest_position = particle.position.copy()

                    temp_heights = heights.copy()

                    # 计算目标函数的惩罚项
                    combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                        particle.position, temp_heights, plates, delivery_times, batches)
                    energy_time_penalty = minimize_outbound_energy_time_with_batch(particle.position, plates,
                                                                                   temp_heights)
                    balance_penalty = maximize_inventory_balance_v2(particle.position, plates)
                    space_utilization = maximize_space_utilization_v3(particle.position, plates, Dki)

                    # 计算当前的总得分
                    current_score = (self.lambda_1 * combined_movement_turnover_penalty +
                                     self.lambda_2 * energy_time_penalty +
                                     self.lambda_3 * balance_penalty -
                                     self.lambda_4 * space_utilization)

                    # 更新粒子的历史最佳位置
                    if current_score < particle.best_score:
                        particle.best_score = current_score
                        particle.best_position = particle.position.copy()

                    # 更新全局最佳位置
                    if current_score < self.gbest_score:
                        improvement_flag = True
                        self.best_improvement = max(self.best_improvement, self.gbest_score - current_score)
                        self.gbest_score = current_score
                        self.gbest_position = particle.position.copy()

                    # 更新最差得分
                    if current_score > self.worst_score:
                        self.worst_score = current_score

                # 如果本次迭代有改进，更新总改进值
                if improvement_flag:
                    self.total_improvement += self.prev_best_score - self.gbest_score
                    self.prev_best_score = self.gbest_score
                    self.stable_iterations = 0  # 重置稳定的迭代次数
                else:
                    self.stable_iterations += 1

                # 更新粒子的位置和速度
                for particle in self.particles:
                    particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                    particle.update_position(self.num_positions)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.gbest_score])

                # 实时更新收敛曲线
                self.update_convergence_plot(iteration + 1)

                # 打印迭代信息
                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.gbest_score}')

            # 更新最终高度
            self.update_final_heights()

            # 记录运行时间
            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed)

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.gbest_position):
                area = position
                heights[area] += plates[plate_idx, 2]

        def update_convergence_plot(self, current_iteration):
            # 动态更新收敛曲线
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Iterations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Iteration {current_iteration}, Best Score {self.gbest_score}')
            plt.legend()

            # 使用 Streamlit 的空占位符更新图表
            convergence_plot_placeholder.pyplot(plt)

            # 根据数据集名称动态生成收敛数据的保存路径
            dataset_folder = self.dataset_name.split('.')[0]  # 移除文件扩展名
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_pso.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_metrics(self, time_elapsed):
            """保存优化算法的性能指标"""
            iterations = len(self.convergence_data)
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(self.gbest_score) / (abs(self.gbest_score) + 1e-6)

            # 计算平均改进
            average_improvement = self.total_improvement / iterations if iterations > 0 else 0

            # Fitness evaluations is marked as None for PSO, as per your request
            fitness_evaluations = None

            # 构建性能指标
            metrics = {
                'Best Score': self.gbest_score,
                'Worst Score': self.worst_score,
                'Best Improvement': self.best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Fitness Evaluations': fitness_evaluations,  # 适应度函数评估次数为 None
                'Convergence Speed (Stable Iterations)': self.stable_iterations  # 稳定迭代次数
            }

            # 动态生成保存路径
            dataset_folder = f"result/comparison_performance/{self.dataset_name}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_metrics_pso.csv"
            file_path = os.path.join(dataset_folder, file_name)

            # 保存性能指标为 CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)
            st.success(f"Metrics saved to {file_path}")


    class SA_with_Batch:
        def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, lambda_1, lambda_2,
                     lambda_3, lambda_4, num_positions, dataset_name):
            self.initial_temperature = initial_temperature
            self.cooling_rate = cooling_rate
            self.min_temperature = min_temperature
            self.max_iterations = max_iterations
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.num_positions = num_positions  # 修复：将 num_positions 添加为属性
            self.dataset_name = dataset_name  # 增加数据集名称属性

            # 初始化位置和最佳解
            self.best_position = None
            self.best_score = np.inf
            self.convergence_data = []
            self.start_time = None  # 用于记录开始时间

            # 新增性能评价指标
            self.worst_score = -np.inf
            self.best_improvement = 0
            self.total_improvement = 0
            self.last_score = None
            self.stable_iterations = 0  # 用于计算收敛速度

        def evaluate(self, position):
            global heights
            temp_heights = heights.copy()

            try:
                # 计算目标函数的惩罚项
                combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                    position, temp_heights, plates, delivery_times, batches)
                energy_time_penalty = minimize_outbound_energy_time_with_batch(position, plates, temp_heights)
                balance_penalty = maximize_inventory_balance_v2(position, plates)
                space_utilization = maximize_space_utilization_v3(position, plates, Dki)

                score = (self.lambda_1 * combined_movement_turnover_penalty +
                         self.lambda_2 * energy_time_penalty +
                         self.lambda_3 * balance_penalty -
                         self.lambda_4 * space_utilization)

                return score

            except Exception as e:
                print(f"Error in evaluation: {e}")
                return np.inf

        def optimize(self):
            # 默认从随机位置开始优化
            return self.optimize_from_position(np.random.randint(0, self.num_positions, size=num_plates))

        def optimize_from_position(self, initial_position):
            global heights
            current_temperature = self.initial_temperature
            current_position = initial_position  # 使用传入的初始位置
            current_score = self.evaluate(current_position)

            self.best_position = current_position.copy()
            self.best_score = current_score
            self.last_score = current_score
            self.start_time = time.time()  # 记录优化开始时间

            for iteration in range(self.max_iterations):
                if current_temperature < self.min_temperature:
                    break

                # 生成新解
                new_position = current_position.copy()
                random_index = np.random.randint(0, len(current_position))
                new_position[random_index] = np.random.randint(0, self.num_positions)
                new_score = self.evaluate(new_position)

                # 计算接受概率
                delta = new_score - current_score
                if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                    current_position = new_position
                    current_score = new_score

                # 更新全局最佳解
                improvement = self.last_score - current_score
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = current_position.copy()
                    self.best_improvement = max(self.best_improvement, improvement)

                # 更新最差解
                if current_score > self.worst_score:
                    self.worst_score = current_score

                # 累积改进值
                self.total_improvement += improvement

                # 判断收敛速度
                if improvement < 1e-6:
                    self.stable_iterations += 1

                self.last_score = current_score

                # 降温
                current_temperature *= self.cooling_rate

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])

                # 实时更新收敛曲线
                self.update_convergence_plot(iteration + 1)

                # 打印每次迭代的最佳得分
                print(
                    f"Iteration {iteration + 1}/{self.max_iterations}, Best Score: {self.best_score}, Temperature: {current_temperature}")

            # 记录运行时间并保存指标
            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed)

            return self.best_position, self.best_score

        def update_convergence_plot(self, current_iteration):
            # 动态更新收敛曲线
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Iterations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}')
            plt.legend()

            # 使用 Streamlit 的空占位符更新图表
            convergence_plot_placeholder.pyplot(plt)

            # 根据数据集名称动态生成收敛数据的保存路径
            dataset_folder = self.dataset_name.split('.')[0]  # 移除文件扩展名
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_sa.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_metrics(self, time_elapsed):
            # 计算优化的各项指标
            best_score = self.best_score
            iterations = len(self.convergence_data)
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)
            fitness_evaluations = None  # SA 没有适应度函数评估次数
            average_improvement = self.total_improvement / iterations if iterations > 0 else 0

            # 创建字典存储所有指标
            metrics = {
                'Best Score': best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': self.best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Fitness Evaluations': fitness_evaluations,  # 适应度函数评估次数为 None
                'Convergence Speed (Stable Iterations)': self.stable_iterations
            }

            # 动态生成保存路径
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)

            # 保存文件名中包含算法名称
            file_name = f"comparison_performance_sa.csv"
            file_path = os.path.join(dataset_folder, file_name)

            # 保存数据为 CSV 文件
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)

            # 打印保存成功信息
            st.success(f"Performance saved to {file_path}")


    class NSGA_II:
        def __init__(self, population_size, num_generations, crossover_rate, mutation_rate, lambda_1, lambda_2,
                     lambda_3, lambda_4, dataset_name):
            self.population_size = population_size
            self.num_generations = num_generations
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.dataset_name = dataset_name
            self.convergence_data = []
            self.best_score = np.inf

        def initialize_population(self):
            # 初始化种群，每个个体的位置表示问题中的位置，例如堆垛位置
            return np.random.randint(0, len(Dki), size=(self.population_size, num_plates))

        def evaluate_objectives(self, population):
            # 评估目标函数1, 2, 3, 4
            objectives = np.zeros((self.population_size, 4))
            for i in range(self.population_size):
                objectives[i, 0] = minimize_stack_movements_and_turnover(population[i], heights.copy(), plates,
                                                                         delivery_times, batches)
                objectives[i, 1] = minimize_outbound_energy_time_with_batch(population[i], plates, heights.copy())
                objectives[i, 2] = maximize_inventory_balance_v2(population[i], plates)
                objectives[i, 3] = maximize_space_utilization_v3(population[i], plates, Dki)
            return objectives

        def non_dominated_sort(self, objectives):
            # 非支配排序
            num_solutions = len(objectives)
            domination_count = np.zeros(num_solutions)
            domination_set = [[] for _ in range(num_solutions)]
            front = [[]]
            rank = np.zeros(num_solutions)

            for i in range(num_solutions):
                for j in range(num_solutions):
                    if i != j:
                        if self.dominates(objectives[i], objectives[j]):
                            domination_set[i].append(j)
                        elif self.dominates(objectives[j], objectives[i]):
                            domination_count[i] += 1
                if domination_count[i] == 0:
                    rank[i] = 0
                    front[0].append(i)

            i = 0
            while len(front[i]) > 0:
                next_front = []
                for p in front[i]:
                    for q in domination_set[p]:
                        domination_count[q] -= 1
                        if domination_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                front.append(next_front)

            return front[:-1], rank

        def dominates(self, obj1, obj2):
            # 检查 obj1 是否支配 obj2
            return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

        def crowding_distance(self, front, objectives):
            # 计算拥挤距离
            distance = np.zeros(len(front))
            for i in range(objectives.shape[1]):
                sorted_indices = np.argsort(objectives[front, i])
                distance[sorted_indices[0]] = np.inf
                distance[sorted_indices[-1]] = np.inf
                for j in range(1, len(front) - 1):
                    distance[sorted_indices[j]] += (objectives[front[sorted_indices[j + 1]], i] - objectives[
                        front[sorted_indices[j - 1]], i]) / (objectives[front[sorted_indices[-1]], i] - objectives[
                        front[sorted_indices[0]], i])
            return distance

        def select(self, population, objectives):
            # 非支配排序
            fronts, rank = self.non_dominated_sort(objectives)

            # 计算拥挤度距离
            distance = np.zeros(self.population_size)
            for front in fronts:
                distance[front] = self.crowding_distance(front, objectives)

            # 基于排名和拥挤距离选择个体
            selected_population = []
            for front in fronts:
                if len(selected_population) + len(front) > self.population_size:
                    front_sorted_by_distance = np.argsort(-distance[front])
                    selected_population.extend([population[front[i]] for i in front_sorted_by_distance[
                                                                              :self.population_size - len(
                                                                                  selected_population)]])
                    break
                else:
                    selected_population.extend([population[i] for i in front])

            return np.array(selected_population)

        def crossover(self, parent1, parent2):
            # 单点交叉
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            else:
                child1, child2 = parent1, parent2
            return child1, child2

        def mutate(self, individual):
            # 随机变异
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(len(individual))
                individual[mutation_point] = np.random.randint(0, len(Dki))
            return individual

        def evolve(self):
            population = self.initialize_population()

            for generation in range(self.num_generations):
                population_objectives = self.evaluate_objectives(population)

                # 假设第一个目标函数的最小值是最佳得分
                best_score = np.min(population_objectives[:, 0])
                self.convergence_data.append([generation + 1, best_score])

                # 控制台打印每代的最佳得分
                print(f"Generation {generation + 1}: Best Score = {best_score}")

                # 选择下一代
                population = self.select(population, population_objectives)

                # 交叉与变异
                new_population = []
                for i in range(0, self.population_size, 2):
                    parent1, parent2 = population[i], population[i + 1]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.append(child1)
                    new_population.append(child2)

                population = np.array(new_population)

                # 每代更新收敛曲线
                self.update_convergence_plot(generation + 1)

            # 保存最终种群和目标值
            self.save_pareto_solutions(population, population_objectives)
            self.calculate_and_save_metrics(population_objectives)

            # 返回最终种群、Pareto前沿和目标
            return population, population_objectives

        def save_pareto_solutions(self, population, objectives):
            # 根据数据集名称生成路径
            dataset_folder = self.dataset_name.split('.')[0]
            pareto_dir = os.path.join("../result/pareto_solutions", dataset_folder)
            os.makedirs(pareto_dir, exist_ok=True)

            # 保存文件名格式：pareto_solutions_nsga_ii.csv
            pareto_file = os.path.join(pareto_dir, f"pareto_solutions_nsga_ii.csv")

            # 保存 Pareto 解集
            pareto_data = []
            for i in range(len(population)):
                pareto_data.append({
                    'Movement Penalty': objectives[i, 0],
                    'Energy Penalty': objectives[i, 1],
                    'Balance Penalty': objectives[i, 2],
                    'Space Utilization': objectives[i, 3],
                    'Position': population[i]
                })
            pareto_df = pd.DataFrame(pareto_data)
            pareto_df.to_csv(pareto_file, index=False)
            print(f"Pareto solutions saved to {pareto_file}")

        def calculate_and_save_metrics(self, objectives):
            """计算 S 指标、IGD 指标、多样性 Δ' 指标和 DPO 指标并保存"""
            dataset_folder = self.dataset_name.split('.')[0]
            metrics_dir = os.path.join("../result/comparison_metrics", dataset_folder)
            os.makedirs(metrics_dir, exist_ok=True)

            n = len(objectives)
            if n < 2:
                print("解集太小，无法计算指标。")
                return

            # 计算均匀性 S 指标
            distances = np.linalg.norm(objectives[:, np.newaxis] - objectives, axis=2)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            mean_min_distance = np.mean(min_distances)
            S = np.sqrt(np.sum((min_distances - mean_min_distance) ** 2) / (n - 1))

            # 计算 IGD 指标
            reference_points = objectives
            igd_values = np.min(np.linalg.norm(reference_points[:, np.newaxis] - objectives, axis=2), axis=1)
            IGD = np.mean(igd_values)

            # 计算多样性 Δ' 指标
            max_distance = np.max(distances)
            Δ_prime = np.sum(min_distances) / (len(min_distances) * mean_min_distance) if max_distance > 0 else 0

            # 计算 DPO 指标
            DPO = np.mean(min_distances)

            # 保存计算后的指标
            metrics_file = os.path.join(metrics_dir, f"comparison_metrics_nsga_ii.csv")
            metrics_data = {
                'S Index': S,
                'IGD Index': IGD,
                'Δ\' Index': Δ_prime,
                'DPO Index': DPO
            }
            metrics_df = pd.DataFrame([metrics_data])
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Comparison metrics saved to {metrics_file}")

        def update_convergence_plot(self, current_generation):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Generations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Generation {current_generation}, Best Score {self.best_score}')
            plt.legend()
            convergence_plot_placeholder.pyplot(plt)

            # 根据数据集名称动态生成收敛数据的保存路径
            dataset_folder = self.dataset_name.split('.')[0]
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_ga.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    if selected_algorithm == "PSO (Particle Swarm Optimization)":
        # Initialize and run PSO_with_Batch
        pso_with_batch = PSO_with_Batch(num_particles=30, num_positions=len(Dki),
                                        w=w, c1=c1, c2=c2, max_iter=max_iter,
                                        lambda_1=lambda_1, lambda_2=lambda_2,
                                        lambda_3=lambda_3, lambda_4=lambda_4,
                                        dataset_name=selected_dataset if data_choice == "Use system dataset" else uploaded_file.name)

        # Start optimization
        pso_with_batch.optimize()

        # 获取最优解并将其映射到df
        final_positions_with_batch = pso_with_batch.gbest_position
        final_x = []
        final_y = []
        for i, position in enumerate(final_positions_with_batch):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_positions_with_batch
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Particle Swarm Optimization completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                         'final_stack_distribution_plates_pso.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_pso.csv')
        df.to_csv(final_stack_distribution_path, index=False)

        # st.success(f"Stacking statistics saved to {output_file_heights}")

    elif selected_algorithm == "GA (Genetic Algorithm)":
        ga_with_batch = GA_with_Batch(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            generations=generations,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4,
            num_positions=len(Dki),
            dataset_name=selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )

        ga_with_batch.optimize()

        # 获取最优解并将其映射到df
        final_positions_with_batch = ga_with_batch.best_individual
        final_x = []
        final_y = []
        for i, position in enumerate(final_positions_with_batch):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_positions_with_batch
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Genetic Algorithm completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                         'final_stack_distribution_plates_ga.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_ga.csv')
        df.to_csv(final_stack_distribution_path, index=False)

        # st.success(f"Stacking statistics saved to {output_file_heights}")


    elif selected_algorithm == "SA (Simulated Annealing)":
        # Initialize and run SA_with_Batch
        sa_with_batch = SA_with_Batch(
            initial_temperature=initial_temperature, cooling_rate=cooling_rate,
            min_temperature=min_temperature, max_iterations=max_iterations_sa,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4,
            num_positions=len(Dki),
            dataset_name=selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )
        best_position_sa, best_score_sa = sa_with_batch.optimize()
        final_x = []
        final_y = []
        for i, position in enumerate(best_position_sa):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]  # 获取该位置的具体坐标
            final_x.append(x)
            final_y.append(y)

        # 确保生成 'Final Area' 列，并将最优解的位置信息保存
        df['Final Area'] = best_position_sa
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Simulated Annealing optimization completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                     'final_stack_distribution_plates_sa.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                             'final_stack_distribution_height_sa.csv')
        df.to_csv(final_stack_distribution_path, index=False)

        # st.success(f"Stacking statistics saved to {output_file_heights}")

    elif selected_algorithm == "PSO + SA (Hybrid Optimization)":
        # 初始化 PSO + SA 优化器
        pso_sa_optimizer = PSO_SA_Optimizer(
            num_particles=30,
            num_positions=len(Dki),
            w=w,
            c1=c1,
            c2=c2,
            max_iter_pso=max_iter_pso,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            max_iterations_sa=max_iterations_sa,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            dataset_name = selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )

        # 开始优化
        best_position_sa, best_score_sa = pso_sa_optimizer.optimize()

        # 显示优化结果
        st.write(f"Optimization complete. Best score: {best_score_sa}")

        # 正确使用 best_position_sa 而不是 best_position
        final_x = []
        final_y = []
        for i, position in enumerate(best_position_sa):  # 使用 best_position_sa 而不是 best_position
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]  # 获取该位置的具体坐标
            final_x.append(x)
            final_y.append(y)

        # 确保生成 'Final Area' 列，并将最优解的位置信息保存
        df['Final Area'] = best_position_sa
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("PSO SA Hybrid Optimization completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                         'final_stack_distribution_plates_psosa.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_psosa.csv')
        df.to_csv(final_stack_distribution_path, index=False)

    elif selected_algorithm == "ACO (Ant Colony Optimization)":
        # Initialize and run ACO_with_Batch
        aco_with_batch = ACO_with_Batch(
            num_ants=num_ants,  # 使用用户输入的蚂蚁数量
            num_positions=len(Dki),
            alpha=alpha,  # 使用用户输入的 alpha 参数
            beta=beta,  # 使用用户输入的 beta 参数
            evaporation_rate=evaporation_rate,  # 使用用户输入的蒸发率
            q=q,  # 使用用户输入的信息素常数 Q
            max_iter=max_iter,  # 使用用户输入的最大迭代次数
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4,
            dataset_name = selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )

        aco_with_batch.optimize()

        # 获取最优解并将其映射到df
        final_positions_with_batch = aco_with_batch.best_position
        final_x = []
        final_y = []
        for i, position in enumerate(final_positions_with_batch):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_positions_with_batch
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Ant Colony Optimization completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                         'final_stack_distribution_plates_aco.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_aco.csv')
        df.to_csv(final_stack_distribution_path, index=False)


    # 在优化中选择差分进化算法
    elif selected_algorithm == "DE (Differential Evolution)":
        # Initialize and run DE_with_Batch
        de_with_batch = DE_with_Batch(
            pop_size=pop_size,
            num_positions=len(Dki),
            F=F,
            CR=CR,
            max_iter=max_iter,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            dataset_name = selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )
        de_with_batch.optimize()

        # 获取最优解并将其映射到df
        final_positions_with_batch = de_with_batch.best_position
        final_x = []
        final_y = []
        for i, position in enumerate(final_positions_with_batch):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_positions_with_batch
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Differential Evolution completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                         'final_stack_distribution_plates_de.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_de.csv')
        df.to_csv(final_stack_distribution_path, index=False)

    elif selected_algorithm == "CoEA (Co-Evolutionary Algorithm)":
        # 初始化并运行 CoEA_with_Batch
        coea_with_batch = CoEA_with_Batch(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            generations=generations,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            num_positions=len(Dki),
            dataset_name=selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )

        # 开始优化
        coea_with_batch.optimize()

        # 获取最优解并将其映射到 df
        final_positions_with_batch = coea_with_batch.best_individual
        final_x = []
        final_y = []
        for i, position in enumerate(final_positions_with_batch):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_positions_with_batch
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Co-Evolutionary Algorithm completed! You can now visualize the results.")

        # 创建高度和钢板计数字典
        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                     'final_stack_distribution_plates_coea.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在 layout 中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

        # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_coea.csv')
        df.to_csv(final_stack_distribution_path, index=False)

    elif selected_algorithm == "EDA (Estimation of Distribution Algorithm)":
        # Initialize and run EDA_with_Batch
        eda_with_batch = EDA_with_Batch(
            pop_size=pop_size,
            num_positions=len(Dki),
            max_iter=max_iter,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            lambda_4=lambda_4,
            dataset_name = selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )
        eda_with_batch.optimize()

        # 获取最优解并将其映射到df
        final_positions_with_batch = eda_with_batch.best_position
        final_x = []
        final_y = []
        for i, position in enumerate(final_positions_with_batch):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_positions_with_batch
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("Estimation of Distribution Algorithm completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                         'final_stack_distribution_plates_eda.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True
        # st.success("Stacking optimization completed！You can now visualize the results.")

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_eda.csv')
        df.to_csv(final_stack_distribution_path, index=False)

    elif selected_algorithm == "NSGA-II (Non-Dominated Sorting Genetic Algorithm II)":
        # Initialize and run NSGA-II
        nsga_ii = NSGA_II(
            population_size=population_size,
            num_generations=num_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4,
            dataset_name=selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )

        # Evolve the population
        final_population, final_objectives = nsga_ii.evolve()

        # 提取最优解中的位置信息
        best_position_nsga = final_population[np.argmin(final_objectives[:, 0])]  # 假设目标1为主要目标
        final_x = []
        final_y = []

        # 获取堆垛结果的具体坐标
        for i, position in enumerate(best_position_nsga):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]  # 获取该位置的具体坐标
            final_x.append(x)
            final_y.append(y)

        # 确保生成 'Final Area' 列，并将最优解的位置信息保存
        df['Final Area'] = best_position_nsga
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'result/final_stack_distribution/final_stack_distribution_nsga.csv'
        df.to_csv(output_file_plates_with_batch, index=False)

        st.success("NSGA-II optimization completed! You can now visualize the results.")

        heights_dict = {}
        df['Stacking Start Height'] = 0.0
        df['Stacking Height'] = 0.0

        # 计算每个垛位的堆垛高度
        for i in range(len(df)):
            area = df.loc[i, 'Final Area']
            x = df.loc[i, 'Final X']
            y = df.loc[i, 'Final Y']
            key = (area, x, y)
            current_height = heights_dict.get(key, 0.0)
            df.loc[i, 'Stacking Start Height'] = current_height
            df.loc[i, 'Stacking Height'] = current_height + df.loc[i, 'Thickness']
            heights_dict[key] = df.loc[i, 'Stacking Height']

        # 保存最终堆垛结果
        output_file_plates_with_batch = os.path.join(output_dir_base, 'final_stack_distribution_plates',
                                                     'final_stack_distribution_plates_nsga.csv')
        df.to_csv(output_file_plates_with_batch, index=False)

        # 设置 session state，允许可视化
        st.session_state['optimization_done'] = True

        # 生成堆垛结果的统计表
        st.write("### Final Stack Distribution Table")

        # 读取 final_stack_distribution_plates.csv 文件
        df = pd.read_csv(output_file_plates_with_batch)

        # 初始化用于统计的字典
        height_dict = {}
        plate_count_dict = {}

        layout = {
            0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第0库区的垛位
            1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],  # 第1库区
            2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 第2库区
            3: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第3库区
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 第4库区
            5: [(0, 0), (0, 1), (1, 0), (1, 1)]  # 第5库区
        }

        # 初始化每个库区的垛位高度和钢板计数
        for area in layout.keys():
            for pos in layout[area]:
                height_dict[(area, pos[0], pos[1])] = 0.0
                plate_count_dict[(area, pos[0], pos[1])] = 0


        # 检查库区和垛位是否在layout中
        def is_valid_position(area, x, y):
            return (area in layout) and ((int(x), int(y)) in layout[area])


        # 使用已有的 Stacking Height，而不是累加 Thickness
        for index, row in df.iterrows():
            area = row['Final Area']
            x = row['Final X']
            y = row['Final Y']
            stacking_height = row['Stacking Height']  # 使用已计算的堆垛高度

            # 确保 X 和 Y 为整数
            x = int(x)
            y = int(y)

            if is_valid_position(area, x, y):
                # 更新该垛位的堆垛高度
                height_dict[(area, x, y)] = stacking_height

                # 更新该垛位的钢板数量
                plate_count_dict[(area, x, y)] += 1
            else:
                print(f"Warning: Invalid position ({area}, {x}, {y}) in row {index}")

        # 初始化列表用于存储最终结果
        results = []

        # 填充每个库区的垛位高度和钢板数量
        for area, positions in layout.items():
            total_plates = 0
            heights = []

            for pos in positions:
                height = height_dict[(area, pos[0], pos[1])]
                heights.append(height)
                total_plates += plate_count_dict[(area, pos[0], pos[1])]

            # 计算平均高度
            average_height = np.mean(heights)

            # 记录每个库区的堆垛信息
            result_entry = {
                'Area': area,
                'Total Plates': total_plates,
                'Average Height': average_height
            }

            # 记录每个垛位的高度
            for i, pos in enumerate(positions):
                result_entry[f'Position {i + 1}'] = height_dict[(area, pos[0], pos[1])]

            results.append(result_entry)

        # 将结果转换为 DataFrame
        result_df = pd.DataFrame(results)

        # 显示统计表
        st.write("Stacking Distribution Statistics Table:")
        st.dataframe(result_df)

        # 保存计算后的数据
        final_stack_distribution_path = os.path.join(output_dir_base, 'final_stack_distribution_height',
                                                     'final_stack_distribution_height_nsga.csv')
        df.to_csv(final_stack_distribution_path, index=False)

        # 更新收敛曲线
        st.write("### Convergence Curve")
        nsga_ii.update_convergence_plot(num_generations)

        st.success(f"NSGA-II optimization completed! Final stack distribution saved to {output_file_plates_with_batch}")


