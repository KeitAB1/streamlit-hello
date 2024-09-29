import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
from deap import base, creator, tools

from concurrent.futures import ThreadPoolExecutor

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



# 单目标优化和多目标优化算法分开模块
optimization_mode = st.radio("Select Optimization Mode", ("Single-Objective", "Multi-Objective"))

# 单目标优化算法选择
if optimization_mode == "Single-Objective":
# 算法选择放在侧边栏
    with st.sidebar:
        algorithms = ["SA (Simulated Annealing)", "GA (Genetic Algorithm)", "PSO (Particle Swarm Optimization)",
                      "PSO + SA (Hybrid Optimization)", "ACO (Ant Colony Optimization)", "DE (Differential Evolution)",
                      "CoEA (Co-Evolutionary Algorithm)", "EDA (Estimation of Distribution Algorithm)"]
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
# 多目标优化算法选择
elif optimization_mode == "Multi-Objective":
    with st.sidebar:
        multi_objective_algorithms = ["NSGA-II (Non-Dominated Sorting Genetic Algorithm II)",
                                      "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)",
                                      "MOPSO (Multi-Objective Particle Swarm Optimization)"]
        selected_multi_algo = st.selectbox("Select Multi-Objective Optimization Algorithm", multi_objective_algorithms)

        # 显示不同算法的相关参数
        if selected_multi_algo == "NSGA-II (Non-Dominated Sorting Genetic Algorithm II)":
            st.subheader("NSGA-II Parameters")
            generations_nsga = st.number_input("Generations", 1, 1000, 200)
            population_size_nsga = st.number_input("Population Size", 50, 500, 100)
            crossover_rate_nsga = st.slider("Crossover Rate", 0.0, 1.0, 0.9)
            mutation_rate_nsga = st.slider("Mutation Rate", 0.0, 1.0, 0.05)

        elif selected_multi_algo == "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)":
            st.subheader("MOEA/D Parameters")
            generations_moead = st.number_input("Generations", 1, 1000, 200)
            population_size_moead = st.number_input("Population Size", 50, 500, 100)
            neighbor_size_moead = st.number_input("Neighborhood Size", 1, 50, 10)
            crossover_rate_moead = st.slider("Crossover Rate", 0.0, 1.0, 0.9)
            mutation_rate_moead = st.slider("Mutation Rate", 0.0, 1.0, 0.05)

        elif selected_multi_algo == "MOPSO (Multi-Objective Particle Swarm Optimization)":
            st.subheader("MOPSO Parameters")
            max_iter_mopso = st.number_input("Max Iterations", 1, 1000, 200)
            population_size_mopso = st.number_input("Population Size", 50, 500, 100)
            inertia_weight_mopso = st.slider("Inertia Weight", 0.0, 1.0, 0.5)
            cognitive_coeff_mopso = st.slider("Cognitive Coefficient", 0.0, 4.0, 2.0)
            social_coeff_mopso = st.slider("Social Coefficient", 0.0, 4.0, 2.0)

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





    def run_nsga2(population_size, generations, crossover_rate, mutation_rate):
        # 定义单目标最小化问题
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # 两个目标最小化
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # 定义个体生成和评价
        toolbox.register("attr_float", random.uniform, 0, 1)  # 解的范围在0到1之间
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)  # 每个解包含2个变量
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 定义目标函数
        def evaluate(individual):
            x, y = individual
            f1 = x ** 2 + y ** 2
            f2 = (x - 1) ** 2 + y ** 2
            return f1, f2

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0,
                         indpb=1.0 / len(toolbox.individual()))
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=population_size)

        # 初始化种群的适应度
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 执行进化
        for gen in range(generations):
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # 交叉与变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 重新评估个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 选择下一代
            population = toolbox.select(population + offspring, population_size)

        return population

    def run_moead(population_size, generations, neighbor_size, crossover_rate, mutation_rate):
        # 定义MOEA/D的权重向量
        weights = [np.array([i / (population_size - 1), (population_size - 1 - i) / (population_size - 1)]) for i in
                   range(population_size)]

        # 初始化种群
        population = [np.random.rand(2) for _ in range(population_size)]

        # 初始化邻域
        neighbors = [random.sample(range(population_size), neighbor_size) for _ in range(population_size)]

        # 定义目标函数
        def evaluate(individual):
            x, y = individual
            f1 = x ** 2 + y ** 2
            f2 = (x - 1) ** 2 + y ** 2
            return f1, f2

        # 计算每个个体的适应度
        fitness = [evaluate(ind) for ind in population]

        # 执行进化
        for gen in range(generations):
            for i, individual in enumerate(population):
                neighbor_indices = neighbors[i]

                # 选择父代个体
                parent1, parent2 = random.sample([population[idx] for idx in neighbor_indices], 2)

                # 交叉与变异
                if random.random() < crossover_rate:
                    child = tools.cxBlend(parent1, parent2, alpha=0.5)
                else:
                    child = parent1

                if random.random() < mutation_rate:
                    tools.mutGaussian(child, mu=0, sigma=0.1, indpb=0.2)

                # 计算子代的适应度
                child_fitness = evaluate(child)

                # 更新邻居中的适应度
                for neighbor_idx in neighbor_indices:
                    neighbor_fitness = fitness[neighbor_idx]
                    if all(child_fitness[i] < neighbor_fitness[i] for i in range(2)):
                        population[neighbor_idx] = child
                        fitness[neighbor_idx] = child_fitness

        return population


    def run_mopso(population_size, max_iterations, inertia_weight, cognitive_coeff, social_coeff):
        # 初始化种群
        population = [np.random.rand(2) for _ in range(population_size)]
        velocities = [np.random.rand(2) for _ in range(population_size)]
        pbest = population.copy()
        pbest_fitness = [evaluate(ind) for ind in population]

        gbest_archive = []

        def evaluate(individual):
            x, y = individual
            f1 = x ** 2 + y ** 2
            f2 = (x - 1) ** 2 + y ** 2
            return f1, f2

        def update_velocity(velocity, position, pbest_position, gbest_position):
            cognitive_component = cognitive_coeff * np.random.rand() * (pbest_position - position)
            social_component = social_coeff * np.random.rand() * (gbest_position - position)
            return inertia_weight * velocity + cognitive_component + social_component

        def dominates(ind1, ind2):
            return all(ind1[i] <= ind2[i] for i in range(2)) and any(ind1[i] < ind2[i] for i in range(2))

        # 执行迭代
        for iteration in range(max_iterations):
            for i in range(population_size):
                velocities[i] = update_velocity(velocities[i], population[i], pbest[i], random.choice(gbest_archive))
                population[i] = population[i] + velocities[i]

                current_fitness = evaluate(population[i])

                # 更新个体最优解
                if dominates(current_fitness, pbest_fitness[i]):
                    pbest[i] = population[i]
                    pbest_fitness[i] = current_fitness

                # 更新全局最优解
                dominated_solutions = [ind for ind in gbest_archive if dominates(current_fitness, evaluate(ind))]
                gbest_archive = [ind for ind in gbest_archive if not dominates(evaluate(ind), current_fitness)]
                if not dominated_solutions:
                    gbest_archive.append(population[i])

        return gbest_archive


    if optimization_mode == "Multi-Objective":
        with st.sidebar:
            multi_objective_algorithms = ["NSGA-II (Non-Dominated Sorting Genetic Algorithm II)",
                                          "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)",
                                          "MOPSO (Multi-Objective Particle Swarm Optimization)"]
            # 通过传递唯一的 `key` 解决 DuplicateWidgetID 错误
            selected_multi_algo = st.selectbox(
                "Select Multi-Objective Optimization Algorithm",
                multi_objective_algorithms,
                key="multi_objective_algo_selection"
            )

        if selected_multi_algo == "NSGA-II (Non-Dominated Sorting Genetic Algorithm II)":
            population = run_nsga2(population_size_nsga, generations_nsga, crossover_rate_nsga, mutation_rate_nsga)
            st.write("NSGA-II Optimization Completed")

        elif selected_multi_algo == "MOEA/D (Multi-Objective Evolutionary Algorithm Based on Decomposition)":
            population = run_moead(population_size_moead, generations_moead, neighbor_size_moead, crossover_rate_moead,
                                   mutation_rate_moead)
            st.write("MOEA/D Optimization Completed")

        elif selected_multi_algo == "MOPSO (Multi-Objective Particle Swarm Optimization)":
            archive = run_mopso(population_size_mopso, max_iter_mopso, inertia_weight_mopso, cognitive_coeff_mopso,
                                social_coeff_mopso)
            st.write("MOPSO Optimization Completed")

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
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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

