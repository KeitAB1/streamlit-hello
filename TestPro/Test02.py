import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from auxiliary import InterfaceLayout as il

# from optimization_functions import  minimize_stack_movements_and_turnover, minimize_outbound_energy_time_with_batch, \
#     maximize_inventory_balance_v2, maximize_space_utilization_v3
#
# from optimizers.pso_optimizer import PSO_with_Batch
# from optimizers.ga_optimizer import GA_with_Batch
# from optimizers.sa_optimizer import SA_with_Batch


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

        # 新增 Pareto 优化选项
        st.write("### Multi-Objective Optimization (Pareto Front)")
        use_pareto = st.checkbox("Enable Pareto Optimization", value=True)
        if use_pareto:
            # 如果启用 Pareto 优化，可能会提供每个目标函数的选择或权重配置
            st.write("Adjust the weights for each objective (if needed):")
            lambda_1 = st.number_input("Lambda 1 (Movement Penalty Weight)", value=1.0)
            lambda_2 = st.number_input("Lambda 2 (Energy/Time Penalty Weight)", value=1.0)
            lambda_3 = st.number_input("Lambda 3 (Balance Penalty Weight)", value=1.0)
            lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

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


    # 定义PSO优化算法
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
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.convergence_data = []  # 初始化收敛数据
            self.dataset_name = dataset_name  # 增加数据集名称属性

        def optimize(self):
            global heights
            for iteration in range(self.max_iter):
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
                        self.gbest_score = current_score
                        self.gbest_position = particle.position.copy()

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


    class GA_with_Batch:

        cache = {}  # 适应度缓存，避免重复计算

        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions, dataset_name):
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.generations = generations
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.num_positions = num_positions
            self.population = [np.random.randint(0, num_positions, size=num_plates) for _ in range(population_size)]
            self.best_individual = None
            self.best_score = np.inf
            self.convergence_data = []
            self.dataset_name = dataset_name  # 增加数据集名称属性

        def fitness(self, individual):
            # 将个体转化为元组以便在字典中使用
            individual_tuple = tuple(individual)

            # 如果在缓存中，直接返回缓存结果
            if individual_tuple in self.cache:
                return self.cache[individual_tuple]

            global heights
            temp_heights = heights.copy()

            combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                individual, temp_heights, plates, delivery_times, batches)
            energy_time_penalty = minimize_outbound_energy_time_with_batch(individual, plates, temp_heights)
            balance_penalty = maximize_inventory_balance_v2(individual, plates)
            space_utilization = maximize_space_utilization_v3(individual, plates, Dki)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            # 将结果存入缓存
            self.cache[individual_tuple] = score
            return score

        def select(self):
            # 使用线程池并行计算适应度
            with ThreadPoolExecutor() as executor:
                fitness_scores = list(executor.map(self.fitness, self.population))

            fitness_scores = np.array(fitness_scores)
            probabilities = np.exp(-fitness_scores / np.sum(fitness_scores))
            probabilities /= probabilities.sum()  # 正规化以得到概率分布
            selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
            return [self.population[i] for i in selected_indices]

        def crossover(self, parent1, parent2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, num_plates)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                return child1, child2
            return parent1, parent2

        def mutate(self, individual):
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = np.random.randint(0, self.num_positions)
            return individual

        def optimize(self):
            for generation in range(self.generations):
                new_population = []
                selected_population = self.select()

                # 使用线程池并行执行交叉和突变
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for i in range(0, self.population_size, 2):
                        parent1 = selected_population[i]
                        parent2 = selected_population[min(i + 1, self.population_size - 1)]
                        futures.append(executor.submit(self.crossover, parent1, parent2))

                    for future in futures:
                        child1, child2 = future.result()
                        new_population.append(self.mutate(child1))
                        new_population.append(self.mutate(child2))

                self.population = new_population
                best_individual_gen = min(self.population, key=self.fitness)
                best_score_gen = self.fitness(best_individual_gen)

                if best_score_gen < self.best_score:
                    self.best_score = best_score_gen
                    self.best_individual = best_individual_gen.copy()

                self.convergence_data.append([generation + 1, self.best_score])
                self.update_convergence_plot(generation + 1)

                print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

            self.update_final_heights()

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.best_individual):
                area = position
                heights[area] += plates[plate_idx, 2]

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
            dataset_folder = self.dataset_name.split('.')[0]  # 移除文件扩展名
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_ga.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


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
            self.num_positions = num_positions
            self.dataset_name = dataset_name

            # 初始化位置和 Pareto 最优解集
            self.best_position = None
            self.best_score = np.inf
            self.pareto_front = []  # Pareto 最优解集
            self.convergence_data = []

        def normalize_objectives(self, objectives, min_values, max_values):
            """ 将目标函数的值归一化到 [0, 100] 之间 """
            normalized = [
                100 * (obj - min_val) / (max_val - min_val) if max_val > min_val else 0
                for obj, min_val, max_val in zip(objectives, min_values, max_values)
            ]
            return normalized

        def evaluate(self, position, heights):
            temp_heights = heights.copy()  # 确保 temp_heights 被正确初始化

            combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                position, temp_heights, plates, delivery_times, batches)
            energy_time_penalty = minimize_outbound_energy_time_with_batch(position, plates, temp_heights)
            balance_penalty = maximize_inventory_balance_v2(position, plates)
            space_utilization = maximize_space_utilization_v3(position, plates, Dki)

            # 获取目标函数最小和最大值，用于归一化
            min_values = [0, 0, 0, 0]  # 目标函数的最小值
            max_values = [1e7, 1e6, 1e18, 100]  # 目标函数的最大可能值

            objectives = [
                combined_movement_turnover_penalty,
                energy_time_penalty,
                balance_penalty,
                space_utilization
            ]

            # 将目标函数值归一化
            normalized_objectives = self.normalize_objectives(objectives, min_values, max_values)

            return normalized_objectives

        def dominates(self, solution1, solution2):
            return all(x <= y for x, y in zip(solution1, solution2)) and any(
                x < y for x, y in zip(solution1, solution2))

        def update_pareto_front(self, position, objectives):
            non_dominated = []
            dominated = False
            for existing_solution in self.pareto_front:
                if self.dominates(existing_solution['objectives'], objectives):
                    dominated = True
                    break
                elif self.dominates(objectives, existing_solution['objectives']):
                    continue
                else:
                    non_dominated.append(existing_solution)

            if not dominated:
                non_dominated.append({'position': position.copy(), 'objectives': objectives})

            self.pareto_front = non_dominated

        def save_pareto_solutions(self):
            """保存 Pareto 解集"""
            dataset_folder = self.dataset_name.split('.')[0]
            pareto_dir = os.path.join("../result/pareto_solutions", dataset_folder)
            os.makedirs(pareto_dir, exist_ok=True)
            pareto_file = os.path.join(pareto_dir, f"pareto_solutions_sa.csv")

            pareto_data = []
            for solution in self.pareto_front:
                pareto_data.append({
                    'Movement Penalty': solution['objectives'][0],
                    'Energy Penalty': solution['objectives'][1],
                    'Balance Penalty': solution['objectives'][2],
                    'Space Utilization': solution['objectives'][3],
                    'Position': solution['position']
                })

            pareto_df = pd.DataFrame(pareto_data)
            pareto_df.to_csv(pareto_file, index=False)
            print(f"Pareto solutions saved to {pareto_file}")

        def optimize(self):
            global heights  # 确保 heights 被全局定义并传递
            self.optimize_from_position(np.random.randint(0, self.num_positions, size=num_plates), heights)
            self.save_pareto_solutions()

            return self.pareto_front

        def calculate_and_save_metrics(self):
            """计算均匀性S指标，IGD指标，多样性Δ'指标和DPO指标并保存"""
            # 加载已保存的Pareto解集
            dataset_folder = self.dataset_name.split('.')[0]
            pareto_dir = os.path.join("../result/pareto_solutions", dataset_folder)
            pareto_file = os.path.join(pareto_dir, f"pareto_solutions_sa.csv")
            pareto_df = pd.read_csv(pareto_file)

            # 获取目标函数的值
            objectives = pareto_df[
                ['Movement Penalty', 'Energy Penalty', 'Balance Penalty', 'Space Utilization']].values
            n = len(objectives)

            if n < 2:
                print("解集太小，无法计算指标。")
                return

            # 计算均匀性 S 指标
            distances = np.linalg.norm(objectives[:, np.newaxis] - objectives, axis=2)
            np.fill_diagonal(distances, np.inf)  # 忽略对角线的自我距离
            min_distances = np.min(distances, axis=1)
            mean_min_distance = np.mean(min_distances)

            # 计算均匀性 S 指标
            S = np.sqrt(np.sum((min_distances - mean_min_distance) ** 2) / (n - 1))

            # 计算 IGD 指标
            reference_points = objectives  # 如果没有真实的 Pareto 前沿数据，用当前解集代替
            igd_values = np.min(np.linalg.norm(reference_points[:, np.newaxis] - objectives, axis=2), axis=1)
            IGD = np.mean(igd_values)

            # 计算多样性 Δ' 指标
            max_distance = np.max(distances)
            if max_distance > 0:
                Δ_prime = np.sum(min_distances) / (len(min_distances) * mean_min_distance)
            else:
                Δ_prime = 0  # 避免除以 0

            # 计算 DPO 指标
            DPO = np.mean(min_distances)

            # 保存计算后的指标
            metrics_dir = os.path.join("../result/comparison_metrics", dataset_folder)
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_file = os.path.join(metrics_dir, f"comparison_metrics_sa.csv")

            metrics_data = {
                'S Index': S,
                'IGD Index': IGD,
                'Δ\' Index': Δ_prime,
                'DPO Index': DPO
            }

            metrics_df = pd.DataFrame([metrics_data])
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Comparison metrics saved to {metrics_file}")

        def optimize_from_position(self, initial_position, heights):
            current_temperature = self.initial_temperature
            current_position = initial_position
            current_objectives = self.evaluate(current_position, heights)

            self.update_pareto_front(current_position, current_objectives)

            for iteration in range(self.max_iterations):
                if current_temperature < self.min_temperature:
                    break

                new_position = current_position.copy()
                random_index = np.random.randint(0, len(current_position))
                new_position[random_index] = np.random.randint(0, self.num_positions)
                new_objectives = self.evaluate(new_position, heights)

                delta = sum(new - curr for new, curr in zip(new_objectives, current_objectives))
                if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                    current_position = new_position
                    current_objectives = new_objectives

                self.update_pareto_front(current_position, current_objectives)

                current_temperature *= self.cooling_rate
                self.convergence_data.append([iteration + 1, current_objectives])

                self.update_convergence_plot(iteration + 1)

                print(f"Iteration {iteration + 1}/{self.max_iterations}, Pareto Front Size: {len(self.pareto_front)}")

            return self.pareto_front

        def update_convergence_plot(self, current_iteration):
            """动态更新收敛曲线"""
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [min(x[1]) for x in self.convergence_data]  # 以最小值作为收敛值的显示

            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Iterations')
            plt.ylabel('Best Score')
            plt.title(
                f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}, Pareto Front Size {len(self.pareto_front)}')
            plt.legend()

            # 使用 Streamlit 的空占位符更新图表
            convergence_plot_placeholder.pyplot(plt)

            # 根据数据集名称动态生成收敛数据的保存路径
            dataset_folder = self.dataset_name.split('.')[0]
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_sa_pareto.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    class PSO_SA_Optimizer:
        def __init__(self, num_particles, num_positions, w, c1, c2, max_iter_pso,
                     initial_temperature, cooling_rate, min_temperature, max_iterations_sa,
                     lambda_1, lambda_2, lambda_3, lambda_4, dataset_name):
            # 初始化PSO参数
            self.num_particles = num_particles
            self.num_positions = num_positions
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.max_iter_pso = max_iter_pso
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4

            # 初始化SA参数
            self.initial_temperature = initial_temperature
            self.cooling_rate = cooling_rate
            self.min_temperature = min_temperature
            self.max_iterations_sa = max_iterations_sa

            # 存储数据集名称，用于动态路径生成
            self.dataset_name = dataset_name

            # 初始化PSO和SA优化器
            self.pso_optimizer = PSO_with_Batch(
                num_particles=num_particles,
                num_positions=num_positions,
                w=w, c1=c1, c2=c2, max_iter=max_iter_pso,
                lambda_1=lambda_1, lambda_2=lambda_2,
                lambda_3=lambda_3, lambda_4=lambda_4,
                dataset_name=dataset_name
            )

            self.sa_optimizer = SA_with_Batch(
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                min_temperature=min_temperature,
                max_iterations=max_iterations_sa,
                lambda_1=lambda_1, lambda_2=lambda_2,
                lambda_3=lambda_3, lambda_4=lambda_4,
                num_positions=num_positions,
                dataset_name=dataset_name
            )

            self.convergence_data_pso_sa = []  # PSO + SA 的单独收敛数据

        def optimize(self):
            # 首先运行PSO优化
            self.pso_optimizer.optimize()

            # 获取PSO的最优解，作为SA的初始解
            initial_position_for_sa = self.pso_optimizer.gbest_position

            # 使用SA在PSO的解基础上进行局部优化
            best_position_sa, best_score_sa = self.sa_optimizer.optimize_from_position(initial_position_for_sa)

            # 将PSO优化的收敛数据存入 PSO + SA 的收敛数据中
            self.convergence_data_pso_sa.extend(self.pso_optimizer.convergence_data)

            # 将SA优化的收敛数据追加到PSO + SA的收敛数据中
            self.convergence_data_pso_sa.extend(self.sa_optimizer.convergence_data)

            # 保存PSO + SA的收敛数据
            self.save_convergence_data_pso_sa()

            # 返回SA优化的最优解和得分
            return best_position_sa, best_score_sa

        def save_convergence_data_pso_sa(self):
            # 根据数据集名称动态生成收敛数据的保存路径
            dataset_folder = self.dataset_name.split('.')[0]  # 移除文件扩展名
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存PSO + SA的收敛数据到新的文件
            convergence_data_df = pd.DataFrame(self.convergence_data_pso_sa, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_psosa.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    class ACO_with_Batch:
        def __init__(self, num_ants, num_positions, alpha, beta, evaporation_rate, q, max_iter, lambda_1, lambda_2,
                     lambda_3, lambda_4, dataset_name):
            self.num_ants = num_ants  # 蚂蚁数量
            self.num_positions = num_positions  # 库区/垛位数量
            self.alpha = alpha  # 信息素重要程度因子
            self.beta = beta  # 启发因子重要程度因子
            self.evaporation_rate = evaporation_rate  # 信息素蒸发速率
            self.q = q  # 信息素强度
            self.max_iter = max_iter  # 最大迭代次数
            self.lambda_1 = lambda_1  # 目标函数中的权重参数
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.pheromone_matrix = np.ones((num_plates, num_positions))  # 信息素矩阵，初始化为1
            self.best_score = np.inf
            self.best_position = None
            self.convergence_data = []  # 用于保存收敛数据
            self.dataset_name = dataset_name  # 增加数据集名称属性

        def optimize(self):
            global heights
            for iteration in range(self.max_iter):
                all_ant_positions = []  # 用于存储每只蚂蚁的路径
                all_ant_scores = []  # 用于存储每只蚂蚁的适应度

                for ant in range(self.num_ants):
                    ant_position = self.construct_solution()  # 为每只蚂蚁构建解
                    all_ant_positions.append(ant_position)

                    # 计算每只蚂蚁的解的得分
                    temp_heights = heights.copy()
                    combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                        ant_position, temp_heights, plates, delivery_times, batches)
                    energy_time_penalty = minimize_outbound_energy_time_with_batch(ant_position, plates, temp_heights)
                    balance_penalty = maximize_inventory_balance_v2(ant_position, plates)
                    space_utilization = maximize_space_utilization_v3(ant_position, plates, Dki)

                    score = (self.lambda_1 * combined_movement_turnover_penalty +
                             self.lambda_2 * energy_time_penalty +
                             self.lambda_3 * balance_penalty -
                             self.lambda_4 * space_utilization)

                    all_ant_scores.append(score)

                    # 更新最优解
                    if score < self.best_score:
                        self.best_score = score
                        self.best_position = ant_position.copy()

                # 信息素更新
                self.update_pheromones(all_ant_positions, all_ant_scores)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)
                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            self.update_final_heights()

        def construct_solution(self):
            # 根据信息素和启发因子为蚂蚁构建解
            solution = []
            for plate_idx in range(num_plates):
                probabilities = self.calculate_transition_probabilities(plate_idx)
                position = np.random.choice(self.num_positions, p=probabilities)
                solution.append(position)
            return np.array(solution)

        def calculate_transition_probabilities(self, plate_idx):
            # 计算从当前钢板到不同堆垛位置的选择概率
            pheromones = self.pheromone_matrix[plate_idx]
            desirability = 1.0 / (np.arange(1, self.num_positions + 1))  # 启发因子

            # 根据当前堆垛高度调整概率，使得堆垛较低的区域更有吸引力
            stack_heights = np.array([heights[pos] for pos in range(self.num_positions)])
            height_bias = 1.0 / (stack_heights + 1)  # 堆垛越低，偏好越高

            probabilities = (pheromones ** self.alpha) * (desirability ** self.beta) * height_bias
            return probabilities / probabilities.sum()  # 归一化概率

        def update_pheromones(self, all_ant_positions, all_ant_scores):
            # 信息素蒸发
            self.pheromone_matrix *= (1 - self.evaporation_rate)

            # 信息素强化
            for ant_idx, score in enumerate(all_ant_scores):
                for plate_idx, position in enumerate(all_ant_positions[ant_idx]):
                    self.pheromone_matrix[plate_idx, position] += self.q / score

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.best_position):
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
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_aco.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    class DE_with_Batch:
        def __init__(self, pop_size, num_positions, F, CR, max_iter, lambda_1, lambda_2, lambda_3, lambda_4,
                     dataset_name):
            self.pop_size = pop_size  # 种群大小
            self.num_positions = num_positions  # 库区/垛位数量
            self.F = F  # 缩放因子 (mutation factor)
            self.CR = CR  # 交叉概率 (crossover rate)
            self.max_iter = max_iter  # 最大迭代次数
            self.lambda_1 = lambda_1  # 目标函数中的权重参数
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.population = np.random.randint(0, num_positions, size=(pop_size, num_plates))  # 随机初始化种群
            self.best_position = None  # 最佳解
            self.best_score = np.inf  # 最优得分
            self.convergence_data = []  # 用于保存收敛数据
            self.dataset_name = dataset_name  # 增加数据集名称属性

        def optimize(self):
            global heights
            for iteration in range(self.max_iter):
                new_population = []
                for i in range(self.pop_size):
                    # 选择三个随机个体
                    indices = list(range(self.pop_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)

                    # 变异操作
                    mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), 0,
                                     self.num_positions - 1).astype(int)

                    # 交叉操作
                    trial = np.copy(self.population[i])
                    for j in range(num_plates):
                        if np.random.rand() < self.CR:
                            trial[j] = mutant[j]

                    # 计算原始个体和新个体的适应度
                    temp_heights_original = heights.copy()
                    temp_heights_trial = heights.copy()

                    original_score = self.calculate_fitness(self.population[i], temp_heights_original)
                    trial_score = self.calculate_fitness(trial, temp_heights_trial)

                    # 选择操作，保留适应度更好的个体
                    if trial_score < original_score:
                        new_population.append(trial)
                        if trial_score < self.best_score:
                            self.best_score = trial_score
                            self.best_position = trial
                    else:
                        new_population.append(self.population[i])

                # 更新种群
                self.population = np.array(new_population)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)

                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            self.update_final_heights()

        def calculate_fitness(self, individual, temp_heights):
            # 计算个体的适应度得分
            combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                individual, temp_heights, plates, delivery_times, batches)
            energy_time_penalty = minimize_outbound_energy_time_with_batch(individual, plates, temp_heights)
            balance_penalty = maximize_inventory_balance_v2(individual, plates)
            space_utilization = maximize_space_utilization_v3(individual, plates, Dki)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            return score

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.best_position):
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
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_de.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    class CoEA_with_Batch:
        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions, dataset_name):
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.generations = generations
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.num_positions = num_positions
            self.dataset_name = dataset_name

            # 创建多个子种群，假设分为两个子种群，CoEA的核心在于各子种群间的交互
            self.subpopulations = [
                [np.random.randint(0, num_positions, size=num_plates) for _ in range(population_size // 2)] for _ in
                range(2)
            ]

            self.best_individual = None
            self.best_score = np.inf
            self.convergence_data = []

        def fitness(self, individual):
            global heights
            temp_heights = heights.copy()

            combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                individual, temp_heights, plates, delivery_times, batches
            )
            energy_time_penalty = minimize_outbound_energy_time_with_batch(individual, plates, temp_heights)
            balance_penalty = maximize_inventory_balance_v2(individual, plates)
            space_utilization = maximize_space_utilization_v3(individual, plates, Dki)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)
            return score

        def select(self, subpopulation):
            fitness_scores = np.array([self.fitness(individual) for individual in subpopulation])
            probabilities = np.exp(-fitness_scores / np.sum(fitness_scores))
            probabilities /= probabilities.sum()
            selected_indices = np.random.choice(len(subpopulation), size=len(subpopulation), p=probabilities)
            return [subpopulation[i] for i in selected_indices]

        def crossover(self, parent1, parent2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, num_plates)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                return child1, child2
            return parent1, parent2

        def mutate(self, individual):
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = np.random.randint(0, self.num_positions)
            return individual

        def optimize(self):
            for generation in range(self.generations):
                for subpopulation in self.subpopulations:
                    new_subpopulation = []
                    selected_subpopulation = self.select(subpopulation)

                    # 交叉与变异
                    for i in range(0, len(subpopulation), 2):
                        parent1 = selected_subpopulation[i]
                        parent2 = selected_subpopulation[min(i + 1, len(subpopulation) - 1)]
                        child1, child2 = self.crossover(parent1, parent2)
                        new_subpopulation.append(self.mutate(child1))
                        new_subpopulation.append(self.mutate(child2))

                    subpopulation[:] = new_subpopulation

                # 通过交换子种群的一部分个体进行协同进化
                self.exchange_subpopulations()

                # 更新全局最优解
                best_individual_gen = min([min(subpop, key=self.fitness) for subpop in self.subpopulations],
                                          key=self.fitness)
                best_score_gen = self.fitness(best_individual_gen)

                if best_score_gen < self.best_score:
                    self.best_score = best_score_gen
                    self.best_individual = best_individual_gen.copy()

                self.convergence_data.append([generation + 1, self.best_score])
                self.update_convergence_plot(generation + 1)

                print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

        def exchange_subpopulations(self):
            # 定义协同进化的交换规则，比如随机交换部分个体
            exchange_size = self.population_size // 10  # 假设交换10%的个体
            for i in range(exchange_size):
                idx1 = np.random.randint(len(self.subpopulations[0]))
                idx2 = np.random.randint(len(self.subpopulations[1]))
                # 交换个体
                self.subpopulations[0][idx1], self.subpopulations[1][idx2] = self.subpopulations[1][idx2], \
                self.subpopulations[0][idx1]

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

            # 保存收敛数据
            dataset_folder = self.dataset_name.split('.')[0]
            convergence_data_dir = os.path.join("../result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_coea.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    class EDA_with_Batch:
        def __init__(self, pop_size, num_positions, max_iter, lambda_1, lambda_2, lambda_3, lambda_4, dataset_name):
            self.pop_size = pop_size  # 种群大小
            self.num_positions = num_positions  # 库区/垛位数量
            self.max_iter = max_iter  # 最大迭代次数
            self.lambda_1 = lambda_1  # 高度相关的权重参数
            self.lambda_2 = lambda_2  # 翻垛相关的权重参数
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.dataset_name = dataset_name  # 数据集名称
            self.population = np.random.randint(0, num_positions, size=(pop_size, num_plates))  # 随机初始化种群
            self.best_position = None  # 最佳解
            self.best_score = np.inf  # 最优得分
            self.convergence_data = []  # 用于保存收敛数据

        def optimize(self):
            global heights
            for iteration in range(self.max_iter):
                # 估计概率分布
                probabilities = self.estimate_distribution()

                # 使用概率分布生成新的种群
                new_population = self.generate_new_population(probabilities)

                # 选择操作：评估新种群并选择表现最好的个体
                for individual in new_population:
                    temp_heights = heights.copy()
                    score = self.calculate_fitness(individual, temp_heights)

                    if score < self.best_score:
                        self.best_score = score
                        self.best_position = individual

                # 更新种群
                self.population = np.copy(new_population)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)
                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            self.update_final_heights()

        def estimate_distribution(self):
            # 估计种群的概率分布
            probabilities = np.zeros((num_plates, self.num_positions))

            # 统计每个位置被选择的频率
            for i in range(num_plates):
                for individual in self.population:
                    probabilities[i, individual[i]] += 1

            # 将频率转换为概率
            probabilities = probabilities / self.pop_size
            return probabilities

        def generate_new_population(self, probabilities):
            # 根据估计的概率分布生成新种群
            new_population = []
            for _ in range(self.pop_size):
                new_individual = []
                for i in range(num_plates):
                    new_position = np.random.choice(self.num_positions, p=probabilities[i])
                    new_individual.append(new_position)
                new_population.append(np.array(new_individual))
            return np.array(new_population)

        def calculate_fitness(self, individual, temp_heights):
            # 计算个体的适应度得分
            combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                individual, temp_heights, plates, delivery_times, batches)
            energy_time_penalty = minimize_outbound_energy_time_with_batch(individual, plates, temp_heights)
            balance_penalty = maximize_inventory_balance_v2(individual, plates)
            space_utilization = maximize_space_utilization_v3(individual, plates, Dki)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            return score

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.best_position):
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
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_eda.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


    def plot_pareto_front(pareto_front):
        objectives = [sol['objectives'] for sol in pareto_front]
        num_objectives = len(objectives[0])

        # 如果有两个或更多目标
        if num_objectives >= 2:
            # 允许用户选择两个目标进行可视化
            st.write("Select two objectives to visualize on the Pareto Front:")
            obj1 = st.selectbox("Objective for X-axis", range(num_objectives),
                                format_func=lambda x: f"Objective {x + 1}")
            obj2 = st.selectbox("Objective for Y-axis", range(num_objectives),
                                format_func=lambda x: f"Objective {x + 1}", index=1)

            # 提取用户选择的两个目标
            x = [obj[obj1] for obj in objectives]
            y = [obj[obj2] for obj in objectives]

            # 绘制散点图
            plt.figure()
            plt.scatter(x, y, label="Pareto Front")
            plt.xlabel(f"Objective {obj1 + 1}")
            plt.ylabel(f"Objective {obj2 + 1}")
            plt.title(f"Pareto Front - Objective {obj1 + 1} vs Objective {obj2 + 1}")
            plt.legend()
            st.pyplot(plt)

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


    elif selected_algorithm == "SA (Simulated Annealing)" and use_pareto:

        sa_with_batch = SA_with_Batch(
            initial_temperature=initial_temperature, cooling_rate=cooling_rate,
            min_temperature=min_temperature, max_iterations=max_iterations_sa,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, lambda_4=lambda_4,
            num_positions=len(Dki),
            dataset_name=selected_dataset if data_choice == "Use system dataset" else uploaded_file.name
        )

        pareto_front = sa_with_batch.optimize()

        # 显示 Pareto 前沿的解集
        st.write("### Pareto Front Solutions:")
        for idx, solution in enumerate(pareto_front):
            st.write(f"Solution {idx + 1}:")
            st.write(f"Objectives: {solution['objectives']}")
            # st.write(f"Position: {solution['position']}")

        st.success("Pareto front solutions saved successfully!")

        # ---- 计算 Pareto 指标并保存 ----
        sa_with_batch.calculate_and_save_metrics()

        # 选择解并将其映射到 df
        selected_solution_idx = st.selectbox("Select a solution to visualize", range(len(pareto_front)))
        final_position = pareto_front[selected_solution_idx]['position']

        # 将位置映射到 X, Y 坐标
        final_x = []
        final_y = []
        for i, position in enumerate(final_position):
            area = position
            x, y = area_positions[area][i % len(area_positions[area])]
            final_x.append(x)
            final_y.append(y)

        df['Final Area'] = final_position
        df['Final X'] = final_x
        df['Final Y'] = final_y

        # 保存最终堆垛结果
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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
        output_file_plates_with_batch = r'../result/final_stack_distribution/final_stack_distribution.csv'
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
