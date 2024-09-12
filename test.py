import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
output_dir = "stack_distribution_plots/final_stack_distribution"
convergence_dir = "result/ConvergenceData"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(convergence_dir, exist_ok=True)


st.title("Steel Plate Stacking Optimization")

# 获取 data 文件夹下的所有 CSV 文件
data_dir = "data"
available_datasets = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# 选择数据集的方式
data_choice = st.selectbox("Choose dataset", ("Use system dataset", "Upload your own dataset"))

# 初始化 df 为 None
df = None

# 如果用户选择上传数据集
if data_choice == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload your steel plate dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset:")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset to proceed.")

# 如果用户选择使用系统自带的数据集
else:
    # 列出可用的数据集供用户选择，初始选择为空
    selected_dataset = st.selectbox("Select a system dataset", [""] + available_datasets)

    if selected_dataset and selected_dataset != "":
        system_dataset_path = os.path.join(data_dir, selected_dataset)
        df = pd.read_csv(system_dataset_path)
        st.write(f"Using system dataset: {selected_dataset}")
        st.write(df.head())
    else:
        st.warning("Please select a system dataset to proceed.")




# 算法选择放在侧边栏
with st.sidebar:
    algorithms = ["PSO (Particle Swarm Optimization)", "GA (Genetic Algorithm)", "SA (Simulated Annealing)"]
    selected_algorithm = st.selectbox("Select Optimization Algorithm", algorithms)

    # 根据选择的算法动态显示相关参数设置
    if selected_algorithm == "PSO (Particle Swarm Optimization)":
        st.subheader("PSO Parameters")
        max_iter = st.number_input("Max Iterations", 1, 1000, 1)
        w = st.slider("Inertia Weight (w)", 0.0, 1.0, 0.5)
        c1 = st.slider("Cognitive Coefficient (c1)", 0.0, 4.0, 2.0)
        c2 = st.slider("Social Coefficient (c2)", 0.0, 4.0, 2.0)
        st.write(f"Note: c1 + c2 should equal 4.0. Current sum: {c1 + c2}")
        lambda_1 = st.number_input("Lambda 1", value=1.0)
        lambda_2 = st.number_input("Lambda 2", value=1.0)
        lambda_3 = st.number_input("Lambda 3", value=1.0)
        lambda_4 = st.number_input("Lambda 4", value=1.0)

    elif selected_algorithm == "GA (Genetic Algorithm)":
        st.subheader("GA Parameters")
        population_size = st.number_input("Population Size", 10, 500, 100)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.05)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        generations = st.slider("Generations", 1, 1000, 100)

    elif selected_algorithm == "SA (Simulated Annealing)":
        st.subheader("SA Parameters")
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)

# 动态显示收敛曲线的占位符
convergence_plot_placeholder = st.empty()

# 如果 df 已经加载，进行堆垛优化分析
if df is not None:
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

    # 传送带位置参数
    conveyor_position_x = 2000  # 距离库区1-3的传送带水平距离
    conveyor_position_y = 14000  # 距离库区4-6的传送带水平距离

    #  将交货时间从字符串转换为数值
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values


    # 计算入库或出库时间的通用函数
    def calculate_movement_time(plate_idx, area, stack_positions, plates, horizontal_speed, vertical_speed,
                                to_conveyor=True):
        x, y = area_positions[area][plate_idx % len(area_positions[area])]
        plate_length, plate_width = plates[plate_idx, 0], plates[plate_idx, 1]

        # 如果是入库，计算到传送带的距离；否则计算到出库口的距离
        if to_conveyor:
            # 水平方向：从传送带中央移动到库区垛位
            if area in [0, 1, 2]:  # 库区1、2、3
                distance_to_location = conveyor_position_x  # 传送带到库区1-3的距离
            else:  # 库区4、5、6
                distance_to_location = conveyor_position_y  # 传送带到库区4-6的距离
        else:
            # 出库口距离：库区1-3距离出库口15000mm，库区4-6距离出库口3000mm
            if area in [0, 1, 2]:
                distance_to_location = 15000
            else:
                distance_to_location = 3000

        # 计算移动距离
        total_distance_x = abs(
            distance_to_location - (x * (stack_dimensions[area][plate_idx % len(stack_dimensions[area])][0] + 500)))
        total_distance_y = y * 1000  # 假设垛位之间的间距为1000mm

        # 计算移动时间
        time_to_move_x = total_distance_x / horizontal_speed
        time_to_move_y = total_distance_y / vertical_speed

        return time_to_move_x + time_to_move_y


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
        total_energy_time = 0

        # 批次排序，确保按Q1-Q15顺序出库
        sorted_batches = sorted(set(batches), key=lambda x: int(x[1:]))
        plate_indices_by_batch = {batch: [] for batch in sorted_batches}

        # 按批次将钢板索引分配
        for plate_idx, plate in enumerate(plates):
            batch = plate[4]  # 批次信息在第5列（索引4）
            plate_indices_by_batch[batch].append(plate_idx)

        # 按批次依次处理出库
        for batch in sorted_batches:
            for plate_idx in plate_indices_by_batch[batch]:
                position = particle_positions[plate_idx]
                area = position  # 获取钢板所在库区
                plate_height = plates[plate_idx, 2]  # 获取钢板厚度

                # 调用出库时间计算
                outbound_time = calculate_movement_time(plate_idx, area, area_positions, plates, horizontal_speed,
                                                        vertical_speed, to_conveyor=False)

                # 更新堆垛高度
                heights[area] -= plate_height
                total_energy_time += outbound_time

        return total_energy_time


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
        def __init__(self, num_particles, num_positions, w, c1, c2, max_iter, lambda_1, lambda_2, lambda_3,
                     lambda_4):
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

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_dir = "result/ConvergenceData"
            os.makedirs(convergence_data_dir, exist_ok=True)
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_pso.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

    class GA_with_Batch:
        def __init__(self, population_size, num_positions, mutation_rate, crossover_rate, generations, lambda_1,
                     lambda_2, lambda_3, lambda_4):
            self.population_size = population_size
            self.num_positions = num_positions
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.generations = generations
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.population = [self.create_individual() for _ in range(self.population_size)]
            self.gbest_position = None
            self.gbest_score = np.inf
            self.convergence_data = []

        def create_individual(self):
            # 创建个体，随机初始化位置
            return np.random.randint(0, self.num_positions, size=num_plates)

        def evaluate(self, individual):
            global heights
            temp_heights = heights.copy()

            # 计算目标函数的惩罚项
            combined_movement_turnover_penalty = minimize_stack_movements_and_turnover(
                individual, temp_heights, plates, delivery_times, batches)
            energy_time_penalty = minimize_outbound_energy_time_with_batch(individual, plates, temp_heights)
            balance_penalty = maximize_inventory_balance_v2(individual, plates)
            space_utilization = maximize_space_utilization_v3(individual, plates, Dki)

            # 计算当前的总得分
            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)
            return score

        def selection(self):
            # 轮盘赌选择
            fitness_values = [1.0 / (1.0 + self.evaluate(ind)) for ind in self.population]
            total_fitness = np.sum(fitness_values)
            probabilities = fitness_values / total_fitness
            selected_idx = np.random.choice(range(self.population_size), p=probabilities)
            return self.population[selected_idx]

        def crossover(self, parent1, parent2):
            if np.random.rand() < self.crossover_rate:
                # 随机选择交叉点
                crossover_point = np.random.randint(1, num_plates - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            return child1, child2

        def mutation(self, individual):
            # 以一定概率进行变异
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_rate:
                    individual[i] = np.random.randint(0, self.num_positions)
            return individual

        def optimize(self):
            global heights
            for generation in range(self.generations):
                new_population = []
                for _ in range(self.population_size // 2):
                    parent1 = self.selection()
                    parent2 = self.selection()
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(self.mutation(child1))
                    new_population.append(self.mutation(child2))

                self.population = new_population

                # 更新全局最佳
                for individual in self.population:
                    current_score = self.evaluate(individual)
                    if current_score < self.gbest_score:
                        self.gbest_score = current_score
                        self.gbest_position = individual.copy()

                # 保存收敛数据
                self.convergence_data.append([generation + 1, self.gbest_score])

                # 更新收敛曲线
                self.update_convergence_plot(generation + 1)

            self.update_final_heights()

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.gbest_position):
                area = position
                heights[area] += plates[plate_idx, 2]

        def update_convergence_plot(self, current_generation):
            # 动态更新收敛曲线
            generation_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            plt.figure(figsize=(8, 4))
            plt.plot(generation_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Generations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Generation {current_generation}, Best Score {self.gbest_score}')
            plt.legend()

            # 使用 Streamlit 的空占位符更新图表
            convergence_plot_placeholder.pyplot(plt)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_dir = "result/ConvergenceData"
            os.makedirs(convergence_data_dir, exist_ok=True)
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_ga.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)


        # 初始化并运行优化算法
    if selected_algorithm == "PSO (Particle Swarm Optimization)":
        # 初始化并运行PSO_with_Batch
        pso_with_batch = PSO_with_Batch(num_particles=30, num_positions=len(Dki),
                                        w=w, c1=c1, c2=c2, max_iter=max_iter,
                                        lambda_1=lambda_1, lambda_2=lambda_2,
                                        lambda_3=lambda_3, lambda_4=lambda_4)

        # 开始优化
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

    elif selected_algorithm == "GA (Genetic Algorithm)":
        ga_with_batch = GA_with_Batch(population_size=population_size, num_positions=len(Dki),
                                      mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                      generations=generations, lambda_1=lambda_1, lambda_2=lambda_2,
                                      lambda_3=lambda_3, lambda_4=lambda_4)
        ga_with_batch.optimize()

        # 获取最优解并将其映射到df
        final_positions_with_batch = GA_with_batch.gbest_position
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


    # st.success(f"Optimization complete. Results saved to {output_file_plates_with_batch}")



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

    # 保存计算后的数据
    final_stack_distribution_path = os.path.join(
        "result/final_stack_distribution/final_stack_distribution_plates.csv")
    df.to_csv(final_stack_distribution_path, index=False)

    # 设置 session state，允许可视化
    st.session_state['optimization_done'] = True
    st.success("Stacking optimization completed！You can now visualize the results.")

    # 生成堆垛结果的统计表
    st.write("### Final Stack Distribution Table")

    # 读取 final_stack_distribution_plates.csv 文件
    df = pd.read_csv(final_stack_distribution_path)

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

    # 保存结果到 CSV 文件
    output_file_heights = r'result/final_stack_distribution/final_stack_distribution_height.csv'
    result_df.to_csv(output_file_heights, index=False)

    # st.success(f"Stacking statistics saved to {output_file_heights}")