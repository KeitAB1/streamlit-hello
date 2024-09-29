import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import time
from optimization_objectives import OptimizationObjectives
from utils import run_optimization
from utils import save_convergence_plot, save_performance_metrics



# Streamlit 页面配置
st.set_page_config(page_title="home", page_icon="⚙")

# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
output_dir = "stack_distribution_plots/final_stack_distribution"
convergence_dir = "result/ConvergenceData"
data_dir = "test/steel_data"  # 统一数据集目录
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
system_data_dir = "data"  # 系统数据集目录
available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]


st.write("Warehouse and Stack Configuration")
use_default_config = st.checkbox("Use default warehouse and stack configuration", value=True)

if not use_default_config:
    # 如果用户选择自定义库区和垛位设置，显示相关的输入框
    num_areas = st.number_input("Number of Areas", 1, 10, 6)

    # 动态生成库区的垛位设置
    area_positions = {}
    stack_dimensions = {}
    for area in range(num_areas):
        st.write(f"### Area {area + 1}")
        num_stacks = st.number_input(f"Number of Stacks in Area {area + 1}", 1, 10, 4,
                                     key=f'num_stacks_area_{area}')

        area_stack_positions = []
        area_stack_dimensions = []

        for stack in range(num_stacks):
            st.write(f"#### Stack {stack + 1}")
            x = st.number_input(f"Stack {stack + 1} X position", key=f'stack_x_area_{area}_{stack}')
            y = st.number_input(f"Stack {stack + 1} Y position", key=f'stack_y_area_{area}_{stack}')
            width = st.number_input(f"Stack {stack + 1} width (mm)", 1000, 20000, 6000,
                                    key=f'stack_width_area_{area}_{stack}')
            length = st.number_input(f"Stack {stack + 1} length (mm)", 1000, 20000, 3000,
                                     key=f'stack_length_area_{area}_{stack}')

            area_stack_positions.append((x, y))
            area_stack_dimensions.append((length, width))

        area_positions[area] = area_stack_positions
        stack_dimensions[area] = area_stack_dimensions

else:
    # 使用默认配置
    area_positions = {
        0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        3: [(0, 0), (0, 1), (1, 0), (1, 1)],
        4: [(0, 0), (0, 1), (1, 0), (1, 1)],
        5: [(0, 0), (0, 1), (1, 0), (1, 1)]
    }

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

# 查看按钮
# 初始化 session_state 用于存储按钮状态
if "show_stack_config" not in st.session_state:
    st.session_state["show_stack_config"] = False

# 查看按钮，点击后切换状态
if st.button("View/Hide Current Stack Configuration"):
    # 切换显示状态
    st.session_state["show_stack_config"] = not st.session_state["show_stack_config"]

# 根据按钮的状态决定是否显示内容
if st.session_state["show_stack_config"]:
    st.write("### Current Area Positions")
    for area, positions in area_positions.items():
        st.write(f"Area {area + 1} Stack Positions: {positions}")

    st.write("### Current Stack Dimensions")
    for area, dimensions in stack_dimensions.items():
        st.write(f"Area {area + 1} Stack Dimensions (LxW in mm): {dimensions}")




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
                  "CoEA (Co-Evolutionary Algorithm)", "EDA (Estimation of Distribution Algorithm)",
                  "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)"]
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
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    elif selected_algorithm == "GA (Genetic Algorithm)":
        st.subheader("GA Parameters")
        generations = st.number_input("Generations", 1, 1000, 3)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.05)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        population_size = st.slider("Population Size", 100, 500, 100)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    elif selected_algorithm == "SA (Simulated Annealing)":
        st.subheader("SA Parameters")
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
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
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    elif selected_algorithm == "ACO (Ant Colony Optimization)":
        st.subheader("ACO Parameters")
        max_iter = st.number_input("Max Iterations", 1, 1000, 5)  # 默认迭代轮数为5
        num_ants = st.slider("Number of Ants", 10, 100, 50)
        alpha = st.slider("Pheromone Importance (α)", 0.0, 5.0, 1.0)
        beta = st.slider("Heuristic Importance (β)", 0.0, 5.0, 2.0)
        evaporation_rate = st.slider("Evaporation Rate", 0.0, 1.0, 0.5)
        q = st.number_input("Pheromone Constant (Q)", 1.0, 1000.0, 100.0)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    elif selected_algorithm == "DE (Differential Evolution)":
        st.subheader("DE Parameters")
        max_iter = st.number_input("Max Iterations", 1, 1000, 100)
        F = st.slider("Mutation Factor (F)", 0.0, 2.0, 0.5)
        CR = st.slider("Crossover Rate (CR)", 0.0, 1.0, 0.9)
        pop_size = st.slider("Population Size", 10, 200, 50)

        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    elif selected_algorithm == "CoEA (Co-Evolutionary Algorithm)":
        st.subheader("CoEA Parameters")

        # CoEA-specific parameters
        generations = st.number_input("Generations", 1, 1000, 100)
        population_size = st.slider("Population Size", 10, 500, 100)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)

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
    elif selected_algorithm == "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)":
        st.subheader("MOEA/D Parameters")
        population_size = st.slider("Population Size", 100, 500, 100)
        generations = st.number_input("Generations", 1, 1000, 200)
        T = st.slider("Neighborhood Size (T)", 2, 50, 20)
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



    # 新增：入库口和出库口的坐标
    inbound_point = (41500, 3000)  # 入库口坐标
    outbound_point = (41500, 38000)  # 出库口坐标

    #  将交货时间从字符串转换为数值
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values

    # 创建 OptimizationObjectives 实例
    objectives = OptimizationObjectives(
        plates=plates,
        heights=heights,
        delivery_times=delivery_times,
        batches=batches,
        Dki=Dki,
        area_positions=area_positions,
        inbound_point=inbound_point,
        outbound_point=outbound_point,
        horizontal_speed=horizontal_speed,
        vertical_speed=vertical_speed
    )

    #



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
                     dataset_name, objectives):
            self.num_particles = num_particles
            self.num_positions = num_positions
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.max_iter = max_iter
            self.particles = [Particle(num_positions) for _ in range(self.num_particles)]
            self.best_position = None
            self.best_score = np.inf
            self.worst_score = -np.inf
            self.best_improvement = 0
            self.total_improvement = 0
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.convergence_data = []
            self.dataset_name = dataset_name
            self.stable_iterations = 0
            self.prev_best_score = np.inf
            self.start_time = None
            self.objectives = objectives

        def optimize(self):
            global heights
            self.start_time = time.time()
            for iteration in range(self.max_iter):
                improvement_flag = False
                for particle in self.particles:
                    if self.best_position is None:
                        self.best_position = particle.position.copy()

                    temp_heights = heights.copy()

                    # 计算目标函数的惩罚项
                    combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                        particle.position)
                    energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(particle.position)
                    balance_penalty = self.objectives.maximize_inventory_balance_v2(particle.position)
                    space_utilization = self.objectives.maximize_space_utilization_v3(particle.position)

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
                    if current_score < self.best_score:
                        improvement_flag = True
                        self.best_improvement = max(self.best_improvement, self.best_score - current_score)
                        self.best_score = current_score
                        self.best_position = particle.position.copy()

                    # 更新最差得分
                    if current_score > self.worst_score:
                        self.worst_score = current_score

                # 更新总改进值
                if improvement_flag:
                    self.total_improvement += self.prev_best_score - self.best_score
                    self.prev_best_score = self.best_score
                    self.stable_iterations = 0
                else:
                    self.stable_iterations += 1

                # 更新粒子的位置和速度
                for particle in self.particles:
                    particle.update_velocity(self.best_position, self.w, self.c1, self.c2)
                    particle.update_position(self.num_positions)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])

                # 实时更新收敛曲线
                self.update_convergence_plot(iteration + 1)

                # 打印迭代信息
                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            # 更新最终高度
            self.update_final_heights()

            # 记录运行时间
            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed)

        def update_final_heights(self):
            global heights
            heights = np.zeros(len(Dki))
            for plate_idx, position in enumerate(self.best_position):
                area = position
                heights[area] += plates[plate_idx, 2]

        def update_convergence_plot(self, current_iteration):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Iterations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}')
            plt.legend()

            convergence_plot_placeholder.pyplot(plt)

            save_convergence_plot(self.convergence_data, current_iteration, self.best_score, "PSO", self.dataset_name)

        def save_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            save_performance_metrics(
                self.best_score, self.worst_score, self.best_improvement, self.total_improvement,
                iterations, time_elapsed, self.convergence_data, self.stable_iterations, self.dataset_name, "PSO"
            )


    class GA_with_Batch:
        cache = {}  # 适应度缓存，避免重复计算

        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions, dataset_name, objectives, plates, delivery_times, batches):
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.generations = generations
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.num_positions = num_positions
            self.population = [np.random.randint(0, num_positions, size=len(plates)) for _ in range(population_size)]
            self.best_individual = None
            self.best_score = np.inf
            self.worst_score = -np.inf
            self.best_improvement = np.inf
            self.prev_best_score = np.inf
            self.total_improvement = 0
            self.convergence_data = []
            self.stable_iterations = 0
            self.dataset_name = dataset_name  # 增加数据集名称属性
            self.start_time = None  # 用于记录优化过程的时间
            self.objectives = objectives  # OptimizationObjectives 实例
            self.plates = plates  # 增加 plates 属性
            self.delivery_times = delivery_times  # 增加 delivery_times 属性
            self.batches = batches  # 增加 batches 属性
            self.heights = np.zeros(num_positions)  # 初始化高度数组

        def fitness(self, individual):
            individual_tuple = tuple(individual)

            if individual_tuple in self.cache:
                return self.cache[individual_tuple]

            # 使用 OptimizationObjectives 实例的方法来计算惩罚项
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(individual)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            self.cache[individual_tuple] = score
            return score

        def select(self):
            with ThreadPoolExecutor() as executor:
                fitness_scores = list(executor.map(self.fitness, self.population))

            fitness_scores = np.array(fitness_scores)
            probabilities = np.exp(-fitness_scores / np.sum(fitness_scores))
            probabilities /= probabilities.sum()
            selected_indices = np.random.choice(len(self.population), size=self.population_size, p=probabilities)
            return [self.population[i] for i in selected_indices]

        def crossover(self, parent1, parent2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, len(self.plates))
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
            self.start_time = time.time()  # 记录开始时间

            for generation in range(self.generations):
                new_population = []
                selected_population = self.select()

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

                self.worst_score = max(self.worst_score, best_score_gen)
                self.best_improvement = min(self.best_improvement, abs(self.prev_best_score - best_score_gen))
                self.total_improvement += abs(self.prev_best_score - best_score_gen)
                self.prev_best_score = best_score_gen

                if best_score_gen < self.best_score:
                    self.best_score = best_score_gen
                    self.best_individual = best_individual_gen.copy()
                    self.stable_iterations = 0  # 重置稳定迭代次数
                else:
                    self.stable_iterations += 1  # 计数稳定迭代次数

                self.convergence_data.append([generation + 1, self.best_score])
                self.update_convergence_plot(generation + 1)

                print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

            time_elapsed = time.time() - self.start_time  # 计算总耗时
            iterations = len(self.convergence_data)
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            average_improvement = self.total_improvement / iterations if iterations > 0 else np.inf
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)

            metrics = {
                'Best Score': self.best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': self.best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': self.stable_iterations
            }

            self.save_metrics(metrics)

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

            save_convergence_plot(self.convergence_data, current_generation, self.best_score, "GA", self.dataset_name)

        def save_metrics(self, metrics):
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_ga.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)


    class SA_with_Batch:
        def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, lambda_1, lambda_2,
                     lambda_3, lambda_4, num_positions, dataset_name, objectives):
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
            self.objectives = objectives  # OptimizationObjectives 实例

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
            try:
                # 使用 objectives 实例调用优化目标函数
                combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(position)
                energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(position)
                balance_penalty = self.objectives.maximize_inventory_balance_v2(position)
                space_utilization = self.objectives.maximize_space_utilization_v3(position)

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

            save_convergence_plot(self.convergence_data, current_iteration, self.best_score, "SA", self.dataset_name)

        def save_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            save_performance_metrics(
                self.best_score, self.worst_score, self.best_improvement, self.total_improvement,
                iterations, time_elapsed, self.convergence_data, self.stable_iterations, self.dataset_name, "SA"
            )


    class PSO_SA_Optimizer:
        def __init__(self, num_particles, num_positions, w, c1, c2, max_iter_pso,
                     initial_temperature, cooling_rate, min_temperature, max_iterations_sa,
                     lambda_1, lambda_2, lambda_3, lambda_4, dataset_name, objectives):
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
            self.objectives = objectives  # OptimizationObjectives 实例

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
                dataset_name=dataset_name,
                objectives=objectives
            )

            self.sa_optimizer = SA_with_Batch(
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                min_temperature=min_temperature,
                max_iterations=max_iterations_sa,
                lambda_1=lambda_1, lambda_2=lambda_2,
                lambda_3=lambda_3, lambda_4=lambda_4,
                num_positions=num_positions,
                dataset_name=dataset_name,
                objectives=objectives
            )

            self.convergence_data_pso_sa = []  # PSO + SA 的收敛数据
            self.start_time = None  # 初始化运行时间记录

        def optimize(self):
            self.start_time = time.time()  # 开始时间记录
            # 首先运行PSO优化
            self.pso_optimizer.optimize()

            # 获取PSO的最优解，作为SA的初始解
            initial_position_for_sa = self.pso_optimizer.best_position

            # 使用SA在PSO的解基础上进行局部优化
            best_position_sa, best_score_sa = self.sa_optimizer.optimize_from_position(initial_position_for_sa)

            # 将PSO优化的收敛数据存入PSO + SA的收敛数据中
            self.convergence_data_pso_sa.extend(self.pso_optimizer.convergence_data)
            # 将SA优化的收敛数据追加到PSO + SA的收敛数据中
            self.convergence_data_pso_sa.extend(self.sa_optimizer.convergence_data)

            # 保存PSO + SA的收敛数据
            self.save_convergence_data_pso_sa()

            # 保存性能指标
            time_elapsed = time.time() - self.start_time  # 记录总时间
            self.save_performance_metrics(time_elapsed, best_score_sa)

            # 返回SA优化的最优解和得分
            return best_position_sa, best_score_sa

        def save_convergence_data_pso_sa(self):
            # 根据数据集名称动态生成收敛数据的保存路径
            dataset_folder = self.dataset_name.split('.')[0]  # 移除文件扩展名
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存PSO + SA的收敛数据到新的文件
            convergence_data_df = pd.DataFrame(self.convergence_data_pso_sa, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_psosa.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_performance_metrics(self, time_elapsed, best_score_sa):
            # 计算各项指标
            iterations = len(self.convergence_data_pso_sa)
            worst_score = max([data[1] for data in self.convergence_data_pso_sa])
            best_improvement = np.inf if iterations == 0 else abs(worst_score - best_score_sa)
            average_improvement = np.inf if iterations == 0 else best_improvement / iterations
            convergence_rate_value = (self.convergence_data_pso_sa[-1][1] - self.convergence_data_pso_sa[0][
                1]) / iterations
            relative_error_value = abs(best_score_sa) / (abs(best_score_sa) + 1e-6)
            stable_iterations = self.get_stable_iterations()

            # 保存指标到CSV
            metrics = {
                'Best Score': best_score_sa,
                'Worst Score': worst_score,
                'Best Improvement': best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': stable_iterations
            }

            # 根据数据集名称动态生成保存路径
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_psosa.csv"
            file_path = os.path.join(dataset_folder, file_name)

            # 保存性能指标数据
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)

        def get_stable_iterations(self):
            # 获取稳定收敛的迭代次数（例如，最后若干次迭代的得分变化不大）
            stable_threshold = 1e-3  # 可以根据具体情况调整
            stable_iterations = 0
            for i in range(1, len(self.convergence_data_pso_sa)):
                if abs(self.convergence_data_pso_sa[i][1] - self.convergence_data_pso_sa[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class ACO_with_Batch:
        def __init__(self, num_ants, num_positions, alpha, beta, evaporation_rate, q, max_iter, lambda_1, lambda_2,
                     lambda_3, lambda_4, dataset_name, objectives):
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
            self.worst_score = -np.inf
            self.best_position = None
            self.convergence_data = []  # 用于保存收敛数据
            self.dataset_name = dataset_name  # 增加数据集名称属性
            self.start_time = None  # 初始化运行时间记录
            self.objectives = objectives  # OptimizationObjectives 实例

        def optimize(self):
            self.start_time = time.time()  # 开始时间记录

            global heights
            for iteration in range(self.max_iter):
                all_ant_positions = []  # 用于存储每只蚂蚁的路径
                all_ant_scores = []  # 用于存储每只蚂蚁的适应度

                for ant in range(self.num_ants):
                    ant_position = self.construct_solution()  # 为每只蚂蚁构建解
                    all_ant_positions.append(ant_position)

                    # 计算每只蚂蚁的解的得分
                    temp_heights = heights.copy()
                    combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                        ant_position, temp_heights)
                    energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(ant_position)
                    balance_penalty = self.objectives.maximize_inventory_balance_v2(ant_position)
                    space_utilization = self.objectives.maximize_space_utilization_v3(ant_position)

                    # 计算最终得分，确保 score 是标量值
                    score = (self.lambda_1 * combined_movement_turnover_penalty +
                             self.lambda_2 * energy_time_penalty +
                             self.lambda_3 * balance_penalty -
                             self.lambda_4 * space_utilization)

                    # 使用 np.sum() 来确保 score 是标量
                    score = np.sum(score)

                    all_ant_scores.append(score)

                    # 更新最优解
                    if score < self.best_score:
                        self.best_score = score
                        self.best_position = ant_position.copy()

                    # 更新最差得分
                    if score > self.worst_score:
                        self.worst_score = score

                # 信息素更新
                self.update_pheromones(all_ant_positions, all_ant_scores)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)
                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            # 计算并保存最终指标
            self.update_final_heights()
            time_elapsed = time.time() - self.start_time  # 记录总时间
            self.save_performance_metrics(time_elapsed)

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
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_aco.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_performance_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            best_improvement = np.inf if iterations == 0 else abs(self.worst_score - self.best_score)
            average_improvement = np.inf if iterations == 0 else best_improvement / iterations
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)
            stable_iterations = self.get_stable_iterations()

            metrics = {
                'Best Score': self.best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': stable_iterations
            }

            # 保存性能指标
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_aco.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)

        def get_stable_iterations(self):
            stable_threshold = 1e-3  # 可以根据具体情况调整
            stable_iterations = 0
            for i in range(1, len(self.convergence_data)):
                if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class DE_with_Batch:
        def __init__(self, pop_size, num_positions, F, CR, max_iter, lambda_1, lambda_2, lambda_3, lambda_4,
                     dataset_name, objectives):
            self.pop_size = pop_size
            self.num_positions = num_positions
            self.F = F
            self.CR = CR
            self.max_iter = max_iter
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.population = np.random.randint(0, num_positions, size=(pop_size, num_plates))
            self.best_position = None
            self.best_score = np.inf
            self.worst_score = -np.inf
            self.convergence_data = []
            self.dataset_name = dataset_name
            self.start_time = None
            self.objectives = objectives  # OptimizationObjectives 实例

        def optimize(self):
            self.start_time = time.time()

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

                    # 计算适应度
                    original_score = self.calculate_fitness(self.population[i])
                    trial_score = self.calculate_fitness(trial)

                    # 选择操作
                    if trial_score < original_score:
                        new_population.append(trial)
                        if trial_score < self.best_score:
                            self.best_score = trial_score
                            self.best_position = trial
                    else:
                        new_population.append(self.population[i])

                    # 更新最差得分
                    if trial_score > self.worst_score:
                        self.worst_score = trial_score

                # 更新种群
                self.population = np.array(new_population)
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)

                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            time_elapsed = time.time() - self.start_time
            self.update_final_heights()
            self.save_performance_metrics(time_elapsed)

        def calculate_fitness(self, individual):
            # 计算个体的适应度得分
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                individual, self.objectives.heights)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            return np.sum(score)  # 确保返回标量值

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
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_de.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_performance_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            best_improvement = np.inf if iterations == 0 else abs(self.worst_score - self.best_score)
            average_improvement = np.inf if iterations == 0 else best_improvement / iterations
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)
            stable_iterations = self.get_stable_iterations()

            metrics = {
                'Best Score': self.best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': stable_iterations
            }

            # 保存性能指标
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_de.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)
            # print(f"Performance metrics saved to {file_path}")

        def get_stable_iterations(self):
            stable_threshold = 1e-3  # 可以根据具体情况调整
            stable_iterations = 0
            for i in range(1, len(self.convergence_data)):
                if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class CoEA_with_Batch:
        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions, dataset_name, objectives):
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
            self.objectives = objectives  # OptimizationObjectives 实例

            # 创建多个子种群，假设分为两个子种群
            self.subpopulations = [
                [np.random.randint(0, num_positions, size=num_plates) for _ in range(population_size // 2)] for _ in
                range(2)
            ]

            self.best_individual = None
            self.best_score = np.inf
            self.best_position = None  # 初始化 best_position
            self.worst_score = -np.inf  # 最差得分
            self.convergence_data = []
            self.start_time = None  # 记录运行时间

        def fitness(self, individual):
            global heights
            temp_heights = heights.copy()

            # 调整调用以匹配参数数量
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                individual, self.objectives.heights)

            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            return np.sum(score)  # 确保返回标量值

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
            self.start_time = time.time()  # 记录优化开始时间

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
                    self.best_position = best_individual_gen  # 更新最佳位置

                if best_score_gen > self.worst_score:
                    self.worst_score = best_score_gen

                self.convergence_data.append([generation + 1, self.best_score])
                self.update_convergence_plot(generation + 1)

                print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

            time_elapsed = time.time() - self.start_time
            self.save_performance_metrics(time_elapsed)

        def exchange_subpopulations(self):
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
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_coea.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_performance_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            best_improvement = np.inf if iterations == 0 else abs(self.worst_score - self.best_score)
            average_improvement = np.inf if iterations == 0 else best_improvement / iterations
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)
            stable_iterations = self.get_stable_iterations()

            metrics = {
                'Best Score': self.best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': stable_iterations
            }

            # 保存性能指标
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_coea.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)

        def get_stable_iterations(self):
            stable_threshold = 1e-3  # 可以根据具体情况调整
            stable_iterations = 0
            for i in range(1, len(self.convergence_data)):
                if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class EDA_with_Batch:
        def __init__(self, pop_size, num_positions, max_iter, lambda_1, lambda_2, lambda_3, lambda_4, dataset_name,
                     objectives):
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
            self.worst_score = -np.inf  # 最差得分
            self.convergence_data = []  # 用于保存收敛数据
            self.start_time = None  # 记录开始时间
            self.objectives = objectives  # OptimizationObjectives 实例

        def optimize(self):
            self.start_time = time.time()  # 记录优化开始时间
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

                    if score > self.worst_score:
                        self.worst_score = score

                # 更新种群
                self.population = np.copy(new_population)

                # 保存收敛数据
                self.convergence_data.append([iteration + 1, self.best_score])
                self.update_convergence_plot(iteration + 1)
                print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

            # 计算优化结束时间
            time_elapsed = time.time() - self.start_time
            # 保存性能指标
            self.save_performance_metrics(time_elapsed)

            # 更新最终的堆垛高度
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
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                individual, temp_heights)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            return np.sum(score)  # 确保返回标量值

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
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_eda.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_performance_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            best_improvement = np.inf if iterations == 0 else abs(self.worst_score - self.best_score)
            average_improvement = np.inf if iterations == 0 else best_improvement / iterations
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(self.best_score) / (abs(self.best_score) + 1e-6)
            stable_iterations = self.get_stable_iterations()

            metrics = {
                'Best Score': self.best_score,
                'Worst Score': self.worst_score,
                'Best Improvement': best_improvement,
                'Average Improvement': average_improvement,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
                'Convergence Speed (Stable Iterations)': stable_iterations
            }

            # 保存性能指标
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_eda.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)

        def get_stable_iterations(self):
            stable_threshold = 1e-3  # 可以根据具体情况调整
            stable_iterations = 0
            for i in range(1, len(self.convergence_data)):
                if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class MOEAD_with_Batch:
        def __init__(self, population_size, generations, lambda_1, lambda_2, lambda_3, lambda_4, num_positions,
                     dataset_name, T=20):
            self.population_size = population_size
            self.generations = generations
            self.lambda_1 = lambda_1
            self.lambda_2 = lambda_2
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.num_positions = num_positions
            self.population = [np.random.randint(0, num_positions, size=num_plates) for _ in range(population_size)]
            self.best_individual = None
            self.best_score = np.inf
            self.T = T  # 邻居大小
            self.weights = self.init_weights()
            self.neighbors = self.init_neighbors()
            self.fitness_values = None  # 初始化 fitness_values
            self.convergence_data = []
            self.dataset_name = dataset_name
            self.start_time = None
            self.objectives = objectives  # 保存 OptimizationObjectives 实例

        def init_weights(self):
            # 初始化每个子问题的权重向量
            weights = np.zeros((self.population_size, 2))
            for i in range(self.population_size):
                weights[i, 0] = i / (self.population_size - 1)
                weights[i, 1] = 1 - weights[i, 0]
            return weights

        def init_neighbors(self):
            # 初始化每个个体的邻居（基于权重向量的距离）
            neighbors = np.zeros((self.population_size, self.T), dtype=int)
            for i in range(self.population_size):
                distances = np.sum(np.abs(self.weights - self.weights[i]), axis=1)
                neighbors[i] = np.argsort(distances)[:self.T]
            return neighbors

        def fitness(self, individual):
            # 初始化适应度值
            global heights
            temp_heights = heights.copy()

            # 计算个体的适应度得分
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                individual, temp_heights)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            score1 = combined_movement_turnover_penalty + energy_time_penalty
            score2 = balance_penalty - space_utilization
            # print(f"Fitness for individual {individual}: Score1={score1}, Score2={score2}")  # 输出适应度值
            return score1, score2

        def update(self, population, fitness_values, weights, neighbors):
            new_population = []
            for i, individual in enumerate(population):
                # 邻居交叉操作
                k, l = np.random.choice(neighbors[i], 2, replace=False)
                parent1 = population[k]
                parent2 = population[l]
                child = self.crossover(parent1, parent2)

                # 变异操作
                child = self.mutate(child)

                # 计算子代的适应度值
                child_fitness = self.fitness(child)

                # 更新邻居解
                for j in neighbors[i]:
                    if self.tchebycheff(child_fitness, weights[j]) < self.tchebycheff(fitness_values[j], weights[j]):
                        population[j] = child
                        fitness_values[j] = child_fitness

                new_population.append(child)
            return new_population

        def tchebycheff(self, fitness, weight):
            # 基于Tchebycheff标量化方法
            return max(weight * np.abs(fitness))

        def crossover(self, parent1, parent2):
            crossover_point = np.random.randint(1, num_plates)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            return child

        def mutate(self, individual):
            for i in range(len(individual)):
                if np.random.rand() < 0.1:  # 设置突变概率
                    individual[i] = np.random.randint(0, self.num_positions)
            return individual

        def optimize(self):
            self.start_time = time.time()

            # 初始化种群的适应度
            self.fitness_values = [self.fitness(individual) for individual in self.population]

            # 迭代优化
            for generation in range(self.generations):
                self.population = self.update(self.population, self.fitness_values, self.weights, self.neighbors)

                # 获取当前最优解
                current_best = min(self.fitness_values, key=lambda f: f[0] + f[1])
                if self.best_individual is None or (current_best[0] + current_best[1]) < self.best_score:
                    self.best_individual = self.population[self.fitness_values.index(current_best)]
                    self.best_score = current_best[0] + current_best[1]

                # 记录收敛数据
                self.convergence_data.append([generation + 1, self.best_score])

                # 实时更新收敛曲线
                self.update_convergence_plot(generation + 1)

                print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')

            # 保存收敛数据
            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed)

        def update_convergence_plot(self, current_generation):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Generations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Generation {current_generation}')
            plt.legend()
            convergence_plot_placeholder.pyplot(plt)

            dataset_folder = self.dataset_name.split('.')[0]
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_moead.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_metrics(self, time_elapsed):
            # 确保 fitness_values 已初始化
            if self.fitness_values is None:
                raise ValueError("Fitness values are not available.")

            best_score = self.best_score
            iterations = len(self.convergence_data)
            convergence_rate_value = (self.convergence_data[-1][1] - self.convergence_data[0][1]) / iterations
            relative_error_value = abs(best_score) / (abs(best_score) + 1e-6)

            metrics = {
                'Best Score': best_score,
                'Iterations': iterations,
                'Time (s)': time_elapsed,
                'Convergence Rate': convergence_rate_value,
                'Relative Error': relative_error_value,
            }

            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)

            file_name = f"comparison_performance_moead.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)


    if selected_algorithm == "PSO (Particle Swarm Optimization)":
        pso_params = {
            'num_particles': 30, 'num_positions': len(Dki), 'w': w, 'c1': c1, 'c2': c2,
            'max_iter': max_iter, 'lambda_1': lambda_1, 'lambda_2': lambda_2, 'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives
        }
        run_optimization(PSO_with_Batch, pso_params, df, area_positions, output_dir_base, "pso")

    elif selected_algorithm == "GA (Genetic Algorithm)":
        ga_params = {
            'population_size': population_size, 'mutation_rate': mutation_rate, 'crossover_rate': crossover_rate,
            'generations': generations, 'lambda_1': lambda_1, 'lambda_2': lambda_2, 'lambda_3': lambda_3,
            'lambda_4': lambda_4, 'num_positions': len(Dki),
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives,
            'plates': plates,  # 确保传递 plates
            'delivery_times': delivery_times,  # 确保传递 delivery_times
            'batches': batches  # 确保传递 batches
        }
        run_optimization(GA_with_Batch, ga_params, df, area_positions, output_dir_base, "ga")

    elif selected_algorithm == "SA (Simulated Annealing)":
        sa_params = {
            'initial_temperature': initial_temperature, 'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations_sa, 'lambda_1': lambda_1, 'lambda_2': lambda_2, 'lambda_3': lambda_3,
            'lambda_4': lambda_4, 'num_positions': len(Dki),
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives
        }

        run_optimization(SA_with_Batch, sa_params, df, area_positions, output_dir_base, "sa")

    elif selected_algorithm == "PSO + SA (Hybrid Optimization)":
        pso_params = {
            'num_particles': 30, 'num_positions': len(Dki), 'w': w, 'c1': c1, 'c2': c2,
            'max_iter': max_iter_pso,  # 使用max_iter_pso
            'lambda_1': lambda_1, 'lambda_2': lambda_2, 'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives
        }

        # 创建PSO优化器实例并进行优化
        pso_optimizer = PSO_with_Batch(**pso_params)
        pso_optimizer.optimize()

        # 在PSO优化完成后，获取最优解并将其传递给SA参数
        best_position = pso_optimizer.best_position

        sa_params = {
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives,
            # 'initial_solution': best_position  # 传递来自PSO的最佳位置
        }

        # 执行SA优化
        run_optimization(SA_with_Batch, sa_params, df, area_positions, output_dir_base, "psosa")

    elif selected_algorithm == "ACO (Ant Colony Optimization)":
        aco_params = {
            'num_ants': num_ants,  # 蚂蚁数量
            'num_positions': len(Dki),  # 库区/垛位数量
            'alpha': alpha,  # 信息素因子
            'beta': beta,  # 启发式因子
            'evaporation_rate': evaporation_rate,  # 蒸发速率
            'q': q,  # 信息素强度
            'max_iter': max_iter,  # 最大迭代次数
            'lambda_1': lambda_1,  # 适应度函数中的权重参数
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives  # OptimizationObjectives 实例
        }

        # 调用优化函数
        run_optimization(ACO_with_Batch, aco_params, df, area_positions, output_dir_base, "aco")

    elif selected_algorithm == "DE (Differential Evolution)":
        de_params = {
            'pop_size': pop_size,
            'num_positions': len(Dki),
            'F': F,
            'CR': CR,
            'max_iter': max_iter,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives
        }

        # 调用优化函数
        run_optimization(DE_with_Batch, de_params, df, area_positions, output_dir_base, "de")

    elif selected_algorithm == "CoEA (Co-Evolutionary Algorithm)":
        coea_params = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'generations': generations,  # 使用从用户输入获取的 generations
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives
        }

        # 调用优化函数
        run_optimization(CoEA_with_Batch, coea_params, df, area_positions, output_dir_base, "CoEA")

    elif selected_algorithm == "EDA (Estimation of Distribution Algorithm)":
        eda_params = {
            'pop_size': pop_size,
            'num_positions': len(Dki),
            'max_iter': max_iter,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives  # 添加 objectives 参数
        }

        # 调用优化函数
        run_optimization(EDA_with_Batch, eda_params, df, area_positions, output_dir_base, "eda")




    elif selected_algorithm == "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)":
        moead_params = {
            'population_size': population_size,
            'generations': generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives  # 添加 objectives 参数
        }

        # 调用优化函数
        run_optimization(EDA_with_Batch, moead_params, df, area_positions, output_dir_base, "moead")