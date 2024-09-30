import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import time
from optimization_objectives import OptimizationObjectives
from utils import run_optimization, save_convergence_plot, save_performance_metrics
from optimizers.sa_optimizer import SA_with_Batch
from optimization_utils import evaluate_parallel, evaluate_with_cache, run_distributed_optimization
from optimization_utils import apply_adaptive_pso, apply_adaptive_sa, apply_adaptive_ga, apply_adaptive_coea, apply_adaptive_eda
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import logging  # 日志模块
from utils import save_convergence_history


# 从 constants 文件中引入常量
from constants import OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH
from constants import DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS, HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE, INBOUND_POINT, OUTBOUND_POINT, Dki

# 日志配置
# logging.basicConfig(filename="optimization.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Streamlit 页面配置
st.set_page_config(page_title="Steel Plate Stacking Optimization", page_icon="⚙")


# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.title("Steel Plate Stacking Optimization")

# 并行计算适应度函数
def evaluate_parallel(positions, evaluate_func):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_func, positions))
    return results

# 获取数据集
system_data_dir = "data"
available_datasets = [f.replace('.csv', '') for f in os.listdir(system_data_dir) if f.endswith('.csv')]

st.write("Warehouse and Stack Configuration")
use_default_config = st.checkbox("Use default warehouse and stack configuration", value=True)

if not use_default_config:
    # 如果用户选择自定义配置，显示相关输入框
    num_areas = st.number_input("Number of Areas", 1, 10, 6)

    area_positions = {}
    stack_dimensions = {}
    for area in range(num_areas):
        st.write(f"### Area {area + 1}")
        num_stacks = st.number_input(f"Number of Stacks in Area {area + 1}", 1, 10, 4, key=f'num_stacks_area_{area}')
        area_stack_positions = []
        area_stack_dimensions = []
        for stack in range(num_stacks):
            x = st.number_input(f"Stack {stack + 1} X position", key=f'stack_x_area_{area}_{stack}')
            y = st.number_input(f"Stack {stack + 1} Y position", key=f'stack_y_area_{area}_{stack}')
            width = st.number_input(f"Stack {stack + 1} width (mm)", 1000, 20000, 6000, key=f'stack_width_area_{area}_{stack}')
            length = st.number_input(f"Stack {stack + 1} length (mm)", 1000, 20000, 3000, key=f'stack_length_area_{area}_{stack}')
            area_stack_positions.append((x, y))
            area_stack_dimensions.append((length, width))

        area_positions[area] = area_stack_positions
        stack_dimensions[area] = area_stack_dimensions
else:
    area_positions = DEFAULT_AREA_POSITIONS
    stack_dimensions = DEFAULT_STACK_DIMENSIONS

# 查看当前配置
if "show_stack_config" not in st.session_state:
    st.session_state["show_stack_config"] = False

if st.button("View/Hide Current Stack Configuration"):
    st.session_state["show_stack_config"] = not st.session_state["show_stack_config"]

if st.session_state["show_stack_config"]:
    # 堆叠区域位置信息表格
    st.write("### Current Area Positions")
    area_positions_data = []
    for area, positions in DEFAULT_AREA_POSITIONS.items():
        area_positions_data.append({
            "Area": f"Area {area + 1}",
            "Positions": str(positions)
        })

    positions_df = pd.DataFrame(area_positions_data)
    st.table(positions_df)


    # 堆叠尺寸信息表格
    st.write("### Current Stack Dimensions")
    stack_dimensions_data = []
    for area, dimensions in DEFAULT_STACK_DIMENSIONS.items():
        for idx, (length, width) in enumerate(dimensions):
            stack_dimensions_data.append({
                "Area": f"Area {area + 1}",
                "Stack": f"Stack {idx + 1}",
                "Length (mm)": length,
                "Width (mm)": width
            })

    dimensions_df = pd.DataFrame(stack_dimensions_data)
    st.table(dimensions_df)

# 数据集选择
data_choice = st.selectbox("Choose dataset", ("Use system dataset", "Upload your own dataset"))
df = None
dataset_name = None

if data_choice == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload your steel plate dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name.split('.')[0]
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset:")
        st.write(df.head())
    else:
        st.warning("Please upload a dataset to proceed.")
elif data_choice == "Use system dataset":
    selected_dataset = st.selectbox("Select a system dataset", [""] + available_datasets)
    if selected_dataset:
        dataset_name = selected_dataset
        system_dataset_path = os.path.join(system_data_dir, f"{selected_dataset}.csv")
        df = pd.read_csv(system_dataset_path)
        st.write(f"Using system dataset: {selected_dataset}")
        st.write(df.head())
    else:
        st.warning("Please select a system dataset to proceed.")


def get_optimization_weights():
    st.write("#### Optimize Target Weight")

    # 优化目标权重
    lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
    lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
    lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
    lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)

    return lambda_1, lambda_2, lambda_3, lambda_4


# 选择算法
with st.sidebar:
    algorithms = ["SA (Simulated Annealing)", "GA (Genetic Algorithm)", "PSO (Particle Swarm Optimization)", "PSO + SA (Hybrid Optimization)", "ACO (Ant Colony Optimization)", "DE (Differential Evolution)", "CoEA (Co-Evolutionary Algorithm)", "EDA (Estimation of Distribution Algorithm)", "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)"]
    selected_algorithm = st.selectbox("Select Optimization Algorithm", algorithms)
    use_adaptive = st.checkbox("Use Adaptive Parameter Adjustment", value=False)

    if selected_algorithm == "SA (Simulated Annealing)":
        st.subheader("SA Parameters")
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)
        # 调用优化权重函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

    elif selected_algorithm == "GA (Genetic Algorithm)":
        st.subheader("GA Parameters")
        # 种群大小
        population_size = st.number_input("Population Size", value=100, min_value=1, step=1)
        # 变异率
        mutation_rate = st.number_input("Mutation Rate", value=0.01, min_value=0.0, max_value=1.0, step=0.01)
        # 交叉率
        crossover_rate = st.number_input("Crossover Rate", value=0.7, min_value=0.0, max_value=1.0, step=0.01)
        # 最大迭代次数
        generations = st.number_input("Max Generations", value=2, min_value=1)
        # 调用优化权重函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

    elif selected_algorithm == "PSO (Particle Swarm Optimization)":
        st.subheader("PSO Parameters")
        # 用户输入的粒子数
        num_particles = st.number_input("Number of Particles", value=30, min_value=1, step=1)
        # PSO 的惯性权重和学习因子
        w = st.number_input("Inertia Weight (w)", value=0.5)
        c1 = st.number_input("Cognitive Component (c1)", value=1.5)
        c2 = st.number_input("Social Component (c2)", value=1.5)
        # 最大迭代次数
        max_iterations_pso = st.number_input("Max Iterations", value=2, min_value=1)
        # 调用优化权重函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

    elif selected_algorithm == "PSO + SA (Hybrid Optimization)":
        st.subheader("PSO + SA Parameters")

        # 用户输入 PSO 参数
        num_particles = st.number_input("Number of Particles", value=30, min_value=1, step=1)
        w = st.number_input("Inertia Weight (w)", value=0.5)
        c1 = st.number_input("Cognitive Component (c1)", value=1.5)
        c2 = st.number_input("Social Component (c2)", value=1.5)
        max_iter_pso = st.number_input("Max Iterations (PSO)", value=100, min_value=1)

        # 用户输入 SA 参数
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations (SA)", value=100, min_value=1)

        # 调用优化目标权重设置函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

    elif selected_algorithm == "DE (Differential Evolution)":
        st.subheader("DE Parameters")
        # 用户可以通过界面输入 max_iter
        max_iter = st.number_input("Max Iterations", min_value=1, max_value=1000, value=100)
        # 其余 DE 参数也可以通过 Streamlit 输入
        pop_size = st.number_input("Population Size", min_value=1, max_value=500, value=50)
        F = st.slider("Mutation Factor (F)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        CR = st.slider("Crossover Rate (CR)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

        # 调用获取优化权重的函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

    elif selected_algorithm == "CoEA (Co-Evolutionary Algorithm)":
        st.subheader("CoEA Parameters")

        # 用户输入的种群大小、变异率、交叉率、代数等参数
        population_size = st.number_input("Population Size", value=50, min_value=1)
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
        generations = st.number_input("Max Generations", value=100, min_value=1)

        # 调用优化权重函数，获取 λ 参数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

    elif selected_algorithm == "EDA (Estimation of Distribution Algorithm)":
        st.subheader("EDA Parameters")

        # 设置 EDA 特有的参数输入框
        pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=100)
        max_iterations_eda = st.number_input("Max Iterations", min_value=1, max_value=1000, value=100)

        # 交叉率与变异率
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.9)

        # 调用权重优化函数
        lambda_1, lambda_2, lambda_3, lambda_4 = get_optimization_weights()

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


# 优化分析
if df is not None:
    output_dir_base = f"result/final_stack_distribution/{dataset_name}"
    os.makedirs(os.path.join(output_dir_base, 'final_stack_distribution_height'), exist_ok=True)
    os.makedirs(os.path.join(output_dir_base, 'final_stack_distribution_plates'), exist_ok=True)

    plates = df[['Length', 'Width', 'Thickness', 'Material_Code', 'Batch', 'Entry Time', 'Delivery Time']].values
    plate_areas = plates[:, 0] * plates[:, 1]
    num_plates = len(plates)
    batches = df['Batch'].values

    heights = np.zeros(len(Dki))

    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values

    objectives = OptimizationObjectives(
        plates=plates,
        heights=heights,
        delivery_times=delivery_times,
        batches=batches,
        Dki=Dki,
        area_positions=area_positions,
        inbound_point=INBOUND_POINT,
        outbound_point=OUTBOUND_POINT,
        horizontal_speed=HORIZONTAL_SPEED,
        vertical_speed=VERTICAL_SPEED
    )


    # PSO 的粒子类定义
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
                     dataset_name, objectives, use_adaptive):
            self.num_particles = num_particles
            self.num_positions = num_positions
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.max_iter = max_iter
            self.use_adaptive = use_adaptive
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
            self.adaptive_param_data = []

            # 初始化图表占位符
            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

        def optimize(self):
            self.start_time = time.time()
            st.info("PSO Optimization started...")
            with st.spinner("Running PSO Optimization..."):
                for iteration in range(self.max_iter):
                    improvement_flag = False
                    for particle in self.particles:
                        # 计算当前粒子的得分
                        current_score = self.evaluate_particle(particle)

                        if current_score < particle.best_score:
                            particle.best_score = current_score
                            particle.best_position = particle.position.copy()

                        if current_score < self.best_score:
                            improvement_flag = True
                            self.best_improvement = max(self.best_improvement, self.best_score - current_score)
                            self.best_score = current_score
                            self.best_position = particle.position.copy()

                        if current_score > self.worst_score:
                            self.worst_score = current_score

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

                    # 自适应调节
                    if self.use_adaptive:
                        self.w, self.c1, self.c2 = apply_adaptive_pso(self.w, self.c1, self.c2,
                                                                      self.best_score - current_score,
                                                                      self.use_adaptive)
                        self.record_adaptive_params()

                    # 更新收敛数据
                    self.convergence_data.append([iteration + 1, self.best_score])
                    self.update_convergence_plot(iteration + 1)
                    # logging.info(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.best_score}')

                time_elapsed = time.time() - self.start_time
                self.save_metrics(time_elapsed)

                # 优化结束后，保存历史收敛数据
                history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "PSO")
                save_convergence_history(self.convergence_data, "PSO", self.dataset_name, history_data_dir)

        def evaluate_particle(self, particle):
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                particle.position)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(particle.position)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(particle.position)
            space_utilization = self.objectives.maximize_space_utilization_v3(particle.position)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)
            return score

        def record_adaptive_params(self):
            self.adaptive_param_data.append({'w': self.w, 'c1': self.c1, 'c2': self.c2})

        def update_convergence_plot(self, current_iteration):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'),
                secondary_y=False
            )
            fig.update_layout(
                title=f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}',
                xaxis_title='Iterations',
                legend=dict(x=0.75, y=1)
            )
            fig.update_yaxes(title_text="Best Score", secondary_y=False)

            self.convergence_plot_placeholder.plotly_chart(fig)

            if self.use_adaptive:
                self.update_adaptive_param_plot()

        def update_adaptive_param_plot(self):
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            w_data = [x['w'] for x in self.adaptive_param_data]
            c1_data = [x['c1'] for x in self.adaptive_param_data]
            c2_data = [x['c2'] for x in self.adaptive_param_data]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iteration_data, y=w_data, mode='lines+markers', name='Inertia Weight (w)')
            )
            fig.add_trace(
                go.Scatter(x=iteration_data, y=c1_data, mode='lines+markers', name='Cognitive Component (c1)')
            )
            fig.add_trace(
                go.Scatter(x=iteration_data, y=c2_data, mode='lines+markers', name='Social Component (c2)')
            )
            fig.update_layout(
                title="Adaptive Parameter Changes",
                xaxis_title="Iterations",
                yaxis_title="Parameter Values",
                legend=dict(x=0.75, y=1)
            )

            self.adaptive_param_plot_placeholder.plotly_chart(fig)

        def save_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            save_performance_metrics(
                self.best_score, self.worst_score, self.best_improvement, self.total_improvement,
                iterations, time_elapsed, self.convergence_data, self.stable_iterations, self.dataset_name, "PSO"
            )


    class GA_with_Batch:
        cache = {}  # 适应度缓存，避免重复计算

        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions, dataset_name, objectives, plates, delivery_times, batches, use_adaptive):
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
            self.dataset_name = dataset_name
            self.start_time = None  # 用于记录优化过程的时间
            self.objectives = objectives  # OptimizationObjectives 实例
            self.plates = plates
            self.delivery_times = delivery_times
            self.batches = batches
            self.heights = np.zeros(num_positions)
            self.use_adaptive = use_adaptive
            self.adaptive_param_data = []

            # Streamlit 占位符
            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

        def fitness(self, individual):
            individual_tuple = tuple(individual)
            # 使用缓存机制避免重复计算
            return evaluate_with_cache(self.cache, individual_tuple, self._evaluate_fitness)

        def _evaluate_fitness(self, individual):
            # 计算适应度得分
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(individual)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            score = (self.lambda_1 * combined_movement_turnover_penalty +
                     self.lambda_2 * energy_time_penalty +
                     self.lambda_3 * balance_penalty -
                     self.lambda_4 * space_utilization)

            return score

        def select(self):
            """
            选择个体：使用并行计算评估适应度
            """
            fitness_scores = evaluate_parallel(self.population, self.fitness)
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
            st.info("GA Optimization started...")  # 提供优化开始信息
            with st.spinner("Running GA Optimization..."):  # 提供运行时加载提示
                self.start_time = time.time()  # 记录开始时间

                for generation in range(self.generations):
                    new_population = []
                    selected_population = self.select()

                    # 交叉和变异操作使用多线程并行处理
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

                    # 记录并更新自适应参数
                    if self.use_adaptive:
                        self.mutation_rate, self.crossover_rate = apply_adaptive_ga(
                            self.mutation_rate, self.crossover_rate, best_score_gen, self.best_score, self.use_adaptive
                        )
                        self.record_adaptive_params()

                        # 每代更新自适应参数调节图
                        self.update_adaptive_param_plot()

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

                # 优化结束后，保存历史收敛数据
                history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "GA")
                save_convergence_history(self.convergence_data, "GA", self.dataset_name, history_data_dir)

        def record_adaptive_params(self):
            self.adaptive_param_data.append(
                {'mutation_rate': self.mutation_rate, 'crossover_rate': self.crossover_rate})

        def update_convergence_plot(self, current_generation):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            # 创建收敛曲线
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'),
                secondary_y=False
            )

            # 设置图表布局
            fig.update_layout(
                title=f'Convergence Curve - Generation {current_generation}, Best Score {self.best_score}',
                xaxis_title='Generations',
                legend=dict(x=0.75, y=1)
            )

            fig.update_yaxes(title_text="Best Score", secondary_y=False)

            # 使用 Streamlit 展示 Plotly 图表
            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

            # 保存收敛图
            save_convergence_plot(self.convergence_data, current_generation, self.best_score, "GA", self.dataset_name)

        def update_adaptive_param_plot(self):
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            mutation_rate_data = [x['mutation_rate'] for x in self.adaptive_param_data]
            crossover_rate_data = [x['crossover_rate'] for x in self.adaptive_param_data]

            # 创建参数变化曲线
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iteration_data, y=mutation_rate_data, mode='lines+markers', name='Mutation Rate')
            )
            fig.add_trace(
                go.Scatter(x=iteration_data, y=crossover_rate_data, mode='lines+markers', name='Crossover Rate')
            )

            # 设置图表布局
            fig.update_layout(
                title="Adaptive Parameter Changes",
                xaxis_title="Generations",
                yaxis_title="Parameter Values",
                legend=dict(x=0.75, y=1)
            )

            # 使用 Streamlit 展示 Plotly 图表
            self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

        def save_metrics(self, metrics):
            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_name = f"comparison_performance_ga.csv"
            file_path = os.path.join(dataset_folder, file_name)

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)


    class SA_with_Batch:
        def __init__(self, initial_temperature, cooling_rate, min_temperature, max_iterations, lambda_1, lambda_2,
                     lambda_3, lambda_4, num_positions, dataset_name, objectives, use_adaptive):
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
            self.objectives = objectives
            self.use_adaptive = use_adaptive

            self.best_position = None
            self.best_score = np.inf
            self.worst_score = -np.inf  # 初始化最差得分
            self.convergence_data = []
            self.temperature_data = []
            self.adaptive_param_data = []
            self.start_time = None

            self.cache = {}
            self.score_changes = []  # 初始化得分变化列表

            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

        def evaluate_with_cache(self, position):
            return evaluate_with_cache(self.cache, position, self.evaluate)

        def evaluate(self, position):
            try:
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
                # logging.error(f"Error in evaluation: {e}")
                return np.inf

        def optimize(self):
            initial_position = np.random.randint(0, self.num_positions, size=num_plates)
            return self.optimize_from_position(initial_position)

        def optimize_from_position(self, initial_position):
            current_temperature = self.initial_temperature
            current_position = initial_position
            current_score = self.evaluate_with_cache(current_position)

            self.best_position = current_position.copy()
            self.best_score = current_score
            self.worst_score = current_score  # 设置初始最差得分
            self.start_time = time.time()

            # 初始化性能指标
            scores = []  # 所有得分
            unsuccessful_attempts = 0  # 失败次数

            st.info("SA Optimization started...")
            with st.spinner("Running SA Optimization..."):
                for iteration in range(self.max_iterations):
                    if current_temperature < self.min_temperature:
                        break

                    new_positions = [current_position.copy() for _ in range(5)]
                    for new_position in new_positions:
                        random_index = np.random.randint(0, len(current_position))
                        new_position[random_index] = np.random.randint(0, self.num_positions)

                    new_scores = evaluate_parallel(new_positions, self.evaluate_with_cache)

                    best_new_score = min(new_scores)
                    best_new_position = new_positions[new_scores.index(best_new_score)]

                    delta = best_new_score - current_score
                    if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                        current_position = best_new_position
                        current_score = best_new_score
                    else:
                        unsuccessful_attempts += 1  # 记录未成功的尝试

                    if current_score < self.best_score:
                        self.best_score = current_score
                        self.best_position = current_position.copy()

                    if current_score > self.worst_score:
                        self.worst_score = current_score  # 更新最差得分

                    # 更新得分列表和得分变化
                    scores.append(current_score)
                    self.score_changes.append(delta)  # 使用实例变量

                    current_temperature, self.cooling_rate = apply_adaptive_sa(
                        current_temperature, self.cooling_rate, delta, self.use_adaptive)

                    if self.use_adaptive:
                        self.record_adaptive_params()

                    self.convergence_data.append([iteration + 1, self.best_score])
                    self.temperature_data.append(current_temperature)

                    self.update_convergence_plot(iteration + 1)
                    # logging.info(
                    #     f"Iteration {iteration + 1}/{self.max_iterations}, Best Score: {self.best_score}, Temperature: {current_temperature}")

                st.success("Optimization complete!")

            # 计算平均得分和标准差
            avg_score = np.mean(scores)
            score_std = np.std(scores)
            total_attempts = len(scores)

            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed, avg_score, score_std, unsuccessful_attempts)  # 保存性能指标

            # 优化结束后，保存历史收敛数据
            history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "SA")
            save_convergence_history(self.convergence_data, "SA", self.dataset_name, history_data_dir)

            return self.best_position, self.best_score

        def record_adaptive_params(self):
            self.adaptive_param_data.append({'cooling_rate': self.cooling_rate})

        def update_convergence_plot(self, current_iteration):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]
            temperature_data = self.temperature_data

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=iteration_data, y=temperature_data, mode='lines+markers', name='Temperature',
                           line=dict(dash='dash')),
                secondary_y=True
            )

            fig.update_layout(
                title=f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}',
                xaxis_title='Iterations',
                legend=dict(x=0.75, y=1)
            )

            fig.update_yaxes(title_text="Best Score", secondary_y=False)
            fig.update_yaxes(title_text="Temperature", secondary_y=True)

            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

            if self.use_adaptive:
                self.update_adaptive_param_plot()

        def update_adaptive_param_plot(self):
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            cooling_rate_data = [x['cooling_rate'] for x in self.adaptive_param_data]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iteration_data, y=cooling_rate_data, mode='lines+markers', name='Cooling Rate')
            )
            fig.update_layout(
                title="Adaptive Parameter Changes",
                xaxis_title="Iterations",
                yaxis_title="Cooling Rate",
                legend=dict(x=0.75, y=1)
            )

            self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

        def save_metrics(self, time_elapsed, avg_score, score_std, unsuccessful_attempts):
            iterations = len(self.convergence_data)
            total_improvement = np.sum(self.score_changes)  # 使用实例变量
            self.worst_score = max(self.score_changes)  # 更新最差得分

            save_performance_metrics(
                self.best_score,
                self.worst_score,
                total_improvement,
                total_improvement,  # 或者替换为您需要的其他参数
                iterations,
                time_elapsed,
                self.convergence_data,
                len(self.adaptive_param_data),
                self.dataset_name,
                "SA"
            )


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
            self.convergence_plot_placeholder = st.empty()  # 用于展示收敛曲线的占位符

        def optimize(self):
            st.info("DE Optimization started...")  # 提供优化开始信息
            with st.spinner("Running DE Optimization..."):  # 提供运行时加载提示
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

                # 优化结束后，保存历史收敛数据
                history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "DE")
                save_convergence_history(self.convergence_data, "DE", self.dataset_name, history_data_dir)

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
            # 动态更新收敛曲线，使用 Plotly
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'))
            fig.update_layout(
                title=f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}',
                xaxis_title='Iterations',
                yaxis_title='Best Score',
                legend=dict(x=0.75, y=1)
            )

            # 使用 Streamlit 的空占位符更新图表
            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

            # 保存收敛数据
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

        def get_stable_iterations(self):
            stable_threshold = 1e-3  # 可以根据具体情况调整
            stable_iterations = 0
            for i in range(1, len(self.convergence_data)):
                if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class PSO_SA_Optimizer:
        def __init__(self, num_particles, num_positions, w, c1, c2, max_iter_pso,
                     initial_temperature, cooling_rate, min_temperature, max_iterations_sa,
                     lambda_1, lambda_2, lambda_3, lambda_4, dataset_name, objectives, use_adaptive):
            # 保存 dataset_name 为类的属性
            self.dataset_name = dataset_name

            # 初始化 PSO 和 SA 参数
            self.pso_optimizer = PSO_with_Batch(
                num_particles=num_particles,
                num_positions=num_positions,
                w=w, c1=c1, c2=c2, max_iter=max_iter_pso,
                lambda_1=lambda_1, lambda_2=lambda_2,
                lambda_3=lambda_3, lambda_4=lambda_4,
                dataset_name=dataset_name,
                objectives=objectives,
                use_adaptive=use_adaptive
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
                objectives=objectives,
                use_adaptive=use_adaptive
            )

            self.best_position = None  # 初始化 best_position 属性
            self.best_score = None  # 保存最终的 best_score
            self.convergence_data_pso_sa = []  # 存储混合优化的收敛数据
            self.start_time = None

        def optimize(self):
            self.start_time = time.time()

            # 运行 PSO 优化

            self.pso_optimizer.optimize()

            # 获取 PSO 最优解，作为 SA 初始解
            initial_position_for_sa = self.pso_optimizer.best_position

            # 运行 SA 进行局部优化
            best_position_sa, best_score_sa = self.sa_optimizer.optimize_from_position(initial_position_for_sa)

            # 保存最终的最佳位置和得分
            self.best_position = best_position_sa
            self.best_score = best_score_sa

            # 保存收敛数据
            self.convergence_data_pso_sa.extend(self.pso_optimizer.convergence_data)
            self.convergence_data_pso_sa.extend(self.sa_optimizer.convergence_data)

            # 保存收敛数据和性能指标
            self.save_convergence_data_pso_sa()
            time_elapsed = time.time() - self.start_time
            self.save_performance_metrics(time_elapsed, best_score_sa)

            return self.best_position, self.best_score

        def save_convergence_data_pso_sa(self):
            # 保存收敛数据
            dataset_folder = self.dataset_name.split('.')[0]  # 确保 dataset_name 已被赋值
            convergence_data_dir = os.path.join("result/ConvergenceData", dataset_folder)
            os.makedirs(convergence_data_dir, exist_ok=True)

            convergence_data_df = pd.DataFrame(self.convergence_data_pso_sa, columns=['Iteration', 'Best Score'])
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_psosa.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

        def save_performance_metrics(self, time_elapsed, best_score_sa):
            # 保存性能指标
            iterations = len(self.convergence_data_pso_sa)
            worst_score = max([data[1] for data in self.convergence_data_pso_sa])
            best_improvement = abs(worst_score - best_score_sa)
            average_improvement = best_improvement / iterations if iterations > 0 else 0
            convergence_rate_value = (self.convergence_data_pso_sa[-1][1] - self.convergence_data_pso_sa[0][
                1]) / iterations
            relative_error_value = abs(best_score_sa) / (abs(best_score_sa) + 1e-6)
            stable_iterations = self.get_stable_iterations()

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

            dataset_folder = f"result/comparison_performance/{self.dataset_name.split('.')[0]}"
            os.makedirs(dataset_folder, exist_ok=True)
            file_path = os.path.join(dataset_folder, 'comparison_performance_psosa.csv')

            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_path, index=False)

        def get_stable_iterations(self):
            # 获取稳定的迭代次数
            stable_threshold = 1e-3
            stable_iterations = 0
            for i in range(1, len(self.convergence_data_pso_sa)):
                if abs(self.convergence_data_pso_sa[i][1] - self.convergence_data_pso_sa[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class CoEA_with_Batch:
        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions, dataset_name, objectives, use_adaptive=False):
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
            self.use_adaptive = use_adaptive
            self.adaptive_param_data = []

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

            # Streamlit 占位符
            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

        def fitness(self, individual):
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
            st.info("CoEA Optimization started...")  # 提供优化开始信息
            with st.spinner("Running CoEA Optimization..."):  # 提供运行时加载提示
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
                    # 自适应调节
                    if self.use_adaptive:
                        self.mutation_rate, self.crossover_rate = apply_adaptive_coea(
                            self.mutation_rate, self.crossover_rate, best_score_gen, self.best_score, self.use_adaptive)
                        self.record_adaptive_params()

                        # 每代更新自适应参数调节图
                        self.update_adaptive_param_plot()


                    self.convergence_data.append([generation + 1, self.best_score])
                    self.update_convergence_plot(generation + 1)
                    print(f'Generation {generation + 1}/{self.generations}, Best Score: {self.best_score}')
                time_elapsed = time.time() - self.start_time
                self.save_performance_metrics(time_elapsed)

                # 优化结束后，保存历史收敛数据
                history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "CoEA")
                save_convergence_history(self.convergence_data, "CoEA", self.dataset_name, history_data_dir)

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

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'),
                          secondary_y=False)

            fig.update_layout(
                title=f'Convergence Curve - Generation {current_generation}, Best Score {self.best_score}',
                xaxis_title='Generations', legend=dict(x=0.75, y=1))
            fig.update_yaxes(title_text="Best Score", secondary_y=False)

            # 使用 Streamlit 展示 Plotly 图表
            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

        def record_adaptive_params(self):
            self.adaptive_param_data.append(
                {'mutation_rate': self.mutation_rate, 'crossover_rate': self.crossover_rate})

        def update_adaptive_param_plot(self):
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            mutation_rate_data = [x['mutation_rate'] for x in self.adaptive_param_data]
            crossover_rate_data = [x['crossover_rate'] for x in self.adaptive_param_data]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iteration_data, y=mutation_rate_data, mode='lines+markers', name='Mutation Rate'))
            fig.add_trace(
                go.Scatter(x=iteration_data, y=crossover_rate_data, mode='lines+markers', name='Crossover Rate'))

            fig.update_layout(title="Adaptive Parameter Changes", xaxis_title="Generations",
                              yaxis_title="Parameter Values",
                              legend=dict(x=0.75, y=1))

            # 使用 Streamlit 展示自适应参数调节图
            self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

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
            stable_threshold = 1e-3
            stable_iterations = 0
            for i in range(1, len(self.convergence_data)):
                if abs(self.convergence_data[i][1] - self.convergence_data[i - 1][1]) < stable_threshold:
                    stable_iterations += 1
            return stable_iterations


    class EDA_with_Batch:
        def __init__(self, pop_size, num_positions, max_iter, mutation_rate, crossover_rate, lambda_1, lambda_2,
                     lambda_3, lambda_4, dataset_name, objectives, use_adaptive):
            self.pop_size = pop_size  # 种群大小
            self.num_positions = num_positions  # 库区/垛位数量
            self.max_iter = max_iter  # 最大迭代次数
            self.lambda_1 = lambda_1  # 高度相关的权重参数
            self.lambda_2 = lambda_2  # 翻垛相关的权重参数
            self.lambda_3 = lambda_3
            self.lambda_4 = lambda_4
            self.mutation_rate = mutation_rate  # 变异率
            self.crossover_rate = crossover_rate  # 交叉率
            self.dataset_name = dataset_name  # 数据集名称
            self.population = np.random.randint(0, num_positions, size=(pop_size, num_plates))  # 随机初始化种群
            self.best_position = None  # 最佳解
            self.best_score = np.inf  # 最优得分
            self.worst_score = -np.inf  # 最差得分
            self.convergence_data = []  # 用于保存收敛数据
            self.start_time = None  # 记录开始时间
            self.objectives = objectives  # OptimizationObjectives 实例
            self.use_adaptive = use_adaptive  # 是否使用自适应参数
            self.adaptive_param_data = []  # 用于保存自适应参数

            # Streamlit 占位符
            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

        def optimize(self):
            # 提供优化开始信息并显示运行时加载提示
            st.info("EDA Optimization started...")
            with st.spinner("Running EDA Optimization..."):
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

                    # 如果使用自适应参数调节
                    if self.use_adaptive:
                        self.mutation_rate, self.crossover_rate = apply_adaptive_eda(
                            self.mutation_rate, self.crossover_rate, self.best_score, self.use_adaptive)
                        self.record_adaptive_params()

                        # 每代更新自适应参数调节图
                        self.update_adaptive_param_plot()

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

                # 优化结束后，保存历史收敛数据
                history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "EDA")
                save_convergence_history(self.convergence_data, "EDA", self.dataset_name, history_data_dir)

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

            # 使用 Plotly 绘制收敛曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'))
            fig.update_layout(title=f'Convergence Curve - Iteration {current_iteration}',
                              xaxis_title='Iterations', yaxis_title='Best Score')

            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

        def record_adaptive_params(self):
            self.adaptive_param_data.append({
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            })

        def update_adaptive_param_plot(self):
            # 绘制自适应参数变化图
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            mutation_rate_data = [x['mutation_rate'] for x in self.adaptive_param_data]
            crossover_rate_data = [x['crossover_rate'] for x in self.adaptive_param_data]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=iteration_data, y=mutation_rate_data, mode='lines+markers', name='Mutation Rate'))
            fig.add_trace(
                go.Scatter(x=iteration_data, y=crossover_rate_data, mode='lines+markers', name='Crossover Rate'))

            fig.update_layout(title="Adaptive Parameter Changes", xaxis_title="Iterations",
                              yaxis_title="Parameter Values")

            self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

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

            # Streamlit 占位符
            self.convergence_plot_placeholder = st.empty()

        def optimize(self):
            st.info("ACO Optimization started...")  # 提供优化开始信息
            with st.spinner("Running ACO Optimization..."):  # 提供运行时加载提示
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

                        score = np.sum(score)  # 使用 np.sum() 来确保 score 是标量
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

                # 优化结束后，保存历史收敛数据
                history_data_dir = os.path.join("result/History_ConvergenceData", self.dataset_name, "ACO")
                save_convergence_history(self.convergence_data, "ACO", self.dataset_name, history_data_dir)

        def construct_solution(self):
            solution = []
            for plate_idx in range(num_plates):
                probabilities = self.calculate_transition_probabilities(plate_idx)
                position = np.random.choice(self.num_positions, p=probabilities)
                solution.append(position)
            return np.array(solution)

        def calculate_transition_probabilities(self, plate_idx):
            pheromones = self.pheromone_matrix[plate_idx]
            desirability = 1.0 / (np.arange(1, self.num_positions + 1))  # 启发因子

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
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            # 使用 Plotly 创建收敛曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'))

            # 设置图表布局
            fig.update_layout(
                title=f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}',
                xaxis_title='Iterations',
                yaxis_title='Best Score'
            )

            # 使用 Streamlit 展示 Plotly 图表
            st.plotly_chart(fig, use_container_width=True)

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


    class MOEAD_with_Batch:
        def __init__(self, population_size, generations, lambda_1, lambda_2, lambda_3, lambda_4, num_positions,
                     dataset_name, objectives, T=20):
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

            # Streamlit 占位符
            self.convergence_plot_placeholder = st.empty()

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
            global heights
            temp_heights = heights.copy()

            # 计算适应度得分
            combined_movement_turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(
                individual, temp_heights)
            energy_time_penalty = self.objectives.minimize_outbound_energy_time_with_batch(individual)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(individual)
            space_utilization = self.objectives.maximize_space_utilization_v3(individual)

            # 确保 score1 和 score2 是标量值
            score1 = np.sum(combined_movement_turnover_penalty) + np.sum(energy_time_penalty)
            score2 = np.sum(balance_penalty) - np.sum(space_utilization)

            # 返回标量值的数组
            return np.array([score1, score2])

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
            return np.max(weight * np.abs(fitness))  # 使用 np.max 逐元素处理

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
            st.info("MOEA/D Optimization started...")  # 提供优化开始信息
            with st.spinner("Running MOEA/D Optimization..."):  # 提供运行时加载提示
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

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'))

            fig.update_layout(
                title=f'Convergence Curve - Generation {current_generation}',
                xaxis_title='Generations',
                yaxis_title='Best Score',
                legend=dict(x=0.75, y=1)
            )

            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

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


    if selected_algorithm == "SA (Simulated Annealing)":
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
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        }

        run_optimization(SA_with_Batch, sa_params, df, area_positions, output_dir_base, "sa")

    elif selected_algorithm == "GA (Genetic Algorithm)":
        ga_params = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'generations': generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),
            'dataset_name': dataset_name,
            'objectives': objectives,
            'plates': plates,  # 假设 plates 是数据集中的钢板相关信息
            'delivery_times': delivery_times,  # 假设是与钢板相关的交货时间信息
            'batches': batches,  # 假设这是与钢板相关的批次信息
            'use_adaptive': use_adaptive  # 自适应参数调整
        }

        run_optimization(GA_with_Batch, ga_params, df, area_positions, output_dir_base, "ga")


    elif selected_algorithm == "PSO (Particle Swarm Optimization)":
        # 调用优化函数
        pso_params = {
            'num_particles': num_particles,  # 用户指定的粒子数量
            'num_positions': len(Dki),
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter': max_iterations_pso,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives,
            'use_adaptive': use_adaptive  # 如果需要自适应调节
        }

        run_optimization(PSO_with_Batch, pso_params, df, area_positions, output_dir_base, "pso")

    elif selected_algorithm == "PSO + SA (Hybrid Optimization)":

        # 创建混合优化参数
        hybrid_params = {
            'num_particles': num_particles,
            'num_positions': len(Dki),
            'w': w,
            'c1': c1,
            'c2': c2,
            'max_iter_pso': max_iter_pso,
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'max_iterations_sa': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives,
            'use_adaptive': use_adaptive  # 将此参数传递给混合优化器
        }

        # 运行混合优化器
        run_optimization(PSO_SA_Optimizer, hybrid_params, df, area_positions, output_dir_base, "psosa")

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
        # 初始化 CoEA 参数字典
        coea_params = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'generations': generations,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),  # 假设 Dki 是库区或垛位的数量
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives  # 假设 objectives 是已定义的目标函数实例
        }
        run_optimization(CoEA_with_Batch, coea_params, df, area_positions, output_dir_base, "coea")

    elif selected_algorithm == "EDA (Estimation of Distribution Algorithm)":
        # 准备 EDA 优化算法的参数
        eda_params = {
            'pop_size': pop_size,
            'num_positions': len(Dki),
            'max_iter': max_iterations_eda,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'dataset_name': selected_dataset if data_choice == "Use system dataset" else uploaded_file.name,
            'objectives': objectives,
            'use_adaptive': use_adaptive
        }

        # 调用优化函数运行 EDA 算法
        run_optimization(EDA_with_Batch, eda_params, df, area_positions, output_dir_base, "eda")

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
        run_optimization(MOEAD_with_Batch, moead_params, df, area_positions, output_dir_base, "moead")