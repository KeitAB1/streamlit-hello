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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import ElementwiseProblem

# from utils import save_convergence_history

# 从 constants 文件中引入常量
from constants import OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH
from constants import DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS, HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE, INBOUND_POINT, OUTBOUND_POINT, Dki

# # 日志配置
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
    st.write("### Current Area Positions")
    for area, positions in area_positions.items():
        st.write(f"Area {area + 1} Stack Positions: {positions}")

    st.write("### Current Stack Dimensions")
    for area, dimensions in stack_dimensions.items():
        st.write(f"Area {area + 1} Stack Dimensions (LxW in mm): {dimensions}")

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
    algorithms = ["SA (Simulated Annealing)", "GA (Genetic Algorithm)", "PSO (Particle Swarm Optimization)",
                  "PSO + SA (Hybrid Optimization)", "ACO (Ant Colony Optimization)", "DE (Differential Evolution)",
                  "CoEA (Co-Evolutionary Algorithm)", "EDA (Estimation of Distribution Algorithm)",
                  "MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)",  "NSGA-II (Non-dominated Sorting Genetic Algorithm II)"]
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



    # 自定义多目标优化问题类
    class StackingProblem(ElementwiseProblem):
        def __init__(self, objectives, num_variables):
            super().__init__(n_var=num_variables,  # 决策变量的数量
                             n_obj=2,  # 目标数量
                             n_constr=0,  # 约束数量
                             xl=0,  # 决策变量的下界
                             xu=len(objectives.Dki) - 1)  # 决策变量的上界，假设决策是选择区域
            self.objectives = objectives  # 包含优化目标的实例

        def _evaluate(self, X, out, *args, **kwargs):
            heights = np.zeros(len(self.objectives.Dki))
            for i, plate_pos in enumerate(X):
                area = int(plate_pos)
                heights[area] += self.objectives.plates[i, 2]  # 累加钢板厚度

            # 定义多目标函数
            turnover_penalty = self.objectives.minimize_stack_movements_and_turnover(X)
            balance_penalty = self.objectives.maximize_inventory_balance_v2(X)

            # 两个目标：最小化翻堆次数和最小化高度
            out["F"] = [turnover_penalty, np.sum(heights)]


    # 运行 NSGA-II 优化
    def run_nsga2_optimization(objectives):
        num_plates = len(objectives.plates)  # 假设这个是钢板的数量
        problem = StackingProblem(objectives, num_plates)
        algorithm = NSGA2(pop_size=100)
        termination = get_termination("n_gen", 200)
        res = minimize(problem, algorithm, termination, seed=1, verbose=True)
        return res


    if selected_algorithm == "NSGA-II (Non-dominated Sorting Genetic Algorithm II)":
        st.write("### 使用 NSGA-II 优化多目标堆垛问题")
        res = run_nsga2_optimization(objectives)

        # 结果可视化
        plot = Scatter()
        plot.add(res.F, facecolor="none", edgecolor="red")  # 可视化 Pareto 前沿解
        st.pyplot(plot.show())

        # 输出 Pareto 前沿解
        st.write("Pareto Front Solutions: ", res.X)
        st.write("Pareto Front Objectives: ", res.F)

