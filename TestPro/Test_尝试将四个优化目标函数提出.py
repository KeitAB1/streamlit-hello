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
from optimizers.sa_optimizer import  SA_with_Batch
from optimization_utils import apply_adaptive_sa  # 引入自适应控制参数调整函数
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from optimization_utils import evaluate_parallel, load_data_async, evaluate_with_cache, run_distributed_optimization

# 从 constants 文件中引入常量
from constants import OUTPUT_DIR, CONVERGENCE_DIR, DATA_DIR, TEST_DATA_PATH
from constants import DEFAULT_AREA_POSITIONS, DEFAULT_STACK_DIMENSIONS, HORIZONTAL_SPEED, VERTICAL_SPEED, STACK_FLIP_TIME_PER_PLATE, INBOUND_POINT, OUTBOUND_POINT, Dki

# Streamlit 页面配置
st.set_page_config(page_title="home", page_icon="⚙")

# 全局变量用于存储结果
heights = None

# 创建用于保存图像的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 定义保存路径为 test_data.csv
# test_data_path = os.path.join(data_dir, "test_data.csv")

st.title("Steel Plate Stacking Optimization")

# 并行计算适应度函数
def evaluate_parallel(positions, evaluate_func):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_func, positions))
    return results


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
    area_positions = DEFAULT_AREA_POSITIONS
    stack_dimensions = DEFAULT_STACK_DIMENSIONS

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
    use_adaptive = st.checkbox("Use Adaptive Parameter Adjustment", value=False)  # 新增选项


    if selected_algorithm == "SA (Simulated Annealing)":
        st.subheader("SA Parameters")
        initial_temperature = st.number_input("Initial Temperature", value=1000.0)
        w = st.slider("Inertia Weight (w)", 0.0, 1.0, 0.5)  # 默认为0.5
        c1 = st.slider("Cognitive Coefficient (c1)", 0.0, 3.0, 2.0)  # 默认为2.0
        c2 = st.slider("Social Coefficient (c2)", 0.0, 3.0, 2.0)  # 默认为2.0
        cooling_rate = st.slider("Cooling Rate", 0.0, 1.0, 0.9)
        min_temperature = st.number_input("Minimum Temperature", value=0.1)
        max_iterations_sa = st.number_input("Max Iterations", 1, 1000, 100)
        st.write("#### Optimize Target Weight")
        lambda_1 = st.number_input("Lambda 1 (Height Weight)", value=1.0)
        lambda_2 = st.number_input("Lambda 2 (Turnover Weight)", value=1.0)
        lambda_3 = st.number_input("Lambda 3 (Balance Weight)", value=1.0)
        lambda_4 = st.number_input("Lambda 4 (Space Utilization Weight)", value=1.0)






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

    # 初始化每个库区中的垛位高度
    heights = np.zeros(len(Dki))

    #  将交货时间从字符串转换为数值
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    delivery_times = (df['Delivery Time'] - df['Entry Time']).dt.days.values

    # 创建 OptimizationObjectives 实例
    objectives = OptimizationObjectives(
        plates=plates,
        heights=heights,
        delivery_times=(pd.to_datetime(df['Delivery Time']) - pd.to_datetime(df['Entry Time'])).dt.days.values,
        batches=df['Batch'].values,
        Dki=Dki,
        area_positions=area_positions,
        inbound_point=INBOUND_POINT,
        outbound_point=OUTBOUND_POINT,
        horizontal_speed=HORIZONTAL_SPEED,
        vertical_speed=VERTICAL_SPEED
    )

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
            self.use_adaptive = use_adaptive  # 自适应开关

            # 初始化位置和最佳解
            self.best_position = None
            self.best_score = np.inf
            self.convergence_data = []
            self.temperature_data = []  # 用于存储每次迭代的温度
            self.adaptive_param_data = []  # 用于存储自适应参数的变化
            self.start_time = None

            # 性能评价指标
            self.worst_score = -np.inf
            self.best_improvement = 0
            self.total_improvement = 0
            self.last_score = None
            self.stable_iterations = 0

            # 缓存结果，避免重复计算
            self.cache = {}

            # Streamlit 的动态图表占位符
            self.convergence_plot_placeholder = st.empty()
            self.adaptive_param_plot_placeholder = st.empty()

        # 使用缓存机制进行适应度评估
        def evaluate_with_cache(self, position):
            return evaluate_with_cache(self.cache, position, self.evaluate)

        # 定义适应度评估函数
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
                print(f"Error in evaluation: {e}")
                return np.inf

        # 优化入口
        def optimize(self):
            initial_position = np.random.randint(0, self.num_positions, size=num_plates)
            return self.optimize_from_position(initial_position)

        # 使用并行计算优化过程
        def optimize_from_position(self, initial_position):
            global heights
            current_temperature = self.initial_temperature
            current_position = initial_position
            current_score = self.evaluate_with_cache(current_position)

            self.best_position = current_position.copy()
            self.best_score = current_score
            self.last_score = current_score
            self.start_time = time.time()

            # Streamlit 提示优化开始
            st.info("Optimization started...")
            with st.spinner("Optimizing, please wait..."):

                for iteration in range(self.max_iterations):
                    if current_temperature < self.min_temperature:
                        break

                    # 生成多个新解进行并行评估
                    new_positions = [current_position.copy() for _ in range(5)]
                    for new_position in new_positions:
                        random_index = np.random.randint(0, len(current_position))
                        new_position[random_index] = np.random.randint(0, self.num_positions)

                    # 使用并行计算评估新解
                    new_scores = evaluate_parallel(new_positions, self.evaluate_with_cache)

                    # 选择最优解并更新
                    best_new_score = min(new_scores)
                    best_new_position = new_positions[new_scores.index(best_new_score)]

                    delta = best_new_score - current_score
                    if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                        current_position = best_new_position
                        current_score = best_new_score

                    improvement = self.last_score - current_score
                    if current_score < self.best_score:
                        self.best_score = current_score
                        self.best_position = current_position.copy()
                        self.best_improvement = max(self.best_improvement, improvement)

                    if current_score > self.worst_score:
                        self.worst_score = current_score

                    self.total_improvement += improvement

                    if improvement < 1e-6:
                        self.stable_iterations += 1

                    self.last_score = current_score

                    # 调整冷却速率（根据自适应开关）
                    current_temperature, self.cooling_rate = apply_adaptive_sa(
                        current_temperature, self.cooling_rate, improvement, self.use_adaptive)

                    # 如果启用了自适应，记录参数变化
                    if self.use_adaptive:
                        self.record_adaptive_params()

                    # 保存收敛数据
                    self.convergence_data.append([iteration + 1, self.best_score])
                    self.temperature_data.append(current_temperature)

                    # 实时更新收敛曲线
                    self.update_convergence_plot(iteration + 1)

                    # 打印每次迭代的最佳得分和温度
                    print(
                        f"Iteration {iteration + 1}/{self.max_iterations}, Best Score: {self.best_score}, Temperature: {current_temperature}")

                # 完成时提供通知
                st.success("Optimization complete!")

            time_elapsed = time.time() - self.start_time
            self.save_metrics(time_elapsed)

            return self.best_position, self.best_score

        # 记录自适应参数
        def record_adaptive_params(self):
            # 记录当前迭代的参数变化（这里假设是冷却速率 cooling_rate，也可以记录其他参数）
            self.adaptive_param_data.append({
                'cooling_rate': self.cooling_rate,
            })

        # 使用 Plotly 绘制交互式收敛曲线，包含温度变化
        def update_convergence_plot(self, current_iteration):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]
            temperature_data = self.temperature_data

            # 创建带有双y轴的子图
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 添加收敛曲线 (最佳得分)
            fig.add_trace(
                go.Scatter(x=iteration_data, y=score_data, mode='lines+markers', name='Best Score'),
                secondary_y=False
            )

            # 添加温度变化曲线
            fig.add_trace(
                go.Scatter(x=iteration_data, y=temperature_data, mode='lines+markers', name='Temperature',
                           line=dict(dash='dash')),
                secondary_y=True
            )

            # 设置布局
            fig.update_layout(
                title=f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}',
                xaxis_title='Iterations',
                legend=dict(x=0.75, y=1)  # 将图例放置在右上角
            )

            # 设置 y 轴标题
            fig.update_yaxes(title_text="Best Score", secondary_y=False)
            fig.update_yaxes(title_text="Temperature", secondary_y=True)

            # 使用 Streamlit 的 Plotly 图表，并使用空占位符实时更新
            self.convergence_plot_placeholder.plotly_chart(fig, use_container_width=True)

            # 如果启用了自适应参数，绘制参数变化图
            if self.use_adaptive:
                self.update_adaptive_param_plot()

            # 保存收敛图
            save_convergence_plot(self.convergence_data, current_iteration, self.best_score, "SA", self.dataset_name)

        # 绘制自适应参数变化图
        def update_adaptive_param_plot(self):
            iteration_data = list(range(1, len(self.adaptive_param_data) + 1))
            cooling_rate_data = [x['cooling_rate'] for x in self.adaptive_param_data]

            # 创建图表
            fig = go.Figure()

            # 绘制冷却速率变化
            fig.add_trace(
                go.Scatter(x=iteration_data, y=cooling_rate_data, mode='lines+markers', name='Cooling Rate')
            )

            # 设置布局
            fig.update_layout(
                title="Adaptive Parameter Changes",
                xaxis_title="Iterations",
                yaxis_title="Cooling Rate",
                legend=dict(x=0.75, y=1)  # 将图例放置在右上角
            )

            # 实时更新自适应参数变化图
            self.adaptive_param_plot_placeholder.plotly_chart(fig, use_container_width=True)

        # 保存性能指标
        def save_metrics(self, time_elapsed):
            iterations = len(self.convergence_data)
            save_performance_metrics(
                self.best_score, self.worst_score, self.best_improvement, self.total_improvement,
                iterations, time_elapsed, self.convergence_data, self.stable_iterations, self.dataset_name, "SA"
            )



    if selected_algorithm == "SA (Simulated Annealing)":
        sa_params = {
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            # 'w': w,  # 用户定义的惯性权重
            # 'c1': c1,  # 用户定义的认知系数
            # 'c2': c2,  # 用户定义的社会系数
            'min_temperature': min_temperature,
            'max_iterations': max_iterations_sa,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'num_positions': len(Dki),
            'dataset_name': dataset_name,
            'objectives': objectives,
            'use_adaptive': use_adaptive  # 如果使用自适应调节
        }

        run_optimization(SA_with_Batch, sa_params, df, area_positions, output_dir_base, "sa")








