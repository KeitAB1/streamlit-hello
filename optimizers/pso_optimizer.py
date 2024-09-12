import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from optimization_functions import minimize_stack_movements_and_turnover, minimize_outbound_energy_time_with_batch, \
    maximize_inventory_balance_v2, maximize_space_utilization_v3





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


# 初始化 df 为 None
df = None

# 优先检查 test_data.csv 是否存在
if os.path.exists(test_data_path):
    # 直接读取已存在的 test_data.csv
    df = pd.read_csv(test_data_path)




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

    # 动态显示收敛曲线的占位符
    convergence_plot_placeholder = st.empty()
    # 动态显示收敛曲线的占位符
    convergence_plot_placeholder = st.empty()

    # 动态显示收敛曲线的占位符
    convergence_plot_placeholder = st.empty()


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

        # PSO类中更新收敛曲线的方法
        def update_convergence_plot(self, current_iteration):
            iteration_data = [x[0] for x in self.convergence_data]
            score_data = [x[1] for x in self.convergence_data]

            # 调整图表的尺寸和显示方式
            plt.figure(figsize=(8, 4))
            plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
            plt.xlabel('Iterations')
            plt.ylabel('Best Score')
            plt.title(f'Convergence Curve - Iteration {current_iteration}, Best Score {self.gbest_score}')
            plt.legend()

            # 使用 Streamlit 的空占位符更新图表，并放入 expander 容器中
            convergence_plot_placeholder.pyplot(plt)

            # 保存收敛数据到 CSV 文件
            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
            convergence_data_dir = "result/ConvergenceData"
            os.makedirs(convergence_data_dir, exist_ok=True)
            convergence_data_path = os.path.join(convergence_data_dir, 'convergence_data_pso.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)

