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

    class GA_with_Batch:
        def __init__(self, population_size, mutation_rate, crossover_rate, generations, lambda_1, lambda_2, lambda_3,
                     lambda_4, num_positions):
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

        def fitness(self, individual):
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

            return score

        def select(self):
            fitness_scores = np.array([self.fitness(ind) for ind in self.population])
            probabilities = np.exp(-fitness_scores / np.sum(fitness_scores))
            probabilities /= probabilities.sum()  # Normalize to get probabilities
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
                for i in range(0, self.population_size, 2):
                    parent1, parent2 = selected_population[i], selected_population[min(i + 1, self.population_size - 1)]
                    child1, child2 = self.crossover(parent1, parent2)
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

            convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Generation', 'Best Score'])
            convergence_data_path = os.path.join(convergence_dir, 'convergence_data_ga.csv')
            convergence_data_df.to_csv(convergence_data_path, index=False)
