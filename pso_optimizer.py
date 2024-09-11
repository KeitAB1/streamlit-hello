import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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
    def __init__(self, num_particles, num_positions, num_plates, w, c1, c2, max_iter, lambda_1, lambda_2, lambda_3, lambda_4,
                 minimize_stack_movements_and_turnover, minimize_outbound_energy_time_with_batch,
                 maximize_inventory_balance_v2, maximize_space_utilization_v3, plates, delivery_times, heights, Dki, batches, cols_per_area):
        self.num_particles = num_particles
        self.num_positions = num_positions
        self.num_plates = num_plates
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.particles = [Particle(num_positions, num_plates) for _ in range(self.num_particles)]
        self.gbest_position = None
        self.gbest_score = np.inf
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.convergence_data = []  # 初始化收敛数据
        self.minimize_stack_movements_and_turnover = minimize_stack_movements_and_turnover
        self.minimize_outbound_energy_time_with_batch = minimize_outbound_energy_time_with_batch
        self.maximize_inventory_balance_v2 = maximize_inventory_balance_v2
        self.maximize_space_utilization_v3 = maximize_space_utilization_v3
        self.plates = plates
        self.delivery_times = delivery_times
        self.heights = heights
        self.Dki = Dki
        self.batches = batches
        self.cols_per_area = cols_per_area  # 添加 cols_per_area 参数

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                if self.gbest_position is None:
                    self.gbest_position = particle.position.copy()

                temp_heights = self.heights.copy()

                # 计算目标函数的惩罚项
                combined_movement_turnover_penalty = self.minimize_stack_movements_and_turnover(
                    particle.position, temp_heights, self.plates, self.delivery_times, self.num_positions,
                    self.cols_per_area, self.batches)
                energy_time_penalty = self.minimize_outbound_energy_time_with_batch(particle.position, self.plates,
                                                                                   temp_heights, area_positions, stack_dimensions,
                                                                                   horizontal_speed, vertical_speed, conveyor_position_x, conveyor_position_y)
                balance_penalty = self.maximize_inventory_balance_v2(particle.position, self.plates, self.Dki, self.num_positions)
                space_utilization = self.maximize_space_utilization_v3(particle.position, self.plates, self.Dki)

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

            # 打印迭代信息
            print(f'Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.gbest_score}')

        # 更新最终高度
        self.update_final_heights()

    def update_final_heights(self):
        self.heights = np.zeros(len(self.Dki))
        for plate_idx, position in enumerate(self.gbest_position):
            area = position
            self.heights[area] += self.plates[plate_idx, 2]

    def save_convergence_to_csv(self, filepath):
        convergence_data_df = pd.DataFrame(self.convergence_data, columns=['Iteration', 'Best Score'])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        convergence_data_df.to_csv(filepath, index=False)



