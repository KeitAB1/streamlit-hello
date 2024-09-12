import numpy as np
from home import minimize_stack_movements_and_turnover, minimize_outbound_energy_time_with_batch, \
    maximize_inventory_balance_v2, maximize_space_utilization_v3


# 定义 Particle 类
class Particle:
    def __init__(self, num_positions, num_plates):
        # 随机初始化粒子的位置
        self.position = np.random.randint(0, num_positions, size=num_plates)
        # 初始化粒子的速度为 0
        self.velocity = np.zeros(num_plates)
        # 记录历史最佳位置
        self.best_position = self.position.copy()
        # 初始化最佳得分为正无穷
        self.best_score = np.inf

    def update_velocity(self, gbest_position, w, c1, c2):
        # 随机生成两个系数用于 PSO 更新
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        # 计算认知部分
        cognitive = c1 * r1 * (self.best_position - self.position)
        # 计算社会部分
        social = c2 * r2 * (gbest_position - self.position)
        # 更新速度
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, num_positions):
        # 更新位置，确保位置在合法范围内
        self.position = np.clip(self.position + self.velocity, 0, num_positions - 1).astype(int)


# 定义 PSO_with_Batch 类
class PSO_with_Batch:
    def __init__(self, num_particles, num_positions, num_plates, w, c1, c2, max_iter, lambda_1, lambda_2, lambda_3,
                 lambda_4):
        # 初始化参数
        self.num_particles = num_particles
        self.num_positions = num_positions
        self.num_plates = num_plates
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        # 初始化粒子群
        self.particles = [Particle(num_positions, num_plates) for _ in range(num_particles)]
        self.gbest_position = None
        self.gbest_score = np.inf

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                if self.gbest_position is None:
                    self.gbest_position = particle.position.copy()

                # 创建一个临时高度矩阵用于计算得分
                temp_heights = np.zeros(self.num_positions)

                # 计算目标函数得分
                combined_score = self.calculate_combined_score(particle.position, temp_heights)

                # 更新粒子历史最佳位置和全局最佳位置
                if combined_score is not None and combined_score < particle.best_score:
                    particle.best_score = combined_score
                    particle.best_position = particle.position.copy()

                if combined_score is not None and combined_score < self.gbest_score:
                    self.gbest_score = combined_score
                    self.gbest_position = particle.position.copy()

            # 更新每个粒子的速度和位置
            for particle in self.particles:
                particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                particle.update_position(self.num_positions)

    def calculate_combined_score(self, particle_position, heights):
        # 目标函数的集成计算，具体逻辑在 home.py 中实现
        movement_turnover_penalty = minimize_stack_movements_and_turnover(particle_position, heights, plates,
                                                                          delivery_times, batches)
        energy_time_penalty = minimize_outbound_energy_time_with_batch(particle_position, plates, heights)
        balance_penalty = maximize_inventory_balance_v2(particle_position, plates)
        space_utilization = maximize_space_utilization_v3(particle_position, plates, Dki)

        # 返回综合得分
        return (self.lambda_1 * movement_turnover_penalty +
                self.lambda_2 * energy_time_penalty +
                self.lambda_3 * balance_penalty -
                self.lambda_4 * space_utilization)
