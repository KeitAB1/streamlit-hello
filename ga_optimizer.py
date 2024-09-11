# ga_optimizer.py
import numpy as np
import pandas as pd
from optimization_functions import minimize_stack_movements_and_turnover, minimize_outbound_energy_time_with_batch, \
    maximize_inventory_balance_v2, maximize_space_utilization_v3

# 适应度函数，用于计算个体的适应度
def fitness_function(particle_positions, plates, heights, delivery_times, Dki, lambda_1, lambda_2, lambda_3, lambda_4):
    # 计算每个目标函数的惩罚项和得分
    movement_turnover_penalty = minimize_stack_movements_and_turnover(particle_positions, heights, plates, delivery_times)
    energy_time_penalty = minimize_outbound_energy_time_with_batch(particle_positions, plates, heights)
    balance_penalty = maximize_inventory_balance_v2(particle_positions, plates)
    space_utilization = maximize_space_utilization_v3(particle_positions, plates, Dki)

    # 综合目标函数得分，返回总适应度分数
    score = (lambda_1 * movement_turnover_penalty +
             lambda_2 * energy_time_penalty +
             lambda_3 * balance_penalty -
             lambda_4 * space_utilization)
    return score

# 选择操作：轮盘赌选择
def selection(population, fitness_values):
    probabilities = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
    return population[selected_indices]

# 交叉操作：单点交叉
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# 变异操作：随机变异
def mutation(individual, num_positions, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(0, num_positions)
    return individual

# GA优化算法类
class GeneticAlgorithm:
    def __init__(self, population_size, num_generations, mutation_rate, num_particles, num_positions,
                 lambda_1, lambda_2, lambda_3, lambda_4):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_particles = num_particles
        self.num_positions = num_positions
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.population = np.random.randint(0, num_positions, size=(self.population_size, self.num_particles))  # 初始化种群
        self.convergence_curve = []

    def evolve(self, plates, heights, delivery_times, Dki):
        for generation in range(self.num_generations):
            # 计算当前种群中的适应度值
            fitness_values = np.array([fitness_function(individual, plates, heights.copy(), delivery_times, Dki,
                                                        self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4)
                                        for individual in self.population])
            # 保存每一代最优适应度
            self.convergence_curve.append(np.min(fitness_values))

            # 选择操作
            self.population = selection(self.population, fitness_values)

            # 生成新种群
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = self.population[i], self.population[i + 1]
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutation(child1, self.num_positions, self.mutation_rate))
                new_population.append(mutation(child2, self.num_positions, self.mutation_rate))

            self.population = np.array(new_population)

            # 打印每一代的最佳适应度值
            best_fitness = np.min(fitness_values)
            print(f'Generation {generation + 1}/{self.num_generations}, Best Fitness: {best_fitness}')

    def get_best_solution(self, plates, heights, delivery_times, Dki):
        fitness_values = np.array([fitness_function(individual, plates, heights.copy(), delivery_times, Dki,
                                                    self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4)
                                    for individual in self.population])
        best_index = np.argmin(fitness_values)
        return self.population[best_index], fitness_values[best_index]

    def save_convergence_to_csv(self, filepath):
        # 将收敛数据保存到CSV文件
        df_convergence = pd.DataFrame(self.convergence_curve, columns=['Best_Fitness'])
        df_convergence.to_csv(filepath, index_label='Generation')
