import numpy as np
import time
import matplotlib.pyplot as plt
from utils import save_convergence_plot, save_performance_metrics


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
        self.num_positions = num_positions
        self.dataset_name = dataset_name
        self.objectives = objectives

        self.best_position = None
        self.best_score = np.inf
        self.convergence_data = []
        self.start_time = None

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

    def optimize(self):
        return self.optimize_from_position(np.random.randint(0, self.num_positions, size=num_plates))

    def optimize_from_position(self, initial_position):
        global heights
        current_temperature = self.initial_temperature
        current_position = initial_position
        current_score = self.evaluate(current_position)

        self.best_position = current_position.copy()
        self.best_score = current_score
        self.start_time = time.time()

        for iteration in range(self.max_iterations):
            if current_temperature < self.min_temperature:
                break

            new_position = current_position.copy()
            random_index = np.random.randint(0, len(current_position))
            new_position[random_index] = np.random.randint(0, self.num_positions)
            new_score = self.evaluate(new_position)

            delta = new_score - current_score
            if delta < 0 or np.random.rand() < np.exp(-delta / current_temperature):
                current_position = new_position
                current_score = new_score

            if current_score < self.best_score:
                self.best_score = current_score
                self.best_position = current_position.copy()

            current_temperature *= self.cooling_rate
            self.convergence_data.append([iteration + 1, self.best_score])
            self.update_convergence_plot(iteration + 1)

        time_elapsed = time.time() - self.start_time
        self.save_metrics(time_elapsed)

        return self.best_position, self.best_score

    def update_convergence_plot(self, current_iteration):
        iteration_data = [x[0] for x in self.convergence_data]
        score_data = [x[1] for x in self.convergence_data]

        plt.figure(figsize=(8, 4))
        plt.plot(iteration_data, score_data, '-o', color='blue', label='Best Score')
        plt.xlabel('Iterations')
        plt.ylabel('Best Score')
        plt.title(f'Convergence Curve - Iteration {current_iteration}, Best Score {self.best_score}')
        plt.legend()

        save_convergence_plot(self.convergence_data, current_iteration, self.best_score, "SA", self.dataset_name)

    def save_metrics(self, time_elapsed):
        iterations = len(self.convergence_data)
        save_performance_metrics(self.best_score, iterations, time_elapsed, self.convergence_data, self.dataset_name,
                                 "SA")
