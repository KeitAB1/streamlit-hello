from concurrent.futures import ThreadPoolExecutor
# import aiofiles
# import asyncio
import pandas as pd
import numpy as np
import dask
from dask import delayed

def apply_adaptive_eda(mutation_rate, crossover_rate, best_score, use_adaptive):

    if not use_adaptive:
        return mutation_rate, crossover_rate

    # 假设根据得分的改善情况，调整变异率和交叉率
    if best_score < 0.1:
        mutation_rate = max(0.01, mutation_rate * 0.95)  # 减小变异率
        crossover_rate = min(1.0, crossover_rate * 1.05)  # 增大交叉率
    else:
        mutation_rate = min(1.0, mutation_rate * 1.05)  # 增大变异率
        crossover_rate = max(0.01, crossover_rate * 0.95)  # 减小交叉率

    return mutation_rate, crossover_rate

def apply_adaptive_coea(mutation_rate, crossover_rate, current_score, best_score, use_adaptive):
    if not use_adaptive:
        return mutation_rate, crossover_rate

    # 如果当前得分较好，减少变异率并增加交叉率
    if current_score < best_score:
        mutation_rate = max(mutation_rate * 0.9, 0.01)
        crossover_rate = min(crossover_rate * 1.1, 1.0)
    else:
        # 如果当前得分较差，增加变异率并减少交叉率
        mutation_rate = min(mutation_rate * 1.1, 1.0)
        crossover_rate = max(crossover_rate * 0.9, 0.01)

    return mutation_rate, crossover_rate


def apply_adaptive_de(F, CR, trial_score, best_score, use_adaptive):
    """
    自适应调节差分进化中的 F 和 CR 参数。

    :param F: 当前变异因子 F
    :param CR: 当前交叉率 CR
    :param trial_score: 当前个体的适应度分数
    :param best_score: 全局最佳适应度分数
    :param use_adaptive: 是否使用自适应调节
    :return: 调整后的 F 和 CR 参数
    """
    if not use_adaptive:
        return F, CR

    # 根据当前适应度差距调整参数
    score_diff = abs(trial_score - best_score)

    # 调整 F（变异因子）：如果当前代的个体没有显著改进，减小 F，增强局部搜索
    if score_diff < 1e-6:
        F = max(0.4, F * 0.9)  # 减小 F，但保持在合理范围内
    else:
        F = min(1.5, F * 1.1)  # 如果有改进，增加 F

    # 调整 CR（交叉率）：如果改进较大，增加 CR，使得探索更多不同的解
    if score_diff > 1e-3:
        CR = min(1.0, CR * 1.1)
    else:
        CR = max(0.1, CR * 0.9)

    # 返回调整后的参数
    return F, CR


def apply_adaptive_ga(mutation_rate, crossover_rate, current_best_score, global_best_score, use_adaptive):
    """
    自适应调节遗传算法中的变异率和交叉率。

    :param mutation_rate: 当前变异率
    :param crossover_rate: 当前交叉率
    :param current_best_score: 当前代的最佳得分
    :param global_best_score: 全局最佳得分
    :param use_adaptive: 是否使用自适应调节
    :return: 调整后的变异率和交叉率
    """
    if not use_adaptive:
        return mutation_rate, crossover_rate

    # 计算当前代与全局最佳得分的差异
    score_diff = abs(current_best_score - global_best_score)

    # 调整变异率 (mutation_rate): 如果改进不明显，增加变异率以提高搜索多样性
    if score_diff < 1e-6:
        mutation_rate = min(0.2, mutation_rate * 1.1)  # 增加变异率，最大值为 0.2
    else:
        mutation_rate = max(0.01, mutation_rate * 0.9)  # 减少变异率，最小值为 0.01

    # 调整交叉率 (crossover_rate): 如果改进显著，增加交叉率以探索更多潜在解
    if score_diff > 1e-3:
        crossover_rate = min(1.0, crossover_rate * 1.1)  # 增加交叉率，最大值为 1.0
    else:
        crossover_rate = max(0.4, crossover_rate * 0.9)  # 减少交叉率，最小值为 0.4

    # 返回调整后的参数
    return mutation_rate, crossover_rate


# 自适应参数调节函数
def apply_adaptive_pso(w, c1, c2, delta, use_adaptive):
    if not use_adaptive:
        return w, c1, c2

    if delta > 0:
        w = min(w * 1.05, 0.9)
        c1 = min(c1 * 1.05, 2.0)
        c2 = max(c2 * 0.95, 1.5)
    else:
        w = max(w * 0.95, 0.4)
        c1 = max(c1 * 0.95, 1.5)
        c2 = min(c2 * 1.05, 2.5)

    return w, c1, c2



def adjust_cooling_rate(current_cooling_rate, improvement, adaptive_factor=0.95):
    """
    自适应控制参数调整：根据改进程度动态调整冷却速率。

    Parameters:
    current_cooling_rate (float): 当前冷却速率
    improvement (float): 本次迭代的解的改进程度
    adaptive_factor (float): 调整冷却速率的自适应因子

    Returns:
    float: 调整后的冷却速率
    """
    if improvement > 0.01:
        # 如果改进显著，降低冷却速率，探索更广泛的解空间
        return max(0.7, current_cooling_rate * adaptive_factor)
    else:
        # 改进不明显，增加冷却速率，加快收敛
        return min(0.99, current_cooling_rate * (1 / adaptive_factor))


def apply_adaptive_sa(temperature, cooling_rate, improvement, adaptive_enabled):
    """
    在模拟退火算法中，根据是否启用自适应调整冷却速率。

    Parameters:
    temperature (float): 当前温度
    cooling_rate (float): 当前冷却速率
    improvement (float): 当前解的改进程度
    adaptive_enabled (bool): 是否启用自适应控制

    Returns:
    float: 调整后的温度
    """
    if adaptive_enabled:
        cooling_rate = adjust_cooling_rate(cooling_rate, improvement)
    temperature *= cooling_rate
    return temperature, cooling_rate



# 并行计算：使用 ThreadPoolExecutor 并行执行多个适应度评估
def evaluate_parallel(positions, evaluate_func):
    """
    并行计算多个解的适应度。

    :param positions: List of positions (solutions) to evaluate
    :param evaluate_func: The function to evaluate a single solution
    :return: List of evaluated results for each position
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_func, positions))
    return results


# 异步 I/O 处理：异步读取 CSV 文件
async def load_data_async(file_path):
    """
    异步读取 CSV 文件。

    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the content of the CSV
    """
    async with aiofiles.open(file_path, mode='r') as f:
        content = await f.read()
    return pd.read_csv(content)


# 动态规划与启发式结合：通过缓存机制避免重复计算
def evaluate_with_cache(cache, position, evaluate_func):
    """
    使用缓存机制避免重复计算相同解的适应度。

    :param cache: A dictionary to store previously calculated results
    :param position: The current solution to evaluate
    :param evaluate_func: The function to evaluate the current solution
    :return: The evaluated result (either from cache or newly computed)
    """
    pos_key = tuple(position)
    if pos_key in cache:
        return cache[pos_key]
    else:
        result = evaluate_func(position)
        cache[pos_key] = result
        return result


# 分布式计算：使用 Dask 进行分布式计算任务
def run_distributed_optimization(positions, evaluate_func):
    """
    使用 Dask 将评估任务分发到多个计算节点进行分布式计算。

    :param positions: List of positions (solutions) to evaluate
    :param evaluate_func: The function to evaluate a single solution
    :return: List of evaluated results for each position
    """
    tasks = [delayed(evaluate_func)(pos) for pos in positions]
    results = dask.compute(*tasks)
    return results
