# 导入更新后的库
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem  # 这里的 factory 已更新为 problems
from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination

# 获取问题定义
problem = get_problem("zdt1")  # zdt1 是一个经典的多目标优化问题

# 定义 NSGA-II 算法
algorithm = NSGA2(pop_size=100)

# 定义终止条件：设置最大代数
termination = get_termination("n_gen", 200)

# 使用 minimize 函数进行优化
res = minimize(problem,
               algorithm,
               termination,
               seed=1,  # 固定随机种子
               verbose=True)  # 输出优化过程中的信息

# 结果可视化
plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")  # 可视化 Pareto 前沿
plot.show()

# 输出 Pareto 最优解
print("Pareto Front Solutions: ", res.X)
print("Pareto Front Objectives: ", res.F)
