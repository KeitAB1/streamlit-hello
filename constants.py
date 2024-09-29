# constants.py

import numpy as np

# 创建用于保存图像的目录
OUTPUT_DIR = "stack_distribution_plots/final_stack_distribution"
CONVERGENCE_DIR = "result/ConvergenceData"
DATA_DIR = "test/steel_data"
TEST_DATA_PATH = f"{DATA_DIR}/test_data.csv"

# 默认的库区和垛位配置
DEFAULT_AREA_POSITIONS = {
    0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
    1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
    2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    3: [(0, 0), (0, 1), (1, 0), (1, 1)],
    4: [(0, 0), (0, 1), (1, 0), (1, 1)],
    5: [(0, 0), (0, 1), (1, 0), (1, 1)]
}

DEFAULT_STACK_DIMENSIONS = {
    0: [(6000, 3000), (9000, 3000), (9000, 3000), (6000, 3000), (9000, 3000), (9000, 3000), (9000, 4000), (15000, 4000)],
    1: [(6000, 3000), (9000, 3000), (9000, 3000), (6000, 3000), (9000, 3000), (9000, 3000), (15000, 4000), (9000, 4000)],
    2: [(12000, 3000), (12000, 3000), (12000, 3000), (12000, 3000), (12000, 4000), (12000, 4000)],
    3: [(9000, 5000), (15000, 5000), (9000, 5000), (15000, 5000)],
    4: [(18000, 5000), (6000, 5000), (18000, 5000), (6000, 5000)],
    5: [(12000, 5000), (12000, 5000), (12000, 5000), (12000, 5000)]
}

# 速度相关常量
HORIZONTAL_SPEED = 72 * 1000 / 60  # 水平速度：72m/min，转换为 mm/s
VERTICAL_SPEED = 15 * 1000 / 60  # 垂直速度：15m/min，转换为 mm/s
STACK_FLIP_TIME_PER_PLATE = 10  # 每次翻垛需要10秒

# 入库口和出库口的坐标
INBOUND_POINT = (41500, 3000)  # 入库口坐标
OUTBOUND_POINT = (41500, 38000)  # 出库口坐标

# 库区最大堆垛容量
STACK_HEIGHT_LIMIT = 3000
Dki = np.array([
    np.sum([dim[0] * dim[1] * STACK_HEIGHT_LIMIT for dim in DEFAULT_STACK_DIMENSIONS[0]]),
    np.sum([dim[0] * dim[1] * STACK_HEIGHT_LIMIT for dim in DEFAULT_STACK_DIMENSIONS[1]]),
    np.sum([dim[0] * dim[1] * STACK_HEIGHT_LIMIT for dim in DEFAULT_STACK_DIMENSIONS[2]]),
    np.sum([dim[0] * dim[1] * STACK_HEIGHT_LIMIT for dim in DEFAULT_STACK_DIMENSIONS[3]]),
    np.sum([dim[0] * dim[1] * STACK_HEIGHT_LIMIT for dim in DEFAULT_STACK_DIMENSIONS[4]]),
    np.sum([dim[0] * dim[1] * STACK_HEIGHT_LIMIT for dim in DEFAULT_STACK_DIMENSIONS[5]])
])
