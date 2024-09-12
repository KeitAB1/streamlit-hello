import os

import numpy as np
import pandas as pd

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


    # 计算入库或出库时间的通用函数
    def calculate_movement_time(plate_idx, area, stack_positions, plates, horizontal_speed, vertical_speed,
                                to_conveyor=True):
        x, y = area_positions[area][plate_idx % len(area_positions[area])]
        plate_length, plate_width = plates[plate_idx, 0], plates[plate_idx, 1]

        # 如果是入库，计算到传送带的距离；否则计算到出库口的距离
        if to_conveyor:
            # 水平方向：从传送带中央移动到库区垛位
            if area in [0, 1, 2]:  # 库区1、2、3
                distance_to_location = conveyor_position_x  # 传送带到库区1-3的距离
            else:  # 库区4、5、6
                distance_to_location = conveyor_position_y  # 传送带到库区4-6的距离
        else:
            # 出库口距离：库区1-3距离出库口15000mm，库区4-6距离出库口3000mm
            if area in [0, 1, 2]:
                distance_to_location = 15000
            else:
                distance_to_location = 3000

        # 计算移动距离
        total_distance_x = abs(
            distance_to_location - (x * (stack_dimensions[area][plate_idx % len(stack_dimensions[area])][0] + 500)))
        total_distance_y = y * 1000  # 假设垛位之间的间距为1000mm

        # 计算移动时间
        time_to_move_x = total_distance_x / horizontal_speed
        time_to_move_y = total_distance_y / vertical_speed

        return time_to_move_x + time_to_move_y


    # 目标函数1：最小化翻垛次数
    # 目标函数1：最小化翻垛次数
    def minimize_stack_movements_and_turnover(particle_positions, heights, plates, delivery_times, batches,
                                              weight_movement=1.0, weight_turnover=1.0):
        num_movements = 0
        total_turnover = 0
        batch_turnover = 0

        for plate_idx, position in enumerate(particle_positions):
            area = position
            plate_height = plates[plate_idx, 2]  # 厚度即为钢板的高度
            current_height = heights[area]

            # 判断是否需要翻垛（按高度限制）
            if current_height + plate_height > 3000:
                num_movements += 1  # 超过限制高度需要翻垛

            heights[area] += plate_height  # 更新当前垛位高度

        for i in range(len(particle_positions)):
            for j in range(i + 1, len(particle_positions)):
                # 考虑钢板的交货时间差来计算翻转次数
                time_diff = abs(delivery_times[i] - delivery_times[j])
                total_turnover += time_diff

                # 如果属于不同批次，增加翻堆次数
                if batches[i] != batches[j]:
                    batch_turnover += 1

        combined_score = weight_movement * num_movements + weight_turnover * (total_turnover + batch_turnover)
        return combined_score


    # 目标函数2：最小化出库能耗与时间
    def minimize_outbound_energy_time_with_batch(particle_positions, plates, heights):
        total_energy_time = 0

        # 批次排序，确保按Q1-Q15顺序出库
        sorted_batches = sorted(set(batches), key=lambda x: int(x[1:]))
        plate_indices_by_batch = {batch: [] for batch in sorted_batches}

        # 按批次将钢板索引分配
        for plate_idx, plate in enumerate(plates):
            batch = plate[4]  # 批次信息在第5列（索引4）
            plate_indices_by_batch[batch].append(plate_idx)

        # 按批次依次处理出库
        for batch in sorted_batches:
            for plate_idx in plate_indices_by_batch[batch]:
                position = particle_positions[plate_idx]
                area = position  # 获取钢板所在库区
                plate_height = plates[plate_idx, 2]  # 获取钢板厚度

                # 调用出库时间计算
                outbound_time = calculate_movement_time(plate_idx, area, area_positions, plates, horizontal_speed,
                                                        vertical_speed, to_conveyor=False)

                # 更新堆垛高度
                heights[area] -= plate_height
                total_energy_time += outbound_time

        return total_energy_time


    # 目标函数3：最大化库存均衡度
    def maximize_inventory_balance_v2(particle_positions, plates):
        total_variance = 0
        total_volume = np.sum(plates[:, 0] * plates[:, 1] * plates[:, 2])
        num_positions = len(Dki)
        mean_volume_per_position = total_volume / num_positions
        area_volumes = np.zeros(num_positions)

        # 计算每个库区的体积占用
        for plate_idx, position in enumerate(particle_positions):
            plate_volume = plates[plate_idx][0] * plates[plate_idx][1] * plates[plate_idx][2]
            area_volumes[position] += plate_volume

        # 计算均衡度的方差，通过减小方差使各个库区的体积更均衡
        for j in range(num_positions):
            total_variance += (area_volumes[j] - mean_volume_per_position) ** 2

        return total_variance / num_positions  # 方差越小，均衡度越好


    # 目标函数4：空间利用率最大化
    def maximize_space_utilization_v3(particle_positions, plates, Dki, alpha_1=1.0, epsilon=1e-6):
        total_space_utilization = 0
        for i in range(len(Dki)):
            used_volume = 0
            max_volume = Dki[i]

            for j in range(len(plates)):
                if particle_positions[j] == i:
                    plate_volume = plates[j][0] * plates[j][1] * plates[j][2]
                    used_volume += plate_volume

            if used_volume > 0:
                utilization = alpha_1 * max((max_volume - used_volume), epsilon) / used_volume
                total_space_utilization += utilization
            else:
                total_space_utilization += 0

        return total_space_utilization
