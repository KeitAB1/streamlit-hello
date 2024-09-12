import numpy as np

# 计算入库或出库时间的通用函数
def calculate_movement_time(plate_idx, area, area_positions, plates, horizontal_speed, vertical_speed,
                            conveyor_position_x, conveyor_position_y, stack_dimensions, to_conveyor=True):
    # 确保 area_positions 和 area 都是有效的
    if area not in area_positions:
        raise ValueError(f"Area {area} not found in area_positions.")

    x, y = area_positions[area][plate_idx % len(area_positions[area])]
    plate_length, plate_width = plates[plate_idx, 0], plates[plate_idx, 1]

    # 如果是入库，计算到传送带的距离；否则计算到出库口的距离
    if to_conveyor:
        if area in [0, 1, 2]:
            distance_to_location = conveyor_position_x
        else:
            distance_to_location = conveyor_position_y
    else:
        if area in [0, 1, 2]:
            distance_to_location = 15000
        else:
            distance_to_location = 3000

    # 计算移动距离
    total_distance_x = abs(
        distance_to_location - (x * (stack_dimensions[area][plate_idx % len(stack_dimensions[area])][0] + 500)))
    total_distance_y = y * 1000  # 假设垛位之间的间距为 1000mm

    # 计算移动时间
    time_to_move_x = total_distance_x / horizontal_speed
    time_to_move_y = total_distance_y / vertical_speed

    return time_to_move_x + time_to_move_y


# 目标函数1：最小化翻垛次数
def minimize_stack_movements_and_turnover(particle_positions, heights, plates, delivery_times, area_positions, batches, weight_movement=1.0, weight_turnover=1.0):
    num_movements = 0
    total_turnover = 0
    batch_turnover = 0

    for plate_idx, position in enumerate(particle_positions):
        # 计算区域、行、列
        area = position // len(area_positions[0])  # 根据区域数量计算库区
        area_position_index = position % len(area_positions[area])  # 获取库区内的位置索引
        row, col = area_positions[area][area_position_index]  # 获取具体的行和列

        # 检查行和列是否超出高度数组的范围
        if row >= heights.shape[1] or col >= heights.shape[2]:
            raise IndexError(f"Row {row} or Column {col} is out of bounds for heights array of shape {heights.shape}")

        # 获取当前钢板的厚度
        current_height = heights[area][row, col]
        plate_height = plates[plate_idx, 2]  # 厚度即为钢板的高度

        # 判断是否需要翻垛（按高度限制）
        if plate_height < current_height:
            num_movements += 1  # 如果钢板在下方，则增加翻垛次数

        # 更新堆垛高度
        heights[area][row, col] += plate_height

    # 计算倒垛量优化公式 (结合交货时间)
    for i in range(len(particle_positions)):
        for j in range(i + 1, len(particle_positions)):
            time_diff = abs(delivery_times[i] - delivery_times[j])
            total_turnover += time_diff

            # 如果属于不同批次，增加翻堆次数
            if batches[i] != batches[j]:
                batch_turnover += 1

    combined_score = weight_movement * num_movements + weight_turnover * (total_turnover + batch_turnover)
    return combined_score





# 目标函数2：最小化出库能耗与时间
def minimize_outbound_energy_time_with_batch(particle_positions, plates, heights, area_positions, stack_dimensions,
                                             horizontal_speed, vertical_speed, conveyor_position_x, conveyor_position_y):
    total_energy_time = 0

    sorted_batches = sorted(set(plates[:, 4]), key=lambda x: int(x[1:]))
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
                                                    vertical_speed, conveyor_position_x, conveyor_position_y,
                                                    stack_dimensions, to_conveyor=False)

            # 更新堆垛高度
            heights[area] -= plate_height
            total_energy_time += outbound_time

    return total_energy_time


# 目标函数3：最大化库存均衡度
def maximize_inventory_balance_v2(particle_positions, plates, Dki, num_positions_per_area):
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
