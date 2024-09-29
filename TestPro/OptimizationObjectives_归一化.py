import numpy as np

class OptimizationObjectives:
    def __init__(self, plates, heights, delivery_times, batches, Dki, area_positions, inbound_point, outbound_point, horizontal_speed, vertical_speed):
        self.plates = plates
        self.heights = heights
        self.delivery_times = delivery_times
        self.batches = batches
        self.Dki = Dki
        self.area_positions = area_positions
        self.inbound_point = inbound_point
        self.outbound_point = outbound_point
        self.horizontal_speed = horizontal_speed
        self.vertical_speed = vertical_speed

    # 归一化函数
    def normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value + 1e-6)

    # 目标函数1：最小化翻垛次数
    def minimize_stack_movements_and_turnover(self, particle_positions, weight_movement=1.0, weight_turnover=1.0):
        num_movements = 0
        total_turnover = 0
        batch_turnover = 0

        for plate_idx, position in enumerate(particle_positions):
            area = position
            plate_height = self.plates[plate_idx, 2]
            current_height = self.heights[area]

            if current_height + plate_height > 3000:
                num_movements += 1

            self.heights[area] += plate_height

        for i in range(len(particle_positions)):
            for j in range(i + 1, len(particle_positions)):
                time_diff = abs(self.delivery_times[i] - self.delivery_times[j])
                total_turnover += time_diff

                if self.batches[i] != self.batches[j]:
                    batch_turnover += 1

        combined_score = weight_movement * num_movements + weight_turnover * (total_turnover + batch_turnover)

        # 归一化翻垛次数与翻转次数
        min_score = 0  # 假设最小值为0
        max_score = combined_score * 1.2  # 适当调整最大值以确保不会有过高值

        return self.normalize(combined_score, min_score, max_score)

    # 目标函数2：最小化出库能耗与时间
    def minimize_outbound_energy_time_with_batch(self, particle_positions):
        total_time_energy = 0

        sorted_batches = sorted(set(self.batches), key=lambda x: int(x[1:]))
        plate_indices_by_batch = {batch: [] for batch in sorted_batches}

        for plate_idx, plate in enumerate(self.plates):
            batch = plate[4]
            plate_indices_by_batch[batch].append(plate_idx)

        def calculate_distance(x1, y1, x2, y2):
            return abs(x2 - x1), abs(y2 - y1)

        def calculate_pick_time(plate_height):
            return plate_height / self.vertical_speed

        def calculate_flip_time(plate_idx):
            area = particle_positions[plate_idx]
            current_height = self.heights[area]
            plate_height = self.plates[plate_idx, 2]

            if current_height > plate_height:
                n_flip = int(current_height // plate_height)
                return n_flip * 10  # 每次翻垛时间
            else:
                return 0

        for plate_idx, position in enumerate(particle_positions):
            area = position
            plate_height = self.plates[plate_idx, 2]
            x, y = self.area_positions[area][plate_idx % len(self.area_positions[area])]

            inbound_horizontal_dist, inbound_vertical_dist = calculate_distance(x, y, self.inbound_point[0], self.inbound_point[1])
            inbound_time = (inbound_horizontal_dist / self.horizontal_speed) + (inbound_vertical_dist / self.vertical_speed)

            self.heights[area] += plate_height
            total_time_energy += inbound_time

        for batch in sorted_batches:
            for plate_idx in plate_indices_by_batch[batch]:
                area = particle_positions[plate_idx]
                plate_height = self.plates[plate_idx, 2]
                x, y = self.area_positions[area][plate_idx % len(self.area_positions[area])]

                outbound_horizontal_dist, outbound_vertical_dist = calculate_distance(x, y, self.outbound_point[0], self.outbound_point[1])
                outbound_time = (outbound_horizontal_dist / self.horizontal_speed) + (outbound_vertical_dist / self.vertical_speed)
                pick_time = calculate_pick_time(plate_height)
                flip_time = calculate_flip_time(plate_idx)

                self.heights[area] -= plate_height
                total_time_energy += (outbound_time + pick_time + flip_time)

        # 归一化能耗与时间
        min_score = 0  # 假设最小值为0
        max_score = total_time_energy * 1.2  # 适当调整最大值以确保不会有过高值

        return self.normalize(total_time_energy, min_score, max_score)

    # 目标函数3：最大化库存均衡度
    def maximize_inventory_balance_v2(self, particle_positions):
        total_variance = 0
        total_volume = np.sum(self.plates[:, 0] * self.plates[:, 1] * self.plates[:, 2])
        num_positions = len(self.Dki)
        mean_volume_per_position = total_volume / num_positions
        area_volumes = np.zeros(num_positions)

        for plate_idx, position in enumerate(particle_positions):
            plate_volume = self.plates[plate_idx][0] * self.plates[plate_idx][1] * self.plates[plate_idx][2]
            area_volumes[position] += plate_volume

        for j in range(num_positions):
            total_variance += (area_volumes[j] - mean_volume_per_position) ** 2

        # 归一化方差
        min_score = 0  # 最小值为0
        max_score = total_variance * 1.2  # 适当调整最大值以确保不会有过高值

        return self.normalize(total_variance, min_score, max_score)

    # 目标函数4：空间利用率最大化
    def maximize_space_utilization_v3(self, particle_positions, alpha_1=1.0, epsilon=1e-6):
        total_space_utilization = 0
        for i in range(len(self.Dki)):
            used_volume = 0
            max_volume = self.Dki[i]

            for j in range(len(self.plates)):
                if particle_positions[j] == i:
                    plate_volume = self.plates[j][0] * self.plates[j][1] * self.plates[j][2]
                    used_volume += plate_volume

            if used_volume > 0:
                utilization = alpha_1 * max((max_volume - used_volume), epsilon) / used_volume
                total_space_utilization += utilization

        # 归一化空间利用率
        min_score = 0  # 最小值为0
        max_score = total_space_utilization * 1.2  # 适当调整最大值以确保不会有过高值

        return self.normalize(total_space_utilization, min_score, max_score)
