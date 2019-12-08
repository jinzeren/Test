import numpy as np
import pandas as pd
from collections import defaultdict

data_all = dict()
data_all['c_3_10000_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Curved Bridge 3 Legs\\行人流量\\10000\\result_data.npy')

# 统计时长
time_slice = 6
# 监测范围，当前时刻往前推算
time_quantum = 5 * 10

# 仅考虑4m桥宽
width = 4

# 区段密度获取
# 初始化结果数据
bridge_pedestrian_density = pd.DataFrame()
current_slice = time_slice
# 循环统计
for data_type, data in data_all.items():
    pedestrian_count_list = list()
    pedestrian_density_list = list()
    pedestrian_count_data = data[()]['distribution_space']
    for time in range(len(pedestrian_count_data)):
        if time >= current_slice:
            pedestrian_density_list.append((np.array(pedestrian_count_list) / 20 / width).mean())
            pedestrian_count_list = list()
            current_slice += time_slice
            if current_slice > len(pedestrian_count_data):
                break
        distribution = pedestrian_count_data[time]
        pedestrian_on_bridge = \
            distribution[:, (distribution[3] == 14) & (distribution[2] <= 40)].shape[1]
        pedestrian_count_list.append(pedestrian_on_bridge)

    tmp_pd = pd.DataFrame({data_type: np.array(pedestrian_density_list)})
    bridge_pedestrian_density = pd.concat((bridge_pedestrian_density, tmp_pd), sort=False, axis=1)

# 截面行人流率获取
bridge_flow_rate = pd.DataFrame()
current_slice = time_slice
# 循环统计
for data_type, data in data_all.items():
    # 跨越断面行人数量记录
    pedestrian_cross_count = 0
    # 行人流率记录
    pedestrian_flow_rate_list = list()
    # 先前在截面左侧的行人
    pedestrian_section_left_pre = None
    # 先前在截面右侧的行人
    pedestrian_section_right_pre = None
    pedestrian_count_data = data[()]['distribution_space']
    for time in range(len(pedestrian_count_data)):
        if time >= current_slice:
            pedestrian_flow_rate_list.append(pedestrian_cross_count / width / time_slice * 60)
            pedestrian_cross_count = 0
            current_slice += time_slice
            if current_slice > len(pedestrian_count_data):
                break
        distribution = pedestrian_count_data[time]
        pedestrian_on_bridge = distribution[:, (distribution[3] == 14) & (distribution[2] <= 40)]
        if pedestrian_section_left_pre is None:
            pedestrian_section_left_pre = pedestrian_on_bridge[0, pedestrian_on_bridge[2] <= 20]
            pedestrian_section_right_pre = pedestrian_on_bridge[0, pedestrian_on_bridge[2] > 20]
        else:
            pedestrian_section_left = pedestrian_on_bridge[0, pedestrian_on_bridge[2] <= 20]
            pedestrian_section_right = pedestrian_on_bridge[0, pedestrian_on_bridge[2] > 20]
            pedestrian_cross_count += np.isin(pedestrian_section_left_pre, pedestrian_section_right).sum() + np.isin(
                pedestrian_section_right_pre, pedestrian_section_left).sum()
            pedestrian_section_left_pre = pedestrian_section_left
            pedestrian_section_right_pre = pedestrian_section_right

    tmp_pd = pd.DataFrame({data_type: np.array(pedestrian_flow_rate_list)})
    bridge_flow_rate = pd.concat((bridge_flow_rate, tmp_pd), sort=False, axis=1)

# 区段行人速率获取
bridge_velocity = pd.DataFrame()
current_slice = time_slice
# 循环统计
for data_type, data in data_all.items():
    # 行人在time_slice内移动信息记录，第一项为移动距离， 第二项为行走时间，之后将补充第三项当前位置
    pedestrian_move_dict = defaultdict(lambda: [0, 0])
    # 行人速度记录
    pedestrian_velocity_list = list()
    pedestrian_count_data = data[()]['distribution_space']
    for time in range(len(pedestrian_count_data)):
        if time >= current_slice:
            # 计算平均速度
            # 初始化速度增量累计
            velocity_sum = 0
            # 初始化有效速度统计
            velocity_valid = 0
            for value in pedestrian_move_dict.values():
                if value[1] > 0:
                    velocity_sum += value[0] / value[1]
                    velocity_valid += 1
            # 考虑元胞尺寸为0.5m
            average_velocity = velocity_sum / velocity_valid / 2 if velocity_valid > 1 else 0
            pedestrian_velocity_list.append(average_velocity)
            pedestrian_move_dict = defaultdict(lambda: [0, 0])
            current_slice += time_slice
            if current_slice > len(pedestrian_count_data):
                break
        distribution = pedestrian_count_data[time]
        pedestrian_on_bridge = distribution[:, (distribution[3] == 14) & (distribution[2] <= 40)]

        # 添加每个行人的信息,或进行时间和距离更新
        for i in range(pedestrian_on_bridge.shape[1]):
            pedestrian_NO = pedestrian_on_bridge[0, i]
            if pedestrian_NO not in pedestrian_move_dict:
                pedestrian_move_dict[pedestrian_NO].append(pedestrian_on_bridge[1:3, i].flatten())
            else:
                pedestrian_move_dict[pedestrian_NO][0] += np.linalg.norm(
                    pedestrian_on_bridge[1:3, i].flatten() - pedestrian_move_dict[pedestrian_NO][2])
                pedestrian_move_dict[pedestrian_NO][1] += 1
                pedestrian_move_dict[pedestrian_NO][2] = pedestrian_on_bridge[1:3, i].flatten()

    tmp_pd = pd.DataFrame({data_type: np.array(pedestrian_velocity_list)})
    bridge_velocity = pd.concat((bridge_velocity, tmp_pd), sort=False, axis=1)


# 风险后果计算
def risk_consequence_calc(crowd_density, crowd_flow_rate, crowd_velocity):
    # 参数上下限
    density_bins = np.array([0.31, 0.43, 0.72, 1.08, 2.17])
    flow_rate_bins = np.array([23, 33, 49, 66, 82])
    velocity_bins = np.array([-1.3, -1.27, -1.22, -1.14, -0.76])

    # 服务状态等级分别计算
    density_levels = np.digitize(crowd_density, density_bins)
    flow_rate_levels = np.digitize(crowd_flow_rate, flow_rate_bins)
    crowd_velocity_neg = -crowd_velocity
    # 排除0的影响
    crowd_velocity_neg[crowd_velocity_neg == 0] = -1.4
    velocity_levels = np.digitize(crowd_velocity_neg, velocity_bins)

    # 服务状态等级最终确定
    final_levels = np.zeros_like(density_levels)
    step_one_mapping = density_levels >= flow_rate_levels
    final_levels[~step_one_mapping] = flow_rate_levels[~step_one_mapping]
    final_levels[step_one_mapping] = np.maximum(density_levels[step_one_mapping], velocity_levels[step_one_mapping])

    return (final_levels + 1).flatten()


final_level = dict()
for data_name in data_all.keys():
    final_level[data_name] = risk_consequence_calc(bridge_pedestrian_density[data_name], bridge_flow_rate[data_name], bridge_velocity[data_name])


# 风险等级计算
def risk_magnitude_calc(risk_level, time_range):
    # 密度参数上下限
    probability_bins = np.arange(0, 1.1, 0.2)
    risk_magnitude_list = list()
    for i in range(risk_level.size):
        if i < time_range - 1:
            continue
        level_range = risk_level[i - time_range + 1: i + 1]
        levels, counts = np.unique(level_range, return_counts=True)
        risk_magnitude_tmp = [levels[j] * (np.digitize(counts[j] / time_range, probability_bins)) for j in range(levels.size)]
        risk_magnitude_list.append(np.array(risk_magnitude_tmp).max())

    return np.array(risk_magnitude_list)


risk_magnitude = dict()
for data_name in data_all.keys():
    risk_magnitude[data_name] = risk_magnitude_calc(final_level[data_name], time_quantum)

risk_magnitude_pd = pd.DataFrame(risk_magnitude['c_3_10000_8_data'], columns=['Risk Magnitude'], index=np.arange(49, 600))
bridge_pedestrian_density.columns = ['Crowd Density (p/m^2)']
bridge_flow_rate.columns = ['Crowd Flow Rate (p/(m*s))']
bridge_velocity.columns = ['Crowd Velocity (m/s)']

all_info = pd.concat((bridge_pedestrian_density, bridge_flow_rate, bridge_velocity, risk_magnitude_pd), axis=1)
writer = pd.ExcelWriter('analysis_output_plan_B.xlsx')
all_info.to_excel(writer, 'sheet1')
writer.close()
