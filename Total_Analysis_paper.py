import numpy as np
import pandas as pd
from collections import defaultdict

# 直桥分析
data_all = dict()
data_all['s_7500_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Straight Bridge\\宽度\\8m\\result_data.npy')
data_all['s_7500_7_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Straight Bridge\\宽度\\7m\\result_data.npy')
data_all['s_7500_9_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Straight Bridge\\宽度\\9m\\result_data.npy')
data_all['s_5000_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Straight Bridge\\行人流量\\5000\\result_data.npy')
data_all['s_10000_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Straight Bridge\\行人流量\\10000\\result_data.npy')

bridge_passing_time = pd.DataFrame(columns=['data_type', 'time'])
for data_type, data in data_all.items():
    passing_time_list = list()
    passing_time_data = data[()]['bridge_passing_time']
    for instance in passing_time_data.values():
        if len(instance) == 3:
            passing_time_list.append(instance[2])
    passing_time_array = np.array(passing_time_list)
    tmp_pd = pd.DataFrame({'time': passing_time_array})
    tmp_pd['data_type'] = data_type
    bridge_passing_time = pd.concat((bridge_passing_time, tmp_pd), sort=False)

bridge_passing_time['time'] = bridge_passing_time['time'].astype(np.int64)
mean_time = bridge_passing_time.groupby('data_type').mean() / 100
q_75_time = bridge_passing_time.groupby('data_type').quantile(q=0.75) / 100
mean_speed = 130 / mean_time


width = {k: int(k[-6]) for k in data_all}
bridge_pedestrian_density = pd.DataFrame(columns=['data_type', 'density'])
for data_type, data in data_all.items():
    pedestrian_count_list = list()
    pedestrian_count_data = data[()]['distribution_space']
    for time, distribution in pedestrian_count_data.items():
        if 900 < time <= 2700:
            pedestrian_on_bridge = distribution[:, distribution[3] == 13].shape[1]
            pedestrian_count_list.append(pedestrian_on_bridge)
    pedestrian_density_array = np.array(pedestrian_count_list) / 130 / width[data_type]
    tmp_pd = pd.DataFrame({'density': pedestrian_density_array})
    tmp_pd['data_type'] = data_type
    bridge_pedestrian_density = pd.concat((bridge_pedestrian_density, tmp_pd), sort=False)

bridge_pedestrian_density['density'] = bridge_pedestrian_density['density'].astype(np.float64)
mean_density = bridge_pedestrian_density.groupby('data_type').mean()


width_mod = {k: int(k[-6]) - 4 for k in data_all}
bridge_pedestrian_density = pd.DataFrame(columns=['data_type', 'density'])
for data_type, data in data_all.items():
    pedestrian_count_list = list()
    pedestrian_count_data = data[()]['distribution_space']
    for time, distribution in pedestrian_count_data.items():
        if 900 < time <= 2700:
            pedestrian_on_bridge = distribution[:, distribution[3] == 13].shape[1]
            pedestrian_count_list.append(pedestrian_on_bridge)
    pedestrian_density_array = np.array(pedestrian_count_list) / 130 / width_mod[data_type]
    tmp_pd = pd.DataFrame({'density': pedestrian_density_array})
    tmp_pd['data_type'] = data_type
    bridge_pedestrian_density = pd.concat((bridge_pedestrian_density, tmp_pd), sort=False)

bridge_pedestrian_density['density'] = bridge_pedestrian_density['density'].astype(np.float64)
mean_density_mod = bridge_pedestrian_density.groupby('data_type').mean()

# ----------------------平面布置分界线----------------------
# Y型桥分析
data_all = dict()
data_all['c_3_7500_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Curved Bridge 3 Legs\\宽度\\8m\\result_data.npy')
data_all['c_3_7500_7_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Curved Bridge 3 Legs\\宽度\\7m\\result_data.npy')
data_all['c_3_7500_9_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Curved Bridge 3 Legs\\宽度\\9m\\result_data.npy')
data_all['c_3_5000_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Curved Bridge 3 Legs\\行人流量\\5000\\result_data.npy')
data_all['c_3_10000_8_data'] = np.load(
    'G:\\OneDrive - business\\20180723杭州人行桥疏散模拟\\Python程序_py\\修正分析\\Curved Bridge 3 Legs\\行人流量\\10000\\result_data.npy')

bridge_passing_time = pd.DataFrame(columns=['data_type', 'time'])
for data_type, data in data_all.items():
    passing_time_list = list()
    passing_time_data = data[()]['bridge_passing_time']
    for instance in passing_time_data.values():
        if len(instance) == 3:
            passing_time_list.append(instance[2])
    passing_time_array = np.array(passing_time_list)
    tmp_pd = pd.DataFrame({'time': passing_time_array})
    tmp_pd['data_type'] = data_type
    bridge_passing_time = pd.concat((bridge_passing_time, tmp_pd), sort=False)

bridge_passing_time['time'] = bridge_passing_time['time'].astype(np.int64)
mean_time = bridge_passing_time.groupby('data_type').mean() / 100
q_75_time = bridge_passing_time.groupby('data_type').quantile(q=0.75) / 100
mean_speed = 176 / mean_time


width = {k: int(k[-6]) for k in data_all}
bridge_pedestrian_density = pd.DataFrame(columns=['data_type', 'density'])
for data_type, data in data_all.items():
    pedestrian_count_list = list()
    pedestrian_count_data = data[()]['distribution_space']
    for time, distribution in pedestrian_count_data.items():
        if 900 < time <= 2700:
            pedestrian_on_bridge = distribution[:, (distribution[3] == 14) & (distribution[2] <= 50)].shape[1]
            pedestrian_count_list.append(pedestrian_on_bridge)
    pedestrian_density_array = np.array(pedestrian_count_list) / 24.761 / width[data_type]
    tmp_pd = pd.DataFrame({'density': pedestrian_density_array})
    tmp_pd['data_type'] = data_type
    bridge_pedestrian_density = pd.concat((bridge_pedestrian_density, tmp_pd), sort=False)

bridge_pedestrian_density['density'] = bridge_pedestrian_density['density'].astype(np.float64)
mean_density = bridge_pedestrian_density.groupby('data_type').mean()


width_mod = {k: int(k[-6]) - 4 for k in data_all}
bridge_pedestrian_density = pd.DataFrame(columns=['data_type', 'density'])
for data_type, data in data_all.items():
    pedestrian_count_list = list()
    pedestrian_count_data = data[()]['distribution_space']
    for time, distribution in pedestrian_count_data.items():
        if 900 < time <= 2700:
            pedestrian_on_bridge = distribution[:, (distribution[3] == 14) & (distribution[2] <= 50)].shape[1]
            pedestrian_count_list.append(pedestrian_on_bridge)
    pedestrian_density_array = np.array(pedestrian_count_list) / 24.761 / width_mod[data_type]
    tmp_pd = pd.DataFrame({'density': pedestrian_density_array})
    tmp_pd['data_type'] = data_type
    bridge_pedestrian_density = pd.concat((bridge_pedestrian_density, tmp_pd), sort=False)

bridge_pedestrian_density['density'] = bridge_pedestrian_density['density'].astype(np.float64)
mean_density_mod = bridge_pedestrian_density.groupby('data_type').mean()
