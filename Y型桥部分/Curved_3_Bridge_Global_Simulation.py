import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os

current_dir = os.getcwd()


class Pivot(object):
    # Simulate the movement of pedestrians in a pivot
    def __init__(self, node_no, size, exits, corresponding_node, corresponding_space, additional_obstacle=None,
                 s_matrix_list=None):
        self.no = node_no
        print('Pivot {} is established'.format(node_no))

        # Geometry setting
        '''In exits, the 4 components means x and y location of the first exit cell, length of the exit and the 
        direction, respectively. Wherein, 0 stands for along x while 1 for along y. size=(100, 100); exits=[(0, 0, 4, 
        0)]; corresponding_node here means the adjacent nodes connecting to this pivot '''

        # Set the exit
        self.size = size
        self.exit_no = len(exits)
        x, y, length, direction = zip(*exits)
        x = np.array(x, dtype=np.int64, ndmin=1)
        y = np.array(y, dtype=np.int64, ndmin=1)
        length = np.array(length, dtype=np.int64, ndmin=1)
        direction = np.array(direction, dtype=np.int64, ndmin=1)
        # 4 rows: x, y, length, direction
        self.exit_data = np.vstack((x, y, length, direction))
        self.corresponding_node = np.array(corresponding_node)
        self.corresponding_space = np.array(corresponding_space)

        # Calculate the simulation interval for different slope degree.
        self.simulation_interval = int(50 / 1.3)

        # Boundary and Occupation status matrix Initialization, set the boundary cells to 1
        b_o_matrix = np.zeros((size[0], size[1]))
        b_o_matrix[1:-1, 1:-1] = -1
        b_o_matrix += 1

        # Set the additional obstacle
        if additional_obstacle is not None:
            obstacle_x, obstacle_y = zip(*additional_obstacle)
            b_o_matrix[obstacle_x, obstacle_y] = 1

        # Calculate S matrix
        if s_matrix_list is None:
            s_matrix_list = []
            if additional_obstacle is not None:
                for i in range(self.exit_no):
                    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=self.exit_data[:, i]))
            else:
                for i in range(self.exit_no):
                    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=self.exit_data[:, i]))
            self.s_matrix_list = np.array(s_matrix_list)
        else:
            self.s_matrix_list = s_matrix_list
            b_o_matrix[s_matrix_list[0] == 1000] = 1

        # Set the exit cell to 2
        for i in range(self.exit_no):
            # Use 2 to represent the exit cells
            if self.exit_data[3, i] == 1:
                b_o_matrix[self.exit_data[0, i]:self.exit_data[0, i] + self.exit_data[2, i], self.exit_data[1, i]] = 2
            else:
                b_o_matrix[self.exit_data[0, i], self.exit_data[1, i]:self.exit_data[1, i] + self.exit_data[2, i]] = 2

        # Pedestrian Distribution Initialization
        self.pedestrian_data = np.empty((4, 0), dtype=np.int64)
        # Set the status matrix
        self.b_o_matrix = b_o_matrix

    def move_simulation(self, time):
        # Check if there is any pedestrian in the space
        if self.pedestrian_data.size == 0:
            return None

        if np.mod(time, self.simulation_interval) != 0:
            return None

        # Calculate relevant matrix
        e_matrix = calc_e_matrix(self.pedestrian_data, self.b_o_matrix, self.exit_data, self.corresponding_node)
        # Calculate the next step
        next_step_data = next_step(self.pedestrian_data, self.corresponding_node, self.s_matrix_list, e_matrix)
        # Update the status
        self.pedestrian_data[:3] = next_step_data

        # Update the status matrix
        self.b_o_matrix[self.b_o_matrix == 3] = 0
        self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] = np.where(
            self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] == 2, 2, 3)

        # Find the pedestrian at the exit
        exit_flag = self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] == 2
        if exit_flag.any():
            index_transfer = defaultdict(list)
            # Update the pedestrian_data
            pedestrian_exit_index = self.pedestrian_data[0, exit_flag]
            # New here
            pedestrian_exit_x = self.pedestrian_data[1, exit_flag]
            pedestrian_exit_y = self.pedestrian_data[2, exit_flag]

            pedestrian_exit_local_destination = self.pedestrian_data[3, exit_flag]

            # New here
            pedestrian_at_exit_index = \
                np.where(pedestrian_exit_local_destination[:, None] == self.corresponding_node[None, :])[1]

            self.pedestrian_data = self.pedestrian_data[:, ~exit_flag]
            for i in range(pedestrian_exit_index.size):
                # Transfer all the pedestrian at exits to the next space
                # index_transfer[pedestrian_exit_local_destination[i]].append()
                exit_data_sample = self.exit_data[:, pedestrian_at_exit_index[i]]
                if exit_data_sample[3] == 1:
                    index_transfer[pedestrian_exit_local_destination[i]].append(
                        [pedestrian_exit_index[i], pedestrian_exit_x[i] - exit_data_sample[0]])
                else:
                    index_transfer[pedestrian_exit_local_destination[i]].append(
                        [pedestrian_exit_index[i], pedestrian_exit_y[i] - exit_data_sample[1]])

            # Transfer the pedestrians to the next space
            for next_local_destination, index_position_pedestrian in index_transfer.items():
                self.pedestrian_transfer_to_space(time, np.array(index_position_pedestrian), next_local_destination)

    def pedestrian_transfer_to_space(self, time, pedestrian_transfer_index_position, next_local_destination):
        # Every pedestrian shares the same next local destination
        # Acquire the space no
        node_combination_tuple = tuple(sorted([self.no, next_local_destination]))
        space_intended = space_node_relationship[node_combination_tuple]
        exit_data = space_dict[space_intended].exit_data[:,
                    space_dict[space_intended].corresponding_node == self.no].flatten()

        # # Acquire the random positions
        # random_position = np.random.choice(np.arange(exit_data[2]), pedestrian_transfer_index_position.size, replace=False)

        if exit_data[3] == 1:
            pedestrian_x = (exit_data[0] + pedestrian_transfer_index_position[:, 1]).astype(np.int64).reshape(1, -1)
            position_y = exit_data[1] + 1 if exit_data[1] == 0 else exit_data[1] - 1
            pedestrian_y = np.ones_like(pedestrian_x, dtype=np.int64) * position_y
        else:
            pedestrian_y = (exit_data[1] + pedestrian_transfer_index_position[:, 1]).astype(np.int64).reshape(1, -1)
            position_x = exit_data[0] + 1 if exit_data[0] == 0 else exit_data[0] - 1
            pedestrian_x = np.ones_like(pedestrian_y, dtype=np.int64) * position_x

        local_destination = np.ones(pedestrian_transfer_index_position.shape[0],
                                    dtype=np.int64) * next_local_destination
        transferred_pedestrian = np.vstack(
            (pedestrian_transfer_index_position[:, 0].T, pedestrian_x, pedestrian_y, local_destination))

        space_dict[space_intended].pedestrian_data = np.hstack(
            (space_dict[space_intended].pedestrian_data, transferred_pedestrian)).astype(np.int64)

        if space_intended in bridge_space and self.no in bridge_end_node:
            for index_position_transfer in pedestrian_transfer_index_position:
                if index_position_transfer[0] not in bridge_passing_time:
                    bridge_passing_time[index_position_transfer[0]].append(time)
                else:
                    del bridge_passing_time[index_position_transfer[0]][1:]

    def pedestrian_transfer_to_pivot(self, pedestrian_transfer_index_position, from_space_no):
        # Transfer the pedestrians from space to pivot
        local_destination = np.empty(pedestrian_transfer_index_position.shape[0], dtype=np.int64)
        for i in range(pedestrian_transfer_index_position.shape[0]):
            new_local_destination_index = pedestrian_global_path[pedestrian_transfer_index_position[i, 0]].index(
                self.no) + 1
            local_destination[i] = pedestrian_global_path[pedestrian_transfer_index_position[i, 0]][
                new_local_destination_index]

        exit_data = self.exit_data[:, self.corresponding_space == from_space_no].flatten()
        # # Acquire the random positions
        # random_position = np.random.choice(np.arange(exit_data[2]), pedestrian_transfer_index_position.size,
        #                                    replace=False)

        if exit_data[3] == 1:
            pedestrian_x = (exit_data[0] + pedestrian_transfer_index_position[:, 1]).astype(np.int64).reshape(1, -1)
            position_y = exit_data[1] + 1 if exit_data[1] == 0 else exit_data[1] - 1
            pedestrian_y = np.ones_like(pedestrian_x, dtype=np.int64) * position_y
        else:
            pedestrian_y = (exit_data[1] + pedestrian_transfer_index_position[:, 1]).astype(np.int64).reshape(1, -1)
            position_x = exit_data[0] + 1 if exit_data[0] == 0 else exit_data[0] - 1
            pedestrian_x = np.ones_like(pedestrian_y, dtype=np.int64) * position_x

        # Set the transferred pedestrians
        transferred_pedestrian = np.vstack(
            (pedestrian_transfer_index_position[:, 0].T, pedestrian_x, pedestrian_y, local_destination))
        self.pedestrian_data = np.hstack((self.pedestrian_data, transferred_pedestrian)).astype(np.int64)


class Road(object):
    # Simulate the pedestrian movement on the bridge.
    def __init__(self, road_no, size, exits, corresponding_node, initial_pedestrian=0, additional_obstacle=None,
                 s_matrix_list=None):
        global pedestrian_start_index

        self.no = road_no
        print('Road {} is established'.format(road_no))

        # Geometry setting
        '''In exits, the 4 components means x and y location of the first exit cell, length of the exit and the 
        direction, respectively. Wherein, 0 stands for along x while 1 for along y. size=(100, 100); exits=[(0, 0, 4, 
        0)] '''

        # Set the exit
        self.size = size
        self.exit_no = len(exits)
        x, y, length, direction = zip(*exits)
        x = np.array(x, dtype=np.int64, ndmin=1)
        y = np.array(y, dtype=np.int64, ndmin=1)
        length = np.array(length, dtype=np.int64, ndmin=1)
        direction = np.array(direction, dtype=np.int64, ndmin=1)
        # 4 rows: x, y, length, direction
        self.exit_data = np.vstack((x, y, length, direction))
        self.corresponding_node = np.array(corresponding_node)

        # Calculate the simulation interval for different slope degree.
        self.simulation_interval = int(50 / 1.3)

        # Boundary and Occupation status matrix Initialization, set the boundary cells to 1
        b_o_matrix = np.zeros((size[0], size[1]))
        b_o_matrix[1:-1, 1:-1] = -1
        b_o_matrix += 1

        # Set the additional obstacle
        if additional_obstacle is not None:
            obstacle_x, obstacle_y = zip(*additional_obstacle)
            b_o_matrix[obstacle_x, obstacle_y] = 1

            # Calculate S matrix
        if s_matrix_list is None:
            s_matrix_list = []
            if additional_obstacle is not None:
                for i in range(self.exit_no):
                    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=self.exit_data[:, i]))
            else:
                for i in range(self.exit_no):
                    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=self.exit_data[:, i]))
            self.s_matrix_list = np.array(s_matrix_list)
        else:
            self.s_matrix_list = s_matrix_list
            b_o_matrix[s_matrix_list[0] == 1000] = 1

        # Set the exit cell to 2
        for i in range(self.exit_no):
            # Use 2 to represent the exit cells
            if self.exit_data[3, i] == 1:
                b_o_matrix[self.exit_data[0, i]:self.exit_data[0, i] + self.exit_data[2, i], self.exit_data[1, i]] = 2
            else:
                b_o_matrix[self.exit_data[0, i], self.exit_data[1, i]:self.exit_data[1, i] + self.exit_data[2, i]] = 2

        # Pedestrian Distribution Initialization
        self.pedestrian_data = np.empty((4, 0), dtype=np.int64)
        # Set the status matrix
        self.b_o_matrix = b_o_matrix

    def move_simulation(self, time):
        # Check if there is any pedestrian in the space
        if self.pedestrian_data.size == 0:
            return None

        if np.mod(time, self.simulation_interval) != 0:
            return None

        # Calculate relevant matrix
        e_matrix = calc_e_matrix(self.pedestrian_data, self.b_o_matrix, self.exit_data, self.corresponding_node)
        # Calculate the next step
        next_step_data = next_step(self.pedestrian_data, self.corresponding_node, self.s_matrix_list, e_matrix)
        # Update the status
        self.pedestrian_data[:3] = next_step_data

        # Update the status matrix
        self.b_o_matrix[self.b_o_matrix == 3] = 0
        self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] = np.where(
            self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] == 2, 2, 3)

        # Find the pedestrian at the exit
        exit_flag = self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] == 2
        if exit_flag.any():
            index_transfer = defaultdict(list)
            # Update the pedestrian_data
            pedestrian_exit_index = self.pedestrian_data[0, exit_flag]

            # New here
            pedestrian_exit_x = self.pedestrian_data[1, exit_flag]
            pedestrian_exit_y = self.pedestrian_data[2, exit_flag]

            pedestrian_exit_local_destination = self.pedestrian_data[3, exit_flag]

            # New here
            pedestrian_at_exit_index = \
                np.where(pedestrian_exit_local_destination[:, None] == self.corresponding_node[None, :])[1]

            self.pedestrian_data = self.pedestrian_data[:, ~exit_flag]
            for i in range(pedestrian_exit_index.size):
                if pedestrian_exit_local_destination[i] != pedestrian_global_path[pedestrian_exit_index[i]][-1]:
                    # index_transfer[pedestrian_exit_local_destination[i]].append(pedestrian_exit_index[i])

                    exit_data_sample = self.exit_data[:, pedestrian_at_exit_index[i]]
                    if exit_data_sample[3] == 1:
                        index_transfer[pedestrian_exit_local_destination[i]].append(
                            [pedestrian_exit_index[i], pedestrian_exit_x[i] - exit_data_sample[0]])
                    else:
                        index_transfer[pedestrian_exit_local_destination[i]].append(
                            [pedestrian_exit_index[i], pedestrian_exit_y[i] - exit_data_sample[1]])

            # Transfer the pedestrians to the pivot
            for target_pivot, index_position_pedestrian in index_transfer.items():
                if target_pivot in pivot_dict.keys():
                    pivot_dict[target_pivot].pedestrian_transfer_to_pivot(np.array(index_position_pedestrian), self.no)
                else:
                    space_to_space_transfer(time, np.array(index_position_pedestrian), target_pivot)

    def pedestrian_generate(self, average_hourly_pedestrian, time_interval):
        global pedestrian_start_index
        lam = average_hourly_pedestrian / 3600 * time_interval
        # Calculate the pedestrian number
        pedestrian_generated = np.random.poisson(lam)
        if pedestrian_generated == 0:
            return None

        # Determine the index range
        index_range = np.arange(pedestrian_start_index, pedestrian_start_index + pedestrian_generated)
        pedestrian_start_index += pedestrian_generated

        # Determine the global path for pedestrians
        global_paths_added = generate_global_path(self.no, index_range)
        # Update the global path dictionary
        pedestrian_global_path.update(global_paths_added)

        # Determine the local destination
        local_destination = np.array([global_paths_added[index_gen][0] for index_gen in index_range], dtype=np.int64)

        # Allocate the pedestrian(s) to the space
        vacant_x, vacant_y = (self.b_o_matrix == 0).nonzero()
        vacant_array = np.arange(len(vacant_x))
        np.random.shuffle(vacant_array)
        random_index = vacant_array[:pedestrian_generated]
        new_pedestrians = np.vstack((index_range, vacant_x[random_index], vacant_y[random_index], local_destination))
        self.pedestrian_data = np.hstack((self.pedestrian_data, new_pedestrians))


class Bridge(object):
    # Simulate the pedestrian movement on the bridge.
    def __init__(self, bridge_no, size, exits, corresponding_node, slope_info, slope_direction, wander_prob,
                 initial_pedestrian=0, additional_obstacle=None, slope_along=1, s_matrix_list=None):
        global pedestrian_start_index

        self.no = bridge_no
        print('Bridge {} is established'.format(bridge_no))

        # Geometry setting
        '''In exits, the 4 components means x and y location of the first exit cell, length of the exit and the 
        direction, respectively. Wherein, 0 stands for along x while 1 for along y. size=(100, 100); exits=[(0, 0, 4, 
        0)]; slope_info={(position1, position2): degree1, (position2, position3): degree2, ...}; slope_direction={(
        position1, position2): (up_destination, down_destination}; wander_prob={(position1, position2): (probability, 
        time_duration)}; slope_along=1(x) or 2(y) '''

        # Set the exit
        self.size = size
        self.exit_no = len(exits)
        x, y, length, direction = zip(*exits)
        x = np.array(x, dtype=np.int64, ndmin=1)
        y = np.array(y, dtype=np.int64, ndmin=1)
        length = np.array(length, dtype=np.int64, ndmin=1)
        direction = np.array(direction, dtype=np.int64, ndmin=1)
        # 4 rows: x, y, length, direction
        self.exit_data = np.vstack((x, y, length, direction))
        self.corresponding_node = np.array(corresponding_node)

        # Calculate the simulation interval for different slope degree.
        self.simulation_interval = simu_gap(slope_info)
        self.wander_interval = 100
        self.slope_direction = slope_direction
        self.slope_along = slope_along

        # Set the probability of pedestrians wandering on the bridge
        self.wander_prob = wander_prob
        self.wander_info = dict()

        # Boundary and Occupation status matrix Initialization
        b_o_matrix = np.zeros((size[0], size[1]))
        b_o_matrix[1:-1, 1:-1] = -1
        b_o_matrix += 1

        # Set the additional obstacle
        if additional_obstacle is not None:
            obstacle_x, obstacle_y = zip(*additional_obstacle)
            b_o_matrix[obstacle_x, obstacle_y] = 1

        # Calculate S matrix
        if s_matrix_list is None:
            s_matrix_list = []
            if additional_obstacle is not None:
                for i in range(self.exit_no):
                    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=self.exit_data[:, i]))
            else:
                for i in range(self.exit_no):
                    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=self.exit_data[:, i]))
            self.s_matrix_list = np.array(s_matrix_list)
        else:
            self.s_matrix_list = s_matrix_list
            b_o_matrix[s_matrix_list[0] == 1000] = 1

        # Set the exit cell
        for i in range(self.exit_no):
            # Use 2 to represent the exit cells
            if self.exit_data[3, i] == 1:
                b_o_matrix[self.exit_data[0, i]:self.exit_data[0, i] + self.exit_data[2, i], self.exit_data[1, i]] = 2
            else:
                b_o_matrix[self.exit_data[0, i], self.exit_data[1, i]:self.exit_data[1, i] + self.exit_data[2, i]] = 2

        # Pedestrian Distribution Initialization
        self.pedestrian_data = np.empty((4, 0), dtype=np.int64)
        # Set the status matrix
        self.b_o_matrix = b_o_matrix

    def move_simulation(self, time):
        # Check if there is any pedestrian in the space
        if self.pedestrian_data.size == 0:
            return None

        # Update the wandering pedestrian
        if len(self.wander_info) > 0:
            stop_wander = list()
            for index_exit, time_until in self.wander_info.items():
                if time > time_until:
                    stop_wander.append(index_exit)
            for index_exit in stop_wander:
                del self.wander_info[index_exit]

        # Preclude the wandering pedestrian
        wander_index = list(self.wander_info.keys())
        wander_map = np.isin(self.pedestrian_data[0], wander_index)
        pedestrian_available = self.pedestrian_data[:, ~wander_map]
        pedestrian_wander = self.pedestrian_data[:, wander_map]

        # Determine the pedestrians to move
        action_index = list()
        for bridge_range, simulation_gap in self.simulation_interval.items():
            if np.mod(time, simulation_gap[0]) == 0:
                # Check the pedestrian who can take action according to the simulation gap
                action_map = np.logical_and(np.logical_and(pedestrian_available[self.slope_along] > bridge_range[0],
                                                           pedestrian_available[self.slope_along] <= bridge_range[1]),
                                            pedestrian_available[3] == self.slope_direction[bridge_range][0])
                if action_map.any():
                    action_no = action_map.nonzero()[0].size
                    # Check whether a pedestrian decides to wander
                    start_wander = np.random.choice([True, False], size=action_no,
                                                    p=[self.wander_prob[bridge_range][0],
                                                       1 - self.wander_prob[bridge_range][0]])
                    if start_wander.any():
                        pedestrian_wander_index = pedestrian_available[0, action_map][start_wander]
                        self.wander_info.update(
                            {index_wander: time + self.wander_prob[bridge_range][1] for index_wander in
                             pedestrian_wander_index})
                        action_map[action_map] = ~start_wander

                    action_index.extend(action_map.nonzero()[0])

            if np.mod(time, simulation_gap[1]) == 0:
                action_map = np.logical_and(np.logical_and(pedestrian_available[self.slope_along] > bridge_range[0],
                                                           pedestrian_available[self.slope_along] <= bridge_range[1]),
                                            pedestrian_available[3] == self.slope_direction[bridge_range][1])
                if action_map.any():
                    action_no = action_map.nonzero()[0].size
                    # Check whether a pedestrian decides to wander
                    start_wander = np.random.choice([True, False], size=action_no,
                                                    p=[self.wander_prob[bridge_range][0],
                                                       1 - self.wander_prob[bridge_range][0]])
                    if start_wander.any():
                        pedestrian_wander_index = pedestrian_available[0, action_map][start_wander]
                        self.wander_info.update(
                            {index_wander: time + self.wander_prob[bridge_range][1] for index_wander in
                             pedestrian_wander_index})
                        action_map[action_map] = ~start_wander

                    action_index.extend(action_map.nonzero()[0])
            else:
                continue

        # Set simulation of wander pedestrian to False
        wander_move = False

        if np.mod(time, self.wander_interval) == 0 and wander_map.any():
            wander_move = True

        # if there is no pedestrian to move, then return None
        if len(action_index) == 0 and (not wander_move):
            return None

        if len(action_index) > 0:
            # Set the moving pedestrian data
            pedestrian_move = pedestrian_available[:, action_index]
            # Calculate relevant matrix
            e_matrix = calc_e_matrix(pedestrian_move, self.b_o_matrix, self.exit_data, self.corresponding_node)
            # Calculate the next step
            next_step_data = next_step(pedestrian_move, self.corresponding_node, self.s_matrix_list, e_matrix,
                                       bridge=True)
            # Update the status
            pedestrian_available[:3, action_index] = next_step_data
            self.pedestrian_data[:, ~wander_map] = pedestrian_available
        #
        if wander_move:
            # Calculate relevant matrix
            e_matrix = calc_e_matrix(pedestrian_wander, self.b_o_matrix, self.exit_data, self.corresponding_node)
            # Calculate the next step
            next_step_data = next_step(pedestrian_wander, self.corresponding_node, self.s_matrix_list, e_matrix,
                                       bridge=True)
            # Update the status
            self.pedestrian_data[:3, wander_map] = next_step_data

        # Update the status matrix
        self.b_o_matrix[self.b_o_matrix == 3] = 0
        self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] = np.where(
            self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] == 2, 2, 3)

        global bridge_passing_no
        exit_flag = self.b_o_matrix[self.pedestrian_data[1], self.pedestrian_data[2]] == 2
        if exit_flag.any():
            index_transfer = defaultdict(list)
            # Update the pedestrian_data
            pedestrian_exit_index = self.pedestrian_data[0, exit_flag]
            # New here
            pedestrian_exit_x = self.pedestrian_data[1, exit_flag]
            pedestrian_exit_y = self.pedestrian_data[2, exit_flag]

            pedestrian_exit_local_destination = self.pedestrian_data[3, exit_flag]

            # New here
            pedestrian_at_exit_index = \
                np.where(pedestrian_exit_local_destination[:, None] == self.corresponding_node[None, :])[1]

            self.pedestrian_data = self.pedestrian_data[:, ~exit_flag]
            for i in range(pedestrian_exit_index.size):
                if pedestrian_exit_local_destination[i] != pedestrian_global_path[pedestrian_exit_index[i]][-1]:
                    # index_transfer[pedestrian_exit_local_destination[i]].append(pedestrian_exit_index[i])

                    exit_data_sample = self.exit_data[:, pedestrian_at_exit_index[i]]
                    if exit_data_sample[3] == 1:
                        index_transfer[pedestrian_exit_local_destination[i]].append(
                            [pedestrian_exit_index[i], pedestrian_exit_x[i] - exit_data_sample[0]])
                    else:
                        index_transfer[pedestrian_exit_local_destination[i]].append(
                            [pedestrian_exit_index[i], pedestrian_exit_y[i] - exit_data_sample[1]])

                if pedestrian_exit_index[i] in bridge_passing_time and pedestrian_exit_local_destination[
                    i] in bridge_end_node:
                    bridge_passing_time[pedestrian_exit_index[i]].extend(
                        [time, time - bridge_passing_time[pedestrian_exit_index[i]][0]])
                    bridge_passing_no += 1

            # Transfer the pedestrians to the pivot
            for target_pivot, index_position_pedestrian in index_transfer.items():
                if target_pivot in pivot_dict.keys():
                    pivot_dict[target_pivot].pedestrian_transfer_to_pivot(np.array(index_position_pedestrian), self.no)
                else:
                    space_to_space_transfer(time, np.array(index_position_pedestrian), target_pivot)


def calc_distance(b_o_matrix, exit_row):
    # Calculate the distance between a cell and the exit cell
    # Create the mesh data
    x = np.arange(0, b_o_matrix.shape[0])
    y = np.arange(0, b_o_matrix.shape[1])
    y_array, x_array = np.meshgrid(y, x)

    # Calculate the distance
    distance_list = []
    for i in range(exit_row[2]):
        if exit_row[3] == 1:
            distance_tmp = np.sqrt(np.square(x_array - (exit_row[0] + i)) + np.square(y_array - exit_row[1]))
        else:
            distance_tmp = np.sqrt(np.square(x_array - exit_row[0]) + np.square(y_array - (exit_row[1] + i)))
        distance_list.append(distance_tmp)

    # Compare the distance
    distance = np.where(distance_list[0] < distance_list[1], distance_list[0], distance_list[1])
    for i in range(1, exit_row[2] - 1):
        distance = np.where(distance < distance_list[i + 1], distance, distance_list[i + 1])

    return distance


def calc_distance_dij(b_o_matrix, exit_row):
    # Acquire the exit_data
    if exit_row[3] == 1:
        exit_data = [(exit_row[0] + i, exit_row[1]) for i in range(exit_row[2])]
    else:
        exit_data = [(exit_row[0], exit_row[1] + i) for i in range(exit_row[2])]

    p_cells = exit_data
    vacant_x, vacant_y = (b_o_matrix == 0).nonzero()
    t_cells = list(zip(vacant_x, vacant_y))
    e_cells = []
    parent_cells = {cell: cell for cell in p_cells}
    cell_to_exit_distance = {exit_cell: 0 for exit_cell in exit_data}
    p_cell_count = defaultdict(lambda: 0)

    cell_total = len(p_cells) + len(t_cells)
    # If all the t_cells are turned into p_cells, then stop iteration
    while len(p_cells) < cell_total:
        for p_cell in p_cells:
            if p_cell_count[p_cell] == 1:
                continue
            p_cell_count[p_cell] = 1
            # Iterate through all the p_cells
            x_adjacent = np.arange(p_cell[0] - 1, p_cell[0] + 2)
            y_adjacent = np.arange(p_cell[1] - 1, p_cell[1] + 2)

            p_cell_array = np.array(p_cell)

            parent_cell = parent_cells[p_cell]
            parent_cell_array = np.array(parent_cell)

            # Iterate through a p_cell's adjacent area
            for adjacent_i in range(x_adjacent.size):
                for adjacent_j in range(y_adjacent.size):
                    adjacent_cell = (x_adjacent[adjacent_i], y_adjacent[adjacent_j])
                    adjacent_cell_array = np.array(adjacent_cell)
                    if adjacent_cell in t_cells:
                        # Check whether this cell is approachable from the parent_cell of this p_cell
                        # Acquire the check range
                        check_range_x = np.arange(parent_cell[0], adjacent_cell[0] + 1) if parent_cell[0] <= \
                                                                                           adjacent_cell[
                                                                                               0] else np.arange(
                            adjacent_cell[0], parent_cell[0] + 1)
                        check_range_y = np.arange(parent_cell[1], adjacent_cell[1] + 1) if parent_cell[1] <= \
                                                                                           adjacent_cell[
                                                                                               1] else np.arange(
                            adjacent_cell[1], parent_cell[1] + 1)
                        # Iterate through check cells to find the cells on the line formed by parent_cell and adjacent_cell,
                        # check whether they are obstacles or not
                        approach_check = True
                        for check_i in range(check_range_x.size):
                            for check_j in range(check_range_y.size):
                                check_cell = (check_range_x[check_i], check_range_y[check_j])
                                if check_cell == parent_cell or check_cell == adjacent_cell:
                                    continue
                                check_cell_array = np.array(check_cell)
                                # Calculate the vector
                                parent_cell_to_check_cell = check_cell_array - parent_cell_array
                                parent_cell_to_adjacent_cell = adjacent_cell_array - parent_cell_array
                                # Calculate the angle
                                angle_cos = np.dot(parent_cell_to_check_cell,
                                                   parent_cell_to_adjacent_cell) / np.linalg.norm(
                                    parent_cell_to_check_cell) / np.linalg.norm(parent_cell_to_adjacent_cell)
                                angle_sin = np.sqrt(np.clip(1 - np.square(angle_cos), 0, 1))
                                # Calculate the distance
                                check_cell_to_the_line = np.linalg.norm(parent_cell_to_check_cell) * angle_sin
                                # Check the approachability
                                if check_cell_to_the_line < np.sqrt(2) / 2 and b_o_matrix[
                                    check_cell[0], check_cell[1]] == 1:
                                    approach_check = False
                                    break

                        # Add the cell to e_cells
                        e_cells.append(adjacent_cell)
                        t_cells.remove(adjacent_cell)

                        if approach_check:
                            parent_cells[adjacent_cell] = parent_cell
                            cell_to_exit_distance[adjacent_cell] = cell_to_exit_distance[parent_cell] + np.linalg.norm(
                                adjacent_cell_array - parent_cell_array)
                        else:
                            parent_cells[adjacent_cell] = p_cell
                            cell_to_exit_distance[adjacent_cell] = cell_to_exit_distance[p_cell] + np.linalg.norm(
                                adjacent_cell_array - p_cell_array)

                    elif adjacent_cell in e_cells:
                        # Check whether this cell is approachable from the parent_cell of this p_cell
                        # Acquire the check range
                        check_range_x = np.arange(parent_cell[0], adjacent_cell[0] + 1) if parent_cell[0] <= \
                                                                                           adjacent_cell[
                                                                                               0] else np.arange(
                            adjacent_cell[0], parent_cell[0] + 1)
                        check_range_y = np.arange(parent_cell[1], adjacent_cell[1] + 1) if parent_cell[1] <= \
                                                                                           adjacent_cell[
                                                                                               1] else np.arange(
                            adjacent_cell[1], parent_cell[1] + 1)
                        # Iterate through check cells to find the cells on the line formed by parent_cell and adjacent_cell,
                        # check whether they are obstacles or not
                        approach_check = True
                        for check_i in range(check_range_x.size):
                            for check_j in range(check_range_y.size):
                                check_cell = (check_range_x[check_i], check_range_y[check_j])
                                if check_cell == parent_cell or check_cell == adjacent_cell:
                                    continue
                                check_cell_array = np.array(check_cell)
                                # Calculate the vector
                                parent_cell_to_check_cell = check_cell_array - parent_cell_array
                                parent_cell_to_adjacent_cell = adjacent_cell_array - parent_cell_array
                                # Calculate the angle
                                angle_cos = np.dot(parent_cell_to_check_cell,
                                                   parent_cell_to_adjacent_cell) / np.linalg.norm(
                                    parent_cell_to_check_cell) / np.linalg.norm(parent_cell_to_adjacent_cell)
                                angle_sin = np.sqrt(np.clip(1 - np.square(angle_cos), 0, 1))
                                # Calculate the distance
                                check_cell_to_the_line = np.linalg.norm(parent_cell_to_check_cell) * angle_sin
                                # Check the approachability
                                if check_cell_to_the_line < np.sqrt(2) / 2 and b_o_matrix[
                                    check_cell[0], check_cell[1]] == 1:
                                    approach_check = False
                                    break

                        if approach_check:
                            test_distance = cell_to_exit_distance[parent_cell] + np.linalg.norm(
                                adjacent_cell_array - parent_cell_array)
                            if test_distance < cell_to_exit_distance[adjacent_cell]:
                                parent_cells[adjacent_cell] = parent_cell
                                cell_to_exit_distance[adjacent_cell] = test_distance
                        else:
                            test_distance = cell_to_exit_distance[p_cell] + np.linalg.norm(
                                adjacent_cell_array - p_cell_array)
                            if test_distance < cell_to_exit_distance[adjacent_cell]:
                                parent_cells[adjacent_cell] = p_cell
                                cell_to_exit_distance[adjacent_cell] = test_distance
                    else:
                        continue
        # Update the e_cells, choose the e_cell with smallest distance to be in the p_cells
        e_cell_distance_min = np.inf
        e_cell_with_min_distance = tuple()
        for e_cell in e_cells:
            if cell_to_exit_distance[e_cell] < e_cell_distance_min:
                e_cell_distance_min = cell_to_exit_distance[e_cell]
                e_cell_with_min_distance = e_cell
        e_cells.remove(e_cell_with_min_distance)
        p_cells.append(e_cell_with_min_distance)

    distance = np.ones_like(b_o_matrix) * 1000
    for cell, dis in cell_to_exit_distance.items():
        distance[cell[0], cell[1]] = dis

    return distance


def calc_e_matrix(pedestrian_move, b_o_matrix, exit_data, corresponding_node):
    pedestrian_no = pedestrian_move.shape[1]
    e_matrix = np.ones((pedestrian_no, 3, 3))
    for i in range(pedestrian_no):
        # Block the exit other than the pedestrian's destination
        b_o_matrix_modified = b_o_matrix.copy()
        # B_O_matrix_modified = np.where(B_O_matrix_modified == 2, 1, B_O_matrix_modified)
        b_o_matrix_modified[b_o_matrix_modified == 2] = 1
        location_destination = pedestrian_move[3, i]
        exit_data_pedestrian = exit_data[:, corresponding_node == location_destination].flatten()
        if exit_data_pedestrian[3] == 1:
            b_o_matrix_modified[exit_data_pedestrian[0]:exit_data_pedestrian[0] + exit_data_pedestrian[2],
            exit_data_pedestrian[1]] = 2
        else:
            b_o_matrix_modified[exit_data_pedestrian[0],
            exit_data_pedestrian[1]:exit_data_pedestrian[1] + exit_data_pedestrian[2]] = 2
        # if sum(pedestrian_stay_count.get(pedestrian_move[0, i], [])) >= 5:
        #     change_map = (~(b_o_matrix_modified[pedestrian_move[1, i] - 1:pedestrian_move[1, i] + 2,
        #                     pedestrian_move[2, i] - 1:pedestrian_move[2, i] + 2] == 1)).astype(np.int64)
        #     pedestrian_stay_count[pedestrian_move[0, i]] = []
        # else:
        #     change_map = (~np.logical_or(b_o_matrix_modified[pedestrian_move[1, i] - 1:pedestrian_move[1, i] + 2,
        #                                  pedestrian_move[2, i] - 1:pedestrian_move[2, i] + 2] == 1,
        #                                  b_o_matrix_modified[pedestrian_move[1, i] - 1:pedestrian_move[1, i] + 2,
        #                                  pedestrian_move[2, i] - 1:pedestrian_move[2, i] + 2] == 3)).astype(np.int64)
        change_map = (~np.logical_or(b_o_matrix_modified[pedestrian_move[1, i] - 1:pedestrian_move[1, i] + 2,
                                     pedestrian_move[2, i] - 1:pedestrian_move[2, i] + 2] == 1,
                                     b_o_matrix_modified[pedestrian_move[1, i] - 1:pedestrian_move[1, i] + 2,
                                     pedestrian_move[2, i] - 1:pedestrian_move[2, i] + 2] == 3)).astype(np.int64)
        change_map[1, 1] = 1
        e_matrix[i] = change_map

    return e_matrix


def next_step(pedestrian_move, corresponding_node, s_matrix_list, e_matrix, bridge=False):
    # Initialize the next_step data
    pedestrian_move = pedestrian_move.astype(np.int64)
    pedestrian_no = pedestrian_move.shape[1]
    next_step_data = np.empty((3, pedestrian_no), dtype=np.int64)
    next_step_data[0] = pedestrian_move[0]

    d = np.ones((3, 3))
    d[::2, ::2] = 0.7
    # Iterate through all the pedestrians
    r_enhance = 0.1 * np.random.randn(pedestrian_no) + r_enhance_basic
    for i in range(pedestrian_no):
        s_matrix = s_matrix_list[corresponding_node == pedestrian_move[3, i]][0]
        s = s_matrix[pedestrian_move[1, i] - 1:pedestrian_move[1, i] + 2,
            pedestrian_move[2, i] - 1:pedestrian_move[2, i] + 2]
        s = s[1, 1] - s
        e = e_matrix[i]
        r = np.ones((3, 3))
        i_matrix = np.ones((3, 3))
        # check if it is the pivot
        if bridge and next_step_data[0, i] in pedestrian_move_direction.keys():
            if pedestrian_move_direction[next_step_data[0, i]] != (0, 0):
                r_change_x = r_change[pedestrian_move_direction[next_step_data[0, i]]][0]
                r_change_y = r_change[pedestrian_move_direction[next_step_data[0, i]]][1]
                r[r_change_x, r_change_y] = r_enhance[i]
                previous_x = int(pedestrian_move_direction[next_step_data[0, i]][0])
                previous_y = int(pedestrian_move_direction[next_step_data[0, i]][1])
                i_matrix[previous_x + 1, previous_y + 1] = i_enhance_basic

        p = np.exp(s) * e * d * np.exp(r) * np.exp(i_matrix)

        # p[::2, ::2] = 0
        p = p / p.sum().sum()
        p = p.flatten()
        next_step_index = np.argmax(p)
        # next_step_index = np.random.choice(np.arange(9).astype(np.int64), p=p)
        next_step_index += 1
        next_step_delta_y = 1 if np.mod(next_step_index, 3) == 0 else np.mod(next_step_index, 3) - 2
        next_step_delta_x = np.ceil(next_step_index / 3) - 2
        next_step_data[1, i] = pedestrian_move[1, i] + next_step_delta_x
        next_step_data[2, i] = pedestrian_move[2, i] + next_step_delta_y

        # Update the pedestrian's moving direction and passing distance
        if bridge:
            pedestrian_move_direction[next_step_data[0, i]] = (next_step_delta_x, next_step_delta_y)
            bridge_passing_distance[next_step_data[0, i]] += np.linalg.norm([next_step_delta_x, next_step_delta_y])

        # Another scenario of counting passing distance
        # TODO 这里可以扩展成两座桥期间要经过多个区段的情况
        # If one has passed a bridge while going to pass another, the distance should be counted
        elif next_step_data[0, i] in bridge_passing_time and pedestrian_move[3, i] in bridge_all_node:
            bridge_passing_distance[next_step_data[0, i]] += np.linalg.norm([next_step_delta_x, next_step_delta_y])

        # Update the pedestrian stay count
        # if next_step_delta_x == 0 and next_step_delta_y == 0:
        #     pedestrian_stay_count[next_step_data[0, i]].append(1)

    # Deal with the collision
    permuted_array = np.random.permutation(pedestrian_no)
    next_step_data_temp = next_step_data[:, permuted_array]
    pedestrian_data_temp = pedestrian_move[:, permuted_array]
    duplicate_mapping = np.array([True] * pedestrian_no)
    unique_destination, unique_index = np.unique(next_step_data_temp[1:], axis=1, return_index=True)
    duplicate_mapping[unique_index] = False
    next_step_data_temp[1:3, duplicate_mapping] = pedestrian_data_temp[1:3, duplicate_mapping]
    next_step_data[:, permuted_array] = next_step_data_temp

    return next_step_data.astype(np.int64)


def generate_global_path(space_no, pedestrian_index):
    pedestrian_no = pedestrian_index.size
    global_path = dict()
    global_destination_array = np.random.choice(np.arange(1, attraction_rate.size + 1), size=pedestrian_no,
                                                p=attraction_rate)

    for i in range(pedestrian_no):
        flag = 0
        # Find the corresponding nodes for the space
        nodes_of_space = node_space_relationship[space_no]
        # Calculate shortest lengths from this nodes to the global destination
        nodes_to_destination_length = np.zeros(len(nodes_of_space))
        for j in range(len(nodes_of_space)):
            if global_destination_array[i] == nodes_of_space[j]:
                flag = 1
                break
            else:
                nodes_to_destination_length[j] = paths_shortest_length_dict[
                    tuple(sorted([global_destination_array[i], nodes_of_space[j]]))]
        if flag == 1:
            global_path[pedestrian_index[i]] = [global_destination_array[i]]
            continue

        # Determine the starting node
        arg_min_length = np.argmin(nodes_to_destination_length)
        chosen_start_node = nodes_of_space[int(arg_min_length)]
        # Determine the path
        node_combination_tuple = tuple(sorted([global_destination_array[i], chosen_start_node]))
        paths_prob = paths_prob_dict[node_combination_tuple]
        # chosen_path = paths_dict[node_combination_tuple][np.argmax(paths_prob)]
        paths_amount = paths_prob.size
        chosen_path_index = np.random.choice(np.arange(paths_amount), p=paths_prob)
        chosen_path = paths_dict[node_combination_tuple][chosen_path_index]
        if nodes_of_space[0] in chosen_path and nodes_of_space[1] in chosen_path:
            chosen_path = paths_dict[node_combination_tuple][np.argmax(paths_prob)]
        global_path[pedestrian_index[i]] = chosen_path if chosen_path[0] == chosen_start_node else chosen_path[::-1]

    return global_path


def space_to_space_transfer(time, pedestrian_transfer_index_position, current_local_destination):
    # Every pedestrian shares the same next local destination
    # Acquire the space no
    next_local_destination_index = pedestrian_global_path[pedestrian_transfer_index_position[0, 0]].index(
        current_local_destination) + 1
    next_local_destination = pedestrian_global_path[pedestrian_transfer_index_position[0, 0]][
        next_local_destination_index]
    node_combination_tuple = tuple(sorted([current_local_destination, next_local_destination]))
    space_intended = space_node_relationship[node_combination_tuple]
    exit_data = space_dict[space_intended].exit_data[:,
                space_dict[space_intended].corresponding_node == current_local_destination].flatten()

    # # Acquire the random positions
    # random_position = np.random.choice(np.arange(exit_data[2]), pedestrian_transfer_index.size, replace=False)

    if exit_data[3] == 1:
        pedestrian_x = (exit_data[0] + pedestrian_transfer_index_position[:, 1]).astype(np.int64).reshape(1, -1)
        position_y = exit_data[1] + 1 if exit_data[1] == 0 else exit_data[1] - 1
        pedestrian_y = np.ones_like(pedestrian_x, dtype=np.int64) * position_y
    else:
        pedestrian_y = (exit_data[1] + pedestrian_transfer_index_position[:, 1]).astype(np.int64).reshape(1, -1)
        position_x = exit_data[0] + 1 if exit_data[0] == 0 else exit_data[0] - 1
        pedestrian_x = np.ones_like(pedestrian_y, dtype=np.int64) * position_x

    local_destination = np.ones(pedestrian_transfer_index_position.shape[0],
                                dtype=np.int64) * next_local_destination
    transferred_pedestrian = np.vstack(
        (pedestrian_transfer_index_position[:, 0].T, pedestrian_x, pedestrian_y, local_destination))

    space_dict[space_intended].pedestrian_data = np.hstack(
        (space_dict[space_intended].pedestrian_data, transferred_pedestrian)).astype(np.int64)

    if space_intended in bridge_space and current_local_destination in bridge_end_node:
        for index_position_transfer in pedestrian_transfer_index_position:
            if index_position_transfer[0] not in bridge_passing_time:
                bridge_passing_time[index_position_transfer[0]].append(time)
            else:
                del bridge_passing_time[index_position_transfer[0]][1:]


def simu_gap(slope_info):
    # Calculate the simulation gap for certain slope degree and moving direction
    slope_velocity_mapping = {10.3: (50 / np.array([1.2, 1.15])).astype(np.int64),
                              9.3: (50 / np.array([1.19, 1.14])).astype(np.int64),
                              7.0: (50 / np.array([1.2, 1.21])).astype(np.int64),
                              1.3: (50 / np.array([1.25, 1.23])).astype(np.int64),
                              5.3: (50 / np.array([1.14, 1.16])).astype(np.int64),
                              7.4: (50 / np.array([1.23, 1.19])).astype(np.int64),
                              7.8: (50 / np.array([1.24, 1.2])).astype(np.int64),
                              7.3: (50 / np.array([1.23, 1.19])).astype(np.int64),
                              6.9: (50 / np.array([1.2, 1.21])).astype(np.int64),
                              6.4: (50 / np.array([1.2, 1.21])).astype(np.int64)}

    simulation_interval = dict()
    for bridge_range, slope in slope_info.items():
        simulation_interval.update({bridge_range: slope_velocity_mapping[slope]})
    return simulation_interval


# Location graph
location_G = nx.Graph()
location_G.add_weighted_edges_from(
    [(1, 12, 37.8), (2, 12, 55.3), (3, 13, 43.5), (4, 13, 43.5), (5, 13, 43.5), (6, 13, 47.9), (7, 14, 62.4),
     (8, 14, 36.9), (8, 9, 22.8), (10, 14, 41.6), (10, 11, 22.5), (12, 15, 44.5), (13, 15, 43.2), (14, 15, 129.7)])
node_array = np.arange(1, 16)

# Create the node combinations of 2 elements
node_combinations = list(combinations(node_array, 2))

paths_shortest_length_dict = dict()
paths_dict = dict()
paths_prob_dict = dict()

# Acquire the paths and weighted lengths(probability) for each combination
for node_combination in node_combinations:
    paths_shortest_length_dict[node_combination] = nx.shortest_path_length(location_G, node_combination[0],
                                                                           node_combination[1], weight='weight')
    paths_in_order = list(
        nx.shortest_simple_paths(location_G, node_combination[0], node_combination[1], weight='weight'))
    paths_dict[node_combination] = paths_in_order
    paths_weighted_lengths = np.zeros(len(paths_in_order))
    for index, path in enumerate(paths_in_order):
        paths_weighted_lengths[index] = location_G.subgraph(path).size(weight='weight')

    paths_weighted_lengths_inverse = 1 / paths_weighted_lengths
    paths_weighted_lengths_prob = paths_weighted_lengths_inverse / paths_weighted_lengths_inverse.sum()
    paths_prob_dict[node_combination] = paths_weighted_lengths_prob

space_node_relationship = {(1, 12): 1, (2, 12): 2, (3, 13): 3, (4, 13): 4, (5, 13): 5, (6, 13): 6, (7, 14): 7,
                           (8, 14): 8, (8, 9): 9, (10, 14): 10, (10, 11): 11, (12, 15): 12, (13, 15): 13, (14, 15): 14}
node_space_relationship = dict(zip(list(space_node_relationship.values()), list(space_node_relationship.keys())))

# Pedestrian global path
pedestrian_global_path = dict()

# Pedestrian stay count
# pedestrian_stay_count = defaultdict(list)

# Attraction rate setting
attraction_rate = np.zeros(14)
# Typical Mode
attraction_rate[[0, 1]] = 0.05
attraction_rate[[2, 3, 4]] = 0.1
attraction_rate[5] = 0.1
attraction_rate[[6, 8]] = 0.1
attraction_rate[10] = 0.2
attraction_rate[13] = 0.1

# Bridge specifications
bridge_space = [12, 13, 14]
bridge_end_node = [12, 13, 14]
bridge_all_node = [12, 13, 14, 15]

# Pedestrian distribution data
pedestrian_distribution_space = dict()
pedestrian_distribution_pivot = dict()

# Bridge passing data
bridge_passing_no = 0
bridge_passing_time = defaultdict(list)
bridge_passing_distance = defaultdict(lambda: 0)
bridge_pedestrian_on = dict()

# Simulation Configuration
pedestrian_start_index = 1
np.random.seed(123)

# Define pedestrians' moving direction to consider the right_inclined effect
pedestrian_move_direction = dict()
r_change = {(-1, -1): ([0, 0, 1], [1, 2, 2]), (-1, 0): ([0, 1, 2], [2] * 3), (-1, 1): ([1, 2, 2], [2, 2, 1]),
            (0, -1): ([0] * 3, [0, 1, 2]), (0, 1): ([2] * 3, [0, 1, 2]),
            (1, -1): ([0, 0, 1], [0, 1, 0]), (1, 0): ([0, 1, 2], [0] * 3), (1, 1): ([1, 2, 2], [0, 0, 1])}

# Initialize space and pivot
bridge_slope_info = dict()
bridge_slope_direction = dict()
bridge_wander_prob = dict()

# Space 12
bridge_slope_info[12] = {(-1, 39): 6.9, (39, 78): 6.4}
bridge_slope_direction[12] = {(-1, 39): (15, 12), (39, 78): (15, 12)}
bridge_wander_prob[12] = {(-1, 39): (0.0001, 50000), (39, 78): (0.0003, 40000)}

# Space 13
bridge_slope_info[13] = {(-1, 38): 7.3, (38, 77): 7.8}
bridge_slope_direction[13] = {(-1, 38): (15, 13), (38, 77): (15, 13)}
bridge_wander_prob[13] = {(-1, 38): (0.0001, 50000), (38, 77): (0.0003, 40000)}

# Space 14
bridge_slope_info[14] = {(-1, 50): 5.3, (50, 100): 1.3, (100, 150): 7.0, (150, 200): 9.3, (200, 253): 10.3}
bridge_slope_direction[14] = {(-1, 50): (15, 14), (50, 100): (15, 14), (100, 150): (15, 14), (150, 200): (15, 14),
                              (200, 253): (15, 14)}
bridge_wander_prob[14] = {(-1, 50): (0.001, 20000), (50, 100): (0.001, 20000), (100, 150): (0.0005, 30000),
                          (150, 200): (0.0003, 40000), (200, 253): (0.0001, 50000)}

s_matrix_all_space = np.load(current_dir + '\\distance\\distance_space.npy')
s_matrix_all_pivot = np.load(current_dir + '\\distance\\distance_pivot.npy')

space_dict = {
    1: Road(1, (77, 81), [(0, 24, 16, 0), (76, 1, 72, 0)], (1, 12), s_matrix_list=s_matrix_all_space[()][1]),
    2: Road(2, (46, 112), [(1, 0, 44, 1), (1, 111, 44, 1)], (2, 12), s_matrix_list=s_matrix_all_space[()][2]),
    3: Road(3, (16, 88), [(1, 0, 14, 1), (1, 87, 14, 1)], (3, 13), s_matrix_list=s_matrix_all_space[()][3]),
    4: Road(4, (21, 88), [(1, 0, 19, 1), (1, 87, 19, 1)], (4, 13), s_matrix_list=s_matrix_all_space[()][4]),
    5: Road(5, (34, 88), [(1, 0, 32, 1), (1, 87, 32, 1)], (5, 13), s_matrix_list=s_matrix_all_space[()][5]),
    6: Road(6, (97, 91), [(96, 1, 89, 0), (0, 1, 89, 0)], (6, 13), s_matrix_list=s_matrix_all_space[()][6]),
    7: Road(7, (126, 14), [(0, 1, 12, 0), (125, 1, 12, 0)], (7, 14), s_matrix_list=s_matrix_all_space[()][7]),
    8: Road(8, (75, 11), [(0, 1, 9, 0), (74, 1, 9, 0)], (8, 14), s_matrix_list=s_matrix_all_space[()][8]),
    9: Road(9, (47, 17), [(46, 1, 9, 0), (0, 6, 10, 0)], (8, 9), s_matrix_list=s_matrix_all_space[()][9]),
    10: Road(10, (84, 11), [(83, 1, 9, 0), (0, 1, 9, 0)], (10, 14), s_matrix_list=s_matrix_all_space[()][10]),
    11: Road(11, (46, 27), [(0, 16, 9, 0), (45, 1, 9, 0)], (10, 11), s_matrix_list=s_matrix_all_space[()][11]),
    12: Bridge(12, (78, 65), [(0, 1, 18, 0), (77, 46, 18, 0)], (12, 15), slope_info=bridge_slope_info[12],
               slope_direction=bridge_slope_direction[12], wander_prob=bridge_wander_prob[12], slope_along=1,
               s_matrix_list=s_matrix_all_space[()][12]),
    13: Bridge(13, (49, 77), [(48, 2, 25, 0), (1, 76, 17, 1)], (13, 15), slope_info=bridge_slope_info[13],
               slope_direction=bridge_slope_direction[13], wander_prob=bridge_wander_prob[13], slope_along=2,
               s_matrix_list=s_matrix_all_space[()][13]),
    14: Bridge(14, (56, 253), [(55, 223, 29, 0), (9, 0, 16, 1)], (14, 15), slope_info=bridge_slope_info[14],
               slope_direction=bridge_slope_direction[14], wander_prob=bridge_wander_prob[14], slope_along=2,
               s_matrix_list=s_matrix_all_space[()][14])
}

pivot_dict = {12: Pivot(12, (46, 104), [(0, 31, 72, 0), (1, 0, 44, 1), (9, 55, 18, 0)], (1, 2, 15), (1, 2, 12),
                        s_matrix_list=s_matrix_all_pivot[()][12]),
              13: Pivot(13, (174, 102),
                        [(1, 0, 14, 1), (58, 0, 19, 1), (105, 0, 32, 1), (173, 11, 89, 0), (79, 46, 25, 0)],
                        (3, 4, 5, 6, 15), (3, 4, 5, 6, 13), s_matrix_list=s_matrix_all_pivot[()][13]),
              14: Pivot(14, (99, 94), [(0, 7, 12, 0), (0, 48, 9, 0), (98, 35, 9, 0), (49, 43, 29, 0)], (7, 8, 10, 15),
                        (7, 8, 10, 14), s_matrix_list=s_matrix_all_pivot[()][14]),
              15: Pivot(15, (48, 59), [(0, 3, 18, 0), (30, 0, 17, 1), (10, 58, 16, 1)], (12, 13, 14),
                        (12, 13, 14), s_matrix_list=s_matrix_all_pivot[()][15])
              }

r_enhance_basic = 1.1
i_enhance_basic = 1.4

pedestrian_flow_factor = 3
# use 10ms as interval
time_stamp = 0
while time_stamp < 360001:
    for i in range(1, 15):
        space_dict[i].move_simulation(time_stamp)
    for i in [12, 13, 14, 15]:
        pivot_dict[i].move_simulation(time_stamp)

    if np.mod(time_stamp, 100) == 0:
        space_dict[1].pedestrian_generate(1395, 1)
        space_dict[2].pedestrian_generate(985, 1)
        space_dict[3].pedestrian_generate(525, 1)
        space_dict[4].pedestrian_generate(525, 1)
        space_dict[5].pedestrian_generate(525, 1)
        space_dict[6].pedestrian_generate(833, 1)
        space_dict[7].pedestrian_generate(808, 1)
        space_dict[9].pedestrian_generate(808, 1)
        space_dict[11].pedestrian_generate(1058, 1)

        # Initialize the temporary pedestrian data
        pedestrian_space = np.empty((4, 0), dtype=np.int64)
        pedestrian_pivot = np.empty((4, 0), dtype=np.int64)
        pedestrian_on_bridge = 0

        for i in range(1, 15):
            pedestrian_i = space_dict[i].pedestrian_data[:3]
            pedestrian_i = np.vstack((pedestrian_i, i * np.ones((1, pedestrian_i.shape[1])))).astype(np.int64)
            pedestrian_space = np.hstack((pedestrian_space, pedestrian_i))
            if i in bridge_space:
                pedestrian_on_bridge += space_dict[i].pedestrian_data.shape[1]
        for i in [12, 13, 14, 15]:
            pedestrian_i = pivot_dict[i].pedestrian_data[:3]
            pedestrian_i = np.vstack((pedestrian_i, i * np.ones((1, pedestrian_i.shape[1])))).astype(np.int64)
            pedestrian_pivot = np.hstack((pedestrian_pivot, pedestrian_i))

        pedestrian_distribution_space[time_stamp // 100] = pedestrian_space
        pedestrian_distribution_pivot[time_stamp // 100] = pedestrian_pivot
        bridge_pedestrian_on[time_stamp // 100] = pedestrian_on_bridge

    if time_stamp % 6000 == 0:
        print(time_stamp)
    time_stamp += 1

data_dict = {'distribution_space': pedestrian_distribution_space, 'distribution_pivot': pedestrian_distribution_pivot,
             'bridge_passing_no': bridge_passing_no, 'bridge_passing_time': dict(bridge_passing_time),
             'bridge_passing_distance': dict(bridge_passing_distance), 'pedestrian_on_bridge': bridge_pedestrian_on,
             'pedestrian_path': pedestrian_global_path}
np.save('result_data.npy', data_dict)
