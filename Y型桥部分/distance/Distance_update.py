import numpy as np
from collections import defaultdict


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
    distance_all = np.where(distance_list[0] < distance_list[1], distance_list[0], distance_list[1])
    for i in range(1, exit_row[2] - 1):
        distance_all = np.where(distance_all < distance_list[i + 1], distance_all, distance_list[i + 1])

    return distance_all


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
    print(cell_total)
    # If all the t_cells are turned into p_cells, then stop iteration
    while len(p_cells) < cell_total:
        print(len(p_cells))
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

    distance_all = np.ones_like(b_o_matrix) * 1000
    for cell, dis in cell_to_exit_distance.items():
        distance_all[cell[0], cell[1]] = dis

    return distance_all


# distance_space_old = np.load('distance_space.npy')
distance_pivot_old = np.load('distance_pivot.npy')

# distance_space_new = dict()
distance_pivot_new = dict()

# for i in range(1, 15):
#     distance_space_new[i] = distance_space_old[()][i]

for i in [12, 13, 15]:
    distance_pivot_new[i] = distance_pivot_old[()][i]

# # Space 12
# # Boundary and Occupation status matrix Initialization
# b_o_matrix = np.zeros((78, 65))
# b_o_matrix[1:-1, 1:-1] = -1
# b_o_matrix += 1
#
# x = np.arange(0, 78)
# y = np.arange(0, 65)
# Y, X = np.meshgrid(y, x)
#
# T = 45.16 * X - 76.7 * Y
#
# vacant_or_blocked = np.where(np.logical_and(T <= 0, T >= -1403.15), 0, 1)
# obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
# obstacle_1 = zip(obstacle_x, obstacle_y)
#
# obstacle_x, obstacle_y = zip(*obstacle_1)
# b_o_matrix[obstacle_x, obstacle_y] = 1
#
# exit_row = [(0, 1, 18, 0), (77, 46, 18, 0)]
# # Calculate S matrix
# s_matrix_list = []
# for i in range(len(exit_row)):
#     s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))
#
# distance_space_new[12] = np.array(s_matrix_list)
#
# # Space 13
# # Boundary and Occupation status matrix Initialization
# b_o_matrix = np.zeros((49, 77))
# b_o_matrix[1:-1, 1:-1] = -1
# b_o_matrix += 1
#
# x = np.arange(0, 49)
# y = np.arange(0, 77)
# Y, X = np.meshgrid(y, x)
#
# T = (X - 289.2) ** 2 + (Y - 207.4) ** 2
#
# vacant_or_blocked = np.where(np.logical_and(T <= 317.6 ** 2, T >= 301.6 ** 2), 0, 1)
# obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
# obstacle_1 = zip(obstacle_x, obstacle_y)
#
# obstacle_x, obstacle_y = zip(*obstacle_1)
# b_o_matrix[obstacle_x, obstacle_y] = 1
#
# exit_row = [(48, 1, 26, 0), (1, 76, 17, 1)]
# # Calculate S matrix
# s_matrix_list = []
# for i in range(len(exit_row)):
#     s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))
#
# distance_space_new[13] = np.array(s_matrix_list)

# # Space 14
# # Boundary and Occupation status matrix Initialization
# b_o_matrix = np.zeros((56, 253))
# b_o_matrix[1:-1, 1:-1] = -1
# b_o_matrix += 1
#
# x = np.arange(0, 56)
# y = np.arange(0, 253)
# Y, X = np.meshgrid(y, x)
#
# T = (X - 317.6) ** 2 + (Y - 74) ** 2
#
# vacant_or_blocked = np.where(np.logical_and(T <= 317.6 ** 2, T >= 301.6 ** 2), 0, 1)
# obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
# obstacle_1 = zip(obstacle_x, obstacle_y)
#
# obstacle_x, obstacle_y = zip(*obstacle_1)
# b_o_matrix[obstacle_x, obstacle_y] = 1
#
# exit_row = [(55, 223, 29, 0), (9, 0, 17, 1)]
# # Calculate S matrix
# s_matrix_list = []
# for i in range(len(exit_row)):
#     s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))
#
# distance_space_new[14] = np.array(s_matrix_list)


# # Pivot 12
# # Boundary and Occupation status matrix Initialization
# b_o_matrix = np.zeros((46, 104))
# b_o_matrix[1:-1, 1:-1] = -1
# b_o_matrix += 1
#
# exit_row = [(0, 31, 72, 0), (1, 0, 44, 1), (9, 55, 18, 0)]
# # Calculate S matrix
# s_matrix_list = []
# for i in range(len(exit_row)):
#     for j in range(len(exit_row)):
#         if i != j:
#             if exit_row[j][3] == 1:
#                 b_o_matrix[exit_row[j][0]:exit_row[j][0] + exit_row[j][2], exit_row[j][1]] = 1
#             else:
#                 b_o_matrix[exit_row[j][0], exit_row[j][1]:exit_row[j][1] + exit_row[j][2]] = 1
#     s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))
#
# distance_pivot_new[12] = np.array(s_matrix_list)
#
# # Pivot 13
# # Boundary and Occupation status matrix Initialization
# b_o_matrix = np.zeros((174, 102))
# b_o_matrix[1:-1, 1:-1] = -1
# b_o_matrix += 1
#
# exit_row = [(1, 0, 14, 1), (58, 0, 19, 1), (105, 0, 32, 1), (173, 11, 89, 0), (79, 46, 26, 0)]
# # Calculate S matrix
# s_matrix_list = []
# for i in range(len(exit_row)):
#     for j in range(len(exit_row)):
#         if i != j:
#             if exit_row[j][3] == 1:
#                 b_o_matrix[exit_row[j][0]:exit_row[j][0] + exit_row[j][2], exit_row[j][1]] = 1
#             else:
#                 b_o_matrix[exit_row[j][0], exit_row[j][1]:exit_row[j][1] + exit_row[j][2]] = 1
#     s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))
#
# distance_pivot_new[13] = np.array(s_matrix_list)

# Pivot 14
# Boundary and Occupation status matrix Initialization


exit_row = [(0, 7, 12, 0), (0, 48, 9, 0), (98, 35, 9, 0), (49, 43, 29, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    b_o_matrix = np.zeros((99, 94))
    b_o_matrix[1:-1, 1:-1] = -1
    b_o_matrix += 1
    if i != 3:
        b_o_matrix[exit_row[3][0], exit_row[3][1]:exit_row[3][1] + exit_row[3][2]] = 1
        s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))
    else:
        s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_pivot_new[14] = np.array(s_matrix_list)
#
# Pivot 15
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((48, 59))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 48)
y = np.arange(0, 59)
Y, X = np.meshgrid(y, x)

T_1 = 2.49 * X + 29.6 * Y
T_2 = 37.0238 * X - 9.6634 * Y
T_3 = (X - 318.6) ** 2 + (Y - 131.8) ** 2

range_map = np.logical_and(np.logical_and(T_1 >= 73.58, T_2 >= -200.84), T_3 >= 301.6 ** 2)

vacant_or_blocked = np.where(range_map, 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(0, 3, 18, 0), (30, 0, 17, 1), (10, 58, 17, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_pivot_new[15] = np.array(s_matrix_list)

# np.save('distance_space.npy', distance_space_new)
np.save('distance_pivot.npy', distance_pivot_new)
