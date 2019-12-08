import numpy as np


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

    # P_cell:0, T_cell:1, E_cell:2
    cell_type = dict()

    # Set initial T_cells
    vacant_x, vacant_y = (b_o_matrix == 0).nonzero()
    for cell in zip(vacant_x, vacant_y):
        cell_type[cell] = 1

    # Set initial P_cells
    for cell in exit_data:
        cell_type[cell] = 0

    # Set parent_cells and distance to the exit:
    parent_cells = dict()
    distance_to_exit = dict()
    for cell, c_type in cell_type.items():
        if c_type == 0:
            parent_cells[cell] = cell
            distance_to_exit[cell] = 0

    # Set it to check whether a cell has been counted as p
    p_cell_counted = dict(zip(list(cell_type.keys()), [False] * len(cell_type)))

    print(len(cell_type))

    p_cell_no = len(exit_data)
    # If all the t_cells are turned into p_cells, then stop iteration
    while True:
        for cell, c_type in cell_type.items():
            # Check whether should we count this cell as p_cell
            if c_type != 0 or p_cell_counted[cell]:
                continue

            # Set the cell to be counted as p_cell
            p_cell_counted[cell] = True

            # Iterate through all the p_cells
            x_adjacent = np.arange(cell[0] - 1, cell[0] + 2)
            y_adjacent = np.arange(cell[1] - 1, cell[1] + 2)

            p_cell_array = np.array(cell)

            parent_cell = parent_cells[cell]
            parent_cell_array = np.array(parent_cell)

            # Iterate through a p_cell's adjacent area
            for adjacent_i in range(x_adjacent.size):
                for adjacent_j in range(y_adjacent.size):
                    adjacent_cell = (x_adjacent[adjacent_i], y_adjacent[adjacent_j])
                    adjacent_cell_array = np.array(adjacent_cell)
                    if adjacent_cell not in cell_type.keys():
                        continue
                    if cell_type[adjacent_cell] == 1:
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
                        cell_type[adjacent_cell] = 2

                        if approach_check:
                            parent_cells[adjacent_cell] = parent_cell
                            distance_to_exit[adjacent_cell] = distance_to_exit[parent_cell] + np.linalg.norm(
                                adjacent_cell_array - parent_cell_array)
                        else:
                            parent_cells[adjacent_cell] = cell
                            distance_to_exit[adjacent_cell] = distance_to_exit[cell] + np.linalg.norm(
                                adjacent_cell_array - p_cell_array)

                    elif cell_type[adjacent_cell] == 2:
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
                            test_distance = distance_to_exit[parent_cell] + np.linalg.norm(
                                adjacent_cell_array - parent_cell_array)
                            if test_distance < distance_to_exit[adjacent_cell]:
                                parent_cells[adjacent_cell] = parent_cell
                                distance_to_exit[adjacent_cell] = test_distance
                        else:
                            test_distance = distance_to_exit[cell] + np.linalg.norm(
                                adjacent_cell_array - p_cell_array)
                            if test_distance < distance_to_exit[adjacent_cell]:
                                parent_cells[adjacent_cell] = cell
                                distance_to_exit[adjacent_cell] = test_distance
                    else:
                        continue
        # Update the e_cells, choose the e_cell with smallest distance to be in the p_cells
        e_cell_distance_min = np.inf
        e_cell_with_min_distance = tuple()

        for cell, c_type in cell_type.items():
            if c_type == 2:
                if distance_to_exit[cell] < e_cell_distance_min:
                    e_cell_distance_min = distance_to_exit[cell]
                    e_cell_with_min_distance = cell

        cell_type[e_cell_with_min_distance] = 0
        p_cell_no += 1
        print(p_cell_no)

        # Check whether the iteration is over
        type_value = list(cell_type.values())
        if (np.array(type_value) == 0).all():
            break

    distance_all = np.ones_like(b_o_matrix) * 1000
    for cell, dis in distance_to_exit.items():
        distance_all[cell[0], cell[1]] = dis

    return distance_all


distance_space = dict()

# Space 1
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((77, 81))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 77)
y = np.arange(0, 81)
Y, X = np.meshgrid(y, x)

T = X + 10.9954 * Y
vacant_or_blocked = np.where(np.logical_and(T <= 878.5548, T >= 75.6), 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(0, 24, 16, 0), (76, 1, 72, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_space[1] = np.array(s_matrix_list)


# Space 2
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((46, 112))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(1, 0, 44, 1), (1, 111, 44, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[2] = np.array(s_matrix_list)


# Space 3
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((16, 88))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(1, 0, 14, 1), (1, 87, 14, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[3] = np.array(s_matrix_list)


# Space 4
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((21, 88))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(1, 0, 19, 1), (1, 87, 19, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[4] = np.array(s_matrix_list)


# Space 5
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((34, 88))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(1, 0, 32, 1), (1, 87, 32, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[5] = np.array(s_matrix_list)


# Space 6
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((97, 91))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(96, 1, 89, 0), (0, 1, 89, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[6] = np.array(s_matrix_list)


# Space 7
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((126, 14))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(0, 1, 12, 0), (125, 1, 12, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[7] = np.array(s_matrix_list)


# Space 8
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((75, 11))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(0, 1, 9, 0), (74, 1, 9, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[8] = np.array(s_matrix_list)


# Space 9
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((47, 17))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 47)
y = np.arange(0, 17)
Y, X = np.meshgrid(y, x)

T = X + 7.9134 * Y
vacant_or_blocked = np.where(np.logical_and(T <= 124.2474, T >= 45.6), 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(46, 1, 9, 0), (0, 6, 10, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_space[9] = np.array(s_matrix_list)


# Space 10
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((84, 11))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(83, 1, 9, 0), (0, 1, 9, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_space[10] = np.array(s_matrix_list)


# Space 11
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((46, 27))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 46)
y = np.arange(0, 27)
Y, X = np.meshgrid(y, x)

T = X + 2.8851 * Y
vacant_or_blocked = np.where(np.logical_and(T <= 75.2722, T >= 45), 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(0, 16, 9, 0), (45, 1, 9, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_space[11] = np.array(s_matrix_list)


# Space 12
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((78, 65))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 78)
y = np.arange(0, 65)
Y, X = np.meshgrid(y, x)

T = 45.16 * X - 76.7 * Y

vacant_or_blocked = np.where(np.logical_and(T <= 0, T >= -1403.15), 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(0, 1, 18, 0), (77, 46, 18, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_space[12] = np.array(s_matrix_list)


# Space 13
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((49, 77))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 49)
y = np.arange(0, 77)
Y, X = np.meshgrid(y, x)

T = (X - 289.2) ** 2 + (Y - 207.4) ** 2

vacant_or_blocked = np.where(np.logical_and(T <= 317.6 ** 2, T >= 301.6 ** 2), 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(48, 2, 25, 0), (1, 76, 17, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_space[13] = np.array(s_matrix_list)


# Space 14
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((56, 253))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

x = np.arange(0, 56)
y = np.arange(0, 253)
Y, X = np.meshgrid(y, x)

T = (X - 317.6) ** 2 + (Y - 74) ** 2

vacant_or_blocked = np.where(np.logical_and(T <= 317.6 ** 2, T >= 301.6 ** 2), 0, 1)
obstacle_x, obstacle_y = vacant_or_blocked.nonzero()
obstacle_1 = zip(obstacle_x, obstacle_y)

obstacle_x, obstacle_y = zip(*obstacle_1)
b_o_matrix[obstacle_x, obstacle_y] = 1

exit_row = [(55, 223, 29, 0), (9, 0, 16, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_space[14] = np.array(s_matrix_list)


distance_pivot = dict()


# Pivot 12
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((46, 104))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(0, 31, 72, 0), (1, 0, 44, 1), (9, 55, 18, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_pivot[12] = np.array(s_matrix_list)


# Pivot 13
# Boundary and Occupation status matrix Initialization
b_o_matrix = np.zeros((174, 102))
b_o_matrix[1:-1, 1:-1] = -1
b_o_matrix += 1

exit_row = [(1, 0, 14, 1), (58, 0, 19, 1), (105, 0, 32, 1), (173, 11, 89, 0), (79, 46, 25, 0)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance(b_o_matrix, exit_row=exit_row[i]))

distance_pivot[13] = np.array(s_matrix_list)


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

distance_pivot[14] = np.array(s_matrix_list)


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

exit_row = [(0, 3, 18, 0), (30, 0, 17, 1), (10, 58, 16, 1)]
# Calculate S matrix
s_matrix_list = []
for i in range(len(exit_row)):
    s_matrix_list.append(calc_distance_dij(b_o_matrix, exit_row=exit_row[i]))

distance_pivot[15] = np.array(s_matrix_list)


np.save('distance_space.npy', distance_space)
np.save('distance_pivot.npy', distance_pivot)
