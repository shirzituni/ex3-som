import csv
import math
import random
import pandas as pd
from sklearn.manifold import TSNE
from hexalattice.hexalattice import *

grid_rows = 8
grid_cols = 8
grid_size = grid_rows * grid_cols
FILE_NAME = 'Elec_24.csv'

class Hexagon:
    Y_OFFSET = 0.86603
    X_OFFSET = 1

    def __init__(self, center, centers_list):
        self.x = float(center[0])
        self.y = float(center[1])
        self.centers_list = centers_list

    '''
    Mapping between values that we know to calculate their ring distance and the real hexagons centers
    '''

    def rotate_axis_60_degree(self):
        all_rows_mapping = []
        for j in range(0, 5):
            for i in range(0 - j, 5):
                all_rows_mapping.append((i, -4 + j))

        for k in range(1, 5):
            for m in range(-4, 5 - k):
                all_rows_mapping.append((m, k))

        map_center_to_rotate_axis = (list(zip(all_rows_mapping, self.centers_list)))
        return map_center_to_rotate_axis

    @classmethod
    def sign(cls, number):
        return 1 - (number <= 0)

    @classmethod
    def calc_ring_distance(cls, point, another_point):
        dx = another_point[0] - point[0]
        dy = another_point[1] - point[1]

        if cls.sign(dx) == cls.sign(dy):
            return abs(dx + dy)
        else:
            return max(abs(dx), abs(dy))

    '''
    Args:
        centers_to_rotated_centers_mapping: Mapping from centers to the 60 degree rotated points
    '''

    def generate_neighbors_rings(self, centers_to_rotated_centers_mapping):
        centers_to_rings = {}
        # (x, y) : {0: [], 1: [], 2: [], 3: [], ...}
        for center, orig_center in centers_to_rotated_centers_mapping:
            key = tuple(orig_center.tolist())
            centers_to_rings[key] = dict(zip(range(1, grid_rows + 1), [[] for _ in range(1, grid_rows + 1)]))
            for other_center, orig_other_center in centers_to_rotated_centers_mapping:
                if all(orig_center[i] == orig_other_center[i] for i in range(len(orig_center))):
                    continue
                centers_to_rings[key][self.calc_ring_distance(center, other_center)].append(orig_other_center)
        return centers_to_rings


def get_data(path):
    # get data from csv file
    data = pd.read_csv(filepath_or_buffer=path, header=None).values
    data2 = pd.read_csv(FILE_NAME)
    cities_list = data2['Municipality'].tolist()
    # convert str to int
    data[1:, 1:] = data[1:, 1:].astype(np.uint8)
    # create 61 random vectors with same dim=15
    random_vectors = np.ndarray(shape=(61, 15), dtype=np.uint8)
    # generate values for the random vectors
    for i in range(15):
        random_vectors[:, i] = [
            random.randint(np.min(data[1:, i + 1].astype(int)), np.max(data[1:, i + 1].astype(int)))
            for x in range(61)]
    return data, random_vectors, cities_list


# find the approximate random vector to input_vector
def algo_approximation(input_points, hexagon_to_randvec, neighbors_data):
    # 10 iterations
    for input_data in input_points:
        # find the distance between the input data to the random vectors
        difference = [math.dist(input_data, list(hex_to_vec.values())[0]) for hex_to_vec in hexagon_to_randvec]

        # save the nearset point to the input point
        index_of_nearest_rand_vector = np.argmin(difference)
        nearest_hexagon_value = list(hexagon_to_randvec[index_of_nearest_rand_vector].keys())[0]
        nearest_rand_point = list(hexagon_to_randvec[index_of_nearest_rand_vector].values())[0]

        # approximate input_vector to the "nearest" vector
        vector_1 = nearest_rand_point - input_data
        vector_1 = vector_1 / np.linalg.norm(vector_1)

        # move the random vector to the point
        dist = np.min(difference)
        hexagon_to_randvec[index_of_nearest_rand_vector][nearest_hexagon_value] = \
            nearest_rand_point - vector_1 * (3 / 10) * dist

        # approx neighbors of nearest_rand_point
        for circle_rank, neighbors_set in neighbors_data[nearest_hexagon_value].items():
            for neighbor in neighbors_set:
                find_helper = [0] * 61
                for i, cur_dict in enumerate(hexagon_to_randvec):
                    if list(cur_dict.keys())[0] == tuple(neighbor):
                        find_helper[i] = 1
                index = np.argmax(find_helper)
                vec_from_rand = list(hexagon_to_randvec[index].values())[0]
                vector_1 = vec_from_rand - input_data
                vector_1 = vector_1 / np.linalg.norm(vector_1)
                dist = np.min(difference)
                if circle_rank <= 2:
                    hexagon_to_randvec[index][tuple(neighbor)] = vec_from_rand - vector_1 * (
                            3 - circle_rank / 10) * dist
    return hexagon_to_randvec


def make_as_points(input_data, rand_vectors):
    # save only numeric data (without headers)
    only_numeric_data = input_data[1:, 1:]
    # initialize TSNE Algo to reduce vector dim from 15 to 2
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    data_to_tsne = np.concatenate((only_numeric_data, rand_vectors))
    tsne_results = tsne.fit_transform(data_to_tsne)
    # save input_points and random_points
    input_data_as_points = tsne_results[:196]
    rand_vector_as_points = tsne_results[196:]
    return input_data_as_points, rand_vector_as_points


def paint():
    hex_centers, _ = create_hex_grid(n=100, crop_circ=4)
    return hex_centers


def color(hex_centers, groups):
    x_hex_coords = hex_centers[:, 0]
    y_hex_coords = hex_centers[:, 1]
    colors = np.zeros([grid_size, 3])
    for i in range(0, grid_size - 3):
        colors[i] = np.array([groups[i][1], 0.8, 0])  # RGB 0-1

    plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=0.9,
                                      plotting_gap=0.05,
                                      rotate_deg=0)


'''
Mapping random vector to specific hexagon (by his center)
'''


def hexagon_to_random_vector_func(centers_list, random_vectors_input):
    return [{tuple(center): rand_vec} for center, rand_vec in zip(centers_list, random_vectors_input)]


def map_city_to_hexagon(input_points, hexagon_to_rand_vector):
    hexagon_to_cities = {}
    for item in hexagon_to_rand_vector:
        hexagon_to_cities[list(item.keys())[0]] = []
    for city_index, input_data in enumerate(input_points):
        # find the distance between the input data to the random vectors
        difference = [math.dist(input_data, list(hex_to_vec.values())[0]) for hex_to_vec in hexagon_to_rand_vector]
        # save the nearset point to the input point
        index_of_nearest_rand_vector = np.argmin(difference)
        nearest_hexagon_value = list(hexagon_to_rand_vector[index_of_nearest_rand_vector].keys())[0]
        hexagon_to_cities[nearest_hexagon_value].append(city_index)
    return hexagon_to_cities


def calc_color_for_hex(hex_to_cities, all_data):
    x = [np.round(list(hex_to_cities.keys())[i][0], 3) for i in range(len(hex_to_cities.keys()))]
    x = np.array(x)
    y = [np.round(list(hex_to_cities.keys())[i][1], 3) for i in range(len(hex_to_cities.keys()))]
    y = np.array(y)
    z = []
    for i in range(len(hex_to_cities.keys())):
        vals = list(hex_to_cities.values())[i]
        if len(vals) != 0:
            res = [all_data[i + 1, 1] for i in vals]
            z.append(np.round(np.mean(res), 3))
        else:
            z.append(0)
    z = z / np.linalg.norm(z)
    colors = np.zeros([grid_size, 3])
    for i in range(0, grid_size - 3):
        if z[i] != 0:
            colors[i] = np.array([0.2, z[i], 2 * z[i]])  # RGB 0-1
        else:
            colors[i] = np.array([0.8, 0.8, 0.8])  # RGB 0-1
    plot_single_lattice_custom_colors(x, y,
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=0.9,
                                      plotting_gap=0.05,
                                      rotate_deg=0)

    plt.show()


def shuffle_csv(filename):
    file = open(filename)
    shuffle_list = file.readlines()
    files_names = ["file1.csv", "file2.csv", "file3.csv", "file4.csv", "file5.csv",
                   "file6.csv", "file7.csv", "file8.csv", "file9.csv"]
    fields = shuffle_list[0]
    shuffle_list = shuffle_list[1:]

    for file in files_names:
        random.shuffle(shuffle_list)
        with open(file, 'w') as f:
            f.write(fields)
            f.write(''.join(shuffle_list))
    return shuffle_list


def main():

    # get data
    data, random_vectors, cities_list = get_data(FILE_NAME)

    # make a rand vector as points, make our data to points
    data_as_points, rand_vec_as_points = make_as_points(data, random_vectors)

    # paint hexagons
    hexagons_centers = paint()

    # map each point of a random vector for each hexagon
    hexagon_to_random_vector = hexagon_to_random_vector_func(hexagons_centers, rand_vec_as_points)

    # create an instance of specific hexagon for test
    hexagon = Hexagon(hexagons_centers[8], hexagons_centers)
    hexagon.rotate_axis_60_degree()

    # calculate the neighbors of all the hexagons
    neighbors_data = hexagon.generate_neighbors_rings(hexagon.rotate_axis_60_degree())

    # make approximation
    hexagon_to_rand_vector = algo_approximation(data_as_points, hexagon_to_random_vector, neighbors_data)

    # map city to hexagon
    hex_to_cities = map_city_to_hexagon(data_as_points, hexagon_to_rand_vector)
    print(hex_to_cities)
    # calculate economic for group
    # calc_color_for_hex(hex_to_cities, data)

    return hex_to_cities, hexagon_to_rand_vector, data_as_points, data


""" -------- MAIN -------- """
grades = []
all_data = []

for run in range(10):
    run_grade = 0
    hex_to_cities, rand_vector_to_hex, points_data, data = main()
    all_data.append((hex_to_cities, data))
    for index, hexagon in enumerate(hex_to_cities):
        cities_indices = hex_to_cities[hexagon]
        hexagon_vec = list(rand_vector_to_hex[index].values())[0]
        total = 0
        for city_index in cities_indices:
            dist = math.dist(points_data[city_index], hexagon_vec)
            total += dist
        total_dist = total / len(cities_indices) if total != 0 else 0
        run_grade += total_dist
    grades.append(run_grade)


index_of_best_run = np.argmin(grades)
'''
Graph creation

shuffle_csv(FILE_NAME)
iter_num = [0, 1, 2, 3, 4, 5, 6, 7, 8]
result_of_each_file = [47.0039, 46.9733, 43.7814, 45.3665, 45.40726, 46.5509, 47.0327, 43.5043, 44.9258]
plt.xlabel('iteration number')
plt.ylabel('file best result')
plt.plot(iter_num, result_of_each_file)
'''

print(f"Run number {index_of_best_run} with grade {min(grades)} was the best run")
calc_color_for_hex(all_data[index_of_best_run][0], all_data[index_of_best_run][1])
