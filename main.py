import math
import random

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from hexalattice.hexalattice import *

# import seaborn as sns

grid_rows = 8
grid_cols = 8
grid_size = grid_rows * grid_cols


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
    data2 = pd.read_csv('Elec_24.csv')
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
            nearest_rand_point - vector_1 * (2 / 10) * dist

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
                hexagon_to_randvec[index][tuple(neighbor)] = vec_from_rand - vector_1 * (9 - circle_rank / 10) * dist
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


def calculate_color(groups):
    data = pd.read_csv('Elec_24.csv')
    merez_votes_list = data['Merez'].tolist()
    # min and max
    # normalize color by the votes
    # 0 - 255
    # min_val - max_val
    # min_val ==> 0
    # max_val ==> 255

    # min_val = 1 ==> 0
    # max_val = 100 ==> 255
    # 50 ==> (50-1)/100 * 255

    min_val = min(merez_votes_list)
    max_val = max(merez_votes_list)
    max_intensity = math.e + np.log((1 * max_val - min_val) / max_val + 1)

    for i in range(len(merez_votes_list)):
        intensity = math.e + np.log(((1 * merez_votes_list[i]) - min_val) / max_val + 1)
        for j in range(len(groups)):
            if i in groups[j][0]:
                groups[j][1] += intensity / len(groups[j][0])
                break

    for j in range(len(groups)):
        groups[j][1] /= max_intensity


def calculate_euclidean_dist(vector1, vector2):
    dist = np.linalg.norm(vector2 - vector1)
    return dist


'''
This function is for clustering groups.
The aim is that the vector in size 196 will be in size 61.
NEED CHANGE -> JUST A DRAFT FOR CLUSTERING TO GROUPS
'''


def cluster_groups_to_distances(vectors):
    max_dist = 0
    origin = np.zeros(shape=(1, 2), dtype=float, order='C')
    min_dist = float('inf')
    dict_vec_euclidean_dist = dict()

    # calculate the min and the max distance from the origin (0,0)
    for i in range(0, 196):
        euclidean_dist = calculate_euclidean_dist(vectors[i], origin)
        # dict_vec_euclidean_dist[vectors[i]] = euclidean_dist
        dict_vec_euclidean_dist[i] = euclidean_dist
        if euclidean_dist > max_dist:
            max_dist = euclidean_dist
        if euclidean_dist < min_dist:
            min_dist = euclidean_dist
    print(min_dist)
    print(max_dist)
    dist_between_max_min = max_dist - min_dist
    print(dist_between_max_min)
    # cluster each vector by his distance to the origin
    offset = dist_between_max_min / 61

    distances = [(s, e, [])  # create couples of start and end
                 for s, e in zip(  # for each couple
            [min_dist + x * ((max_dist - min_dist) / 61) for x in range(0, 61)],  # start
            [min_dist + (x + 1) * ((max_dist - min_dist) / 61) for x in range(0, 61)]  # end
        )]

    for city_idx, city_dist in dict_vec_euclidean_dist.items():
        for section in distances:
            if section[0] <= city_dist <= section[1]:
                section[2].append(city_idx)
                break

    # Verify all were mapped
    count = 0
    for section in distances:
        count += len(section[2])

    # assert count == 196

    groups = [[distances[i][2], 0] for i in range(0, len(distances))]
    calculate_color(groups)
    return groups

def cluster_groups_to_hexagons(groups, centers_list):
    return list(zip(centers_list.tolist(), groups))


'''
Mapping random vector to specific hexagon (by his center)
'''


def hexagon_to_random_vector_func(centers_list, random_vectors_input):
    return [{tuple(center): rand_vec} for center, rand_vec in zip(centers_list, random_vectors_input)]
    # return list(zip(random_vectors_input, centers_list.tolist()))


def map_each_vector_to_hexagon(vector_as_points, vector_hexagon_centers):
    map_of_vector_to_hexagon = np.column_stack((vector_hexagon_centers, vector_as_points))
    return map_of_vector_to_hexagon


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
        # try:
        hexagon_to_cities[nearest_hexagon_value].append(city_index)
        # except:
        #     hexagon_to_cities[nearest_hexagon_value] = [city_index]
    return hexagon_to_cities


def calc_color_for_hex(hex_to_cities, all_data):
    x = [np.round(list(hex_to_cities.keys())[i][0], 3) for i in range(len(hex_to_cities.keys()))]
    x = np.array(x)
    y = [np.round(list(hex_to_cities.keys())[i][1], 3) for i in range(len(hex_to_cities.keys()))]
    y = np.array(y)
    # z = [np.round(np.mean(list(hex_to_cities.values())[i]), 3) for i in range(len(hex_to_cities.keys())) if len(hex_to_cities.values()) != 0 else 1]
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
    edge_colors = np.zeros([grid_size, 3])
    for i in range(0, grid_size - 3):
        if z[i] != 0:
            colors[i] = np.array([0.5, z[i], 0.1])  # RGB 0-1
        else:
            colors[i] = np.array([0.9, 0.9, 0.8])  # RGB 0-1
    plot_single_lattice_custom_colors(x, y,
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=0.9,
                                      plotting_gap=0.05,
                                      rotate_deg=0)

    plt.show()
    return 0


def main():
    # get data
    data, random_vectors, cities_list = get_data('Elec_24.csv')

    # make a rand vector as points, make our data to points
    data_as_points, rand_vec_as_points = make_as_points(data, random_vectors)

    # paint hexagons
    hexagons_centers = paint()

    # map each point of a random vector for each hexagon
    hexagon_to_random_vector = hexagon_to_random_vector_func(hexagons_centers, rand_vec_as_points)

    # NEED CHANGE -> JUST A DRAFT FOR CLUSTERING TO GROUPS
    vector_to_groups = cluster_groups_to_distances(data_as_points)

    # create an instance of specific hexagon for test
    hexagon = Hexagon(hexagons_centers[8], hexagons_centers)
    hexagon.rotate_axis_60_degree()

    # calculate the neighbors of all the hexagons
    neighbors_data = hexagon.generate_neighbors_rings(hexagon.rotate_axis_60_degree())

    # make approximation
    hexagon_to_rand_vector = algo_approximation(data_as_points, hexagon_to_random_vector, neighbors_data)

    # map city to hexagon
    hex_to_cities = map_city_to_hexagon(data_as_points, hexagon_to_rand_vector)

    # calculate economic for group
    hex_colors = calc_color_for_hex(hex_to_cities, data)

    # color the hexagons
    # color(hexagons_centers, vector_to_groups)
    # plt.show()


main()
