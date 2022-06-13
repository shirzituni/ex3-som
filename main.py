import math
import random
import sys

import numpy
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
# import seaborn as sns
from scipy.spatial import distance


grid_rows = 8
grid_cols = 8
grid_size = grid_rows * grid_cols

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

'''
def draw_reg_poly(surface, color, vertex_count, radius, position, width=0):
    n, r = vertex_count, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * np.cos(2 * np.pi * i / n),
         y + r * np.sin(2 * np.pi * i / n))
        for i in range(n)], width)


def plot_data(tsne_results):
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(x[0], y[0], c='red')
    plt.scatter(x[1:], y[1:])
    plt.show()
'''


# find the approximate random vector to input_vector
def algo_approximation(data, rand_vectors):
    for input_data in data:
        # find the distance between the input data to the random vectors
        difference = [math.dist(input_data, rand_point) for rand_point in rand_vectors]
        # save the closet point to the input point
        closet_rand_point = rand_vectors[np.argmin(difference)]
        # curr_data = np.concatenate((input_data.reshape(1, 2), rand_vectors))
        dist = np.min(difference)

        # approximate input_vector to the "closest" vector
        vector_1 = closet_rand_point - input_data
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        # move the random vector to the point
        closest_rand_point_after_approx = closet_rand_point - vector_1*(2/10)*dist
        # we need to know who are the neighborhoods in order to approx them too
        return closest_rand_point_after_approx


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
    max_intensity = math.e + np.log((1*max_val - min_val)/max_val + 1)

    for i in range(len(merez_votes_list)):
        intensity = math.e + np.log(((1*merez_votes_list[i]) - min_val)/max_val + 1)
        for j in range(len(groups)):
            if i in groups[j][0]:
                groups[j][1] += intensity/len(groups[j][0])
                break

    for j in range(len(groups)):
        groups[j][1] /= max_intensity

    a = 5



def calculate_euclidean_dist(vector1, vector2):
    dist = numpy.linalg.norm(vector2-vector1)
    return dist


# This function is for clustering groups.
# The aim is that the vector in size 196 will be in size 61.
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
                    [min_dist+x*((max_dist-min_dist)/61) for x in range(0, 61)],  # start
                    [min_dist+(x+1)*((max_dist-min_dist)/61) for x in range(0, 61)]  # end
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

    assert count == 196

    groups = [[distances[i][2], 0] for i in range(0, len(distances))]
    calculate_color(groups)
    return groups
    '''
    map_vector_to_group = dict()
    for k in range(0,60):
        for j in range (0,195):
            if (k-1) * offset < dict_vec_euclidean_dist.get(k-1) < k * offset:
                map_vector_to_group[k].append(dict_vec_euclidean_dist[j-1])
            #if euclidean_dist
    '''


def cluster_groups_to_hexagons(groups, centers_list):
    return list(zip(centers_list.tolist(), groups))

#todo: fix the number of hexagons


def map_each_vector_to_hexagon(vector_as_points, vector_hexagon_centers):
    map_of_vector_to_hexagon = np.column_stack((vector_hexagon_centers, vector_as_points))
    return map_of_vector_to_hexagon


if __name__ == '__main__':
    data, random_vectors, cities_list = get_data('Elec_24.csv')
    data_as_points, rand_vec_as_points = make_as_points(data, random_vectors)
    hexagons_centers = paint()
    #map_vector_hexagon = map_each_vector_to_hexagon(data_as_points, paint())
    #print(map_vector_hexagon)
    algo_approximation(data_as_points, rand_vec_as_points)

    vector_to_groups = cluster_groups_to_distances(data_as_points)
    color(hexagons_centers, vector_to_groups)
    group_to_hexagon = cluster_groups_to_hexagons(vector_to_groups, hexagons_centers)
    plt.show()

