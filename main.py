import math
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(path):
    data = pd.read_csv(filepath_or_buffer=path, header=None).values
    random_vectors = np.ndarray(shape=(61, 15), dtype=np.uint8)
    data[1:, 1:] = data[1:, 1:].astype(np.uint8)
    for i in range(15):
        random_vectors[:, i] = [
            random.randint(np.min(data[1:, i + 1].astype(int)), np.max(data[1:, i + 1].astype(int)))
            for x in range(61)]
    return data, random_vectors


# def draw_reg_poly(surface, color, vertex_count, radius, position, width=0):
#     n, r = vertex_count, radius
#     x, y = position
#     pygame.draw.polygon(surface, color, [
#         (x + r * np.cos(2 * np.pi * i / n),
#          y + r * np.sin(2 * np.pi * i / n))
#         for i in range(n)], width)


def plot_data(tsne_results):
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(x[0], y[0], c='red')
    plt.scatter(x[1:], y[1:])
    plt.show()


def algo_approximation(data, rand_vectors):
    for input_data in data:
        # find the approximate random vector to input_vector
        difference = [math.dist(input_data, rand_point) for rand_point in rand_vectors]
        closet_rand_point = rand_vectors[np.argmin(difference)]
        curr_data = np.concatenate((input_data.reshape(1, 2), rand_vectors))
        # plot_data(curr_data)

        # approximate input_vector to the "closest" vector
        pass


def make_as_points(input_data, rand_vectors):
    only_numeric_data = input_data[1:, 1:]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    data_to_tsne = np.concatenate((only_numeric_data, rand_vectors))
    tsne_results = tsne.fit_transform(data_to_tsne)
    input_data_as_points = tsne_results[:196]
    rand_vector_as_points = tsne_results[196:]
    return input_data_as_points, rand_vector_as_points


if __name__ == '__main__':
    data, random_vectors = get_data('Elec_24.csv')
    data_as_points, rand_vec_as_points = make_as_points(data, random_vectors)
    algo_approximation(data_as_points, rand_vec_as_points)
