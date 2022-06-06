import random

import numpy as np
import pandas as pd
import pygame as pygame


def get_data(path):
    data = pd.read_csv(filepath_or_buffer=path, header=None).values
    random_vectors = np.ndarray(shape=(61, 15))
    for i in range(15):
        random_vectors[:, i] = [
            random.randint(int(np.min(data[1:, i + 1].astype(int))), int(np.max(data[1:, i + 1].astype(int))))
            for x in range(61)]
    return data, random_vectors


def draw_reg_poly(surface, color, vertex_count, radius, position, width=0):
    n, r = vertex_count, radius
    x, y = position
    pygame.draw.polygon(surface, color, [
        (x + r * np.cos(2 * np.pi * i / n),
         y + r * np.sin(2 * np.pi * i / n))
        for i in range(n)], width)


def algo_approximation(data, random_vectors):
    for line_number in range(1, data.shape[0]):
        input_vector = data[line_number, 1:]
        # find the approximate random vector to input_vector

        # approximate input_vector to the "closest" vector


if __name__ == '__main__':
    data, random_vectors = get_data('Elec_24.csv')
    algo_approximation(data, random_vectors)
