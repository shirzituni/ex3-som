from hexalattice.hexalattice import *
import numpy as np

grid_rows = 8
grid_cols = 8
grid_size = grid_rows * grid_cols

colors = np.zeros([grid_size, 3])
hex_centers, grid = create_hex_grid(nx=grid_rows, ny=grid_cols, do_plot=False)
x_hex_coords = hex_centers[:, 0]
y_hex_coords = hex_centers[:, 1]

# keys = list(range(0, grid_size))  # Replace by Municipality column from CSV
# grid_dict = dict(zip(keys), colors))

for i in range(0, grid_size):
    colors[i] = np.array([val / 255 for val in [255 - i, 255 - 2 * i, 0]])  # RGB 0-1

plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=0.9,
                                      plotting_gap=0.05,
                                      rotate_deg=0)
plt.show()


