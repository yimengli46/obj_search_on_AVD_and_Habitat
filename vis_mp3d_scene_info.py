import numpy as np 
import matplotlib.pyplot as plt 
import math
from baseline_utils import get_class_mapper, apply_color_to_map

mp3d_info_folder = '/home/yimeng/Datasets/habitat-lab/habitat_nav/code_for_habitat/mp3d_scenes_info'
scene_name = '2t7WUuJeko7_0'
IGNORED_CLASS = ['void', 'wall', 'floor', 'railing', 'blinds']

cell_per_meter = 10
cat2idx_dict = get_class_mapper()
idx2cat_dict = {v: k for k, v in cat2idx_dict.items()}

npy_file = np.load(f'{mp3d_info_folder}/{scene_name}.npy', allow_pickle=True)

min_X = math.inf
min_Y = math.inf 
min_Z = math.inf
max_X = -math.inf
max_Y = -math.inf 
max_Z = -math.inf 


for i, a_inst in enumerate(npy_file):
	a_center_coords = a_inst['center']
	a_cat = a_inst['category']
	a_dims = a_inst['dims']
	if a_inst['objectness']:
		x, y, z = a_center_coords
		x_dims, y_dims, z_dims = a_dims
		x_min = x - x_dims/2
		x_max = x + x_dims/2
		y_min = y - y_dims/2
		y_max = y + y_dims/2
		z_min = z - z_dims/2
		z_max = z + z_dims/2
		if min_X > x_min:
			min_X = x_min
		if min_Y > y_min:
			min_Y = y_min
		if min_Z > z_min:
			min_Z = z_min
		if max_X < x_max:
			max_X = x_max
		if max_Y < y_max:
			max_Y = y_max
		if max_Z < z_max:
			max_Z = z_max

min_X -= 1.
min_Y -= 1.
min_Z -= 1.
max_X += 1.
max_Y += 1.
max_Z += 1.

H = math.ceil((max_Z - min_Z) * cell_per_meter)
T = math.ceil((max_Y - min_Y) * cell_per_meter)
W = math.ceil((max_X - min_X) * cell_per_meter)

grid = np.zeros((H, W), dtype=int)


for i, a_inst in enumerate(npy_file):
	a_center_coords = a_inst['center']
	a_cat = a_inst['category']
	a_dims = a_inst['dims']
	if a_inst['objectness'] and a_cat in list(cat2idx_dict.keys()):
		if a_cat not in IGNORED_CLASS:
			x, y, z = a_center_coords
			x_dims, y_dims, z_dims = a_dims
			x_min = x - x_dims/2
			x_max = x + x_dims/2
			y_min = y - y_dims/2
			y_max = y + y_dims/2
			z_min = z - z_dims/2
			z_max = z + z_dims/2
			
			x1 = math.floor((x_min - min_X) * cell_per_meter)
			x2 = math.floor((x_max - min_X) * cell_per_meter)
			y1 = math.floor((y_min - min_Y) * cell_per_meter)
			y2 = math.floor((y_max - min_Y) * cell_per_meter)
			z1 = math.floor((z_min - min_Z) * cell_per_meter)
			z2 = math.floor((z_max - min_Z) * cell_per_meter)

			cat_idx = cat2idx_dict[a_cat]
			grid[z1:z2, x1:x2] = cat_idx

'''
top_down_grid = np.zeros((H, W), dtype=int)
for y in range(grid.shape[1]):
	current_level_grid = grid[:, y, :]
	top_down_grid = np.where(current_level_grid > 0, current_level_grid, top_down_grid)
'''

color_semantic_map = apply_color_to_map(grid)
plt.imshow(color_semantic_map)
plt.show()