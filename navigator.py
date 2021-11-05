import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
import random
from navigation_utils import get_obs, random_move, get_obs_panor, read_map_npy, get_pose, change_brightness
from baseline_utils import apply_color_to_map
from map_utils import SemanticMap

dataset_dir = '/Users/yimengli/Work/object_search/Datasets/MP3D'
scene_name = '2t7WUuJeko7_0'
SEED = 5
NUM_STEPS = 2000
THRESH_MATCHES = 10
cell_size = 0.1
flag_vis = False
saved_folder = 'output/explore_PF'
vis_observed_area_from_panorama = False

np.random.seed(SEED)
random.seed(SEED)

# load img list
img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
img_names = list(img_act_dict.keys())

ins2cat_dict = np.load('{}/{}/dict_ins2category.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()


sem_map = SemanticMap()
traverse_lst = []

# randomly pick a start point
cur_img_id = random.choice(img_names)

step = 0
while step < NUM_STEPS:
	print(f'step = {step}, img_id = {cur_img_id}')

	obs_rgb, obs_depth = get_obs(cur_img_id)
	traverse_lst.append(cur_img_id)

	# add the observed area
	sem_map.build_semantic_map(traverse_lst, panorama=vis_observed_area_from_panorama)

	if step % 10 == 0:
		# visualize the path on the map
		map_npy = np.load(f'output/semantic_map/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
		semantic_map, pose_range, coords_range = read_map_npy(map_npy)
		observed_map = sem_map.get_semantic_map()
		cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]
		color_semantic_map = apply_color_to_map(cropped_semantic_map)

		observed_area_flag = (observed_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1] > 0)
		color_semantic_map = change_brightness(color_semantic_map, observed_area_flag, value=100)
		#assert 1==2

		# visualize the agent pose list
		x_coord_lst = []
		z_coord_lst = []
		for img_name in traverse_lst:
			cur_pose = get_pose(img_name)
			tx, tz, theta = cur_pose
			
			x_coord = int(floor((tx - pose_range[0]) / cell_size) - coords_range[0])
			z_coord = int(floor((tz - pose_range[1]) / cell_size) - coords_range[1])

			x_coord_lst.append(x_coord)
			z_coord_lst.append(z_coord)

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
		ax.imshow(color_semantic_map)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.scatter(x_coord_lst, z_coord_lst, s=30, c='red', zorder=2)
		ax.plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=1)
		fig.tight_layout()
		#plt.show()
		plt.savefig('{}/observed_area_{}_steps.jpg'.format(saved_folder, step))
		plt.close()


	step += 1
	cur_img_id = random_move(cur_img_id)