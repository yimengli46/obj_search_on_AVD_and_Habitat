import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from baseline_utils import apply_color_to_map, get_class_mapper, read_map_npy, create_folder, pxl_coords_to_pose
import skimage.measure
from math import floor, sqrt
import csv
import math
from sklearn.mixture import GaussianMixture

#scene_id = 3
dataset_dir = '/home/yimeng/Datasets/MP3D'

scene_list = ['2t7WUuJeko7_0']
scene_list += ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1', 'e9zR4mvMWw7_0',]
scene_list += ['GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1', 'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0',]
scene_list += ['V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0', 'RPmz2sHmrrY_0',]
scene_list += ['WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0']

IGNORED_CLASS = ['void', 'wall', 'floor', 'railing', 'blinds']
thresh_dist = math.inf

semantic_map_output_folder = f'output/semantic_map'
create_folder(semantic_map_output_folder, clean_up=False)

# create obj-obj dict
cat2idx_dict = get_class_mapper()
idx2cat_dict = {v: k for k, v in cat2idx_dict.items()}
obj_obj_dict = {}
for k1 in list(cat2idx_dict.keys()):
	if k1 in IGNORED_CLASS:
		continue
	else:
		obj_obj_dict[k1] = {}
		for k2 in list(cat2idx_dict.keys()):
			if k2 in IGNORED_CLASS:
				continue
			else:
				obj_obj_dict[k1][k2] = []

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'

	# load semantic map
	map_npy = np.load(f'{saved_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(map_npy)

	# load instance centers
	list_insts = np.load(f'{saved_folder}/instances_centers.npy', allow_pickle=True)

	for i, a_inst in enumerate(list_insts[:-1]):
		a_center_coords = a_inst['center']
		a_cat = idx2cat_dict[a_inst['cat']]
		a_center_pose = pxl_coords_to_pose(a_center_coords, pose_range, coords_range)
		if a_cat in IGNORED_CLASS:
			continue
		else:
			for j, b_inst in enumerate(list_insts[i+1:]):
				b_center_coords = b_inst['center']
				b_cat = idx2cat_dict[b_inst['cat']]
				b_center_pose = pxl_coords_to_pose(b_center_coords, pose_range, coords_range)
				
				x_diff = b_center_pose[0] - a_center_pose[0]
				z_diff = b_center_pose[1] - a_center_pose[1]
				obj_obj_dict[a_cat][b_cat].append((x_diff, z_diff))
				obj_obj_dict[b_cat][a_cat].append((-x_diff, -z_diff))

for i, k1 in enumerate(list(obj_obj_dict.keys())):
	for j, k2 in enumerate(list(obj_obj_dict[k1].keys())):
		arr_dist = np.array(obj_obj_dict[k1][k2])
		gm = GaussianMixture(n_components=3).fit(arr_dist)
		params = gm.get_params()
		'''
		logprob = gm.score_samples(np.array([[-14., 1.]]))
		pdf = np.exp(logprob)
		gm.set_params(**params)
		'''
		assert 1==2




