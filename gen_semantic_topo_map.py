'''
combine topological map with semantic map.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from baseline_utils import apply_color_to_map, get_class_mapper, read_map_npy, create_folder
import skimage.measure
from math import floor

#scene_id = 3
dataset_dir = '/home/yimeng/Datasets/MP3D'

scene_list = ['2t7WUuJeko7_0']
scene_list = ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1', 'e9zR4mvMWw7_0',]
#scene_list = ['GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1', 'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0',]
#scene_list = ['V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0', 'RPmz2sHmrrY_0',]
#scene_list = ['WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0']

semantic_map_output_folder = f'output/semantic_map'
create_folder(semantic_map_output_folder, clean_up=False)

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'

	cat_dict = get_class_mapper()
	num_classes = cat_dict[list(cat_dict.keys())[-1]] + 1

	#======================================== load scene semantic occupancy map ====================================
	map_npy = np.load(f'{saved_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(map_npy)
	cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

	semantic_occupancy_map = cropped_semantic_map.copy()
	H, W = semantic_occupancy_map.shape
	semantic_occupancy_map = cv2.resize(semantic_occupancy_map, (int(W*10), int(H*10)), interpolation=cv2.INTER_NEAREST)
	H, W = semantic_occupancy_map.shape
	x = np.linspace(0, W-1, W)
	y = np.linspace(0, H-1, H)
	xv, yv = np.meshgrid(x, y)

	#======================================== load the topo map ==========================================
	topo_V_E = np.load(f'{saved_folder}/v_and_e.npy', allow_pickle=True).item()
	v_lst, e_lst = topo_V_E['vertices'], topo_V_E['edges']

	color_semantic_map = apply_color_to_map(semantic_occupancy_map, num_classes=num_classes)

	#'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
	ax.imshow(color_semantic_map)
	x, y = [], []
	for ed in e_lst:
		v1 = v_lst[ed[0]]
		v2 = v_lst[ed[1]]
		y.append(v1[1])
		x.append(v1[0])
		y.append(v2[1])
		x.append(v2[0])
		ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 
	            'k-', lw=1)
	ax.scatter(x=x, y=y, c='r', s=2)
	fig.tight_layout()
	#plt.show()
	#'''
	#assert 1==2
	#====================================== compute centers of semantic classes =====================================
	idx2cat_dict = {v: k for k, v in cat_dict.items()}
	IGNORED_CLASS = [0, 1, 2, 17]
	cat_binary_map = semantic_occupancy_map.copy()
	for cat in IGNORED_CLASS:
		cat_binary_map = np.where(cat_binary_map==cat, -1, cat_binary_map)
	# run skimage to find the number of objects belong to this class
	instance_label, num_ins = skimage.measure.label(cat_binary_map, background=-1, connectivity=1, return_num=True)

	list_instances = []
	for idx_ins in range(1, num_ins+1):
		mask_ins = (instance_label==idx_ins)
		if np.sum(mask_ins) > 50: # should have at least 50 pixels
			print(f'idx_ins = {idx_ins}')
			x_coords = xv[mask_ins]
			y_coords = yv[mask_ins]
			ins_center = (floor(np.median(x_coords)), floor(np.median(y_coords)))
			ins_cat = semantic_occupancy_map[int(y_coords[0]), int(x_coords[0])]
			ins = {}
			ins['center'] = ins_center
			ins['cat'] = ins_cat
			list_instances.append(ins)

	np.save(f'{saved_folder}/instances_centers.npy', list_instances)
	#assert 1==2

	#================================== link the instances to the topo nodes ======================================
	v_arr = np.array(v_lst)
	x, y = [], []
	for ins in list_instances:
		center = ins['center']
		cat = ins['cat']

		x.append(center[0])
		y.append(center[1])

		dist = np.sqrt((center[0] - v_arr[:, 0])**2 + (center[1] - v_arr[:, 1])**2)
		closest_v_idx = np.argmin(dist)
		vertex = v_lst[closest_v_idx]

		ax.plot([center[0], vertex[0]], [center[1], vertex[1]], 
	            'k-', lw=1)

		try:
			cat_name = idx2cat_dict[cat]
		except:
			cat_name = 'unknown'
		ax.text(center[0], center[1], cat_name)

	ax.scatter(x=x, y=y, c='b', s=5)

	#assert 1==2
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	fig.savefig(f'{saved_folder}/topo_semantic_map.png')
	plt.close()
	#plt.show() 
