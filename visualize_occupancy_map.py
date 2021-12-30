import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import read_map_npy, pose_to_coords, apply_color_to_map, save_fig_through_plt, create_folder

dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_list = ['Allensville_0']
scene_list = ['Beechwood_0']
scene_list = ['Hanson_0', 'Stockman_0', 'Pinesdale_0', 'Collierville_1', 'Shelbyville_2', 'Coffeen_0', 'Corozal_1', 'Stockman_2']
scene_list = ['Woodbine_0', 'Ranchester_0', 'Mifflinburg_1', 'Lakeville_1', 'Hanson_2', 'Pomaria_2', 'Wainscott_1', 'Hiteman_2', 'Coffeen_2', 'Onaga_0', 'Pomaria_0', 'Newfields_1', 'Shelbyville_0', 'Klickitat_0']
scene_list = ['Darden_1', 'Merom_1', 'Lindenwood_0', 'Coffeen_3', 'Klickitat_2', 'Hiteman_1', 'Forkland_2', 'Newfields_0', 'Mifflinburg_2', 'Marstons_1', 'Shelbyville_1', 'Tolstoy_1', 'Darden_0', 'Tolstoy_0']
scene_list = ['Marstons_3', 'Forkland_1', 'Hanson_1', 'Klickitat_1', 'Markleeville_1', 'Merom_0', 'Leonardo_2', 'Benevolence_2', 'Hiteman_0', 'Pinesdale_1', 'Collierville_0', 'Cosmos_0', 'Newfields_2']
scene_list = ['Forkland_0', 'Collierville_2', 'Woodbine_1', 'Wainscott_0', 'Coffeen_1', 'Markleeville_0', 'Wiconisco_0', 'Mifflinburg_0', 'Lindenwood_1', 'Stockman_1']
scene_list = ['Corozal_0', 'Pomaria_1', 'Onaga_1', 'Wiconisco_2', 'Darden_2', 'Ranchester_1', 'Cosmos_1', 'Benevolence_1', 'Leonardo_0', 'Beechwood_1', 'Lakeville_0', 'Marstons_0', 'Wiconisco_1', 'Benevolence_0', 'Leonardo_1', 'Marstons_2']
sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'

cell_size = 0.1
UNIGNORED_CLASS = []

semantic_map_output_folder = f'output/semantic_map'
#create_folder(semantic_map_output_folder, clean_up=False)

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'
	#create_folder(saved_folder, clean_up=False)

	# load img list
	img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	img_names = list(img_act_dict.keys())

	#================================== load scene npz and semantic map ======================================
	scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name[:-2]}.npz', allow_pickle=True)['output'].item()
	sem_map_npy = np.load(f'{saved_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)
	cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

	occ_map = np.zeros(cropped_semantic_map.shape, dtype=int)

	#===================================== traverse the observations ===============================
	for idx, img_name in enumerate(img_names):

		print('idx = {}'.format(idx))
		#====================================== load pose ==================================
		pose = img_act_dict[img_name]['pose'] # x, z, theta
		print('pose = {}'.format(pose))

		x_coord, z_coord = pose_to_coords((pose[0], pose[1]), pose_range, coords_range)
		occ_map[z_coord-2:z_coord+3, x_coord-2:x_coord+3] = 1

	# save the final results
	map_dict = {}
	map_dict['occupancy'] = occ_map
	np.save(f'{saved_folder}/BEV_occupancy_map.npy', occ_map)

	# save the final color image
	save_fig_through_plt(occ_map, f'{saved_folder}/occ_map.jpg')
