'''
	generate topological map vertices and edges given an occupancy map
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from topo_map.img_to_skeleton import img2skeleton
from topo_map.skeleton_to_topoMap import skeleton2topoMap
from topo_map.utils import drawtoposkele_with_VE,  build_VE_from_graph
from baseline_utils import read_map_npy, create_folder, semanticMap_to_binary, apply_color_to_map

#scene_id = 3
dataset_dir = '/home/yimeng/Datasets/MP3D'

scene_list = ['2t7WUuJeko7_0']
scene_list = ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1', 'e9zR4mvMWw7_0',]
#scene_list = ['GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1', 'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0',]
#scene_list = ['V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0', 'RPmz2sHmrrY_0',]
#scene_list = ['WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0',]

semantic_map_output_folder = f'output/semantic_map'
create_folder(semantic_map_output_folder, clean_up=False)

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'

	map_npy = np.load(f'{saved_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(map_npy)

	# reduce the size of occupancy map
	cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]
	color_semantic_map = apply_color_to_map(cropped_semantic_map)
	#cv2.imwrite(f'{saved_folder}/fixed_final_semantic_map.jpg', color_semantic_map[:, :, ::-1])

	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())
	ax.imshow(color_semantic_map)
	fig.savefig(f'{saved_folder}/fixed_final_semantic_map.jpg', format='jpg', dpi=500, bbox_inches='tight', pad_inches=0)

	occupancy_map = semanticMap_to_binary(cropped_semantic_map)
	gray1, gau1, skeleton = img2skeleton(occupancy_map)
	graph = skeleton2topoMap(skeleton)

	v_lst, e_lst = build_VE_from_graph(graph, skeleton, vertex_dist=10)

	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())
	drawtoposkele_with_VE(graph, skeleton + (1 - gray1[:, :] / 255) * 2, v_lst, e_lst, ax=ax)
	fig.savefig(f'{saved_folder}/topo_map.png', format='png', dpi=500, bbox_inches='tight', pad_inches=0)

	result = {}
	result['vertices'] = v_lst
	result['edges'] = e_lst

	np.save(f'{saved_folder}/v_and_e.npy', result)
