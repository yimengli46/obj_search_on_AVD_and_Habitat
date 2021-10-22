import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertInsSegToSSeg, apply_color_to_map, create_folder


dataset_dir = '/home/yimeng/Datasets/MP3D'
scene_list = ['2t7WUuJeko7_0']
#scene_list = ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1', 'e9zR4mvMWw7_0',]
#scene_list = ['GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1', 'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0',]
#scene_list = ['V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0', 'RPmz2sHmrrY_0',]
#scene_list = ['WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0', ]

cell_size = 0.1
UNIGNORED_CLASS = [1, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 22, 23, 25, 27, 28, 31, 33, 34, 36, 37, 38, 39, 40]
step_size = 500
map_boundary = 10
y_coord_size = 1000

IGNORED_CLASS = []
for i in range(41):
	if i not in UNIGNORED_CLASS:
		IGNORED_CLASS.append(i)

semantic_map_output_folder = f'output/semantic_map'
create_folder(semantic_map_output_folder, clean_up=False)

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'
	create_folder(saved_folder, clean_up=True)

	# load img list
	img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	img_names = list(img_act_dict.keys())

	ins2cat_dict = np.load('{}/{}/dict_ins2category.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()

	min_X = 1000.0
	max_X = -1000.0
	min_Z = 1000.0
	max_Z = -1000.0
	for idx, img_name in enumerate(img_names):
		pose = img_act_dict[img_name]['pose'] # x, z, theta
		x, z, _ = pose
		if x < min_X:
			min_X = x
		if x > max_X:
			max_X = x
		if z < min_Z:
			min_Z = z
		if z > max_Z:
			max_Z = z
	min_X -= 10.0
	max_X += 10.0
	min_Z -= 10.0
	max_Z += 10.0
	x_grid = np.arange(min_X, max_X, cell_size)
	z_grid = np.arange(min_Z, max_Z, cell_size)

	four_dim_grid = np.zeros((len(z_grid), y_coord_size, len(x_grid), 41)) # x, y, z, C
	H, W = len(z_grid), len(x_grid)

	for idx, img_name in enumerate(img_names):
		#if idx == 100:
		#	break

		print('idx = {}'.format(idx))
		# load rgb image, depth and sseg
		rgb_img = cv2.imread('{}/{}/images/{}.jpg'.format(dataset_dir, scene_name, img_name), 1)[:, :, ::-1]
		npy_file = np.load('{}/{}/others/{}.npy'.format(dataset_dir, scene_name, img_name), allow_pickle=True).item()
		InsSeg_img = npy_file['sseg']
		sseg_img = convertInsSegToSSeg(InsSeg_img, ins2cat_dict)
		depth_img = npy_file['depth']
		pose = img_act_dict[img_name]['pose'] # x, z, theta
		print('pose = {}'.format(pose))

		if idx % step_size == 0:
			#'''
			fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
			ax[0].imshow(rgb_img)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb")
			ax[1].imshow(apply_color_to_map(sseg_img))
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg")
			ax[2].imshow(depth_img)
			ax[2].get_xaxis().set_visible(False)
			ax[2].get_yaxis().set_visible(False)
			ax[2].set_title("depth")
			fig.tight_layout()
			#plt.show()
			fig.savefig('{}/step_{}.jpg'.format(saved_folder, idx))
			plt.close()
			#assert 1==2
			#'''

		xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, pose, gap=2, ignored_classes=IGNORED_CLASS)

		mask_X = np.logical_and(xyz_points[0, :] > min_X, xyz_points[0, :] < max_X) 
		mask_Y = np.logical_and(xyz_points[1, :] > 0.0, xyz_points[1, :] < 100.0)
		mask_Z = np.logical_and(xyz_points[2, :] > min_Z, xyz_points[2, :] < max_Z)  
		mask_XYZ = np.logical_and.reduce((mask_X, mask_Y, mask_Z))
		xyz_points = xyz_points[:, mask_XYZ]
		sseg_points = sseg_points[mask_XYZ]

		x_coord = np.floor((xyz_points[0, :] - min_X) / cell_size).astype(int)
		y_coord = np.floor(xyz_points[1, :] / cell_size).astype(int)
		z_coord = np.floor((xyz_points[2, :] - min_Z) / cell_size).astype(int)
		mask_y_coord = y_coord < y_coord_size
		x_coord = x_coord[mask_y_coord]
		y_coord = y_coord[mask_y_coord]
		z_coord = z_coord[mask_y_coord]
		sseg_points = sseg_points[mask_y_coord]
		four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1
		#assert 1==2

		# sum over the height axis
		grid_sum_height = np.sum(four_dim_grid, axis=1)
		# argmax over the category axis
		semantic_map = np.argmax(grid_sum_height, axis=2)

		if x_coord.shape[0] == 0:
			continue

		# get the local map
		if idx == 0:
			min_x_coord = max(np.min(x_coord)-map_boundary, 0)
			max_x_coord = min(np.max(x_coord)+map_boundary, W-1)
			min_z_coord = max(np.min(z_coord)-map_boundary, 0)
			max_z_coord = min(np.max(z_coord)+map_boundary, H-1)
		else:
			min_x_coord = min(max(np.min(x_coord)-map_boundary, 0), min_x_coord)
			max_x_coord = max(min(np.max(x_coord)+map_boundary, W-1), max_x_coord)
			min_z_coord = min(max(np.min(z_coord)-map_boundary, 0), min_z_coord)
			max_z_coord = max(min(np.max(z_coord)+map_boundary, H-1), max_z_coord)
		
		cropped_semantic_map = semantic_map[min_z_coord:max_z_coord+1, min_x_coord:max_x_coord+1]
		color_semantic_map = apply_color_to_map(cropped_semantic_map)

		if idx % step_size == 0:
			# write the map with cv2 so the map image can be load directly
			cv2.imwrite('{}/step_{}_semantic.jpg'.format(saved_folder, idx), color_semantic_map[:, :, ::-1])

	# save the final results
	map_dict = {}
	map_dict['min_x'] = min_x_coord
	map_dict['max_x'] = max_x_coord
	map_dict['min_z'] = min_z_coord
	map_dict['max_z'] = max_z_coord
	map_dict['min_X'] = min_X
	map_dict['max_X'] = max_X
	map_dict['min_Z'] = min_Z
	map_dict['max_Z'] = max_Z
	map_dict['semantic_map'] = semantic_map
	print(f'semantic_map.shape = {semantic_map.shape}')
	np.save(f'{saved_folder}/BEV_semantic_map.npy', map_dict)

	# save the final color image
	cv2.imwrite('{}/final_semantic_map.jpg'.format(saved_folder), color_semantic_map[:, :, ::-1])
			