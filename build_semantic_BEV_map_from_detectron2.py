import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertDetectron2ToSSeg, apply_color_to_map, create_folder, save_fig_through_plt

#from semantic_prediction import SemanticPredMaskRCNN

dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_list = ['Allensville_0']
#scene_list = ['Beechwood_0']
#scene_list = ['Hanson_0', 'Stockman_0', 'Pinesdale_0', 'Collierville_1', 'Shelbyville_2', 'Coffeen_0', 'Corozal_1', 'Stockman_2']
#scene_list = ['Woodbine_0', 'Ranchester_0', 'Mifflinburg_1', 'Lakeville_1', 'Hanson_2', 'Pomaria_2', 'Wainscott_1', 'Hiteman_2', 'Coffeen_2', 'Onaga_0', 'Pomaria_0', 'Newfields_1', 'Shelbyville_0', 'Klickitat_0']
#scene_list = ['Darden_1', 'Merom_1', 'Lindenwood_0', 'Coffeen_3', 'Klickitat_2', 'Hiteman_1', 'Forkland_2', 'Newfields_0', 'Mifflinburg_2', 'Marstons_1', 'Shelbyville_1', 'Tolstoy_1', 'Darden_0', 'Tolstoy_0']
#scene_list = ['Marstons_3', 'Forkland_1', 'Hanson_1', 'Klickitat_1', 'Markleeville_1', 'Merom_0', 'Leonardo_2', 'Benevolence_2', 'Hiteman_0', 'Pinesdale_1', 'Collierville_0', 'Cosmos_0', 'Newfields_2']
#scene_list = ['Forkland_0', 'Collierville_2', 'Woodbine_1', 'Wainscott_0', 'Coffeen_1', 'Markleeville_0', 'Wiconisco_0', 'Mifflinburg_0', 'Lindenwood_1', 'Stockman_1']
#scene_list = ['Corozal_0', 'Pomaria_1', 'Onaga_1', 'Wiconisco_2', 'Darden_2', 'Ranchester_1', 'Cosmos_1', 'Benevolence_1', 'Leonardo_0', 'Beechwood_1', 'Lakeville_0', 'Marstons_0', 'Wiconisco_1', 'Benevolence_0', 'Leonardo_1', 'Marstons_2']
sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'

cell_size = 0.1
UNIGNORED_CLASS = []
step_size = 50
map_boundary = 5
y_coord_size = 1000
flag_first_time_having_pixels = True
IGNORED_CLASS = [0]
'''
for i in range(41):
	if i not in UNIGNORED_CLASS:
		IGNORED_CLASS.append(i)
'''

# initialize object detector
#sem_pred = SemanticPredMaskRCNN()

semantic_map_output_folder = f'output/semantic_map_detectron2'
create_folder(semantic_map_output_folder, clean_up=False)

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'
	create_folder(saved_folder, clean_up=False)

	# load img list
	img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	img_names = list(img_act_dict.keys())

	#================================== load scene npz and category dict ======================================
	scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name[:-2]}.npz', allow_pickle=True)['output'].item()
	cat2id_dict = np.load('{}/{}/category_id_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	detectron2_folder = f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset/{scene_name}/detectron2_pred'

	#======================================= initialize the grid ===========================================
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

	#===================================== traverse the observations ===============================
	for idx, img_name in enumerate(img_names):
	#for idx, img_name in enumerate(['077122135', '077122180', '079120135', '089126315']):
		#if idx == 100:
		#	break

		print('idx = {}'.format(idx))
		#====================================== load rgb image, depth and sseg ==================================
		rgb_img = cv2.imread(f'{dataset_dir}/{scene_name}/rgb/{img_name}.jpg', 1)[:, :, ::-1]
		depth_img = cv2.imread(f'{dataset_dir}/{scene_name}/depth/{img_name}.png', cv2.IMREAD_UNCHANGED)
		depth_img = depth_img/256.
		depth_img = cv2.blur(depth_img, (3,3))
		detectron2_npy = np.load(f'{detectron2_folder}/{img_name}.npy', allow_pickle=True).item()
		sseg_img = convertDetectron2ToSSeg(detectron2_npy, det_thresh=0.9)
		pose = img_act_dict[img_name]['pose'] # x, z, theta
		print('pose = {}'.format(pose))

		#assert 1==2

		if idx % step_size == 0:
			'''
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
			plt.show()
			#fig.savefig('{}/step_{}.jpg'.format(saved_folder, idx))
			#plt.close()
			#assert 1==2
			'''

		#========================================= start the projection ================================
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
			print('hhhhhh it happens as most pixels do not have labels')
			continue

		# ==================================== get the local map ===================================
		if flag_first_time_having_pixels:
			min_x_coord = max(np.min(x_coord)-map_boundary, 0)
			max_x_coord = min(np.max(x_coord)+map_boundary, W-1)
			min_z_coord = max(np.min(z_coord)-map_boundary, 0)
			max_z_coord = min(np.max(z_coord)+map_boundary, H-1)
			flag_first_time_having_pixels = False
		else:
			min_x_coord = min(max(np.min(x_coord)-map_boundary, 0), min_x_coord)
			max_x_coord = max(min(np.max(x_coord)+map_boundary, W-1), max_x_coord)
			min_z_coord = min(max(np.min(z_coord)-map_boundary, 0), min_z_coord)
			max_z_coord = max(min(np.max(z_coord)+map_boundary, H-1), max_z_coord)
		
		cropped_semantic_map = semantic_map[min_z_coord:max_z_coord+1, min_x_coord:max_x_coord+1]
		color_semantic_map = apply_color_to_map(cropped_semantic_map)

		if idx % step_size == 0:
			# write the map with cv2 so the map image can be load directly
			#cv2.imwrite('{}/step_{}_semantic.jpg'.format(saved_folder, idx), color_semantic_map[:, :, ::-1])
			save_fig_through_plt(color_semantic_map, f'{saved_folder}/step_{idx}_semantic.jpg')

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
	save_fig_through_plt(color_semantic_map, f'{saved_folder}/final_semantic_map.jpg')

			