import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertInsSegToSSeg, convertMaskRCNNToSSeg, convertPanopSegToSSeg, apply_color_to_map, create_folder, save_fig_through_plt
import habitat
import habitat_sim
#from semantic_prediction import SemanticPredMaskRCNN
from panoptic_prediction import PanopPred
from navigation_utils import SimpleRLEnv
import random

from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector


dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_list = ['Allensville_0']

cell_size = 0.1
UNIGNORED_CLASS = []
step_size = 20
map_boundary = 5
y_coord_size = 1000
flag_first_time_having_pixels = True
IGNORED_CLASS = [0, 23] # 'unlabeld', 'ceiling'
detector = 'PanopticSeg' #'InstanceSeg'
max_STEPS = 2000
random.seed(10)

panop_pred = PanopPred()

semantic_map_output_folder = f'output/semantic_map_continuous'
create_folder(semantic_map_output_folder, clean_up=False)

#================================ load habitat env============================================
config = habitat.get_config(config_paths="/home/yimeng/Datasets/habitat-lab/configs/tasks/devendra_objectnav_gibson.yaml")
config.defrost()
config.DATASET.DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/objectnav/gibson/all.json.gz'
config.DATASET.SCENES_DIR = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
#config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#config.TASK.SENSORS.append("HEADING_SENSOR")
config.freeze()
env = SimpleRLEnv(config=config)

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'
	create_folder(saved_folder, clean_up=False)

	# load img list
	img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	img_names = list(img_act_dict.keys())
	id2class_mapper = np.load('configs/COCO_PanopticSeg_labels_dict.npy', allow_pickle=True).item()

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

	obs = env.reset()
	agent_pos = np.array([6.6, 0.17, -6.9])
	agent_rot = habitat_sim.utils.common.quat_from_angle_axis(2.36, habitat_sim.geo.GRAVITY)
	obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)
	#===================================== traverse the observations ===============================
	for idx in range(max_STEPS):
		print('idx = {}'.format(idx))

		agent_pos = env.habitat_env.sim.get_agent_state().position
		agent_rot = env.habitat_env.sim.get_agent_state().rotation
		#angle = habitat_sim.utils.common.quat_to_angle_axis(agent_rot)
		heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))

		phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
		
		angle = phi
		print(f'angle = {angle}')
		#====================================== load rgb image, depth and sseg ==================================
		rgb_img = obs['rgb']
		depth_img = 5. * obs['depth']
		depth_img = cv2.blur(depth_img, (3,3))
		if detector == 'PanopticSeg':
			panopSeg_img, _ = panop_pred.get_prediction(rgb_img, flag_vis=False)
			sseg_img = convertPanopSegToSSeg(panopSeg_img, id2class_mapper)
		pose = (agent_pos[0], agent_pos[2], angle)
		print('pose = {}'.format(pose))
		sem_map_pose = (pose[0], -pose[1], -pose[2])
		#print(f'obs = {obs}')

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
			plt.show()
			#fig.savefig('{}/step_{}.jpg'.format(saved_folder, idx))
			#plt.close()
			#assert 1==2
			#'''

		#========================================= start the projection ================================
		xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, sem_map_pose, gap=2, ignored_classes=IGNORED_CLASS)

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

		#================================= take actions =======================================
		action = random.choice(["TURN_LEFT", "MOVE_FORWARD"])
		print(f'action = {action}')
		obs = env.step(action)[0]
		#print(f'obs = {obs}')

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
			'''
			semantic_pred, rgb_vis = sem_pred.get_prediction(rgb_img, flag_vis=True)

			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
			ax[0][0].imshow(rgb_img)
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title("rgb")
			ax[0][1].imshow(apply_color_to_map(sseg_img))
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title("sseg")
			ax[1][0].imshow(depth_img)
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title("depth")
			ax[1][1].imshow(rgb_vis)
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title("sem_pred")
			fig.tight_layout()
			plt.show()
			'''
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

			