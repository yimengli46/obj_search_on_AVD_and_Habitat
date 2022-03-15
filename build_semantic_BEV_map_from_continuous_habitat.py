import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertPanopSegToSSeg, apply_color_to_map, create_folder
import habitat
import habitat_sim
from build_map_utils import SemanticMap
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import random
from panoptic_prediction import PanopPred
from navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg

#=========================================== fix the habitat scene shuffle ===============================
random.seed(cfg.GENERAL.RANDOM_SEED)
np.random.seed(cfg.GENERAL.RANDOM_SEED)

scene_list = ['Allensville_0']
#scene_list = ['Collierville_1']
#scene_list = ['Darden_0', 'Markleeville_0', 'Wiconisco_0']

scene_dict = {}
for scene in scene_list:
	scene_name = scene[:-2]
	floor = int(scene[-1])
	temp = {}
	temp['name'] = scene
	temp['floor'] = floor 
	scene_dict[scene_name] = temp

output_folder = cfg.SAVE.SEM_MAP_PATH
# after testing, using 8 angles is most efficient
theta_lst = [0, pi/4, pi/2, pi*3./4, pi, pi*5./4, pi*3./2, pi*7./4]
#theta_lst = [0]
str_theta_lst = ['000', '090', '180', '270']

scene_heights_dict = np.load(cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH, allow_pickle=True).item()

#============================= build a grid =========================================
x = np.arange(-30, 30, 0.3)
z = np.arange(-30, 30, 0.3)
xv, zv = np.meshgrid(x, z)
#xv = xv.flatten()
#zv = zv.flatten()
grid_H, grid_W = zv.shape

#============================ traverse each scene =============================

panop_pred = PanopPred()

config = habitat.get_config(config_paths=cfg.GENERAL.HABITAT_CONFIG_PATH)
config.defrost()
config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_EPISODE_DATA_PATH
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()
env = SimpleRLEnv(config=config)

for episode_id in range(5):
	env.reset()
	print('episode_id = {}'.format(episode_id))
	print('env.current_episode = {}'.format(env.current_episode))

	scene_name_no_floor = get_scene_name(env.current_episode)

	if scene_name_no_floor in scene_dict:
		scene_name = scene_dict[scene_name_no_floor]['name']
		floor_id   = scene_dict[scene_name_no_floor]['floor']
	
		height = scene_heights_dict[scene_name_no_floor][floor_id]
		#=============================== traverse each floor ===========================
		
		print(f'*****scene_name = {scene_name}***********')

		saved_folder = f'{output_folder}/{scene_name}'
		create_folder(saved_folder, clean_up=False)

		id2class_mapper = np.load('configs/COCO_PanopticSeg_labels_dict.npy', allow_pickle=True).item()

		#================================ Building a map ===============================
		SemMap = SemanticMap(saved_folder)

		count_ = 0
		#========================= generate observations ===========================
		for grid_z in range(grid_H):
			for grid_x in range(grid_W):
				x = xv[grid_z, grid_x]
				z = zv[grid_z, grid_x]
				y = height
				agent_pos = np.array([x, y, z])

				flag_nav = env.habitat_env.sim.is_navigable(agent_pos)
				#print(f'after teleportation, flag_nav = {flag_nav}')

				if flag_nav:
					#==================== traverse theta ======================
					for idx_theta, theta in enumerate(theta_lst):
						agent_rot = habitat_sim.utils.common.quat_from_angle_axis(theta, habitat_sim.geo.GRAVITY)
						observations = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)
						rgb_img = observations["rgb"]
						depth_img = observations["depth"][:,:,0]
						depth_img = 5. * depth_img
						depth_img = cv2.blur(depth_img, (3,3))
						panopSeg_img, _ = panop_pred.get_prediction(rgb_img, flag_vis=False)
						sseg_img = convertPanopSegToSSeg(panopSeg_img, id2class_mapper)
						#print(f'rgb_img.shape = {rgb_img.shape}')

						'''
						fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(100, 300))

						ax[0].imshow(rgb_img)
						ax[1].imshow(depth_img)
						ax[2].imshow(sseg_img)

						fig.tight_layout()
						plt.show()
						'''

						#=============================== get agent global pose on habitat env ========================#
						agent_pos = env.habitat_env.sim.get_agent_state().position
						agent_rot = env.habitat_env.sim.get_agent_state().rotation
						heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
						phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
						angle = phi
						print(f'agent position = {agent_pos}, angle = {angle}')
						pose = (agent_pos[0], agent_pos[2], angle)

						SemMap.build_semantic_map(rgb_img, depth_img, sseg_img, pose, count_)
						count_ += 1


		SemMap.save_final_sem_map()

env.close()