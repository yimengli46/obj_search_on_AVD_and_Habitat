import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertPanopSegToSSeg, apply_color_to_map, create_folder
import habitat
import habitat_sim
import random
from baseline_utils import read_map_npy, pose_to_coords, apply_color_to_map, save_fig_through_plt, create_folder

#=========================================== fix the habitat scene shuffle ===============================
SEED = 5
random.seed(SEED)
np.random.seed(SEED)

output_folder = 'output/semantic_map'
scene_list = ['Allensville_0']
# after testing, using 8 angles is most efficient
theta_lst = [0]
#theta_lst = [0]
built_scenes = [] 
cell_size = 0.1

#============================= build a grid =========================================
x = np.arange(-30, 30, cell_size)
z = np.arange(-30, 30, cell_size)
xv, zv = np.meshgrid(x, z)
#xv = xv.flatten()
#zv = zv.flatten()
grid_H, grid_W = zv.shape

#============================ traverse each scene =============================
class SimpleRLEnv(habitat.RLEnv):
	def get_reward_range(self):
		return [-1, 1]

	def get_reward(self, observations):
		return 0

	def get_done(self, observations):
		return self.habitat_env.episode_over

	def get_info(self, observations):
		return self.habitat_env.get_metrics()

def get_scene_name(episode):
	idx_right_most_slash = episode.scene_id.rfind('/')
	return episode.scene_id[idx_right_most_slash+1:-4]

def verify_img(img):
    sum_img = np.sum((img[:,:,0] > 0))
    h, w = img.shape[:2]
    return sum_img > h*w*0.75


config = habitat.get_config(config_paths="/home/yimeng/Datasets/habitat-lab/configs/tasks/devendra_objectnav_gibson.yaml")
config.defrost()
#assert 1==2
config.DATASET.DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/objectnav/gibson/all.json.gz'
config.DATASET.SCENES_DIR = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
config.freeze()
env = SimpleRLEnv(config=config)

for episode_id in range(1):
	env.reset()
	print('episode_id = {}'.format(episode_id))
	print('env.current_episode = {}'.format(env.current_episode))

	scene_name = scene_list[episode_id] #get_scene_name(env.current_episode)
	
	height = 0.17

	#=============================== traverse each floor ===========================
	print(f'*****scene_name = {scene_name}***********')

	saved_folder = f'{output_folder}/{scene_name}'
	#create_folder(saved_folder, clean_up=False)

	#'''
	sem_map_npy = np.load(f'{saved_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	_, pose_range, coords_range = read_map_npy(sem_map_npy)
	#cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]
	#'''

	occ_map = np.zeros((grid_H, grid_W), dtype=int)

	count_ = 0
	#========================= generate observations ===========================
	for grid_z in range(grid_H):
		for grid_x in range(grid_W):
			x = xv[grid_z, grid_x] + cell_size/2.
			z = zv[grid_z, grid_x] + cell_size/2.
			y = height

			agent_pos = np.array([x, y, z])

			flag_nav = env.habitat_env.sim.is_navigable(agent_pos)
			#print(f'after teleportation, flag_nav = {flag_nav}')

			x_coord, z_coord = pose_to_coords((x, -z), pose_range, coords_range, flag_cropped=False)

			if flag_nav:
				occ_map[z_coord, x_coord] = 1

	#assert 1==2
	occ_map = occ_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

	# save the final results
	map_dict = {}
	map_dict['occupancy'] = occ_map
	np.save(f'{saved_folder}/BEV_occupancy_map.npy', occ_map)

	# save the final color image
	save_fig_through_plt(occ_map, f'{saved_folder}/occ_map.jpg')
