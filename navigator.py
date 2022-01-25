import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
import random
from navigation_utils import get_obs, random_move, get_obs_panor, read_map_npy, get_pose, change_brightness, SimpleRLEnv
from baseline_utils import apply_color_to_map, pose_to_coords
from map_utils import SemanticMap
from PF_continuous_utils import ParticleFilter
import habitat
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import random

dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_name = 'Allensville_0'
SEED = 5
NUM_STEPS = 2000
cell_size = 0.1
flag_vis = False
saved_folder = 'output/explore_PF_continuous'
vis_observed_area_from_panorama = False
flag_gt_semantic_map = True
NUM_STEPS_EXPLORE = 10
NUM_STEPS_vis = 10
random.seed(10)
detector = 'PanopticSeg'

np.random.seed(SEED)
random.seed(SEED)

if flag_gt_semantic_map:
	sem_map_npy = np.load(f'output/gt_semantic_map_from_SceneGraph/{scene_name}/gt_semantic_map.npy', allow_pickle=True).item()
gt_semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)
H, W = gt_semantic_map.shape[:2]
#occ_map = np.load(f'output/semantic_map/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True)

'''
PF = ParticleFilter(10000, gt_semantic_map.copy(), pose_range, coords_range)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
PF.visualizeBelief(ax)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title('initial particle distribution')
plt.show()
'''

semMap_module = SemanticMap(coords_range) # build the observed sem map
traverse_lst = []

#================================ load habitat env============================================
config = habitat.get_config(config_paths="/home/yimeng/Datasets/habitat-lab/configs/tasks/devendra_objectnav_gibson.yaml")
config.defrost()
config.DATASET.DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/objectnav/gibson/all.json.gz'
config.DATASET.SCENES_DIR = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
#config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#config.TASK.SENSORS.append("HEADING_SENSOR")
config.freeze()
env = SimpleRLEnv(config=config)

#===================================== setup the start location ===============================#
obs = env.reset()
agent_pos = np.array([6.6, 0.17, -6.9])
agent_rot = habitat_sim.utils.common.quat_from_angle_axis(2.36, habitat_sim.geo.GRAVITY)
obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

step = 0
while step < NUM_STEPS:
	print(f'step = {step}')

	#=============================== get agent global pose on habitat env ========================#
	agent_pos = env.habitat_env.sim.get_agent_state().position
	agent_rot = env.habitat_env.sim.get_agent_state().rotation
	heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
	phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
	angle = phi
	print(f'agent position = {agent_pos}, rot = {agent_rot}, angle = {angle}')
	pose = (agent_pos[0], agent_pos[2], angle)
	traverse_lst.append(pose)

	# add the observed area
	semMap_module.build_semantic_map(obs, pose)
	#assert 1==2

	#============================================= visualize semantic map ===========================================#
	if step % NUM_STEPS_vis == 0:
		#==================================== visualize the path on the map ==============================
		built_semantic_map, observed_area_flag, occupancy_map = semMap_module.get_semantic_map()

		observed_area_flag = (observed_area_flag[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1])
		## for the explored free space visualization
		mask_observed_and_non_obj = np.logical_and(observed_area_flag, gt_semantic_map == 0)
		gt_semantic_map[mask_observed_and_non_obj] = 59 # class index for explored non-object area

		color_gt_semantic_map = apply_color_to_map(gt_semantic_map)
		color_gt_semantic_map = change_brightness(color_gt_semantic_map, observed_area_flag, value=60)

		built_semantic_map = built_semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]
		color_built_semantic_map = apply_color_to_map(built_semantic_map)
		color_built_semantic_map = change_brightness(color_built_semantic_map, observed_area_flag, value=60)

		occupancy_map = occupancy_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

		#=================================== visualize the agent pose as red nodes =======================
		x_coord_lst = []
		z_coord_lst = []
		for cur_pose in traverse_lst:
			x_coord, z_coord = pose_to_coords((cur_pose[0], -cur_pose[1]), pose_range, coords_range, cell_size=.1)
			x_coord_lst.append(x_coord)
			z_coord_lst.append(z_coord)

		#'''
		fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(200, 200))
		# visualize gt semantic map
		ax[0][0].imshow(color_gt_semantic_map)
		ax[0][0].get_xaxis().set_visible(False)
		ax[0][0].get_yaxis().set_visible(False)
		ax[0][0].scatter(x_coord_lst, z_coord_lst, s=30, c='red', zorder=2)
		ax[0][0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=1)
		ax[0][0].set_title('gt semantic map')
		# visualize built semantic map
		ax[0][1].imshow(color_built_semantic_map)
		ax[0][1].get_xaxis().set_visible(False)
		ax[0][1].get_yaxis().set_visible(False)
		ax[0][1].set_title('built semantic map')

		ax[1][0].imshow(occupancy_map, vmax=3)
		ax[1][0].get_xaxis().set_visible(False)
		ax[1][0].get_yaxis().set_visible(False)
		ax[1][0].set_title('occupancy map')
		plt.title('observed area')
		plt.show()
		#fig.savefig(f'{saved_folder}/temp.jpg')
		#plt.close()
		#assert 1==2
		#'''
	
	#==================================== update particle filter =============================
	'''
	if step % NUM_STEPS_EXPLORE == 0:
		PF.observeUpdate(observed_area_flag)
		# get the peak global coordinates from particle filter
		subgoal_pose = PF.getPeak()
	'''
		
	#====================================== take next action ================================
	step += 1
	#action = semantic_map.move_towards_subgoal(subgoal_pose)
	action = random.choice(["TURN_LEFT", "MOVE_FORWARD"])
	obs = env.step(action)[0]