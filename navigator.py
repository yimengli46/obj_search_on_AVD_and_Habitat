import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor, degrees
import random
from navigation_utils import get_obs, random_move, get_obs_panor, read_map_npy, get_pose, change_brightness, SimpleRLEnv
from baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, pose_to_coords_numpy
from map_utils import SemanticMap
from PF_continuous_utils import ParticleFilter
from localNavigator_Astar import localNav_Astar
import habitat
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import random

def nav(episode_id, scene_name, start_pose, goal_poses, target_cat, saved_folder):
	dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
	#scene_name = 'Allensville_0'
	SEED = 10
	NUM_STEPS = 300
	cell_size = 0.1
	flag_vis = False
	#saved_folder = 'output/TEST_RESULTS'
	vis_observed_area_from_panorama = False
	flag_gt_semantic_map = True
	NUM_STEPS_EXPLORE = 30
	NUM_STEPS_vis = 10
	detector = 'PanopticSeg'
	THRESH_REACH = 0.8

	np.random.seed(SEED)
	random.seed(SEED)

	if flag_gt_semantic_map:
		sem_map_npy = np.load(f'output/gt_semantic_map_from_SceneGraph/{scene_name}/gt_semantic_map.npy', allow_pickle=True).item()
	gt_semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)
	H, W = gt_semantic_map.shape[:2]
	#occ_map = np.load(f'output/semantic_map/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True)

	PF = ParticleFilter(target_cat, 10000, gt_semantic_map.copy(), pose_range, coords_range)
	LN = localNav_Astar(pose_range, coords_range)

	semMap_module = SemanticMap(scene_name, pose_range, coords_range) # build the observed sem map
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

	#===================================== visualize initial particles ===============================
	if False:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
		PF.visualizeBelief(ax)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.title('initial particle distribution')
		plt.show()
		#fig.savefig(f'{saved_folder}/temp.jpg')
		#plt.close()

	#===================================== setup the start location ===============================#
	obs = env.reset()
	agent_pos = np.array([start_pose[0], 0.17, start_pose[1]]) # (6.6, -6.9), (3.6, -4.5)
	agent_rot = habitat_sim.utils.common.quat_from_angle_axis(2.36, habitat_sim.geo.GRAVITY)
	# check if the start point is navigable
	if not env.habitat_env.sim.is_navigable(agent_pos):
		print(f'start pose is not navigable ...')
		return
	obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

	step = 0
	subgoal_coords = None
	subgoal_pose = None 
	MODE_FIND_SUBGOAL = True
	explore_steps = 0
	MODE_FIND_GOAL = False
	GOAL_POSE_list = goal_poses

	while step < NUM_STEPS:
		print(f'step = {step}')

		#=============================== get agent global pose on habitat env ========================#
		agent_pos = env.habitat_env.sim.get_agent_state().position
		agent_rot = env.habitat_env.sim.get_agent_state().rotation
		heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
		phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
		angle = phi
		print(f'agent position = {agent_pos}, angle = {angle}')
		pose = (agent_pos[0], agent_pos[2], angle)
		agent_map_pose = (pose[0], -pose[1], -pose[2])
		traverse_lst.append(agent_map_pose)

		# add the observed area
		semMap_module.build_semantic_map(obs, pose, step=step, saved_folder=saved_folder)
		#assert 1==2

		#======================================= decide if the agent run PF or not ================================
		if subgoal_pose is not None:
		#if not MODE_FIND_SUBGOAL:
			assert subgoal_pose is not None
			# condition 1: reach the subgoal
			dist_to_subgoal = math.sqrt(
				(agent_map_pose[0] - subgoal_pose[0])**2 + (agent_map_pose[1] - subgoal_pose[1])**2) # pose might be wrong here
			print(f'dist_to_subgoal = {dist_to_subgoal}')
			if dist_to_subgoal <= THRESH_REACH:
				print(f'condition 1: reach the subgoal')
				explore_steps = 0
				MODE_FIND_SUBGOAL = True
				# check if the agent has reached the goal
				for GOAL_Pose in GOAL_POSE_list:
					dist_subgoal_to_goal = math.sqrt(
						(GOAL_Pose[0] - subgoal_pose[0])**2 + (GOAL_Pose[1] - subgoal_pose[1])**2)
					print(f'dist from subgoal to goal is {dist_subgoal_to_goal}')
					if dist_subgoal_to_goal <= 1.:
						print(f'==========================REACH THE GOAL =============================')
						MODE_FIND_GOAL = True
						break

			# condition 2: run out of exploration steps
			elif explore_steps >= NUM_STEPS_EXPLORE:
				print(f'condition 2: running out exploration steps')
				explore_steps = 0
				MODE_FIND_SUBGOAL = True

		if MODE_FIND_GOAL:
			break

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

			#occupancy_map = occupancy_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

			
		
		#==================================== update particle filter =============================
		if MODE_FIND_SUBGOAL:
			PF.observeUpdate(observed_area_flag, step, saved_folder)
			# get the peak global coordinates from particle filter
			peak_pose = PF.getPeak(step, saved_folder)
			#assert 1==2
			#peak_pose = (7.38, 3.42)
			print(f'peak_pose = {peak_pose}')
			MODE_FIND_SUBGOAL = False

			# peak is not subgoal
			# subgoal is the closest/reachable(bfs) free position/coordinates to the peak on the occupancy map
			subgoal_coords, subgoal_pose = LN.plan(peak_pose, agent_map_pose, occupancy_map, step, saved_folder)
			print(f'subgoal_coords = {subgoal_coords}')

		if step % NUM_STEPS_vis == 0:
			#=================================== visualize the agent pose as red nodes =======================
			x_coord_lst, z_coord_lst, theta_lst = [], [], []
			for cur_pose in traverse_lst:
				x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, cell_size=.1)
				x_coord_lst.append(x_coord)
				z_coord_lst.append(z_coord)			
				theta_lst.append(cur_pose[2])

			#'''
			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
			# visualize gt semantic map
			ax[0][0].imshow(color_gt_semantic_map)
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			#ax[0][0].scatter(x_coord_lst, z_coord_lst, s=30, c='red', zorder=2)
			marker, scale = gen_arrow_head_marker(theta_lst[-1])
			ax[0][0].scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=2)
			ax[0][0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=1)
			# draw the subgoal
			if subgoal_coords is not None:
				ax[0][0].scatter(subgoal_coords[0], subgoal_coords[1], marker='X', s=50, c='yellow', zorder=3)
			# draw the object goal
			np_target_poses = np.array(list(map(list, GOAL_POSE_list)))
			vis_target_coords = pose_to_coords_numpy(np_target_poses, pose_range, coords_range)
			ax[0][0].scatter(vis_target_coords[:, 0], vis_target_coords[:, 1], marker='*', s=50, c='yellow', zorder=3)
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
			fig.tight_layout()
			plt.title('observed area')
			#plt.show()
			fig.savefig(f'{saved_folder}/step_{step}_semmap.jpg')
			plt.close()
			#assert 1==2
			#'''
			
		#====================================== take next action ================================
		step += 1
		explore_steps += 1
		action, next_pose = LN.next_action(occupancy_map, env, 0.17)
		print(f'action = {action}')
		if action == "collision":
			#assert next_pose is None
			# input next_pose is environment pose, not sem_map pose
			semMap_module.add_occupied_cell_pose(next_pose)
			# redo the planning
			print(f'redo planning')
			_, _, occupancy_map = semMap_module.get_semantic_map()
			'''
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
			ax.imshow(occupancy_map, vmax=5)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.title('collision occupancy_map')
			plt.show()
			'''
			
			subgoal_coords, subgoal_pose = LN.plan(peak_pose, agent_map_pose, occupancy_map, step, saved_folder)
			# do not take any actions
		elif action == "": # finished navigating to the subgoal
			MODE_FIND_SUBGOAL = True
		else:
			print(f'next_pose = {next_pose}')
			agent_pos = np.array([next_pose[0], 0.17, next_pose[1]])
			# output rot is negative of the input angle
			agent_rot = habitat_sim.utils.common.quat_from_angle_axis(-next_pose[2], habitat_sim.geo.GRAVITY)
			obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

	env.close()