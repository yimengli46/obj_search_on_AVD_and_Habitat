import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor, degrees
import random
from navigation_utils import change_brightness, SimpleRLEnv
from baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, pose_to_coords_numpy, read_map_npy
from map_utils import SemanticMap
from PF_continuous_utils import ParticleFilter
from localNavigator_Astar import localNav_Astar
import habitat
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import random
from core import cfg
import frontier_utils as fr_utils

def nav(env, episode_id, scene_name, scene_height, start_pose, targets, target_cat, saved_folder):

	#=================================== start original navigation code ========================
	np.random.seed(cfg.GENERAL.RANDOM_SEED)
	random.seed(cfg.GENERAL.RANDOM_SEED)

	if cfg.NAVI.FLAG_GT_SEM_MAP:
		sem_map_npy = np.load(f'{cfg.SAVE.SEM_MAP_FROM_SCENE_GRAPH_PATH}/{scene_name}/gt_semantic_map.npy', allow_pickle=True).item()
	gt_semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)
	H, W = gt_semantic_map.shape[:2]

	PF = ParticleFilter(scene_name, target_cat, 10000, gt_semantic_map.copy(), pose_range, coords_range)
	LN = localNav_Astar(pose_range, coords_range, scene_name)

	semMap_module = SemanticMap(scene_name, pose_range, coords_range) # build the observed sem map
	traverse_lst = []

	#===================================== setup the start location ===============================#

	agent_pos = np.array([start_pose[0], scene_height, start_pose[1]]) # (6.6, -6.9), (3.6, -4.5)
	agent_rot = habitat_sim.utils.common.quat_from_angle_axis(2.36, habitat_sim.geo.GRAVITY)
	# check if the start point is navigable
	if not env.habitat_env.sim.is_navigable(agent_pos):
		print(f'start pose is not navigable ...')
		assert 1==2
	obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

	step = 0
	subgoal_coords = None
	subgoal_pose = None 
	MODE_FIND_SUBGOAL = True
	explore_steps = 0
	MODE_FIND_GOAL = False
	visited_frontier = set()
	chosen_frontier = None
	SUBGOAL = None
	GOAL_list = targets
	GOAL_POSE_list = [a for (a, b) in GOAL_list]

	while step < cfg.NAVI.NUM_STEPS:
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

		#======================================= decide if the agent run PF or not ================================
		if subgoal_pose is not None and MODE_FIND_SUBGOAL:
			# condition 1: reach the subgoal
			dist_to_subgoal = math.sqrt(
				(agent_map_pose[0] - subgoal_pose[0])**2 + (agent_map_pose[1] - subgoal_pose[1])**2) # pose might be wrong here
			print(f'dist_to_subgoal = {dist_to_subgoal}')
			if dist_to_subgoal <= cfg.NAVI.THRESH_REACH:
				print(f'condition 1: reach the subgoal')
				# check if the agent has reached the goal
				for (GOAL_Pose, GOAL_size) in GOAL_list:
					dist_subgoal_to_goal = math.sqrt(
						(GOAL_Pose[0] - subgoal_pose[0])**2 + (GOAL_Pose[1] - subgoal_pose[1])**2)
					print(f'dist from subgoal to goal is {dist_subgoal_to_goal}')
					if dist_subgoal_to_goal <= 1. + GOAL_size:
						print(f'==========================REACH THE GOAL =============================')
						MODE_FIND_GOAL = True
						return MODE_FIND_GOAL, step


		if MODE_FIND_SUBGOAL:
			observed_occupancy_map, gt_occupancy_map, observed_area_flag = semMap_module.get_observed_occupancy_map()

			PF.observeUpdate(observed_area_flag, step, saved_folder)
			# get the peak global coordinates from particle filter
			peak_pose, _ = PF.getPeak(step, saved_folder)
			peak_coords = pose_to_coords(peak_pose, pose_range, coords_range)

			if not observed_area_flag[peak_coords[1], peak_coords[0]]:
				frontiers = fr_utils.get_frontiers(observed_occupancy_map, gt_occupancy_map, observed_area_flag)
				frontiers = frontiers - visited_frontier

				frontiers = LN.filter_unreachable_frontiers(frontiers, agent_map_pose, observed_occupancy_map)

				if cfg.NAVI.STRATEGY == 'Greedy':
					chosen_frontier = fr_utils.get_frontier_with_maximum_area(frontiers, gt_occupancy_map)
				elif cfg.NAVI.STRATEGY == 'DP':
					top_frontiers = fr_utils.select_top_frontiers(frontiers, top_n=5)
					chosen_frontier = fr_utils.get_frontier_with_DP(top_frontiers, agent_map_pose, observed_occupancy_map, \
						cfg.NAVI.NUM_STEPS-step, LN)

				if chosen_frontier is not None:
					SUBGOAL = (int(chosen_frontier.centroid[1]), int(chosen_frontier.centroid[0]))
				else:
					print(f'cannot find a peak and cannot find a frontier')
					return False, step
			else:
				chosen_frontier = None
				frontiers = None
				SUBGOAL = LN.find_reachable_loc_to_peak(peak_coords, agent_map_pose, observed_occupancy_map)

			#============================================= visualize semantic map ===========================================#
			if True:
				#==================================== visualize the path on the map ==============================
				built_semantic_map, observed_area_flag, occupancy_map = semMap_module.get_semantic_map()

				## for the explored free space visualization
				mask_observed_and_non_obj = np.logical_and(observed_area_flag, gt_semantic_map == 0)
				gt_semantic_map[mask_observed_and_non_obj] = 59 # class index for explored non-object area

				color_gt_semantic_map = apply_color_to_map(gt_semantic_map)
				color_gt_semantic_map = change_brightness(color_gt_semantic_map, observed_area_flag, value=60)

				built_semantic_map = built_semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]
				color_built_semantic_map = apply_color_to_map(built_semantic_map)
				color_built_semantic_map = change_brightness(color_built_semantic_map, observed_area_flag, value=60)

				#=================================== visualize the agent pose as red nodes =======================
				x_coord_lst, z_coord_lst, theta_lst = [], [], []
				for cur_pose in traverse_lst:
					x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range)
					x_coord_lst.append(x_coord)
					z_coord_lst.append(z_coord)			
					theta_lst.append(cur_pose[2])

				#'''
				fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
				# visualize gt semantic map
				ax[0].imshow(color_gt_semantic_map)
				ax[0].get_xaxis().set_visible(False)
				ax[0].get_yaxis().set_visible(False)
				marker, scale = gen_arrow_head_marker(theta_lst[-1])
				ax[0].scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=2)
				ax[0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=1)
				# draw the subgoal
				ax[0].scatter(SUBGOAL[0], SUBGOAL[1], marker='X', s=70, c='yellow', zorder=4)
				# draw the object goal
				np_target_poses = np.array(list(map(list, GOAL_POSE_list)))
				vis_target_coords = pose_to_coords_numpy(np_target_poses, pose_range, coords_range)
				ax[0].scatter(vis_target_coords[:, 0], vis_target_coords[:, 1], marker='*', s=50, c='yellow', zorder=5)
				ax[0].set_title('gt semantic map')

				ax[1].imshow(observed_occupancy_map)
				if frontiers is not None:
					for f in frontiers:
						ax[1].scatter(f.points[1], f.points[0], c='white', zorder=2)
						ax[1].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
				if chosen_frontier is not None:
					ax[1].scatter(chosen_frontier.points[1], chosen_frontier.points[0], c='green', zorder=2)
					ax[1].scatter(chosen_frontier.centroid[1], chosen_frontier.centroid[0], c='red', zorder=2)
				ax[1].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].set_title('improved observed_occ_map + frontiers')

				fig.tight_layout()
				plt.title('observed area')
				#plt.show()
				fig.savefig(f'{saved_folder}/step_{step}_semmap.jpg')
				plt.close()
				#assert 1==2
				#'''

		#==================================== update particle filter =============================
		if MODE_FIND_SUBGOAL:
			MODE_FIND_SUBGOAL = False
			explore_steps = 0
			flag_plan, subgoal_coords, subgoal_pose = LN.plan_to_reach_frontier(SUBGOAL, agent_map_pose, observed_occupancy_map, step, saved_folder)

			print(f'subgoal_coords = {subgoal_coords}')
			
		#====================================== take next action ================================
		action, next_pose = LN.next_action(occupancy_map, env, scene_height)
		print(f'action = {action}')
		if action == "collision":
			step += 1
			explore_steps += 1
			#assert next_pose is None
			# input next_pose is environment pose, not sem_map pose
			semMap_module.add_occupied_cell_pose(next_pose)
			# redo the planning
			print(f'redo planning')
			observed_occupancy_map, gt_occupancy_map, observed_area_flag = semMap_module.get_observed_occupancy_map()
			'''
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
			ax.imshow(occupancy_map, vmax=5)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.title('collision occupancy_map')
			plt.show()
			'''
			
			flag_plan, subgoal_coords, subgoal_pose = LN.plan_to_reach_frontier(SUBGOAL, agent_map_pose, observed_occupancy_map, step, saved_folder)

			# do not take any actions
		elif action == "": # finished navigating to the subgoal
			print(f'reached the subgoal')
			MODE_FIND_SUBGOAL = True
			if chosen_frontier is not None:
				visited_frontier.add(chosen_frontier)
		else:
			step += 1
			explore_steps += 1
			print(f'next_pose = {next_pose}')
			agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
			# output rot is negative of the input angle
			agent_rot = habitat_sim.utils.common.quat_from_angle_axis(-next_pose[2], habitat_sim.geo.GRAVITY)
			obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

		if explore_steps == cfg.NAVI.NUM_STEPS_EXPLORE:
			explore_steps = 0
			MODE_FIND_SUBGOAL = True

	return False, step

