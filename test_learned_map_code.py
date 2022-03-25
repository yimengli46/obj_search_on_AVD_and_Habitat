import numpy as np
from navigator import nav
from baseline_utils import create_folder
import habitat
import habitat_sim
from navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg
from localNavigator_Astar import localNav_Astar
from baseline_utils import read_map_npy
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import matplotlib.pyplot as plt

from models.predictors import get_predictor_from_options
from utils.test_utils import get_latest_model, load_model
from models.semantic_grid import SemanticGrid

import torch
import torch.nn as nn
import torch.nn.functional as F

import habitat
import habitat_sim

import utils.utils as utils
import utils.test_utils as tutils
import utils.m_utils as mutils
import utils.viz_utils as vutils

device = 'cuda'

scene_list = ['Allensville_0']
scene_dict = {}
for scene in scene_list:
	scene_name = scene[:-2]
	floor = int(scene[-1])
	temp = {}
	temp['name'] = scene
	temp['floor'] = floor 
	scene_dict[scene_name] = temp

scene_heights_dict = np.load(cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH, allow_pickle=True).item()

#========================= load the occupancy map =============================
predictor = get_predictor_from_options(cfg).to('cuda')
# Needed only for models trained with multi-gpu setting
predictor = nn.DataParallel(predictor)

checkpoint_dir = "trained_weights/resnet_unet_occ_ensemble0_dataPer0-7_4"
latest_checkpoint = get_latest_model(save_dir=checkpoint_dir)
print(f"loading checkpoint: {latest_checkpoint}")
load_model(model=predictor, checkpoint_file=latest_checkpoint)
predictor.eval()


class Param():
	def __init__(self):
		self.hfov = float(90.) * np.pi / 180.
		self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1, 1, cfg.MAP.IMG_SIZE), 
			np.linspace(1, -1, cfg.MAP.IMG_SIZE)), device='cuda')
		self.xs = self.xs.reshape(1, cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE)
		self.ys = self.ys.reshape(1, cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE)
		K = np.array([
			[1 / np.tan(self.hfov / 2.), 0., 0., 0.],
			[0., 1 / np.tan(self.hfov / 2.), 0., 0.],
			[0., 0.,  1, 0],
			[0., 0., 0, 1]])
		self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')

		self.grid_dim = (cfg.MAP.GRID_DIM, cfg.MAP.GRID_DIM)
		self.img_size = (cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE)
		self.crop_size = (cfg.MAP.CROP_SIZE, cfg.MAP.CROP_SIZE)

def run_map_predictor(step_ego_grid_crops):
	input_batch = step_ego_grid_crops.to(device).unsqueeze(0)

	### Estimate average predictions from the ensemble
	print(f'input_batch.shape = {input_batch.shape}')
	mean_ensemble_spatial = predictor(input_batch)
	return mean_ensemble_spatial


def learned_map_nav(env, episode_id, scene_name, scene_height, start_pose, targets, target_cat):
	with torch.no_grad():
		par = Param()
		# For each episode we need a new instance of a fresh global grid
		sg = SemanticGrid(1, par.grid_dim, cfg.MAP.CROP_SIZE, cfg.MAP.CELL_SIZE,
			spatial_labels=cfg.MAP.N_SPATIAL_CLASSES)

		if cfg.NAVI.FLAG_GT_SEM_MAP:
			sem_map_npy = np.load(f'{cfg.SAVE.SEM_MAP_FROM_SCENE_GRAPH_PATH}/{scene_name}/gt_semantic_map.npy', allow_pickle=True).item()
		_, pose_range, coords_range = read_map_npy(sem_map_npy)
		LN = localNav_Astar(pose_range, coords_range, scene_name)

		abs_poses = []
		rel_poses_list = []
		abs_poses_noisy = []
		pose_coords_list = []
		pose_coords_noisy_list = []
		stg_pos_list = []
		agent_height = []

		#agent_pos = np.array([start_pose[0], scene_height, start_pose[1]]) # (6.6, -6.9), (3.6, -4.5)
		agent_pos = np.array([6.6, scene_height, -6.9])
		agent_rot = habitat_sim.utils.common.quat_from_angle_axis(2.36, habitat_sim.geo.GRAVITY)
		heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
		phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
		angle = phi
		pose = (agent_pos[0], agent_pos[2], angle)
		agent_map_pose = (pose[0], -pose[1], -pose[2])
		
		
		GOAL_Pose, GOAL_size = targets[0]
		goal_steps = LN.get_gt_number_steps(GOAL_Pose, agent_map_pose)

		t = 0
		obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

		while t < goal_steps:

			rgb_img = obs['rgb']
			depth_img = 10. * obs['depth'][:, :, 0]
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
			ax[0].imshow(rgb_img)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb")
			ax[1].imshow(depth_img)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("depth")
			fig.tight_layout()
			plt.show()
			#fig.savefig(f'{saved_folder}/step_{step}_obs.jpg')
			#plt.close()
			
			depth_abs = obs['depth'].reshape(cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE, 1)
			depth_abs = torch.tensor(depth_abs).to(device)
			local3D_step = utils.depth_to_3D(depth_abs, par.img_size, par.xs, par.ys, par.inv_K)

			agent_pose, y_height = utils.get_sim_location(agent_state=env.habitat_env.sim.get_agent_state())

			abs_poses.append(agent_pose)
			agent_height.append(y_height)

			# get the relative pose with respect to the first pose in the sequence
			rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
			_rel_pose = torch.Tensor(rel).unsqueeze(0).float()
			_rel_pose = _rel_pose.to(device)
			rel_poses_list.append(_rel_pose.clone())

			pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[0], par.grid_dim[0], cfg.MAP.CELL_SIZE, device) # B x T x 3
			pose_coords_list.append(pose_coords.clone().cpu().numpy())

			'''
			if t==0:
				# get gt map from initial agent pose for visualization at end of episode
				x, y, label_seq = map_utils.slice_scene(x=self.test_ds.pcloud[0].copy(),
														y=self.test_ds.pcloud[1].copy(),
														z=self.test_ds.pcloud[2].copy(),
														label_seq=self.test_ds.label_seq_spatial.copy(),
														height=agent_height[0])
				gt_map_initial = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
															grid_dim=self.test_ds.grid_dim, cell_size=self.test_ds.cell_size)
			'''

			# do ground-projection, update the map
			ego_grid_sseg_3 = mutils.est_occ_from_depth([local3D_step], grid_dim=par.grid_dim, cell_size=cfg.MAP.CELL_SIZE, 
																			device=device, occupancy_height_thresh=cfg.MAP.OCCUPANCY_HEIGHT_THRESH)
			#print(f'ego_grid_sseg_3.shape = {ego_grid_sseg_3.shape}')

			# Transform the ground projected egocentric grids to geocentric using relative pose
			geo_grid_sseg = sg.spatialTransformer(grid=ego_grid_sseg_3, pose=rel_poses_list[t], abs_pose=torch.tensor(abs_poses).to(device))
			# step_geo_grid contains the map snapshot every time a new observation is added
			step_geo_grid_sseg = sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
			# transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
			step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=rel_poses_list[t], abs_pose=torch.tensor(abs_poses).to(device))
			# Crop the grid around the agent at each timestep
			#print(f'step_ego_grid_sseg.shape = {step_ego_grid_sseg.shape}')
			step_ego_grid_crops = mutils.crop_grid(grid=step_ego_grid_sseg, crop_size=par.crop_size)
			#print(f'step_ego_grid_crops.shape = {step_ego_grid_crops.shape}')

			mean_ensemble_spatial = run_map_predictor(step_ego_grid_crops)

			# add occupancy prediction to semantic map
			sg.register_occ_pred(prediction_crop=mean_ensemble_spatial, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=device))
			

			#============ take action, move
			t += 1
			'''
			action, next_pose = LN.next_action(env, scene_height)
			agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
			# output rot is negative of the input angle
			agent_rot = habitat_sim.utils.common.quat_from_angle_axis(-next_pose[2], habitat_sim.geo.GRAVITY)
			obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)
			'''
			obs = env.step('TURN_RIGHT')[0]

			color_spatial_pred = vutils.colorize_grid(mean_ensemble_spatial, color_mapping=3)
			im_spatial_pred = color_spatial_pred[0,0,:,:,:].permute(1,2,0).cpu().numpy()
			plt.imshow(im_spatial_pred)
			plt.show()
			
			print(f'sg.occ_grid.shape = {sg.occ_grid.shape}')
			color_occ_grid = vutils.colorize_grid(sg.occ_grid.unsqueeze(1), color_mapping=3)
			im = color_occ_grid[0,0,:,:,:].permute(1,2,0).cpu().numpy()
			plt.imshow(im)
			plt.show()


#================================ load habitat env============================================
config = habitat.get_config(config_paths="/home/yimeng/Datasets/habitat-lab/configs/tasks/devendra_objectnav_gibson_for_GG.yaml" )
config.defrost()
config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_EPISODE_DATA_PATH
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()
env = SimpleRLEnv(config=config)

for episode_id in range(1, 2):
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

		output_folder = 'output/TESTING_RESULTS_TEST_LEARNED_OCC'
		scene_output_folder = f'{output_folder}/{scene_name}'
		create_folder(scene_output_folder)
		testing_data = np.load(f'{cfg.SAVE.TESTING_DATA_FOLDER}/testing_episodes_{scene_name}.npy', allow_pickle=True)

		#'''
		results = {}
		#for idx, data in enumerate(testing_data):
		for idx in range(0, 1):
			data = testing_data[idx]
			print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
			start_pose, target_cat, targets = data
			#start_pose = (6.6, -6.9)
			#target_cat = 'refrigerator'
			saved_folder = f'{scene_output_folder}/eps_{idx}_{target_cat}'
			create_folder(saved_folder, clean_up=True)

			learned_map_nav(env, idx, scene_name, height, start_pose, targets, target_cat)
			
env.close()









