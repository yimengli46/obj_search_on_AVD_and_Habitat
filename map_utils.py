import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertPanopSegToSSeg, apply_color_to_map, pose_to_coords
from panoptic_prediction import PanopPred
from baseline_utils import pose_to_coords_frame, pxl_coords_to_pose


class SemanticMap:
	def __init__(self, pose_range, coords_range):

		self.dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
		self.scene_name = 'Allensville_0'
		self.cell_size = 0.1
		self.UNIGNORED_CLASS = []
		self.saved_folder = 'results'
		self.step_size = 100
		self.first_num_images = 100
		self.map_boundary = 5
		self.detector = 'PanopticSeg'
		self.panop_pred = PanopPred()
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.occupied_poses = [] # detected during navigation

		self.IGNORED_CLASS = [54] # ceiling class is ignored
		self.UNDETECTED_PIXELS_CLASS = 59

		# load img list
		self.img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(self.dataset_dir, self.scene_name), allow_pickle=True).item()
		self.img_names = list(self.img_act_dict.keys())
		self.id2class_mapper = np.load('configs/COCO_PanopticSeg_labels_dict.npy', allow_pickle=True).item()

		self.min_X = 1000.0
		self.max_X = -1000.0
		self.min_Z = 1000.0
		self.max_Z = -1000.0
		for idx, img_name in enumerate(self.img_names):
			pose = self.img_act_dict[img_name]['pose'] # x, z, theta
			x, z, _ = pose
			if x < self.min_X:
				self.min_X = x
			if x > self.max_X:
				self.max_X = x
			if z < self.min_Z:
				self.min_Z = z
			if z > self.max_Z:
				self.max_Z = z

		self.min_X -= 10.0
		self.max_X += 10.0
		self.min_Z -= 10.0
		self.max_Z += 10.0
		self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
		self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)

		self.four_dim_grid = np.zeros((len(self.z_grid), 2000, len(self.x_grid), 60), dtype=np.int16) # x, y, z, C
		self.H, self.W = len(self.z_grid), len(self.x_grid)

	def build_semantic_map(self, obs, pose):
		# load rgb image, depth and sseg
		rgb_img = obs['rgb']
		depth_img = 5. * obs['depth']
		depth_img = cv2.blur(depth_img, (3,3))
		if self.detector == 'PanopticSeg':
			panopSeg_img, _ = self.panop_pred.get_prediction(rgb_img, flag_vis=False)
			sseg_img = convertPanopSegToSSeg(panopSeg_img, self.id2class_mapper)
		sseg_img = np.where(sseg_img==0, self.UNDETECTED_PIXELS_CLASS, sseg_img) # label 59 for pixels observed but undetected by the detector
		sem_map_pose = (pose[0], -pose[1], -pose[2]) # x, z, theta
		print('pose = {}'.format(pose))

		#'''
		if True:
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
		#'''

		xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, sem_map_pose, gap=2, ignored_classes=self.IGNORED_CLASS)

		mask_X = np.logical_and(xyz_points[0, :] > self.min_X, xyz_points[0, :] < self.max_X) 
		mask_Y = np.logical_and(xyz_points[1, :] > 0.0, xyz_points[1, :] < 100.0)
		mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z, xyz_points[2, :] < self.max_Z)  
		mask_XYZ = np.logical_and.reduce((mask_X, mask_Y, mask_Z))
		xyz_points = xyz_points[:, mask_XYZ]
		sseg_points = sseg_points[mask_XYZ]

		x_coord = np.floor((xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
		y_coord = np.floor(xyz_points[1, :] / self.cell_size).astype(int)
		z_coord = np.floor((xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)
		mask_y_coord = y_coord < 1000
		x_coord = x_coord[mask_y_coord]
		y_coord = y_coord[mask_y_coord]
		z_coord = z_coord[mask_y_coord]
		sseg_points = sseg_points[mask_y_coord]
		self.four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1
		#assert 1==2

	def get_semantic_map(self):

		# sum over the height axis
		grid_sum_height = np.sum(self.four_dim_grid, axis=1)
		grid_undetected_class = grid_sum_height[:, :, self.UNDETECTED_PIXELS_CLASS]
		grid_detected_class = grid_sum_height[:, :, :self.UNDETECTED_PIXELS_CLASS]
		# argmax over the detected category axis
		semantic_map = np.argmax(grid_detected_class, axis=2)
		mask_explored_undetected_area = np.logical_and(semantic_map==0, grid_undetected_class > 0)
		semantic_map[mask_explored_undetected_area] = self.UNDETECTED_PIXELS_CLASS

		grid_sum_cat = np.sum(grid_sum_height, axis=2)
		observed_area_flag = (grid_sum_cat > 0)

		# get occupancy map
		occupancy_map = np.zeros(semantic_map.shape, dtype=np.int8)
		occupancy_map = np.where(semantic_map==57, 3, occupancy_map) # floor index 57, free space index 3
		occupancy_map = np.where(semantic_map==self.UNDETECTED_PIXELS_CLASS, 2, occupancy_map) # explored but undetected area, index 2
		# occupied area are the explored area but not floor
		mask_explored_occupied_area = np.logical_and(observed_area_flag, occupancy_map==0)
		occupancy_map[mask_explored_occupied_area] = 1 # occupied space index

		occupancy_map = occupancy_map[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1]

		# add occupied cells
		for pose in self.occupied_poses:
			coords = pose_to_coords(pose, self.pose_range, self.coords_range, flag_cropped=True)
			print(f'coords = {coords}')
			occupancy_map[coords[1], coords[0]] = 1

		'''
		temp_semantic_map = semantic_map[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1]
		temp_occupancy_map = occupancy_map[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1]
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
		# visualize gt semantic map
		ax[0].imshow(temp_semantic_map)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('semantic map')
		ax[1].imshow(temp_occupancy_map, vmax=3)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('occupancy map')
		plt.show()
		'''

		return semantic_map, observed_area_flag, occupancy_map

	def add_occupied_cell_pose(self, pose):
		agent_map_pose = (pose[0], -pose[1], -pose[2])
		self.occupied_poses.append(agent_map_pose)

	

