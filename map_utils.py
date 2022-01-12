import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertPanopSegToSSeg, apply_color_to_map

class SemanticMap:
	def __init__(self):

		self.dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
		self.scene_name = 'Allensville_0'
		self.cell_size = 0.1
		self.UNIGNORED_CLASS = []
		self.saved_folder = 'results'
		self.step_size = 100
		self.first_num_images = 100
		self.map_boundary = 5

		self.IGNORED_CLASS = [23+1]

		# load img list
		self.img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(self.dataset_dir, self.scene_name), allow_pickle=True).item()
		self.img_names = list(self.img_act_dict.keys())
		self.detectron2_folder = f'{self.dataset_dir}/{self.scene_name}/panoptic_pred'
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

		self.four_dim_grid = np.zeros((len(self.z_grid), 2000, len(self.x_grid), 41)) # x, y, z, C
		self.H, self.W = len(self.z_grid), len(self.x_grid)

	def build_semantic_map(self, img_name, panorama=False):
		img_names = []
		if panorama:
			cur_img_loc = img_name[:6]
			for angle in ['000', '030', '060', '090', '120', '150', '180', '210', '240', '270', '300', '330']:
				img_names.append(cur_img_loc+angle)
		else:
			img_names.append(img_name)
		
		for img_name in img_names:
			# load rgb image, depth and sseg
			rgb_img = cv2.imread(f'{self.dataset_dir}/{self.scene_name}/rgb/{img_name}.jpg', 1)[:, :, ::-1]
			depth_img = cv2.imread(f'{self.dataset_dir}/{self.scene_name}/depth/{img_name}.png', cv2.IMREAD_UNCHANGED)
			depth_img = depth_img/256.
			depth_img = cv2.blur(depth_img, (3,3))
			panopSeg_img = cv2.imread(f'{self.detectron2_folder}/{img_name}.png', cv2.IMREAD_UNCHANGED)
			sseg_img = convertPanopSegToSSeg(panopSeg_img, self.id2class_mapper)
			sseg_img += 1
			pose = self.img_act_dict[img_name]['pose'] # x, z, theta
			#print('pose = {}'.format(pose))

			xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, pose, gap=2, ignored_classes=self.IGNORED_CLASS)

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
		# argmax over the category axis
		semantic_map = np.argmax(grid_sum_height, axis=2)

		grid_sum_cat = np.sum(grid_sum_height, axis=2)
		observed_area_flag = (grid_sum_cat > 0)

		return semantic_map, observed_area_flag

		'''
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

		if idx % step_size == 0 or idx == len(img_names)-1:
			# write the map with cv2 so the map image can be load directly
			cv2.imwrite('{}/step_{}_semantic.jpg'.format(saved_folder, idx), color_semantic_map[:, :, ::-1])

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

			np.save(f'{saved_folder}/step_{idx}_semantic.npy', map_dict)

			#assert 1==2
		'''