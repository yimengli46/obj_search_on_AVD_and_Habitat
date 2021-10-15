'''
build top-down-view semantic map from depth and sseg egocentric observations.
'''

import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from utils import project_pixels_to_world_coords, convertInsSegToSSeg, apply_color_to_map, create_folder, apply_color_to_pointCloud
from baseline_utils import readDepthImage, read_all_poses, read_cached_data, load_structs, get_pose
import open3d as o3d

scene_list = ['Home_002_1']

step_size = 1
map_boundary = 10
cell_size = 0.1
max_height = 5.0 # maximum height is 5.0 meter
avd_folder = '/home/yimeng/Datasets/ActiveVisionDataset'
base_saved_folder = 'avd_semantic_map'
avd_minimal_folder = f'{avd_folder}/AVD_Minimal'
detection_folder = f'{avd_folder}/detectron2_panop'
flag_vis = True


IGNORED_CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', \
	'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', \
	'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'frisbee', 'skis', 'snowboard', 'skateboard', 'surfboard', \
	'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'banner', 'bridge', 'gravel', 'house', 'net', \
	'playingfield', 'railroad', 'river', 'road', 'sand', 'sea', 'snow', 'tent', 'water','tree', 'fence', 'sky', \
	'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'rug', 'ceiling', 'teddy bear']


for scene_id in range(len(scene_list)):
	print(f'scene = {scene_list[scene_id]}')
	scene = scene_list[scene_id]

	all_poses = read_all_poses(avd_minimal_folder, scene)
	cached_data = read_cached_data(False, avd_minimal_folder, targets_file_name=None, output_size=224, Home_name=scene.encode()) ## encode() convert string to byte
	img_structs = load_structs(f'{avd_minimal_folder}/{scene}')

	cat2id_mapper = np.load(f'{detection_folder}/cat2id.npy', allow_pickle=True).item()
	num_classes = len(list(cat2id_mapper.keys()))

	saved_folder = f'{base_saved_folder}/{scene}'
	create_folder(saved_folder, clean_up=False)

	IGNORED_CLASS = []
	for i in range(num_classes):
		if list(cat2id_mapper.keys())[i] in IGNORED_CATEGORIES:
			IGNORED_CLASS.append(i)


	# load img list
	img_names = cached_data['world_id_dict'][scene.encode()]

	# decide size of the grid
	min_X = -20
	max_X = 20
	min_Z = -20
	max_Z = 20
	x_grid = np.arange(min_X, max_X, cell_size)
	z_grid = np.arange(min_Z, max_Z, cell_size)
	four_dim_grid = np.zeros((len(z_grid), int(max_height/cell_size), len(x_grid), num_classes+1), dtype=np.int32) # x, y, z, C
	H, W = four_dim_grid.shape[0], four_dim_grid.shape[2]

	img_names = [b'000210005520101', b'000210002840101', ]

	for idx, img_name in enumerate(img_names):
		#if idx == 100:
		#	break

		print(f'idx = {idx}, img_name = {img_name}')
		# load rgb image, depth and sseg
		rgb_img = cv2.imread(f'{avd_folder}/{scene}/jpg_rgb/{img_name.decode()}.jpg', 1)[:, :, ::-1]
		depth_img = readDepthImage(scene, img_name.decode(), avd_folder, resolution=0)
		sseg_img = cv2.imread(f'{detection_folder}/{scene}/{img_name.decode()}_panop.png', 0)
		
		rgb_img = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_LINEAR)
		depth_img = cv2.resize(depth_img, (512, 512), interpolation=cv2.INTER_NEAREST)
		sseg_img = cv2.resize(sseg_img, (512, 512), interpolation=cv2.INTER_NEAREST)

		if idx % step_size == 0:
			#'''
			mask_depth = depth_img < .01
			vis_rgb_img = rgb_img.copy()
			vis_rgb_img[mask_depth, :] = 0
			vis_sseg_img = sseg_img.copy()
			vis_sseg_img[mask_depth] = 0
			fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
			ax[0].imshow(vis_rgb_img)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb")
			ax[1].imshow(apply_color_to_map(vis_sseg_img, num_classes=num_classes))
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg")
			ax[2].imshow(depth_img)
			ax[2].get_xaxis().set_visible(False)
			ax[2].get_yaxis().set_visible(False)
			ax[2].set_title("depth")
			fig.tight_layout()
			#plt.show()
			fig.savefig(f'{saved_folder}/step_{idx}.jpg')
			plt.close()
			#assert 1==2
			#'''

		pose = get_pose(img_name.decode(), all_poses, img_structs) # x, z, theta
		print(f'pose = {pose}')
		#assert 1==2

		xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, pose, gap=4, focal_length=256, resolution=512, ignored_classes=IGNORED_CLASS)
		'''
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(xyz_points.transpose())
		# Flip it, otherwise the pointcloud will be upside down
		pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		color_sseg_points = apply_color_to_pointCloud(sseg_points, num_classes=num_classes)/255.0
		pcd.colors = o3d.utility.Vector3dVector(color_sseg_points)
		o3d.visualization.draw_geometries([pcd])
		'''

		# filter out 3d points out of the range of the environment
		mask_X = np.logical_and(xyz_points[0, :] > min_X, xyz_points[0, :] < max_X) 
		mask_Y = np.logical_and(xyz_points[1, :] > 0.0, xyz_points[1, :] < max_height)
		mask_Z = np.logical_and(xyz_points[2, :] > min_Z, xyz_points[2, :] < max_Z)  
		mask_XYZ = np.logical_and.reduce((mask_X, mask_Y, mask_Z))
		xyz_points = xyz_points[:, mask_XYZ]
		sseg_points = sseg_points[mask_XYZ]

		# discretize 3d points into 3d coordinates
		# assign 3d points to voxels
		x_coord = np.floor((xyz_points[0, :] - min_X) / cell_size).astype(int)
		y_coord = np.floor(xyz_points[1, :] / cell_size).astype(int)
		z_coord = np.floor((xyz_points[2, :] - min_Z) / cell_size).astype(int)
		
		# filter out points with too large height (y value)
		mask_y_coord = y_coord < int(max_height/cell_size)
		x_coord = x_coord[mask_y_coord]
		y_coord = y_coord[mask_y_coord]
		z_coord = z_coord[mask_y_coord]
		
		sseg_points = sseg_points[mask_y_coord]
		four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1

		# sum over the height axis
		grid_sum_height = np.sum(four_dim_grid, axis=1)

		# argmax over the category axis
		semantic_map = np.argmax(grid_sum_height, axis=2)
		#assert 1==2

		# get the local map
		if idx == 0:
			min_x_coord = max(np.min(x_coord)-map_boundary, 0)
			max_x_coord = min(np.max(x_coord)+map_boundary, W-1)
			min_z_coord = max(np.min(z_coord)-map_boundary, 0)
			max_z_coord = min(np.max(z_coord)+map_boundary, H-1)
		else:
			if x_coord.shape[0] > 0:
				min_x_coord = min(max(np.min(x_coord)-map_boundary, 0), min_x_coord)
				max_x_coord = max(min(np.max(x_coord)+map_boundary, W-1), max_x_coord)
			if z_coord.shape[0] > 0:
				min_z_coord = min(max(np.min(z_coord)-map_boundary, 0), min_z_coord)
				max_z_coord = max(min(np.max(z_coord)+map_boundary, H-1), max_z_coord)

		semantic_map = semantic_map[min_z_coord:max_z_coord+1, min_x_coord:max_x_coord+1]
		color_semantic_map = apply_color_to_map(semantic_map, num_classes=num_classes)

		if idx % step_size == 0:
			#color_semantic_map = cv2.resize(color_semantic_map, (int(z_span/cell_size), int(x_span/cell_size)), interpolation = cv2.INTER_NEAREST)
			#cv2.imwrite(f'{saved_folder}/step_{idx}_semantic.jpg', color_semantic_map[:,:,::-1])
			plt.imshow(color_semantic_map)
			plt.show()
			#assert 1==2

	'''
	# save final color_semantic_map and semantic_map
	color_semantic_map = apply_color_to_map(semantic_map, num_classes=num_classes)
	color_semantic_map = cv2.resize(color_semantic_map, (int(z_span/cell_size), int(x_span/cell_size)), interpolation = cv2.INTER_NEAREST)
	cv2.imwrite(f'{saved_folder}/step_{idx}_semantic.jpg', color_semantic_map[:,:,::-1])

	semantic_map = semantic_map.astype('uint')
	semantic_map = cv2.resize(semantic_map, (int(z_span/cell_size), int(x_span/cell_size)), interpolation = cv2.INTER_NEAREST)
	cv2.imwrite(f'{saved_folder}/BEV_semantic_map.png', semantic_map)
	'''