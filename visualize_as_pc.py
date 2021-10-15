import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertInsSegToSSeg, apply_color_to_map, project_pixels_to_camera_coords, apply_color_to_pointCloud


import os
import open3d as o3d
import math
import scipy.io
import matplotlib.pyplot as plt



dataset_dir = '/home/reza/Datasets/MP3D'
scene_name = '2t7WUuJeko7_0'
cell_size = 0.3
UNIGNORED_CLASS = [2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 22, 23, 25, 27, 28, 31, 33, 34, 36, 37, 38, 39, 40]
saved_folder = 'results'
step_size = 1

IGNORED_CLASS = []
'''
for i in range(41):
	if i not in UNIGNORED_CLASS:
		IGNORED_CLASS.append(i)
'''

# load img list
img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
img_names = list(img_act_dict.keys())
ins2cat_dict = np.load('{}/{}/dict_ins2category.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()

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
x_grid = np.arange(min_X, max_X, 0.3)
z_grid = np.arange(min_Z, max_Z, 0.3)

four_dim_grid = np.zeros((len(x_grid), 2000, len(z_grid), 41)) # x, y, z, C

#for idx, img_name in enumerate(img_names):
for idx in [6]:
	print('idx = {}'.format(idx))
	img_name = img_names[idx]
	# load rgb image, depth and sseg
	rgb_img = cv2.imread('{}/{}/images/{}.jpg'.format(dataset_dir, scene_name, img_name), 1)[:, :, ::-1]
	npy_file = np.load('{}/{}/others/{}.npy'.format(dataset_dir, scene_name, img_name), allow_pickle=True).item()
	InsSeg_img = npy_file['sseg']
	sseg_img = convertInsSegToSSeg(InsSeg_img, ins2cat_dict)
	color_sseg_img = apply_color_to_map(sseg_img)
	depth_img = npy_file['depth']
	pose = img_act_dict[img_name]['pose'] # x, z, theta

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
		plt.close()
		#assert 1==2
		#'''

	xyz_points, sseg_points = project_pixels_to_camera_coords(sseg_img, depth_img, pose, gap=2, ignored_classes=IGNORED_CLASS)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz_points.transpose())
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	color_sseg_points = apply_color_to_pointCloud(sseg_points)/255.0
	pcd.colors = o3d.utility.Vector3dVector(color_sseg_points)
	o3d.visualization.draw_geometries([pcd])



	''' 
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()
	image = vis.capture_screen_float_buffer()
	plt.imshow(np.asarray(image))
	plt.show()
	'''
	#assert 1==2

	'''
	color_image = o3d.io.read_image('{}/{}/images/{}.jpg'.format(dataset_dir, scene_name, img_name))
	#orig_depth_image = o3d.io.read_image('/Users/kosecka/research/Negar/Resize/Data/depth/000110000100103.png')
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_img)
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,cam1)
	o3d.visualization.draw_geometries([pcd])
	'''

