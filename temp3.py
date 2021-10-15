import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, convertInsSegToSSeg, apply_color_to_map


dataset_dir = '/home/yimeng/ARGO_datasets/MP3D'
scene_name = 'TbHJrupSAjP_1'
cell_size = 0.1
UNIGNORED_CLASS = [1, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 22, 23, 25, 27, 28, 31, 33, 34, 36, 37, 38, 39, 40]
saved_folder = 'results'
step_size = 1
first_num_images = 100
map_boundary = 10

IGNORED_CLASS = []
for i in range(41):
	if i not in UNIGNORED_CLASS:
		IGNORED_CLASS.append(i)

# load img list
img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
img_names = list(img_act_dict.keys())

ins2cat_dict = np.load('{}/{}/dict_ins2category.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()


for idx, img_name in enumerate(img_names):
	#if idx == 100:
	#	break

	print('idx = {}'.format(idx))
	# load rgb image, depth and sseg
	rgb_img = cv2.imread('{}/{}/images/{}.jpg'.format(dataset_dir, scene_name, img_name), 1)[:, :, ::-1]
	npy_file = np.load('{}/{}/others/{}.npy'.format(dataset_dir, scene_name, img_name), allow_pickle=True).item()
	InsSeg_img = npy_file['sseg']
	sseg_img = convertInsSegToSSeg(InsSeg_img, ins2cat_dict)
	depth_img = npy_file['depth']
	pose = img_act_dict[img_name]['pose'] # x, z, theta
	print('pose = {}'.format(pose))

	b = sseg_img[sseg_img < 0]
	if b.shape[0] > 0:
		print('b = {}'.format(b))

		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
		ax[0].imshow(rgb_img)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb")
		ax[1].imshow(sseg_img)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg")
		ax[2].imshow(depth_img)
		ax[2].get_xaxis().set_visible(False)
		ax[2].get_yaxis().set_visible(False)
		ax[2].set_title("depth")
		fig.tight_layout()
		plt.show()
		
