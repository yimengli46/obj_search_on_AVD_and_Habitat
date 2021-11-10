import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from baseline_utils import apply_color_to_map, get_class_mapper, read_map_npy, create_folder, pxl_coords_to_pose
import skimage.measure
from math import floor, sqrt
import csv
import math
from sklearn.mixture import GaussianMixture
import itertools
from scipy.stats import norm

#scene_id = 3
dataset_dir = '/home/yimeng/Datasets/MP3D'
GMM_param_save_folder = 'output/GMM_obj_obj_1d_prior'
num_GMM_components = 3

scene_list = ['2t7WUuJeko7_0']
scene_list += ['7y3sRwLe3Va_1', '8WUmhLawc2A_0', '29hnd4uzFmX_0', 'cV4RVeZvu5T_0', 'cV4RVeZvu5T_1', 'e9zR4mvMWw7_0',]
scene_list += ['GdvgFV5R1Z5_0', 'i5noydFURQK_0', 's8pcmisQ38h_0', 's8pcmisQ38h_1', 'S9hNv5qa7GM_0', 'V2XKFyX4ASd_0',]
scene_list += ['V2XKFyX4ASd_1', 'V2XKFyX4ASd_2', 'TbHJrupSAjP_0', 'TbHJrupSAjP_1', 'zsNo4HB9uLZ_0', 'RPmz2sHmrrY_0',]
scene_list += ['WYY7iVyf5p8_0', 'WYY7iVyf5p8_1', 'YFuZgdQ5vWj_0']

IGNORED_CLASS = ['void', 'wall', 'floor', 'railing', 'blinds']
thresh_dist = math.inf

semantic_map_output_folder = f'output/semantic_map'
create_folder(semantic_map_output_folder, clean_up=False)

# create obj-obj dict
cat2idx_dict = get_class_mapper()
idx2cat_dict = {v: k for k, v in cat2idx_dict.items()}
obj_obj_dict = {}
for k1 in list(cat2idx_dict.keys()):
	if k1 in IGNORED_CLASS:
		continue
	else:
		obj_obj_dict[k1] = {}
		for k2 in list(cat2idx_dict.keys()):
			if k2 in IGNORED_CLASS:
				continue
			else:
				obj_obj_dict[k1][k2] = []

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{semantic_map_output_folder}/{scene_name}'

	# load semantic map
	map_npy = np.load(f'{saved_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(map_npy)

	# load instance centers
	list_insts = np.load(f'{saved_folder}/instances_centers.npy', allow_pickle=True)

	for i, a_inst in enumerate(list_insts):
		a_center_coords = a_inst['center']
		a_cat = idx2cat_dict[a_inst['cat']]
		a_center_pose = pxl_coords_to_pose(a_center_coords, pose_range, coords_range, cell_size=.01, flag_cropped=True)
		if a_cat in IGNORED_CLASS:
			continue
		else:
			for j, b_inst in enumerate(list_insts):
				b_center_coords = b_inst['center']
				b_cat = idx2cat_dict[b_inst['cat']]
				b_center_pose = pxl_coords_to_pose(b_center_coords, pose_range, coords_range, cell_size=.01, flag_cropped=True)
				
				dist = sqrt((b_center_pose[0] - a_center_pose[0])**2 + (b_center_pose[1] - a_center_pose[1])**2)
				obj_obj_dict[a_cat][b_cat].append(dist)
				obj_obj_dict[b_cat][a_cat].append(dist)

def confirm_nComponents(X):
	bics = []
	min_bic = 0
	counter = 1
	maximum_nComponents = min(len(X), 10)
	for i in range (1, maximum_nComponents): # test the AIC/BIC metric between 1 and 10 components
		gmm = GaussianMixture(n_components=counter, max_iter=1000, random_state=0, covariance_type = 'full').fit(X)
		bic = gmm.bic(X)
		bics.append(bic)
		if bic < min_bic or min_bic == 0:
			min_bic = bic
			opt_bic = counter

		counter += 1
	return opt_bic


def visualize_GMM_dist(arr_dist, gm, nComponents, k1, k2, h=400, w=400):
	min_X = -w/2 * .1
	max_X = w/2 * .1
	min_Z = -h/2 * .1
	max_Z = w/2 * .1
	x_grid = np.arange(min_X, max_X, 0.1)
	z_grid = np.arange(min_Z, max_Z, 0.1)
	xv, yv = np.meshgrid(x_grid, z_grid)
	xv = xv.flatten()
	yv = yv.flatten()
	locs = np.stack((yv, xv), axis=1)
	dists = np.sqrt(locs[:, 0]**2 + locs[:, 1]**2)
	dists = dists.reshape(-1, 1)
	logprob = gm.score_samples(dists)
	pdf = np.exp(logprob)
	prob_dist = pdf.reshape((h, w))

	x_axis = np.arange(0, 30, 0.1)
	y_axis_all = np.zeros((nComponents, x_axis.shape[0]))
	for i in range(nComponents):
		y_axis = norm.pdf(x_axis, float(gm.means_[i][0]), np.sqrt(float(gm.covariances_[i][0][0])))*gm.weights_[i] # 1st gaussian
		y_axis_all[i] = y_axis

	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))
	ax[0].hist(arr_dist, 60, range=[0, 30], density=True, facecolor='g', alpha=0.75)
	ax[0].set_title(f'instance count between {k1} and {k2}')
	ax[0].plot(x_axis, np.sum(y_axis_all, axis=0), lw=3, c='b', ls='dashed')
	im = ax[1].imshow(prob_dist, vmin=.0, vmax=.2)
	ax[1].set_title(f'GMM 1d prior between {k1} and {k2}, {nComponents} compons')
	fig.colorbar(im)
	#plt.show()
	#'''
	fig.tight_layout()
	fig.savefig(f'{GMM_param_save_folder}/GMM_dist_{k1}_{k2}.jpg')
	plt.close()
	#'''
	
for i, k1 in enumerate(list(obj_obj_dict.keys())):
	for j, k2 in enumerate(list(obj_obj_dict[k1].keys())):
		arr_dist = np.array(obj_obj_dict[k1][k2])
		if len(arr_dist) > 5:
			arr_dist = arr_dist.reshape(-1, 1)
			num_GMM_components = confirm_nComponents(arr_dist)
			print(f'k1 = {k1}, k2 = {k2}, nComponents = {num_GMM_components}')
			gm = GaussianMixture(n_components=num_GMM_components).fit(arr_dist)
			params = {}
			params['nComponents'] = num_GMM_components
			params['weights'] = gm.weights_
			params['means'] = gm.means_
			params['covariances'] = gm.covariances_
			np.save(f'{GMM_param_save_folder}/GMM_params_{k1}_{k2}.npy', params)

			visualize_GMM_dist(arr_dist, gm, num_GMM_components, k1, k2)
			#assert 1==2
			#y_axis0 = norm.pdf(x_axis, float(params['means'][0][0]), np.sqrt(float(params['covariances'][0][0][0])))*params['weights'][0] # 1st gaussian


		'''
		logprob = gm.score_samples(np.array([[-14., 1.]]))
		pdf = np.exp(logprob)
		gm.set_params(**params)
		'''
		#assert 1==2




