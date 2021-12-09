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
import glob

#scene_id = 3
scene_info_folder = '/home/yimeng/Datasets/habitat-lab/habitat_nav/code_for_habitat/mp3d_scenes_info'
GMM_param_save_folder = 'output/GMM_obj_obj_1d_prior_all_scenes'
num_GMM_components = 3

scene_list = glob.glob(f'{scene_info_folder}/*.npy')

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

def get_scene_name(scene):
	dot_index = scene.rfind('.')
	slash_index = scene.rfind('/')
	return scene[slash_index+1:dot_index]

for scene_id in range(len(scene_list)):
	scene_name = get_scene_name(scene_list[scene_id])
	print(f'scene_id = {scene_id}, scene_name = {scene_name}')
	
	npy_file = np.load(scene_list[scene_id], allow_pickle=True)

	for i, a_inst in enumerate(npy_file):
		a_center_coords = a_inst['center']
		a_cat = a_inst['category']
		if a_inst['objectness'] and a_cat in list(cat2idx_dict.keys()):
			if a_cat not in IGNORED_CLASS:
				a_center_pose = (a_center_coords[0], a_center_coords[2])
		
				for j, b_inst in enumerate(npy_file):
					b_center_coords = b_inst['center']
					b_cat = b_inst['category']
					if b_inst['objectness']:
						if b_cat not in IGNORED_CLASS and b_cat in list(cat2idx_dict.keys()):
							b_center_pose = (b_center_coords[0], b_center_coords[2])
							
							dist = sqrt((b_center_pose[0] - a_center_pose[0])**2 + (b_center_pose[1] - a_center_pose[1])**2)
							obj_obj_dict[a_cat][b_cat].append(dist)
							obj_obj_dict[b_cat][a_cat].append(dist)

#assert 1==2

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




