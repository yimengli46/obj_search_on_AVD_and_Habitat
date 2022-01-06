import numpy as np
import matplotlib.pyplot as plt

scene_heights_dict = np.load(f'../Datasets/Discretized_Gibson/scene_heights.npy', allow_pickle=True).item()
scene_names = list(scene_heights_dict.keys())

semantic_prior_folder = f'output/semantic_prior'
obj_list = np.load(f'{semantic_prior_folder}/all_objs_list.npy', allow_pickle=True)

'''
room_types = ['bathroom', 'bedroom', 'corridor', 'dining_room', 'kitchen', 'living_room', 'home_office']

for room_type in room_types:
	obj_obj_dict = {}
	for k1 in obj_list:
		obj_obj_dict[k1] = {}
		for k2 in obj_list:
			obj_obj_dict[k1][k2] = 0

	for scene_name in scene_names:
		npy_file = np.load(f'{semantic_prior_folder}/{scene_name}_prior.npy', allow_pickle=True)

		for idx in range(len(npy_file)):
			tup = npy_file[idx]
			if len(tup) > 4:
				o1 = tup[0]
				o2 = tup[3]
				room = tup[6]
				if room == room_type:
					obj_obj_dict[o1][o2] += 1
					obj_obj_dict[o2][o1] += 1

	num_classes = len(obj_list)
	corr_mat_mu = np.zeros((num_classes, num_classes))
	for idx_k1, k1 in enumerate(obj_list):
		for idx_k2, k2 in enumerate(obj_list):
			corr_mat_mu[idx_k1][idx_k2] = obj_obj_dict[k1][k2]

	max_val = np.max(corr_mat_mu)
	corr_mat_mu /= max_val

	# write correlation matrix image
	ticks = []
	for i in range(len(obj_list)):
		ticks.append(1*i + 0.5)

	plt.figure()
	plt.title(f'{room_type}')
	plt.imshow(corr_mat_mu)
	plt.yticks(ticks, list(obj_obj_dict.keys()))
	plt.xticks(ticks, list(obj_obj_dict.keys()), rotation='vertical')
	plt.colorbar()
	plt.tight_layout()
	plt.show()
	#plt.savefig(f'{semantic_prior_folder}/{room_type}_corr.jpg', dpi=400)
	#plt.close()
	#assert 1==2
'''


obj_obj_dict = {}
for k1 in obj_list:
	obj_obj_dict[k1] = {}
	for k2 in obj_list:
		obj_obj_dict[k1][k2] = 0

for scene_name in scene_names:
	npy_file = np.load(f'{semantic_prior_folder}/{scene_name}_prior.npy', allow_pickle=True)

	for idx in range(len(npy_file)):
		tup = npy_file[idx]
		if len(tup) > 4:
			o1 = tup[0]
			o2 = tup[3]
			room = tup[6]
			
			obj_obj_dict[o1][o2] += 1
			obj_obj_dict[o2][o1] += 1

num_classes = len(obj_list)
corr_mat_mu = np.zeros((num_classes, num_classes))
for idx_k1, k1 in enumerate(obj_list):
	for idx_k2, k2 in enumerate(obj_list):
		corr_mat_mu[idx_k1][idx_k2] = obj_obj_dict[k1][k2]

sum_corr = np.sum(corr_mat_mu, axis=1)

# compute weight
weight_prior = {}
for idx, k, in enumerate(obj_list):
	weight_prior[k] = []

	sum_row = sum_corr[idx] - corr_mat_mu[idx][idx]
	if sum_row > 0:
		for j, k2 in enumerate(obj_list):
			if corr_mat_mu[idx][j] > 0 and j != idx:
				weight = corr_mat_mu[idx][j] / sum_row
				weight_prior[k].append((k2, weight))

np.save(f'{semantic_prior_folder}/weight_prior.npy', weight_prior)

