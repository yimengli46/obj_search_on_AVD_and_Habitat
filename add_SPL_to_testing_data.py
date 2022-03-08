import numpy as np


scene_list = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0']


for scene_name in scene_list:
	appended_testing_data = []

	testing_data = np.load(f'output/TESTING_DATA/testing_episodes_{scene_name}.npy', allow_pickle=True)
	results_npy = np.load(f'output/TESTING_RESULTS/results_{scene_name}.npy', allow_pickle=True).item()

	for idx, data in enumerate(testing_data):
		#for idx in range(1, 2):
		data = testing_data[idx]
		result = results_npy[idx]
		print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
		start_pose, target_cat, targets = data
		gt_steps = result['optim_steps']

		data_tuple = (start_pose, target_cat, targets, gt_steps)

		appended_testing_data.append(data_tuple)

	np.save(f'output/TESTING_DATA/testing_episodes_{scene_name}.npy', appended_testing_data)
	#assert 1==2