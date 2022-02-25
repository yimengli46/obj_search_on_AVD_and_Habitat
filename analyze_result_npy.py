import numpy as np 

#scene_list = ['Allensville_0']
scene_list = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0']
for scene_name in scene_list:
	output_folder = 'output/TESTING_RESULTS'

	results_npy = np.load(f'{output_folder}/results_{scene_name}.npy', allow_pickle=True).item()
	num_test = len(results_npy.keys())

	suc_list = []
	for i in range(num_test):
		result = results_npy[i]
		flag_suc = result['success']
		suc_list.append(flag_suc)

	suc_list = np.array(suc_list)
	suc_rate = np.sum(suc_list) / num_test

	# compute SPL
	SPL = 0.5 * np.sum(suc_list) / num_test

	print(f'scene_name = {scene_name}, suc_rate = {suc_rate}, SPL = {SPL}')


