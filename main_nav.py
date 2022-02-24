import numpy as np
from navigator import nav
from baseline_utils import create_folder

scene_name = 'Allensville_0'

output_folder = 'output/TESTING_RESULTS'
scene_output_folder = f'{output_folder}/{scene_name}'
create_folder(scene_output_folder)
testing_data = np.load(f'output/TESTING_DATA/testing_episodes_{scene_name}.npy', allow_pickle=True)

#for idx, data in enumerate(testing_data):
for idx in [3]:
	data = testing_data[idx]
	print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
	start_pose, target_cat, target_poses = data
	#start_pose = (6.6, -6.9)
	#target_cat = 'refrigerator'
	saved_folder = f'{scene_output_folder}/eps_{idx}_{target_cat}'
	create_folder(saved_folder, clean_up=True)
	#try:
	nav(idx, scene_name, start_pose, target_poses, target_cat, saved_folder)
	#except:
	#print(f'CCCCCCCCCCCCCC failed EPS {idx} DDDDDDDDDDDDDDD')