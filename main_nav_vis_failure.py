import numpy as np
from navigator import nav
from baseline_utils import create_folder
import habitat
import habitat_sim
from navigation_utils import SimpleRLEnv, get_scene_name

#scene_list = ['Allensville_0']
scene_list = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0']
scene_dict = {}
for scene in scene_list:
	scene_name = scene[:-2]
	floor = int(scene[-1])
	temp = {}
	temp['name'] = scene
	temp['floor'] = floor 
	scene_dict[scene_name] = temp

scene_heights_dict = np.load(f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy', allow_pickle=True).item()

#================================ load habitat env============================================
config = habitat.get_config(config_paths="/home/yimeng/Datasets/habitat-lab/configs/tasks/devendra_objectnav_gibson.yaml")
config.defrost()
config.DATASET.DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/objectnav/gibson/all.json.gz'
config.DATASET.SCENES_DIR = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
#config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#config.TASK.SENSORS.append("HEADING_SENSOR")
config.freeze()
env = SimpleRLEnv(config=config)

for episode_id in range(5):
	env.reset()
	print('episode_id = {}'.format(episode_id))
	print('env.current_episode = {}'.format(env.current_episode))

	scene_name_no_floor = get_scene_name(env.current_episode)

	if scene_name_no_floor in scene_dict:
		scene_name = scene_dict[scene_name_no_floor]['name']
		floor_id   = scene_dict[scene_name_no_floor]['floor']
	
		height = scene_heights_dict[scene_name_no_floor][floor_id]
	
		#=============================== traverse each floor ===========================
		print(f'*****scene_name = {scene_name}***********')

		output_folder = 'output/TESTING_RESULTS_VIS_FAILURE'
		scene_output_folder = f'{output_folder}/{scene_name}'
		create_folder(scene_output_folder)
		testing_data = np.load(f'output/TESTING_DATA/testing_episodes_{scene_name}.npy', allow_pickle=True)

		existing_results = np.load(f'output/TESTING_RESULTS/results_{scene_name}.npy', allow_pickle=True).item()

		#'''
		results = {}
		for idx in range(len(testing_data)):
		#for idx in range(1, 2):
			if not existing_results[idx]['success']:
				data = testing_data[idx]
				print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
				start_pose, target_cat, targets = data
				#start_pose = (6.6, -6.9)
				#target_cat = 'refrigerator'
				saved_folder = f'{scene_output_folder}/eps_{idx}_{target_cat}'
				create_folder(saved_folder, clean_up=True)
				flag = False
				steps = 0
				#try:
				flag, steps = nav(env, idx, scene_name, height, start_pose, targets, target_cat, saved_folder)
				#except:
				print(f'CCCCCCCCCCCCCC failed EPS {idx} DDDDDDDDDDDDDDD')

				result = {}
				result['eps_id'] = idx
				result['target'] = target_cat
				result['steps'] = steps
				result['success'] = flag

				results[idx] = result

		np.save(f'{output_folder}/results_{scene_name}.npy', results)
		#'''

env.close()