import numpy as np
from navigator import nav
from baseline_utils import create_folder
import habitat
import habitat_sim
from navigation_utils import SimpleRLEnv

scene_name = 'Allensville_0'

output_folder = 'output/TESTING_RESULTS'
scene_output_folder = f'{output_folder}/{scene_name}'
create_folder(scene_output_folder)
testing_data = np.load(f'output/TESTING_DATA/testing_episodes_{scene_name}.npy', allow_pickle=True)

#================================ load habitat env============================================
config = habitat.get_config(config_paths="/home/yimeng/Datasets/habitat-lab/configs/tasks/devendra_objectnav_gibson.yaml")
config.defrost()
config.DATASET.DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/objectnav/gibson/all.json.gz'
config.DATASET.SCENES_DIR = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
#config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
#config.TASK.SENSORS.append("HEADING_SENSOR")
config.freeze()
env = SimpleRLEnv(config=config)
obs = env.reset()

results = {}
for idx, data in enumerate(testing_data):
#for idx in range(2):
	data = testing_data[idx]
	print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
	start_pose, target_cat, targets = data
	#start_pose = (6.6, -6.9)
	#target_cat = 'refrigerator'
	saved_folder = f'{scene_output_folder}/eps_{idx}_{target_cat}'
	create_folder(saved_folder, clean_up=True)
	flag = False
	steps = 0
	try:
		flag, steps = nav(env, idx, scene_name, start_pose, targets, target_cat, saved_folder)
	except:
		print(f'CCCCCCCCCCCCCC failed EPS {idx} DDDDDDDDDDDDDDD')

	result = {}
	result['eps_id'] = idx
	result['target'] = target_cat
	result['steps'] = steps
	result['success'] = flag

	results[idx] = result

np.save(f'{output_folder}/results_{scene_name}', results)

env.close()