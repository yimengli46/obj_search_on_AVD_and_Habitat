import numpy as np
import numpy.linalg as LA
import cv2
from semantic_prediction import SemanticPredMaskRCNN
from baseline_utils import create_folder
from panoptic_prediction import PanopPred
import matplotlib.pyplot as plt

'''
dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_list = ['Allensville_0']

# initialize object detector
sem_pred = SemanticPredMaskRCNN()

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{dataset_dir}/{scene_name}/detectron2_pred'
	create_folder(saved_folder, clean_up=False)

	# load img list
	img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	img_names = list(img_act_dict.keys())

	for idx, img_name in enumerate(img_names):

		print('idx = {}'.format(idx))
		#====================================== load rgb image, depth and sseg ==================================
		rgb_img = cv2.imread(f'{dataset_dir}/{scene_name}/rgb/{img_name}.jpg', 1)[:, :, ::-1]

		semantic_pred, _ = sem_pred.get_prediction(rgb_img, flag_vis=False)

		np.save(f'{saved_folder}/{img_name}.npy', semantic_pred)

		#assert 1==2
'''

dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_list = ['Allensville_0']

# initialize object detector
sem_pred = PanopPred()

for scene_id in range(len(scene_list)):
	print(f'scene_id = {scene_id}')
	scene_name = scene_list[scene_id]

	saved_folder = f'{dataset_dir}/{scene_name}/panoptic_pred'
	create_folder(saved_folder, clean_up=False)

	# load img list
	img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
	img_names = list(img_act_dict.keys())

	for idx, img_name in enumerate(img_names):
	#for idx, img_name in enumerate(['077122135', '077122180', '079120135', '089126315']):

		print('idx = {}'.format(idx))
		#====================================== load rgb image, depth and sseg ==================================
		rgb_img = cv2.imread(f'{dataset_dir}/{scene_name}/rgb/{img_name}.jpg', 1)[:, :, ::-1]

		semantic_pred, img = sem_pred.get_prediction(rgb_img, flag_vis=True)
		#plt.imshow(img)
		#plt.show()

		cv2.imwrite(f'{saved_folder}/{img_name}.png', semantic_pred)

		#assert 1==2