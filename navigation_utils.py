import numpy as np
import numpy.linalg as LA
import cv2
import math
import matplotlib.patches as patches
import networkx as nx
import random
import habitat

dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
scene_name = 'Allensville_0'

img_act_dict = np.load('{}/{}/img_act_dict.npy'.format(dataset_dir, scene_name), allow_pickle=True).item()
all_img_names = list(img_act_dict.keys())

graph = nx.DiGraph()
for img_id in list(img_act_dict.keys()):
	graph.add_node(img_id)
for img_id in list(img_act_dict.keys()):
	for action in ['forward', 'rotate_ccw', 'rotate_cw']:
		next_img_id = img_act_dict[img_id][action]
		if next_img_id != '':
			graph.add_edge(img_id, next_img_id)

def get_obs(img_name):
	rgb_img = cv2.imread(f'{dataset_dir}/{scene_name}/rgb/{img_name}.jpg', 1)[:, :, ::-1]
	depth_img = cv2.imread(f'{dataset_dir}/{scene_name}/depth/{img_name}.png', cv2.IMREAD_UNCHANGED)
	depth_img = depth_img/256.
	depth_img = cv2.blur(depth_img, (3,3))
	return rgb_img, depth_img

def get_pose(img_name):
	pose = img_act_dict[img_name]['pose'] # x, z, theta
	return pose

def choose_direction(img_name, depth_panor, cur_node=None, num_directions=12):
	img_id = img_name[:6]
	arr_slice_depth = np.zeros(12)
	for i in range(num_directions):
		slice_depth = depth_panor[100:150, 64*i:64*(i+1)]
		arr_slice_depth[i] = np.mean(slice_depth.flatten())

	arr_idx_maximum_depth = np.argsort(arr_slice_depth)[::-1]
	print(arr_idx_maximum_depth)
	print(cur_node.angle_flag)
	angle_lst = ['000', '030', '060', '090', '120', '150', '180', '210', '240', '270', '300', '330']
	angle_lst = angle_lst[::-1]
	for idx_maximum_depth in list(arr_idx_maximum_depth):
		angle_maximum_depth = angle_lst[idx_maximum_depth]
		if not cur_node.angle_flag[angle_maximum_depth]:
			img_name_maximum_depth = img_id + angle_maximum_depth
			#print('***************start from here')
			cur_node.angle_flag[angle_maximum_depth] = True
			return img_name_maximum_depth

		'''
		img_name_maximum_depth = img_id + angle_maximum_depth
		# make sure the node is not None
		if img_name_maximum_depth in all_img_names:
			if traverse_lst is not None and img_name_maximum_depth not in traverse_lst:
				print('***************start from here')
				return img_name_maximum_depth
		'''

def random_move(img_name):
	while True:
		action = random.choice(['forward', 'rotate_ccw', 'rotate_cw'])
		next_img_name = img_act_dict[img_name][action]
		if len(next_img_name) > 0:
			return next_img_name

def move(img_name):
	next_img_name = img_act_dict[img_name]['forward']
	return next_img_name

def move_towards_target(cur_img_id, target_img_id):
	path = nx.shortest_path(graph, cur_img_id, target_img_id)
	next_img_id = path[1]
	return next_img_id

def get_obs_panor(img_name):
	rgb_panor = np.zeros((256, 64*12, 3), dtype='uint8')
	depth_panor = np.zeros((256, 64*12), dtype='float32')

	img_id = img_name[:6]
	for i, angle in enumerate(['000', '030', '060', '090', '120', '150', '180', '210', '240', '270', '300', '330']):
		cur_img_name = img_id + angle
		if cur_img_name in list(img_act_dict.keys()):
			rgb_img = cv2.imread('{}/{}/images/{}.jpg'.format(dataset_dir, scene_name, cur_img_name), 1)[:, :, ::-1]
			npy_file = np.load('{}/{}/others/{}.npy'.format(dataset_dir, scene_name, cur_img_name), allow_pickle=True).item()
			depth_img = npy_file['depth']
			j = 11 - i
			rgb_panor[:, 64*j:64*(j+1), :] = cylindrical_panorama_rgb(rgb_img)
			depth_panor[:, 64*j:64*(j+1)] = cylindrical_panorama_depth(depth_img)

	return rgb_panor, depth_panor

def vis_chosen_direction(depth_panor, cur_img_id, robot, ax):
	cur_img_angle = cur_img_id[6:]
	# visualize the explored direction
	node = robot.get_node(robot.cur_img_id)
	for i, angle in enumerate(['000', '030', '060', '090', '120', '150', '180', '210', '240', '270', '300', '330']):
		if node.angle_flag[angle]:
			j = 11 - i
			rect = patches.Rectangle((64*j, 0), 64, 256, linewidth=5, edgecolor='y', facecolor='none')
			ax.add_patch(rect)
	# visualize the selected direction
	for i, angle in enumerate(['000', '030', '060', '090', '120', '150', '180', '210', '240', '270', '300', '330']):
		if cur_img_angle == angle:
			j = 11 - i
			rect = patches.Rectangle((64*j, 0), 64, 256, linewidth=5, edgecolor='r', facecolor='none')
			ax.add_patch(rect)
	return ax

def cylindrical_panorama_rgb(img, focal_length=128):
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])

	"""This function returns the cylindrical warp for a given image and intrinsics matrix K"""
	h, w = img.shape[:2]
	y_i, x_i = np.indices((h, w))
	X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h * w, 3) # to homog
	Kinv = np.linalg.inv(K) 
	X = Kinv.dot(X.T).T # normalized coords
	# calculate cylindrical coords (sin\theta, h, cos\theta)
	A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w * h, 3)
	# project back to image-pixels plane
	B = K.dot(A.T).T 
	# back from homog coords
	B = B[:, :-1] / B[:, [-1]]
	# make sure warp coords only within image bounds
	B[(B[:, 0] < 0) | (B[:, 0] >= w) | (B[:, 1] < 0) | (B[:, 1] >= h)] = -1
	B = B.reshape(h, w, -1)

	img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # for transparent borders...
	# warp the image according to cylindrical coords
	result =  cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, 
		borderMode=cv2.BORDER_TRANSPARENT)

	result = result[:, 96:256-96, :3]

	return result

def cylindrical_panorama_depth(img, focal_length=128):
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])

	"""This function returns the cylindrical warp for a given image and intrinsics matrix K"""
	h, w = img.shape[:2]
	
	y_i, x_i = np.indices((h, w))
	X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h * w, 3) # to homog
	Kinv = np.linalg.inv(K) 
	X = Kinv.dot(X.T).T # normalized coords
	# calculate cylindrical coords (sin\theta, h, cos\theta)
	A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w * h, 3)
	# project back to image-pixels plane
	B = K.dot(A.T).T 
	# back from homog coords
	B = B[:, :-1] / B[:, [-1]]
	# make sure warp coords only within image bounds
	B[(B[:, 0] < 0) | (B[:, 0] >= w) | (B[:, 1] < 0) | (B[:, 1] >= h)] = -1
	B = B.reshape(h, w, -1)

	# warp the image according to cylindrical coords
	result =  cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, 
		borderMode=cv2.BORDER_CONSTANT)

	result = result[:, 96:256-96]

	return result

def read_map_npy(map_npy):
	min_x = map_npy['min_x']
	max_x = map_npy['max_x']
	min_z = map_npy['min_z']
	max_z = map_npy['max_z']
	min_X = map_npy['min_X']
	max_X = map_npy['max_X']
	min_Z = map_npy['min_Z']
	max_Z = map_npy['max_Z']
	semantic_map = map_npy['semantic_map']
	return semantic_map, (min_X, min_Z, max_X, max_Z), (min_x, min_z, max_x, max_z)

def change_brightness(img, flag, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #lim = 255 - value
    #v[v > lim] = 255
    #v[v <= lim] += value

    v[np.logical_and(flag == False, v > value)] -= value
    v[np.logical_and(flag == False, v <= value)] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

class SimpleRLEnv(habitat.RLEnv):
	def get_reward_range(self):
		return [-1, 1]

	def get_reward(self, observations):
		return 0

	def get_done(self, observations):
		return self.habitat_env.episode_over

	def get_info(self, observations):
		return self.habitat_env.get_metrics()
