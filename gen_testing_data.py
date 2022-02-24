import numpy as np
import cv2
import matplotlib.pyplot as plt
from baseline_utils import read_map_npy, pose_to_coords, apply_color_to_map, pxl_coords_to_pose, get_class_mapper, pose_to_coords_numpy
from navigation_utils import change_brightness
import skimage.measure
from math import floor, sqrt
import random
from localNavigator_Astar import TreeList, PriorityQueue, AStarSearch

scene_name = 'Allensville_0'

NUM_EXAMPLES = 100
cat2idx_dict = get_class_mapper()
idx2cat_dict = {v: k for k, v in cat2idx_dict.items()}
ALLOWED_CATS = ['couch', 'potted_plant', 'refrigerator', 'oven', 'tv', 'chair', 'vase', 'potted plant', \
	'toilet', 'clock', 'cup', 'bottle', 'bed']

SEED = 5
random.seed(SEED)
np.random.seed(SEED)

#============================== load gt semantic map and find obj centers =============================
sem_map_npy = np.load(f'output/gt_semantic_map_from_SceneGraph/{scene_name}/gt_semantic_map.npy', allow_pickle=True).item()
gt_semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)

H, W = gt_semantic_map.shape
observed_semantic_map = cv2.resize(gt_semantic_map, (int(W*10), int(H*10)), interpolation=cv2.INTER_NEAREST)
H, W = observed_semantic_map.shape
x = np.linspace(0, W-1, W)
y = np.linspace(0, H-1, H)
xv, yv = np.meshgrid(x, y)
IGNORED_CLASS = [0, 59]
cat_binary_map = observed_semantic_map.copy()
for cat in IGNORED_CLASS:
	cat_binary_map = np.where(cat_binary_map==cat, -1, cat_binary_map)
# run skimage to find the number of objects belong to this class
instance_label, num_ins = skimage.measure.label(cat_binary_map, background=-1, connectivity=1, return_num=True)

list_instances = []
for idx_ins in range(1, num_ins+1):
	mask_ins = (instance_label == idx_ins)
	if np.sum(mask_ins) > 100: # should have at least 50 pixels
		#print(f'idx_ins = {idx_ins}')
		x_coords = xv[mask_ins]
		y_coords = yv[mask_ins]
		ins_center = (floor(np.median(x_coords)*.1), floor(np.median(y_coords)*.1))
		ins_cat = observed_semantic_map[int(y_coords[0]), int(x_coords[0])]
		ins = {}
		ins['center'] = ins_center # in coordinates, rather than size
		ins['center_pose'] = pxl_coords_to_pose(ins_center, pose_range, coords_range, flag_cropped=True) # in pose
		ins['cat'] = idx2cat_dict[ins_cat]

		# compute object radius size
		dist = np.sqrt((x_coords * .1 - ins_center[0])**2 + (y_coords * .1 - ins_center[1])**2)
		size = np.max(dist)
		ins['size'] = size * .1 # in meters, not coordinates

		print(f'ins_center = {ins_center}, cat = {ins_cat}, class = {idx2cat_dict[ins_cat]}, size = {size}')
		list_instances.append(ins)

#================================ load occupancy map ===========================
# load occupancy map
occ_map_path = f'output/semantic_map/{scene_name}'
occupancy_map = np.load(f'{occ_map_path}/BEV_occupancy_map.npy')

H, W = occupancy_map.shape
x = np.linspace(0, W-1, W)
y = np.linspace(0, H-1, H)
xv, yv = np.meshgrid(x, y)
map_coords = np.stack((xv, yv), axis=2).astype(np.int16)
# erode the occupancy map so the free space is smaller
kernel = np.ones((3,3), np.uint8)
occupancy_map_erosion = cv2.erode(occupancy_map.astype(np.uint8), kernel, iterations=1)
mask_free = (occupancy_map_erosion == 1)
free_map_coords = map_coords[mask_free].tolist()

# ============================== FIND allowed categories in the current scene
EXISTING_CATS = set()
for idx, inst in enumerate(list_instances):
	cat = inst['cat']
	print(f'cat = {cat}')
	if cat in ALLOWED_CATS:
		EXISTING_CATS.add(cat)
EXISTING_CATS = list(EXISTING_CATS)

#=============================== sample examples ===================================
sem_map_npy = np.load(f'output/gt_semantic_map_from_SceneGraph/{scene_name}/gt_semantic_map.npy', allow_pickle=True).item()
gt_semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)
observed_area_flag = (occupancy_map > 0)
gt_semantic_map[observed_area_flag] = 59
color_gt_semantic_map = apply_color_to_map(gt_semantic_map)
#observed_area_flag = (occupancy_map > 0)
#color_gt_semantic_map = change_brightness(color_gt_semantic_map, observed_area_flag, value=60)
#color_gt_semantic_map[observed_area_flag] =

testing_data = []

for i in range(NUM_EXAMPLES):
	# sample a cat
	cat = random.choice(EXISTING_CATS)
	print(f'eps = {i}, target = {cat}')
	
	targets = []
	for idx, inst in enumerate(list_instances):
		if inst['cat'] == cat:
			targets.append((inst['center_pose'], inst['size'])) # (map pose, size)

	sampled_coords = random.choice(free_map_coords)
	map_pose = pxl_coords_to_pose(sampled_coords, pose_range, coords_range, flag_cropped=True)
	env_pose = (map_pose[0], -map_pose[1]) # starting pose is environment pose
	#env_pose = random.choice([(6.6, -6.9), (3.6, -4.5)])

	data_tuple = (env_pose, cat, targets)
	testing_data.append(data_tuple)

	# =============================== visualize the sampled data point ==================================
	'''
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
	ax.imshow(color_gt_semantic_map, vmax=5)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	# visualize target_poses
	np_target_poses = np.array(list(map(list, target_poses)))
	vis_target_coords = pose_to_coords_numpy(np_target_poses, pose_range, coords_range)
	ax.scatter(vis_target_coords[:, 0], vis_target_coords[:, 1], marker='*', s=50, c='yellow', zorder=1)
	# visualize start position
	vis_start_coords = pose_to_coords(map_pose, pose_range, coords_range)
	ax.scatter(vis_start_coords[0], vis_start_coords[1], marker='P', s=50, c='yellow', zorder=2)
	plt.title(f'target category: {cat}')
	plt.show()
	'''

np.save(f'output/TESTING_DATA/testing_episodes_{scene_name}.npy', testing_data)


