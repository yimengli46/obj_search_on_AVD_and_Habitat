import numpy as np
import cv2
import matplotlib.pyplot as plt
from baseline_utils import read_map_npy, pose_to_coords

scene_name = 'Allensville_0'

sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'
scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name[:-2]}.npz', allow_pickle=True)['output'].item()

dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'

scene_height_npy = np.load(f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy', allow_pickle=True).item()
sem_map_folder = f'output/semantic_map/{scene_name}'
#================================================================================================================

sem_map_npy = np.load(f'{sem_map_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)

cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

roomType_to_id_dict = {}
roomType_to_id_dict['bathroom'] = 1
roomType_to_id_dict['bedroom'] = 2
roomType_to_id_dict['corridor'] = 3
roomType_to_id_dict['dining_room'] = 4
roomType_to_id_dict['kitchen'] = 5
roomType_to_id_dict['living_room'] = 6
roomType_to_id_dict['lobby'] = 7

room_map = np.zeros(cropped_semantic_map.shape, dtype=int)

room_ids = list(scene_graph_npz['room'].keys())
rooms = []
for room_id in room_ids:
	room = scene_graph_npz['room'][room_id]
	print(room['scene_category'])

	x, z, y = room['location']
	cat = room['scene_category']
	size_x, size_z, size_y = room['size']

	x_coord, z_coord = pose_to_coords((x, z), pose_range, coords_range)
	x1 = x_coord - int(size_x/2/0.1)
	x2 = x_coord + int(size_x/2/0.1)
	z1 = z_coord - int(size_z/2/0.1)
	z2 = z_coord + int(size_z/2/0.1)
	print(f'x1 = {x1}, x2 = {x2}, z1 = {z1}, z2 = {z2}')

	room_map[z1:z2, x1:x2] = roomType_to_id_dict[cat]

	my_room = {}
	my_room['cat'] = cat
	my_room['center'] = (x_coord, z_coord)
	rooms.append(my_room)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.imshow(room_map)

for room in rooms:
	cat = room['cat']
	center = room['center']
	ax.text(center[0], center[1], cat)

fig.tight_layout()
plt.show()