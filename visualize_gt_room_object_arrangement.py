import numpy as np
import cv2
import matplotlib.pyplot as plt
from baseline_utils import read_map_npy, pose_to_coords, apply_color_to_map

scene_name = 'Coffeen'

sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'
scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name}.npz', allow_pickle=True)['output'].item()

floor_list = ['A', 'B', 'C', 'D']
scene_heights_dict = np.load(f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy', allow_pickle=True).item()
height_list = scene_heights_dict[scene_name]
sorted_height_idx = np.argsort(height_list) # height idx will show up in increasing order

for floor_idx, height_idx in enumerate(sorted_height_idx):
	print(f'floor = {floor_list[floor_idx]}, height_idx = {height_idx}')

	#================================================================================================================
	dataset_dir = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/Gibson_Discretized_Dataset'
	cat2id_dict = np.load('{}/{}/category_id_dict.npy'.format(dataset_dir, f'{scene_name}_{height_idx}'), allow_pickle=True).item()

	sem_map_folder = f'output/semantic_map/{scene_name}_{height_idx}'
	sem_map_npy = np.load(f'{sem_map_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)

	cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

	#occ_map = np.load(f'{sem_map_folder}/BEV_occupancy_map.npy', allow_pickle=True)


	full_map = np.zeros(cropped_semantic_map.shape, dtype=int)
	#full_map[occ_map > 0] = 41 # free space on the map has class label 41

	#=========================================== visualize room as rectangles=========================================
	roomType_to_id_dict = {}
	roomType_to_id_dict['bathroom'] = 1
	roomType_to_id_dict['bedroom'] = 2
	roomType_to_id_dict['corridor'] = 3
	roomType_to_id_dict['dining_room'] = 4
	roomType_to_id_dict['kitchen'] = 5
	roomType_to_id_dict['living_room'] = 6
	roomType_to_id_dict['lobby'] = 7
	roomType_to_id_dict['childs_room'] = 8
	roomType_to_id_dict['closet'] = 9
	roomType_to_id_dict['home_office'] = 10
	roomType_to_id_dict['playroom'] = 11
	roomType_to_id_dict['staircase'] = 12
	roomType_to_id_dict['utility_room'] = 13
	roomType_to_id_dict['television_room'] = 14
	roomType_to_id_dict['empty_room'] = 14


	room_ids = list(scene_graph_npz['room'].keys())
	rooms = []
	room_id_at_this_floor = []
	for room_id in room_ids:
		room = scene_graph_npz['room'][room_id]
		if room['floor_number'] == floor_list[floor_idx]:
			room_id_at_this_floor.append(room['id'])

			#print(room['scene_category'])

			x, z, y = room['location']
			cat = room['scene_category']
			size_x, size_z, size_y = room['size']

			x_coord, z_coord = pose_to_coords((x, z), pose_range, coords_range)
			x1 = x_coord - int(size_x/2/0.1)
			x2 = x_coord + int(size_x/2/0.1)
			z1 = z_coord - int(size_z/2/0.1)
			z2 = z_coord + int(size_z/2/0.1)
			#print(f'x1 = {x1}, x2 = {x2}, z1 = {z1}, z2 = {z2}')

			# draw rectangle
			full_map[z1:z2, x1] = roomType_to_id_dict[cat]
			full_map[z1:z2, x2] = roomType_to_id_dict[cat]
			full_map[z1, x1:x2] = roomType_to_id_dict[cat]
			full_map[z2, x1:x2+1] = roomType_to_id_dict[cat]

			#full_map[z1:z2, x1:x2] = roomType_to_id_dict[cat]

			my_room = {}
			my_room['cat'] = cat
			my_room['center'] = (x_coord, z_coord)
			my_room['coords'] = (x1, z1, x2, z2)
			rooms.append(my_room)

	#======================================visualize objects as full rectangles========================
	obj_ids = list(scene_graph_npz['object'].keys())
	objs = []
	for obj_id in obj_ids:
		obj = scene_graph_npz['object'][obj_id]
		if obj['parent_room'] in room_id_at_this_floor:
			#print(obj['class_'])

			x, z, y = obj['location']
			cat = obj['class_']
			size_x, size_z, size_y = obj['size']

			x_coord, z_coord = pose_to_coords((x, z), pose_range, coords_range)
			x1 = x_coord - int(size_x/2/0.1)
			x2 = x_coord + int(size_x/2/0.1)
			z1 = z_coord - int(size_z/2/0.1)
			z2 = z_coord + int(size_z/2/0.1)
			#print(f'x1 = {x1}, x2 = {x2}, z1 = {z1}, z2 = {z2}')

			full_map[z1:z2, x1:x2] = cat2id_dict[cat]

			my_obj = {}
			my_obj['cat'] = cat
			my_obj['coords'] = (x1, z1, x2, z2)
			my_obj['center'] = (x_coord, z_coord)
			objs.append(my_obj)

	color_full_map = apply_color_to_map(full_map, num_classes=42)

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
	ax.imshow(color_full_map)

	for room in rooms:
		cat = room['cat']
		center = room['center']
		x1, z1, x2, _ = room['coords']
		text_x = x1 + (x2-x1)*0.3
		text_z = z1-1
		ax.text(text_x, text_z, cat, color='yellow', size=20)

	for obj in objs:
		cat = obj['cat']
		x1, z1, x2, _ = obj['coords']
		center = obj['center']
		text_x = x1 + (x2-x1)*0.4
		text_z = (z1 + z2)/2-1
		ax.text(center[0], center[1], cat, color='red', size=10)

	fig.tight_layout()
	plt.show()