import numpy as np
import cv2
import matplotlib.pyplot as plt
from baseline_utils import read_map_npy, pose_to_coords, apply_color_to_map, create_folder

scene_list        = ['Allensville_0']
graph_scene_list  = ['Allensville_0']
#scene_list       = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0']
#graph_scene_list = ['Collierville_1', 'Darden_1', 'Markleeville_1', 'Wiconisco_1']

floor_list = ['A', 'B', 'C', 'D']
scene_heights_dict = np.load(f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy', allow_pickle=True).item()
all_objs_list = list(np.load(f'output/semantic_prior/all_objs_list.npy', allow_pickle=True))
cat2id_dict = {all_objs_list[i]:i+1 for i in range(len(all_objs_list))}

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
roomType_to_id_dict['storage_room'] = 15

for scene_idx, scene_name in enumerate(scene_list):
	print(f'scene_name = {scene_name}')
	saved_folder = f'output/gt_semantic_map_from_SceneGraph/{scene_name}'
	create_folder(saved_folder)

	sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'
	scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name[:-2]}.npz', allow_pickle=True)['output'].item()

	sem_map_folder = f'output/semantic_map/{scene_name}'

	floor_idx = int(graph_scene_list[scene_idx][-1])

	#==================================== initialize the map ============================================================
	sem_map_npy = np.load(f'{sem_map_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
	semantic_map, pose_range, coords_range = read_map_npy(sem_map_npy)

	#cropped_semantic_map = semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]

	room_map = np.zeros(semantic_map.shape, dtype=int)*40

	#=========================================== visualize room as rectangles=========================================
	room_ids = list(scene_graph_npz['room'].keys())
	rooms = []
	room_id_at_this_floor = []
	for room_id in room_ids:
		room = scene_graph_npz['room'][room_id]
		if room['floor_number'] == floor_list[floor_idx]:
			room_id_at_this_floor.append(room['id'])

			#print(room['scene_category'])
			'''
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
			room_map[z1:z2, x1] = roomType_to_id_dict[cat]
			room_map[z1:z2, x2] = roomType_to_id_dict[cat]
			room_map[z1, x1:x2] = roomType_to_id_dict[cat]
			room_map[z2, x1:x2+1] = roomType_to_id_dict[cat]

			#full_map[z1:z2, x1:x2] = roomType_to_id_dict[cat]

			my_room = {}
			my_room['cat'] = cat
			my_room['center'] = (x_coord, z_coord)
			my_room['coords'] = (x1, z1, x2, z2)
			rooms.append(my_room)
			'''

	#======================================visualize objects as full rectangles========================
	obj_ids = list(scene_graph_npz['object'].keys())
	objs = []
	for obj_id in obj_ids:
		obj = scene_graph_npz['object'][obj_id]
		if obj['parent_room'] in room_id_at_this_floor:
			print(obj['class_'])

			x, z, y = obj['location']
			cat = obj['class_']
			size_x, size_z, size_y = obj['size']

			x_coord, z_coord = pose_to_coords((x, z), pose_range, coords_range)
			x1 = x_coord - int(size_x/2/0.1)
			x2 = x_coord + int(size_x/2/0.1)
			z1 = z_coord - int(size_z/2/0.1)
			z2 = z_coord + int(size_z/2/0.1)
			print(f'x1 = {x1}, x2 = {x2}, z1 = {z1}, z2 = {z2}')

			room_map[z1:z2, x1:x2] = cat2id_dict[cat]

			my_obj = {}
			my_obj['cat'] = cat
			my_obj['center'] = (x_coord, z_coord)
			objs.append(my_obj)

	color_room_map = apply_color_to_map(room_map)

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
	ax.imshow(color_room_map)

	'''
	for room in rooms:
		cat = room['cat']
		center = room['center']
		x1, z1, x2, _ = room['coords']
		text_x = x1 + (x2-x1)*0.3
		text_z = z1-1
		ax.text(text_x, text_z, cat, color='yellow', size=20)
	'''

	for obj in objs:
		cat = obj['cat']
		center = obj['center']
		ax.text(center[0], center[1], cat)

	fig.tight_layout()
	plt.show()

	map_dict = {}
	map_dict['min_x'] = coords_range[0]
	map_dict['max_x'] = coords_range[2]
	map_dict['min_z'] = coords_range[1]
	map_dict['max_z'] = coords_range[3]
	map_dict['min_X'] = pose_range[0]
	map_dict['max_X'] = pose_range[2]
	map_dict['min_Z'] = pose_range[1]
	map_dict['max_Z'] = pose_range[3]
	map_dict['semantic_map'] = room_map
	np.save(f'{saved_folder}/gt_semantic_map.npy', map_dict)

	#assert 1==2