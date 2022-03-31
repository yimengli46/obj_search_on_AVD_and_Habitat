import numpy as np
import cv2
import matplotlib.pyplot as plt
from baseline_utils import read_map_npy, pose_to_coords, apply_color_to_map, create_folder

#scene_list        = ['Allensville_0']
#graph_scene_list  = ['Allensville_0']
scene_list       = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0']
graph_scene_list = ['Collierville_1', 'Darden_1', 'Markleeville_1', 'Wiconisco_1']

floor_list = ['A', 'B', 'C', 'D']
scene_heights_dict = np.load(f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy', allow_pickle=True).item()
semantic_prior_folder = f'output/semantic_prior'
all_objs_list = list(np.load(f'{semantic_prior_folder}/all_objs_list.npy', allow_pickle=True))
room_list = np.load(f'{semantic_prior_folder}/room_type_list.npy', allow_pickle=True)

roomType_to_id_dict = {room_list[i]:i+1 for i in range(len(room_list))}

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
			if cat != 'corridor':
				room_map[z1:z2, x1:x2] = roomType_to_id_dict[cat]

				my_room = {}
				my_room['cat'] = cat
				my_room['center'] = (x_coord, z_coord)
				my_room['coords'] = (x1, z1, x2, z2)
				rooms.append(my_room)
	
	# search for corridor
	for room_id in room_ids:
		room = scene_graph_npz['room'][room_id]
		if room['floor_number'] == floor_list[floor_idx]:
			room_id_at_this_floor.append(room['id'])

			x, z, y = room['location']
			cat = room['scene_category']
			size_x, size_z, size_y = room['size']

			x_coord, z_coord = pose_to_coords((x, z), pose_range, coords_range)
			x1 = x_coord - int(size_x/2/0.1)
			x2 = x_coord + int(size_x/2/0.1)
			z1 = z_coord - int(size_z/2/0.1)
			z2 = z_coord + int(size_z/2/0.1)
			
			if cat == 'corridor':
				corridor_map = np.zeros(semantic_map.shape, dtype=int)
				corridor_map[z1:z2, x1:x2] = roomType_to_id_dict[cat]
				mask_unoccupied = (room_map == 0)
				mask_corridor = np.logical_and(mask_unoccupied, corridor_map > 0)
				room_map[mask_corridor] = roomType_to_id_dict[cat]

				my_room = {}
				my_room['cat'] = cat
				my_room['center'] = (x_coord, z_coord)
				my_room['coords'] = (x1, z1, x2, z2)
				rooms.append(my_room)

	color_room_map = apply_color_to_map(room_map)

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
	ax.imshow(color_room_map)

	#'''
	for room in rooms:
		cat = room['cat']
		center = room['center']
		x1, z1, x2, _ = room['coords']
		text_x = x1 + (x2-x1)*0.3
		text_z = z1-1
		ax.text(text_x, text_z, cat, color='yellow', size=20)
	#'''

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
	np.save(f'{saved_folder}/gt_semantic_map_rooms.npy', map_dict)

	#assert 1==2