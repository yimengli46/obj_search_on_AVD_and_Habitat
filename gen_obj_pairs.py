import numpy as np
import cv2
import matplotlib.pyplot as plt
from baseline_utils import create_folder

scene_heights_dict = np.load(f'/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy', allow_pickle=True).item()
scene_names = list(scene_heights_dict.keys())

saved_folder = f'output/semantic_prior'
create_folder(saved_folder, clean_up=False)

'''
for scene_name in scene_names:
	print(f'scene_name = {scene_name}')

	list_obj_tuples = []
	sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'
	scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name}.npz', allow_pickle=True)['output'].item()

	room_ids = list(scene_graph_npz['room'].keys())
	#==========================go through rooms ===========================#
	for room_id in room_ids:
		room_type = scene_graph_npz['room'][room_id]['scene_category']
		
		#====================== collect objs in this room =================#
		obj_ids = list(scene_graph_npz['object'].keys())
		objs = []
		for obj_id in obj_ids:
			obj = scene_graph_npz['object'][obj_id]
			if obj['parent_room'] == room_id:
				x, z, y = obj['location']
				cat = obj['class_']
				size_x, size_z, size_y = obj['size']
				objs.append((cat, (x,z,y), (size_x, size_z, size_y)))

		#===================== form obj pairs in this room =================#
		if len(objs) == 1: # only one object in the room
			obj_tuple = (objs[0][0], objs[0][1], objs[0][2], room_type)
			list_obj_tuples.append(obj_tuple)
		elif len(objs) > 1:
			for i in range(len(objs)-1):
				for j in range(i+1, len(objs)):
					o1 = objs[i]
					o2 = objs[j]
					obj_tuple = (o1[0], o1[1], o1[2], o2[0], o2[1], o2[2], room_type)
					list_obj_tuples.append(obj_tuple)

	np.save(f'{saved_folder}/{scene_name}_prior.npy', list_obj_tuples)
'''

'''
list_obj = []
for scene_name in scene_names:
	print(f'scene_name = {scene_name}')

	sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'
	scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name}.npz', allow_pickle=True)['output'].item()
	
	obj_ids = list(scene_graph_npz['object'].keys())
	objs = []
	for obj_id in obj_ids:
		obj = scene_graph_npz['object'][obj_id]	
		cat = obj['class_']
		if cat not in list_obj:
			list_obj.append(cat)		

np.save(f'{saved_folder}/all_objs_list.npy', list_obj)
'''

list_room = []
for scene_name in scene_names:
	print(f'scene_name = {scene_name}')

	sceneGraph_npz_folder = '/home/yimeng/Datasets/3DSceneGraph/3DSceneGraph_tiny/data/automated_graph'
	scene_graph_npz = np.load(f'{sceneGraph_npz_folder}/3DSceneGraph_{scene_name}.npz', allow_pickle=True)['output'].item()

	room_ids = list(scene_graph_npz['room'].keys())
	#==========================go through rooms ===========================#
	for room_id in room_ids:
		room_type = scene_graph_npz['room'][room_id]['scene_category']
		if room_type not in list_room:
			list_room.append(room_type)

np.save(f'{saved_folder}/room_type_list.npy', list_room)