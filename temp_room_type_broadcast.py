import numpy as np 
import matplotlib.pyplot as plt 
import skimage.measure
from core import cfg
from baseline_utils import read_map_npy, apply_color_to_map

sem_map_room_npy = np.load(f'{cfg.SAVE.SEM_MAP_FROM_SCENE_GRAPH_PATH}/Allensville_0/gt_semantic_map_rooms.npy', allow_pickle=True).item()
gt_semantic_map_room, _, _ = read_map_npy(sem_map_room_npy)
observed_room = gt_semantic_map_room.copy()

observed_area_flag = np.zeros(gt_semantic_map_room.shape, dtype=bool)
observed_area_flag[35:50, 60:75] = True

IGNORED_CLASS = cfg.SEM_MAP.IGNORED_ROOM_CLASS
cat_binary_map = gt_semantic_map_room.copy()
for cat in IGNORED_CLASS:
	cat_binary_map = np.where(cat_binary_map==cat, -1, cat_binary_map)
# run skimage to find the number of objects belong to this class
instance_label, num_ins = skimage.measure.label(cat_binary_map, background=-1, connectivity=1, return_num=True)

list_instances = []
for idx_ins in range(1, num_ins+1):
	mask_ins = (instance_label == idx_ins)
	
	mask_ins_and_observed_area = np.logical_and(mask_ins, observed_area_flag)
	if np.sum(mask_ins_and_observed_area) == 0:
		observed_room[mask_ins] = 0


color_gt_semantic_map_room = apply_color_to_map(gt_semantic_map_room)
color_observed_room = apply_color_to_map(observed_room)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
ax[0].imshow(color_gt_semantic_map_room)
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title("rgb")
ax[1].imshow(observed_area_flag)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title("sseg")
ax[2].imshow(color_observed_room)
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title("depth")
fig.tight_layout()
plt.show()