#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import os
import random
import sys
from tracemalloc import start

import git
import imageio
import magnum as mn
import numpy as np
import json
import csv
import cv2
import open3d as o3d

#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

try:
	from habitat.utils.visualizations import maps
except:
	from habitat.utils.visualizations import maps


# In[2]:


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
	from habitat_sim.utils.common import d3_40_colors_rgb

	rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

	arr = [rgb_img]
	titles = ["rgb"]
	if semantic_obs.size != 0:
		semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
		semantic_img.putpalette(d3_40_colors_rgb.flatten())
		semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
		semantic_img = semantic_img.convert("RGBA")
		arr.append(semantic_img)
		titles.append("semantic")

	if depth_obs.size != 0:
		depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
		arr.append(depth_img)
		titles.append("depth")

	plt.figure(figsize=(12, 8))
	for i, data in enumerate(arr):
		ax = plt.subplot(1, 3, i + 1)
		ax.axis("off")
		ax.set_title(titles[i])
		plt.imshow(data)
	plt.show()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--no-display", dest="display", action="store_false")
	parser.add_argument("--no-make-video", dest="make_video", action="store_false")
	parser.set_defaults(show_video=True, make_video=True)
	args, _ = parser.parse_known_args()
	show_video = args.display
	display = args.display
	do_make_video = args.make_video
else:
	show_video = False
	do_make_video = False
	display = False


# In[3]:


# @title Configure Sim Settings

test_scene = "../mp3d_example_scene/17DRP5sb8fy.glb"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

sim_settings = {
	"width": 512,  # Spatial resolution of the observations
	"height": 512,
	"scene": test_scene,  # Scene path
	"default_agent": 0,
	"sensor_height": 1.5,  # Height of sensors in meters
	"color_sensor": rgb_sensor,  # RGB sensor
	"depth_sensor": depth_sensor,  # Depth sensor
	"semantic_sensor": semantic_sensor,  # Semantic sensor
	"seed": 1,  # used in the random navigation
	"enable_physics": False,  # kinematics only
}


# In[4]:


def make_cfg(settings):
	sim_cfg = habitat_sim.SimulatorConfiguration()
	sim_cfg.gpu_device_id = 0
	sim_cfg.scene_id = settings["scene"]
	sim_cfg.enable_physics = settings["enable_physics"]

	# Note: all sensors must have the same resolution
	sensor_specs = []

	color_sensor_spec = habitat_sim.CameraSensorSpec()
	color_sensor_spec.uuid = "color_sensor"
	color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
	color_sensor_spec.resolution = [settings["height"], settings["width"]]
	color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
	color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
	sensor_specs.append(color_sensor_spec)

	depth_sensor_spec = habitat_sim.CameraSensorSpec()
	depth_sensor_spec.uuid = "depth_sensor"
	depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
	depth_sensor_spec.resolution = [settings["height"], settings["width"]]
	depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
	depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
	sensor_specs.append(depth_sensor_spec)

	semantic_sensor_spec = habitat_sim.CameraSensorSpec()
	semantic_sensor_spec.uuid = "semantic_sensor"
	semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
	semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
	semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
	semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
	sensor_specs.append(semantic_sensor_spec)

	# Here you can specify the amount of displacement in a forward action and the turn angle
	agent_cfg = habitat_sim.agent.AgentConfiguration()
	agent_cfg.sensor_specifications = sensor_specs
	agent_cfg.action_space = {
		"move_forward": habitat_sim.agent.ActionSpec(
			"move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
		),
		"turn_left": habitat_sim.agent.ActionSpec(
			"turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"turn_right": habitat_sim.agent.ActionSpec(
			"turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
		),
		"do_nothing": habitat_sim.agent.ActionSpec(
			"turn_right", habitat_sim.agent.ActuationSpec(amount=0.0)
		),
	}

	return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# In[5]:


cfg = make_cfg(sim_settings)
# Needed to handle out of order cell run in Colab
try:  # Got to make initialization idiot proof
	sim.close()
except NameError:
	pass
sim = habitat_sim.Simulator(cfg)


# In[6]:


def print_scene_recur(scene, limit_output=10):
	print(
		f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
	)
	print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

	count = 0
	for level in scene.levels:
		print(
			f"Level id:{level.id}, center:{level.aabb.center},"
			f" dims:{level.aabb.sizes}"
		)
		for region in level.regions:
			print(
				f"Region id:{region.id}, category:{region.category.name()},"
				f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
			)
			for obj in region.objects:
				print(
					f"Object id:{obj.id}, category:{obj.category.name()},"
					f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
				)
				count += 1
				if count >= limit_output:
					return


# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene
print_scene_recur(scene)


# In[7]:


# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)


# In[8]:


output_path = '../output'


# In[9]:


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
	points_topdown = []
	bounds = pathfinder.get_bounds()
	for point in points:
		# convert 3D x,z to topdown x,y
		px = (point[0] - bounds[0][0]) / meters_per_pixel
		py = (point[2] - bounds[0][2]) / meters_per_pixel
		points_topdown.append(np.array([px, py]))
	return points_topdown

def convert_topdown_to_points(pathfinder, topdown, meters_per_pixel):
	points = []
	bounds = pathfinder.get_bounds()
	for point in topdown:
		# convert 3D x,z to topdown x,y
		px = (point[0] * meters_per_pixel) + bounds[0][0]
		py = (point[1] * meters_per_pixel) - bounds[0][2]
		points.append(np.array([px, height, py]))
	return points


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, show=True, store=False, output_path = None):
	plt.figure(figsize=(12, 8))
	ax = plt.subplot(1, 1, 1)
	ax.axis("off")
	plt.imshow(topdown_map)
	
	# plot points on map
	if key_points is not None:
		for point in key_points:
			#topdown_map[int(point[0])][int(point[1])] = 10
			#plt.show()
			plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
	if show == True:
		plt.show()
	if store == True:
		plt.savefig(output_path)
	plt.close()

meters_per_pixel = 0.1 

custom_height = False  # @param {type:"boolean"}
height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
# @markdown If not using custom height, default to scene lower limit.
# @markdown (Cell output provides scene height range from bounding box for reference.)

print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
if not custom_height:
	# get bounding box minumum elevation for automatic height
	height = sim.pathfinder.get_bounds()[0][1]

if not sim.pathfinder.is_loaded:
	print("Pathfinder not initialized, aborting.")
else:
	# @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
	# This map is a 2D boolean array
	sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

	if display:
		# @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-api/blob/master/habitat/utils/visualizations/maps.py)
		hablab_topdown_map = maps.get_topdown_map(
			sim.pathfinder, height, meters_per_pixel=meters_per_pixel
		)
		recolor_map = np.array(
			[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
		)
		hablab_topdown_map = recolor_map[hablab_topdown_map]
		print("Displaying the raw map from get_topdown_view:")
		#display_map(sim_topdown_map)
		print("Displaying the map from the Habitat-Lab maps module:")
		#display_map(hablab_topdown_map)

		# easily save a map to file:
		map_filename = os.path.join(output_path, "top_down_map.png")
		imageio.imsave(map_filename, hablab_topdown_map)


# In[10]:


if not sim.pathfinder.is_loaded:
	print("Pathfinder not initialized, aborting.")
else:
	# @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
	print("NavMesh area = " + str(sim.pathfinder.navigable_area))
	print("Bounds = " + str(sim.pathfinder.get_bounds()))

	# @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
	pathfinder_seed = 1  # @param {type:"integer"}
	sim.pathfinder.seed(pathfinder_seed)
	nav_point = sim.pathfinder.get_random_navigable_point()
	#nav_points = convert_topdown_to_points(sim.pathfinder, sim_topdown_map, meters_per_pixel)
	#print(nav_points)
	print("Random navigable point : " + str(nav_point))
	print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

	bounds = sim.pathfinder.get_bounds()
	min_x = bounds[0][0]
	min_z = bounds[0][2]
	max_x = bounds[1][0]
	max_z = bounds[1][2]

	navigable_points = []
	while min_x <= max_x:
	  z = min_z
	  while z <= max_z:
		point = np.array((min_x, height, z))
		if sim.pathfinder.is_navigable(point):
		  navigable_points.append(point)
		z += 0.1
	  min_x += 0.1


	# @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
	# @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
	print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

	# @markdown The closest boundary point can also be queried (within some radius).
	max_search_radius = 2.0  # @param {type:"number"}
	print(
		"Distance to obstacle: "
		+ str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
	)
	hit_record = sim.pathfinder.closest_obstacle_surface_point(
		nav_point, max_search_radius
	)
	print("Closest obstacle HitRecord:")
	print(" point: " + str(hit_record.hit_pos))
	print(" normal: " + str(hit_record.hit_normal))
	print(" distance: " + str(hit_record.hit_dist))

	vis_points = [nav_point]

	# HitRecord will have infinite distance if no valid point was found:
	if math.isinf(hit_record.hit_dist):
		print("No obstacle found within search radius.")
	else:
		# @markdown Points near the boundary or above the NavMesh can be snapped onto it.
		perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
		print("Perturbed point : " + str(perturbed_point))
		print(
			"Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
		)
		snapped_point = sim.pathfinder.snap_point(perturbed_point)
		print("Snapped point : " + str(snapped_point))
		print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
		vis_points.append(snapped_point)

	# @markdown ---
	# @markdown ### Visualization
	# @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlayed.
	meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

	if display:
		xy_vis_points = convert_points_to_topdown(
			sim.pathfinder, navigable_points, meters_per_pixel
		)
		
		# use the y coordinate of the sampled nav_point for the map height slice
		top_down_map = maps.get_topdown_map(
			sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
		)
		recolor_map = np.array(
			[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
		)
		top_down_map = recolor_map[top_down_map]
		print("\nDisplay the map with key_point overlay:")
		#display_map(top_down_map, key_points=xy_vis_points)


# In[11]:


"""Panorama creation using opencv's stitch method"""

"""From every navigable point, the robot rotates clockwise 12 times, takes observations and saves images.
Using the 12 images, a panorama is created and saved.
"""
'''
observation_path = '../observations'
if not os.path.exists(observation_path):
	os.mkdir(observation_path)
for index, point in enumerate(navigable_points[0:]):
  agent_state = habitat_sim.AgentState()
  agent_state.position = point  # world space
  agent_state.sensor_states = {}
  agent.set_state(agent_state)
  
  print(agent.get_state().position)
  observations = sim.step('turn_left')
  rgb = observations["color_sensor"]
  semantic = observations["semantic_sensor"]
  depth = observations["depth_sensor"]
  print(agent.get_state().position)  
  images = []
  display_sample(rgb, semantic, depth)  
  rgb_path = os.path.join(observation_path, f'rgb/{index}')
  if not os.path.exists(rgb_path):
	os.mkdir(rgb_path)
  semantic_path = os.path.join(observation_path, f'semantic/{index}')
  if not os.path.exists(semantic_path):
	os.mkdir(semantic_path)
  depth_path = os.path.join(observation_path, f'depth_test/{index}')
  if not os.path.exists(depth_path):
	os.mkdir(depth_path)
  panorama_path = os.path.join(observation_path, f'panorama')
  if not os.path.exists(panorama_path):
	os.mkdir(panorama_path)
  for i in range(12):
	observations = sim.step('turn_right')
	#print(agent.get_state())
	rgb = observations["color_sensor"]
	semantic = observations["semantic_sensor"]
	depth = observations["depth_sensor"]
	display_sample(rgb, semantic, depth)
	#images.append(Image.fromarray(rgb))
	#break
	
	#cv2.imwrite(f'{rgb_path}/{i}.png', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
	#cv2.imwrite(f'{semantic_path}/{i}.png', semantic)#cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB))
	#cv2.imwrite(f'{depth_path}/{i}.png', depth)#, cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
	#cv2_imshow(Image.fromarray(rgb))
	#plt.figure()
	#plt.imshow(depth);plt.show()
  if index == 1:
	break
  #panorama = create_cylindrical_panorama(images, index, rgb_path)
  #cv2.imwrite(f'{panorama_path}/{index}.png', panorama)


# In[248]:


depth/10*255


# In[11]:
'''

def transformation_matrix(R, t):
	"""
	Make Transformation matrix T, from rotation matrix R and translation matrix t
	"""
	
	T = np.empty((4,4))
	T[:3,:3] = R
	try:
		T[:3,3] = t.reshape(3)
	except:
		T[:3,3] = None
	T[3:,] = [0,0,0,1]
	return T


# In[12]:


def distance(point1, point2):
	return np.linalg.norm(point1 - point2)


# In[14]:

"""
color_images_path = '../observations/rgb/0'
depth_images_path = '../observations/depth/0'

color_images = sorted(os.listdir(color_images_path), key=lambda name: int(name.split('.')[0]))
depth_images = sorted(os.listdir(depth_images_path), key=lambda name: int(name.split('.')[0]))

cam1 = o3d.camera.PinholeCameraIntrinsic()
cam1.intrinsic_matrix =  [[256, 0.00, 256] , [0.00, 256, 256], [0.00, 0.00, 1]]
pcds = []
for i in range(len(color_images)):
	theta = np.radians(i* -30)
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c, 0, s), (0, 1, 0), (-s,0,c)))
	t = np.zeros((3,1))
	T = transformation_matrix(R,t)
	
	color_image_path = os.path.join(color_images_path, color_images[i])
	depth_image_path = os.path.join(depth_images_path, depth_images[i])
	
	color_image = o3d.io.read_image(color_image_path)
	depth_image = o3d.io.read_image(depth_image_path)
	
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_image,
		depth_image
	)
	'''
	pcd = o3d.geometry.PointCloud.create_from_depth_image(
		o3d.geometry.Image(np.asarray(depth_image).astype(np.float)),
		cam1,
		depth_scale = 1000,
		depth_trunc = 1000)
	'''
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
		rgbd_image,
		cam1)
	
	pcd_np = (np.asarray(pcd.points)*10000)
	#print(pcd_np.shape)
	pcd_np_xz = (np.asarray(pcd.points)*10000)[:,[0,2]]
	pcd_distance = np.linalg.norm(pcd_np_xz,axis=1)
	pcd_np = pcd_np[pcd_distance < 7]
	#print(pcd_np.shape)
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pcd_np)
	#print(pcd)
	#pcd = pcd.voxel_down_sample(voxel_size=0.01)
	'''
	diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
	camera = [0, 0, diameter]
	radius = diameter * 100

	print("Get all points that are visible from given view point")
	_, pt_map = pcd.hidden_point_removal(camera, radius)

	print("Visualize result")
	pcd = pcd.select_by_index(pt_map)
	'''
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	pcd.transform(T)
	pcds.append(pcd)
	#break
	#o3d.visualization.draw_geometries([pcd])
	
	#o3d.io.write_point_cloud(f"{path}/{i}.pcd", pcd)


# In[50]:


np.asarray(depth_image)


# In[15]:


np.asarray(color_image)


# In[48]:


o3d.visualization.draw_geometries(pcds)


# In[27]:

"""
def convert_points_to_topdown_(pathfinder, pcds, meters_per_pixel):
	points_topdown = []
	bounds = pathfinder.get_bounds()
	for pcd in pcds:
		for point in pcd:
			if sim.pathfinder.is_navigable(point):
				# convert 3D x,z to topdown x,y
				px = (point[0] - bounds[0][0]) / meters_per_pixel
				py = (point[2] - bounds[0][2]) / meters_per_pixel
				points_topdown.append(np.array([px, py]))
				
	return points_topdown

'''
# In[101]:


point = navigable_points[0]
pcds_np = [(np.asarray(pcd.points)) + point for pcd in pcds]
xy_vis_points = convert_points_to_topdown_(
			sim.pathfinder, pcds_np[::20,], meters_per_pixel
		)
		

print("\nDisplay the map with key_point overlay:")
display_map(top_down_map, key_points=xy_vis_points)


# In[14]:


point = navigable_points[0]
#point = [point[0], 0, point[2]]
max_x = -np.inf
max_y = -np.inf
max_z = -np.inf

min_x = np.inf
min_y = np.inf
min_z = np.inf
for pcd in pcds:
	pc_points = (np.asarray(pcd.points)*1) + point
	
	max_x_pc = np.max(pc_points[:,0])
	max_y_pc = np.max(pc_points[:,1])
	max_z_pc = np.max(pc_points[:,2])

	min_x_pc = np.min(pc_points[:,0])
	min_y_pc = np.min(pc_points[:,1])
	min_z_pc = np.min(pc_points[:,2])
	
	if max_x_pc > max_x:
		max_x = max_x_pc
	if max_y_pc > max_y:
		max_y = max_y_pc
	if max_z_pc > max_z:
		max_z = max_z_pc
		
	if min_x_pc < min_x:
		min_x = min_x_pc
	if min_y_pc < min_y:
		min_y = min_y_pc
	if min_z_pc < min_z:
		min_z = min_z_pc 


# In[15]:


bounds = sim.pathfinder.get_bounds()
bounds


# In[16]:


(min_x_pc, max_x_pc), (min_y_pc, max_y_pc), (min_z_pc, max_z_pc)


# In[17]:


RESOLUTION = 0.1
x_range = int((max_x - min_x) / RESOLUTION)
y_range = int((max_y - min_y) / RESOLUTION)
z_range = int((max_z - min_z) / RESOLUTION)
x_range, y_range, z_range


# In[18]:


THRESHOLD_LOW = min_y_pc + 0.3
THRESHOLD_HIGH = 0


# In[14]:

'''
def not_inside_boundary(point):
	bounds = sim.pathfinder.get_bounds()
	if point[0] < bounds[0][0] or point[0]>bounds[1][0]:
		#print(point[0])
		return True
	#if point[1] < bounds[0][1] or point[1]>bounds[1][1]:
	#	print(point[1])
	#	return True
	if point[2] < bounds[0][2] or point[2]>bounds[1][2]:
		#print(point[2])
		return True
	return False


# In[13]:


def get_pcd_from_numpy(points):
	pcd_n = o3d.geometry.PointCloud()
	pcd_n.points = o3d.utility.Vector3dVector(points)
	return pcd_n


# In[20]:

'''
occupancy_grid = np.ones((z_range+1, x_range+1)) * -1 #unexlpored
l = []
l1 = []
for i,pcd in enumerate(pcds):
	pc_points = (np.asarray(pcd.points)*1) + point
	#o3d.visualization.draw_geometries([pcd])
	#pc_points = np.array([pc_point for pc_point in pc_points if not_inside_boundary(pc_point) == False])
	#pc_points = np.array([pc_point for pc_point in pc_points if sim.pathfinder.is_navigable([pc_point[0], height, pc_point[2]])])
	
	if pc_points.size == 0:
		continue
	l1.append(pcd)
	print(i)
	#o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_points)])
	
	x = ((pc_points[:,0] - min_x) / RESOLUTION).astype(int)
	z = ((pc_points[:,2] - min_z) / RESOLUTION).astype(int)
	obj = np.logical_and(THRESHOLD_LOW < pc_points[:,1], pc_points[:,1] < THRESHOLD_HIGH)
	free_space = pc_points[:,1] <= THRESHOLD_LOW
	
	#occupancy_grid[x[~y], z[~y]] = 1 #freespace
	occupancy_grid[z[free_space], x[free_space]] = 0 #freespace
	occupancy_grid[z[obj],x[obj]] = 1 #occupied
	
	#o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_points[~y])])
	l.extend(pc_points[free_space])


# In[255]:


pc_n = np.array(l)
pcd_n = o3d.geometry.PointCloud()
pcd_n.points = o3d.utility.Vector3dVector(pc_n)
pcd_n.paint_uniform_color([0,0,0])
o3d.visualization.draw_geometries([pcd_n])


# In[86]:


pc_n = np.array(l)
pcd_n = o3d.geometry.PointCloud()
pcd_n.points = o3d.utility.Vector3dVector(pc_n)
pcd_n.paint_uniform_color([0,0,0])
pcds.append(pcd_n)
o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_n)])


# In[42]:


np.asarray(depth_image)


# In[15]:

'''
def nearest_value_og(occupancy_grid, i, j, threshold=4):
	d = {0:0, -1:0, 1:0}
	d[occupancy_grid[i-1][j]] += 1
	d[occupancy_grid[i+1][j]] += 1
	d[occupancy_grid[i][j-1]] += 1
	d[occupancy_grid[i][j+1]] += 1
	  
	for occupancy_value, count in d.items():
	  if count >= threshold:
		  return occupancy_value
	return occupancy_grid[i][j]


# In[16]:


def is_not_on_boundary(size, i, j):
	if i > 0 and j > 0 and i < size[0] - 1 and j < size[1] -1:
		return True
	return False


# In[17]:


def remove_isolated_points(occupancy_grid, threshold=2):
	for i in range(len(occupancy_grid)):
		for j in range(len(occupancy_grid[i])):
			if occupancy_grid[i][j] == -1 and is_not_on_boundary(occupancy_grid.shape, i, j):
				occupancy_grid[i][j] = nearest_value_og(occupancy_grid, i, j, threshold = threshold)
	return occupancy_grid


# In[24]:
'''

plt.figure(dpi=600)
plt.imshow(occupancy_grid)
plt.show()


# In[49]:


plt.figure(dpi=600)
plt.imshow(remove_isolated_points(occupancy_grid))
#plt.show()


# In[253]:


xy_vis_points = convert_points_to_topdown(
			sim.pathfinder, (np.array(l)[::10,]), meters_per_pixel
		)
display_map(top_down_map, key_points=xy_vis_points)


# In[76]:


p = np.asarray([-10.89344006,  -0.127553  ,  0])
xy_vis_points = convert_points_to_topdown(
			sim.pathfinder, [p], meters_per_pixel
		)
display_map(top_down_map, key_points=xy_vis_points)


# In[65]:


navigable_points[0]


# In[18]:
'''

def get_purturbed_point(points):
	result = []
	height = sim.pathfinder.get_bounds()[0][1]
	for i in points.shape[1]:
		point = [points[0][i], height, points[1][i]]
		max_searh_radius = 2
		hit_record = sim.pathfinder.closest_obstacle_surface_point(
			point, max_search_radius
		)
		if math.isinf(hit_record.hit_dist):
			print("No obstacle found within search radius.")
			return point
		else:
			perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
			snapped_point = sim.pathfinder.snap_point(perturbed_point)
			return snapped_point
		


# In[19]:


class Frontier:
	def __init__(self,x,y):
		self.frontier = np.vstack((x,y))
		self.centroid = np.mean(self.frontier, axis=1)
	def __eq__(self, __o: object) -> bool:
		#print(self.centroid)
		#print(__o.centroid)
		if self.centroid[0] == __o.centroid[0] and self.centroid[1] == __o.centroid[1]:
			return True
		else:
			return False

import scipy.ndimage
def get_frontiers(occupancy_grid):
	filtered_grid = scipy.ndimage.maximum_filter((occupancy_grid == 0), size=3)
	frontier_point_mask = np.logical_and(filtered_grid,
										 occupancy_grid == -1)
	'''
	if group_inflation_radius < 1:
		inflated_frontier_mask = frontier_point_mask
	else:
		inflated_frontier_mask = gridmap.utils.inflate_grid(
			frontier_point_mask,
			inflation_radius=group_inflation_radius,
			obstacle_threshold=0.5,
			collision_val=1.0) > 0.5
	'''
	# Group the frontier points into connected components
	labels, nb = scipy.ndimage.label(frontier_point_mask)

	# Extract the frontiers
	frontiers = []
	for ii in range(nb):
		raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask))
		#return raw_frontier_indices
		#print(raw_frontier_indices)
		#frontiers.add(\
		#	np.concatenate((raw_frontier_indices[0][None, :],
		#					raw_frontier_indices[1][None, :]),
		#				   axis=0))
		frontiers.append(Frontier(raw_frontier_indices[0], raw_frontier_indices[1]))
		#print(frontiers)

	return frontiers


# In[275]:


#frontiers = get_frontiers(occupancy_grid)


# In[120]:


#np.vstack((f[0][0],f[0][1])).shape


# In[21]:


def show_frontiers(occupancy_grid, frontiers, show=True, store=False, output_path=None):
	#plt.figure(dpi=600)
	plt.imshow(occupancy_grid)
	for f in frontiers:
		#print(f.centroid)
		#f.fron
		plt.scatter(f.frontier[1],f.frontier[0])
		plt.scatter(f.centroid[1], f.centroid[0])
	
	if store == True:
		plt.savefig(output_path)
	if show == True:
		plt.show()
	plt.close()

# In[22]:


def get_shortest_path(start_point, end_point):
	shortest_path = habitat_sim.ShortestPath()
	shortest_path.requested_start = start_point
	shortest_path.requested_end = end_point
	found_path = sim.pathfinder.find_path(shortest_path)
	if found_path:
		return shortest_path
	else:
		return 0


# In[41]:

'''
points = []
current_point = navigable_points[0]
min_distance = 10000000
for f in frontiers:
	x = (f.centroid[1] * RESOLUTION)+ min_x
	z = (f.centroid[0] * RESOLUTION) + min_z 
	point = np.array([x,height,z])
	shortest_path = get_shortest_path(current_point, point)
	if shortest_path.geodesic_distance < min_distance:
		min_distance = shortest_path.geodesic_distance
		min_point = point
	#points.append(point)


# In[43]:


xy_vis_points = convert_points_to_topdown(
			sim.pathfinder, [current_point], meters_per_pixel
		)
display_map(top_down_map, key_points=xy_vis_points)


# In[23]:

'''
def get_point_clouds(rgb_path, depth_path, starting_point, current_point, look_down=False):
	cam1 = o3d.camera.PinholeCameraIntrinsic()
	cam1.intrinsic_matrix =  [[256, 0.00, 254] , [0.00, 256, 254], [0.00, 0.00, 1]]
	pcds = []
	
	color_images = sorted(os.listdir(rgb_path), key=lambda name: int(name.split('.')[0]))
	depth_images = sorted(os.listdir(depth_path), key=lambda name: int(name.split('.')[0]))
	
	for i in range(len(color_images)):
		#theta = np.radians(i* -30)
		#c, s = np.cos(theta), np.sin(theta)
		#R = np.array(((c, 0, s), (0, 1, 0), (-s,0,c)))
		r = R_scipy.from_euler('zxy', [0,0,i*-30], degrees=True)
		R = r.as_dcm()

		if look_down == True:
			#theta = np.radians(-75)
			#c_look_down, s_look_down = np.cos(theta), np.sin(theta)
			#R_look_down = np.array(((c_look_down, -s_look_down, 0), (s_look_down, c_look_down, 0), (0,0,1)))
			#R = np.dot(R_look_down, R)

			r = R_scipy.from_euler('zxy', [0,-45,i*-30], degrees=True)
			R = r.as_dcm()

		
		#t = np.zeros((3,1))
		t = np.array([current_point[0]-starting_point[0], current_point[1]- starting_point[1], current_point[2] - starting_point[2]])
		T = transformation_matrix(R,t)
		
		color_image_path = os.path.join(rgb_path, color_images[i])
		depth_image_path = os.path.join(depth_path, depth_images[i])

		color_image = o3d.io.read_image(color_image_path)
		depth_image = o3d.io.read_image(depth_image_path)

		#color_image = o3d.geometry.Image((color_images[i]).astype(np.uint8))
		#depth_image = o3d.geometry.Image((depth_images[i]).astype(np.uint8))
		
		#print(np.asarray(color_image))
		#print(np.asarray(depth_image))

		rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
			color_image,
			depth_image
		)	
		pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
			rgbd_image,
			cam1)
		
		pcd_np = (np.asarray(pcd.points)*10000)
		#print(pcd_np.shape)
		pcd_np_xz = (np.asarray(pcd.points)*10000)[:,[0,2]]
		pcd_distance = np.linalg.norm(pcd_np_xz,axis=1)
		pcd_np = pcd_np[pcd_distance < 7]
		#print(pcd_np.shape)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(pcd_np)
		#pcd = pcd.voxel_down_sample(voxel_size=0.000001)
		# Flip it, otherwise the pointcloud will be upside down
		pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		pcd.transform(T)
		#o3d.visualization.draw_geometries([pcd])
		pcds.append(pcd)
	return pcds


# In[24]:


def get_min_max_pcd(pcds, point, min_x,max_x,min_y,max_y,min_z,max_z, scale=1):
	max_x = -np.inf
	max_y = -np.inf
	max_z = -np.inf

	min_x = np.inf
	min_y = np.inf
	min_z = np.inf
	
	for pcd in pcds:
		pc_points = (np.asarray(pcd.points)*scale) + point

		try:
			max_x_pc = np.max(pc_points[:,0])
			max_y_pc = np.max(pc_points[:,1])
			max_z_pc = np.max(pc_points[:,2])
		except:
			continue

		min_x_pc = np.min(pc_points[:,0])
		min_y_pc = np.min(pc_points[:,1])
		min_z_pc = np.min(pc_points[:,2])

		if max_x_pc > max_x:
			max_x = max_x_pc
		if max_y_pc > max_y:
			max_y = max_y_pc
		if max_z_pc > max_z:
			max_z = max_z_pc

		if min_x_pc < min_x:
			min_x = min_x_pc
		if min_y_pc < min_y:
			min_y = min_y_pc
		if min_z_pc < min_z:
			min_z = min_z_pc 
	return min_x,max_x,min_y,max_y,min_z,max_z


# In[25]:


def initialize():
	cfg = make_cfg(sim_settings)
	# Needed to handle out of order cell run in Colab
	try:  # Got to make initialization idiot proof
		sim.close()
	except NameError:
		pass
	sim = habitat_sim.Simulator(cfg)
	# the randomness is needed when choosing the actions
	'''
	random.seed(sim_settings["seed"])
	sim.seed(sim_settings["seed"])

	# Set agent state
	agent = sim.initialize_agent(sim_settings["default_agent"])
	agent_state = habitat_sim.AgentState()
	agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
	agent.set_state(agent_state)
	'''


# In[41]:


def create_occupancy_grid(current_pcds,current_point,min_x,max_x,min_y,max_y,min_z,max_z, index, RESOLUTION = 0.1, scale=1):
	x_range = int((max_x - min_x) / RESOLUTION)
	y_range = int((max_y - min_y) / RESOLUTION)
	z_range = int((max_z - min_z) / RESOLUTION)
	
	THRESHOLD_LOW = min_y + 0.5
	THRESHOLD_HIGH = max_y - 1
	
	occupancy_grid = np.ones((z_range+1, x_range+1)) * -1 #unexlpored
	if index != 0:
		xy_vis_points = convert_points_to_topdown(
		 sim.pathfinder, np.array([current_point]), meters_per_pixel
		)
		display_map(top_down_map,xy_vis_points, show=False)
		#occupancy_grid[intxy_vis_points[0][0]][xy_vis_points[0][1]] = 0
	l = []
	l1 = []
	l2 = []
	for i,pcd in enumerate(current_pcds):
		pc_points = (np.asarray(pcd.points) * scale) + current_point
		#o3d.visualization.draw_geometries([pcd])
		#pc_points = np.array([pc_point for pc_point in pc_points if not_inside_boundary(pc_point) == False])
		#pc_points = np.array([pc_point for pc_point in pc_points if sim.pathfinder.is_navigable([pc_point[0], height, pc_point[2]])])

		if pc_points.size == 0:
			continue
		l1.append(pcd)
		#print(i)
		#o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_points)])

		x = ((pc_points[:,0] - min_x) / RESOLUTION).astype(int)
		z = ((pc_points[:,2] - min_z) / RESOLUTION).astype(int)
		obj = np.logical_and(THRESHOLD_LOW < pc_points[:,1], pc_points[:,1] < THRESHOLD_HIGH)
		free_space = pc_points[:,1] <= THRESHOLD_LOW

		#occupancy_grid[x[~y], z[~y]] = 1 #freespace
		occupancy_grid[z[free_space], x[free_space]] = 0 #freespace
		occupancy_grid[z[obj],x[obj]] = 1 #occupied

		#o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_points[free_space])])
		l.extend(pc_points[free_space])
		l2.extend(pc_points[obj])
	'''
	if index == 1:
		o3d.visualization.draw_geometries(current_pcds)
		o3d.visualization.draw_geometries([get_pcd_from_numpy(np.array(l))])
		o3d.visualization.draw_geometries([get_pcd_from_numpy(np.array(l2))])
	'''
	return occupancy_grid, l



# In[26]:


def display_occupancy_grid(occupancy_grid, show=True, store=False, output_path=None):
	plt.figure(dpi=600)
	plt.imshow(occupancy_grid)
	
	
	if store == True:
		plt.savefig(output_path)
	if show == True:
		plt.show()
	plt.close()


# In[39]:

def get_largest_frontier(frontiers, current_point, visited_frontiers):
	points = []
	RESOLUTION = 0.1
	max_length = 0
	max_point = np.array([])
	final_path = None
	max_frontier = None
	#print(visited_frontiers)
	for f in frontiers:
		if f in visited_frontiers:
			continue
		
		print("starting_point",starting_point)
		x = (f.centroid[1] * RESOLUTION)+ fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z
		point = np.array([x,height,z])
		#print(f.centroid, point)
		#show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)

		xy_vis_points = convert_points_to_topdown(
		 sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
		)
	
		#path = f'./example4/closest_frontier{i}'
		path = f'./example5/largest_frontier{i}'
		#display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
		#break
		#print(f.frontier)

		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0: #Path not found
			print("Shortest path not found")
			#print(current_point, point)
			continue
		if len(f.frontier[0]) > max_length:
			max_length = len(f.frontier[0])
			max_point = point
			max_frontier = f
			final_path = shortest_path
	#points.append(point)
	return max_point, final_path, max_frontier

def get_smallest_frontier(frontiers, current_point, visited_frontiers):
	points = []
	RESOLUTION = 0.1
	min_length = np.inf
	min_point = np.array([])
	final_path = None
	min_frontier = None
	#print(visited_frontiers)
	for f in frontiers:
		if f in visited_frontiers:
			continue
		
		print("starting_point",starting_point)
		x = (f.centroid[1] * RESOLUTION)+ fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z
		point = np.array([x,height,z])
		#print(f.centroid, point)
		#show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)

		#xy_vis_points = convert_points_to_topdown(
		# sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
		#)
	
		#path = f'./example4/closest_frontier{i}'
		#path = f'./example5/largest_frontier{i}'
		#display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
		#break
		#print(f.frontier)

		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0: #Path not found
			print("Shortest path not found")
			#print(current_point, point)
			continue
		if len(f.frontier[0]) < min_length:
			min_length = len(f.frontier[0])
			min_point = point
			min_frontier = f
			final_path = shortest_path
	#points.append(point)
	return min_point, final_path, min_frontier

def get_farthest_frontier(frontiers, current_point, visited_frontiers):
	points = []
	RESOLUTION = 0.1
	max_distance = np.inf * -1
	max_point = np.array([])
	final_path = None
	max_frontier = None
	#print(visited_frontiers)
	for f in frontiers:
		if f in visited_frontiers:
			continue
		#print("centroid", f.centroid)
		#print("curremt_point", current_point)
		x = ((start_z- f.centroid[1]) * RESOLUTION) + current_point[0]
		z = ((f.centroid[0]-start_x) * RESOLUTION) + current_point[2]

		x = (f.centroid[1] * RESOLUTION)+ fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z

		point = np.array([x,height,z])
		'''
		print("start_x, start_z", start_x, start_z)
		print("point",point)
		show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)
		xy_vis_points = convert_points_to_topdown(
		 sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
		)
		display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
		print(top_down_map.shape)
		'''
		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0: #Path not found
			print("Shortest path not found")
			#print(current_point, point)
			continue
		if shortest_path.geodesic_distance > max_distance:
			max_distance = shortest_path.geodesic_distance
			max_point = point
			max_frontier = f
			final_path = shortest_path
	#points.append(point)
	return max_point, final_path, max_frontier

def get_closest_frontier(frontiers, current_point, visited_frontiers):
	points = []
	RESOLUTION = 0.1
	min_distance = np.inf
	min_point = np.array([])
	final_path = None
	min_frontier = None
	#print(visited_frontiers)
	for f in frontiers:
		if f in visited_frontiers:
			continue
		#print("centroid", f.centroid)
		#print("curremt_point", current_point)
		#x = ((start_z- f.centroid[1]) * RESOLUTION) + current_point[0]
		#z = ((f.centroid[0]-start_x) * RESOLUTION) + current_point[2]

		x = (f.centroid[1] * RESOLUTION)+ fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z

		point = np.array([x,height,z])
		'''
		print("start_x, start_z", start_x, start_z)
		print("point",point)
		show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)
		xy_vis_points = convert_points_to_topdown(
		 sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
		)
		display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
		print(top_down_map.shape)
		'''
		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0: #Path not found
			print("Shortest path not found")
			#print(current_point, point)
			continue
		if shortest_path.geodesic_distance < min_distance:
			min_distance = shortest_path.geodesic_distance
			min_point = point
			min_frontier = f
			final_path = shortest_path
	#points.append(point)
	return min_point, final_path, min_frontier

def get_anticipated_frontier(frontiers, current_point, visited_frontiers):
	points = []
	RESOLUTION = 0.1
	max_count = 0
	max_point = np.array([])
	final_path = None
	max_frontier = None
	max_anticipated_rec = None
	for f in frontiers:
		if f in visited_frontiers:
			continue
		
		#print("starting_point",starting_point)
		x = (f.centroid[1] * RESOLUTION)+ fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z
		point = np.array([x,height,z])
		#print(f.centroid, point)
		#show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)
		left, right = x - 0.5, x + 0.5
		top, bottom = z - 0.5, z + 0.5
		
		anticipation_count = 0
		anticipation_rec = []
		
		i = top
		while i <= bottom:
			j = left
			while j <= right:
				anticipation_point = np.array((j, height, i))
				if sim.pathfinder.is_navigable(anticipation_point):
					anticipation_count += 1
				anticipation_rec.append(anticipation_point)
				j += 0.2
			i += 0.2
		#xy_vis_points = convert_points_to_topdown(
		# sim.pathfinder, anticipation_rec, meters_per_pixel=meters_per_pixel
		#)
		#display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=None)
		#path = f'./example4/closest_frontier{i}'
		#path = f'./example5/largest_frontier{i}'
		#display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
		#break
		#print(f.frontier)
		if anticipation_count > max_count:
			max_count = anticipation_count
			shortest_path = get_shortest_path(current_point, point)
			if shortest_path == 0: #Path not found
				print("Shortest path not found")
				#print(current_point, point)
				continue
			else:
				max_point = point
				max_frontier = f
				final_path = shortest_path
				max_anticipated_rec = anticipation_rec
	#points.append(point)
	return max_point, final_path, max_frontier, max_anticipated_rec

def check_nearest_cc(f, connected_components):
	min = np.inf
	for cc in connected_components:
		x = (cc.frontier[0] - f.centroid[0])**2
		y = (cc.frontier[1] - f.centroid[1])**2

		k = np.sqrt(x + y)
		min_distance = np.min(k)
		if min_distance < min:
			min = min_distance
			min_cc = cc
	return min, min_cc

def get_potential_frontier(occupancy_grid, frontiers, current_point, visited_frontiers):
	hablab_topdown_map = maps.get_topdown_map(
			sim.pathfinder, height, meters_per_pixel=meters_per_pixel
		)
	gt_map = np.ones((hablab_topdown_map.shape)) * -1
	for i in range(len(hablab_topdown_map)):
		for j in range(len(hablab_topdown_map[0])):
			if hablab_topdown_map[i][j] == 1 or hablab_topdown_map[i][j] == 2:
				gt_map[i][j] = 0
	xy_vis_points = convert_points_to_topdown(
		 sim.pathfinder, [starting_point], meters_per_pixel=meters_per_pixel
		)
	display_map(gt_map,xy_vis_points, show=True, store=False, output_path=None)
	#display_map(occupancy_grid,[np.array([start_x,start_z])], show=True, store=False, output_path=None)
	print("starting_point", xy_vis_points)
	print("startx,startz", start_x, start_z)
	x_difference = int(xy_vis_points[0][0] - start_x)
	y_difference = int(xy_vis_points[0][1] - start_z)

	for i in range(occupancy_grid.shape[0]):
		for j in range(occupancy_grid.shape[1]):
			if occupancy_grid[i][j] == 0: 
				gt_map[i+x_difference][j+y_difference] = -1
	
	display_map(gt_map,xy_vis_points, show=True, store=False, output_path=None)
	
	filtered_grid = scipy.ndimage.maximum_filter((gt_map == 0), size=1)
	#print(filtered_grid)
	frontier_point_mask = np.logical_and(filtered_grid,
										 gt_map == 0)
	# Group the frontier points into connected components
	labels, nb = scipy.ndimage.label(filtered_grid)

	# Extract the frontiers
	connected_components = []
	for ii in range(nb):
		raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask))
		#return raw_frontier_indices
		#print(raw_frontier_indices)
		#frontiers.add(\
		#	np.concatenate((raw_frontier_indices[0][None, :],
		#					raw_frontier_indices[1][None, :]),
		#				   axis=0))
		connected_components.append(Frontier(raw_frontier_indices[0], raw_frontier_indices[1]))
		#print(frontiers)

	#show_frontiers(gt_map, connected_components, show=True)
	max_cc = -np.inf
	highest_potential_frontier_distance = np.inf
	for f in frontiers:
		if f in visited_frontiers:
			continue
		x = (f.centroid[1] * RESOLUTION) + fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z
		point = np.array([x,height,z])

		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0: #Path not found
			print("Shortest path not found")
			#print(current_point, point)
			continue

		nearest_cc_distance, nearest_cc = check_nearest_cc(f, connected_components)
		cc_len = len(nearest_cc.frontier[0])
		#show_frontiers(occupancy_grid, [nearest_cc, f], show=True) 
		if cc_len > max_cc:
			max_cc = cc_len
			highest_potential_frontier = f
			highest_potential_frontier_distance = nearest_cc_distance

			highest_potential_frontier_point = point
			final_path = shortest_path
		elif cc_len == max_cc:
			if shortest_path.geodesic_distance < highest_potential_frontier_distance:
				max_cc = cc_len
				highest_potential_frontier = f
				highest_potential_frontier_distance = shortest_path.geodesic_distance

				highest_potential_frontier_point = point
				final_path = shortest_path

	#show_frontiers(occupancy_grid, [highest_potential_frontier], show=True)	
	return highest_potential_frontier_point, final_path, highest_potential_frontier


from itertools import permutations, combinations
def tsp(frontiers, current_point, visited_frontiers, path_length):
	valid_frontiers = [(None, current_point)]
	for f in frontiers:
		if f in visited_frontiers:
			continue
		x = (f.centroid[1] * RESOLUTION) + fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z
		point = np.array([x,height,z])
		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0:
			continue
		if shortest_path.geodesic_distance > target_path_length - path_length:
			continue
		valid_frontiers.append((f, point))
	
	if len(valid_frontiers) == 1:
		print("NO VALID FRONTIERS FOUND FOR TSP")

	adj_matrix = np.zeros((len(valid_frontiers), len(valid_frontiers)))
	for i in range(adj_matrix.shape[0]):
		for j in range(adj_matrix.shape[1]):
			if i==j:
				adj_matrix[i][j] = 0
			else:
				shortest_path = get_shortest_path(valid_frontiers[i][1], valid_frontiers[j][1])
				adj_matrix[i][j] = shortest_path.geodesic_distance
				adj_matrix[j][i] = shortest_path.geodesic_distance
	
	vertex = list(range(1, len(valid_frontiers)))
	permutation = permutations(vertex)
	min_path_length = np.inf
	for p in permutation:
		current_path_length = 0
		k = 0
		for j in p:
			current_path_length += adj_matrix[k][j]
			if current_path_length > target_path_length - path_length:
				current_path_length -= adj_matrix[k][j]
				break
			k = j
		if current_path_length < min_path_length:
			min_path_length = current_path_length
			min_perm = p
	
	return valid_frontiers, min_perm

def TSP_dynamic(frontiers, current_point, visited_frontiers, path_length):
	valid_frontiers = [(None, current_point)]
	for f in frontiers:
		if f in visited_frontiers:
			continue
		x = (f.centroid[1] * RESOLUTION)+ fixed_min_x
		z = (f.centroid[0] * RESOLUTION) + fixed_min_z
		point = np.array([x,height,z])
		shortest_path = get_shortest_path(current_point, point)
		if shortest_path == 0:
			continue
		if shortest_path.geodesic_distance > target_path_length - path_length:
			continue
		valid_frontiers.append((f, point))
	
	if len(valid_frontiers) == 1:
		print("NO VALID FRONTIERS FOUND FOR TSP")


	adj_matrix = np.zeros((len(valid_frontiers), len(valid_frontiers)))
	
	for i in range(adj_matrix.shape[0]):
		for j in range(adj_matrix.shape[1]):
			if i==j:
				adj_matrix[i][j] = 0
			else:
				shortest_path = get_shortest_path(valid_frontiers[i][1], valid_frontiers[j][1])
				adj_matrix[i][j] = shortest_path.geodesic_distance
				adj_matrix[j][i] = shortest_path.geodesic_distance
	n = len(adj_matrix)
	C = [[np.inf for _ in range(n)] for __ in range(1 << n)]
	C[1][0] = 0 # {0} <-> 1
	G = adj_matrix
	for size in range(1, n):
		for S in combinations(range(1, n), size):
			S = (0,) + S
			k = sum([1 << i for i in S])
			for i in S:
				if i == 0: continue
				for j in S:
					#print(1)
					if j == i: continue
					cur_index = k ^ (1 << i)
					C[k][i] = min(C[k][i], C[cur_index][j]+ G[j][i])	 
													#C[S−{i}][j]
	all_index = (1 << n) - 1
	min_ = np.inf
	for i in range(len(C[all_index])):
		if C[all_index][i] < min_:
			min_ = C[all_index][i]
			min_index = i
	return valid_frontiers, [min_index]
	return min([(C[all_index][i] + G[0][i], i) \
								for i in range(n)])


import copy
def inflate_occupancy_grid(occupancy_grid):
	occupancy_grid_copy = copy.deepcopy(occupancy_grid)
	for i in range(len(occupancy_grid_copy)):
		for j in range(len(occupancy_grid_copy[i])):
			if occupancy_grid_copy[i][j] == 1 and is_not_on_boundary(occupancy_grid_copy.shape, i, j):
				occupancy_grid[i-1][j] = 1
				occupancy_grid[i][j-1] = 1
				occupancy_grid[i+1][j] = 1
				occupancy_grid[i][j+1] = 1
	return occupancy_grid

# In[42]:
def initialize_trajectory_map():
	meters_per_pixel = 0.025
	scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
	height = scene_bb.y().min
	top_down_map = maps.get_topdown_map(
		sim.pathfinder, height, meters_per_pixel=meters_per_pixel
	)
	recolor_map = np.array(
		[[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
	)
	top_down_map = recolor_map[top_down_map]
	return top_down_map

def draw_trajectory(top_down_map, path_points, index):
	grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
	trajectory = [
		maps.to_grid(
			path_point[2],
			path_point[0],
			grid_dimensions,
			pathfinder=sim.pathfinder,
		)
		for path_point in path_points
	]
	grid_tangent = mn.Vector2(
		trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
	)
	path_initial_tangent = grid_tangent / grid_tangent.length()
	initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
	# draw the agent and trajectory on the map
	maps.draw_path(top_down_map, trajectory)
	maps.draw_agent(
		top_down_map, trajectory[0], initial_angle, agent_radius_px=8
	)
	print("\nDisplay the map with agent and path overlay:")
	path = f'./{output_folder}/trajectory_{index}'
	display_map(top_down_map, show=SHOW, store=STORE, output_path=path)

#initialize()
path_length = 0
target_path_length = 15
RESOLUTION = 0.1
starting_point = navigable_points[0]
current_point = navigable_points[0]
pcds = []
min_x,max_x,min_y,max_y,min_z,max_z = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf

i = 0
rgb_path = '../observations/temp/rgb'
depth_path = '../observations/temp/depth'

rgb_path_look_down = '../observations/temp_look_down/rgb'
depth_path_look_down = '../observations/temp_look_down/depth'

visited_frontiers = []
trajectory_points = []
trajectory_map = initialize_trajectory_map()

SHOW = False
STORE = True
DISPLAY = True
methods = ['closest', 'largest', 'farthest', 'smallest', 'tsp', 'anticipation', 'TSP_dynamic', 'potential']
frontier_selection_method = methods[-1]
output_folder = f'example18_{frontier_selection_method}'
if os.path.isdir(output_folder) == False:
	os.mkdir(output_folder)
while True:
	print(i)
	#initialize()
	agent_state.position = current_point  # world space
	agent.set_state(agent_state)
	print(agent.get_state().rotation)
	agent_state_initial = agent.get_state()
	observations = sim.step('turn_left')
	rgb = observations["color_sensor"]
	semantic = observations["semantic_sensor"]
	depth = observations["depth_sensor"]
	#display_sample(rgb, semantic, depth)
	color_images = []
	depth_images = []
	for j in range(12):
		'''
		observations = sim.step('turn_right')
		rgb = observations["color_sensor"]
		semantic = observations["semantic_sensor"]
		depth = observations["depth_sensor"]
		'''
		r = R_scipy.from_euler('zxy', [0,0,j*-30], degrees=True)
		agent_state.rotation = r.as_quat()
		agent.set_state(agent_state)
		
		observations = sim.get_sensor_observations()
		rgb = observations["color_sensor"]
		semantic = observations["semantic_sensor"]
		depth = observations["depth_sensor"]

		color_images.append(np.asarray(rgb[:,:,:-1]))
		depth_images.append(depth/10*255)
		
		cv2.imwrite(f'{rgb_path}/{j}.png', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
		cv2.imwrite(f'{depth_path}/{j}.png', depth/10*255)#, cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
		
		#if i == 1:
		#	display_sample(rgb, semantic, depth)
	
	'''
	agent_state.position = current_point  # world space
	agent.set_state(agent_state_initial)
	print(agent.get_state().rotation)
	observations = sim.step('turn_left')
	rgb = observations["color_sensor"]
	semantic = observations["semantic_sensor"]
	depth = observations["depth_sensor"]
	display_sample(rgb, semantic, depth)
	color_images = []
	depth_images = []
	for j in range(12):
		r = R_scipy.from_euler('zxy', [0,-45,j*-30], degrees=True)
		agent_state.rotation = r.as_quat()
		agent.set_state(agent_state)
		
		observations = sim.get_sensor_observations()
		rgb = observations["color_sensor"]
		semantic = observations["semantic_sensor"]
		depth = observations["depth_sensor"]
		
		color_images.append(np.asarray(rgb[:,:,:-1]))
		depth_images.append(depth/10*255)
		
		cv2.imwrite(f'{rgb_path_look_down}/{j}.png', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
		cv2.imwrite(f'{depth_path_look_down}/{j}.png', depth/10*255)#, cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
		#display_sample(rgb, semantic, depth)
	'''  
	current_pcds = get_point_clouds(rgb_path, depth_path, starting_point, current_point)  
	#o3d.visualization.draw_geometries(current_pcds)
	#sys.exit()
	pcds.extend(current_pcds)
	#if i == 1:
	#	o3d.visualization.draw_geometries(pcds)
	'''
	current_pcds = get_point_clouds(rgb_path_look_down, depth_path_look_down, starting_point, current_point, look_down=True)
	pcds.extend(current_pcds)
	o3d.visualization.draw_geometries(current_pcds)
	o3d.visualization.draw_geometries(pcds)
	'''
	min_x,max_x,min_y,max_y,min_z,max_z = get_min_max_pcd(pcds, current_point, min_x,max_x,min_y,max_y,min_z,max_z, scale=1)
	if i == 0:
		fixed_min_x = min_x
		fixed_min_y = min_y
		fixed_min_z = min_z
		fixed_max_y = max_y
	start_x = ((starting_point[0] - min_x) / RESOLUTION).astype(int)
	start_z = ((starting_point[2] - min_z) / RESOLUTION).astype(int)
	#print(min_x,max_x,min_y,max_y,min_z,max_z)
	occupancy_grid, free_space = create_occupancy_grid(pcds, current_point, min_x,max_x,fixed_min_y,fixed_max_y,min_z,max_z, i)
	#sys.exit()

	path = f'./{output_folder}/og_{i}'
	if DISPLAY:
		display_occupancy_grid(occupancy_grid, show=SHOW, store=STORE, output_path=path)
	
	path = f'./{output_folder}/inflate_of{i}'
	occupancy_grid = inflate_occupancy_grid(occupancy_grid)
	if DISPLAY:
		display_occupancy_grid(occupancy_grid, show=SHOW, store=STORE, output_path=path)
	
	frontiers = get_frontiers(occupancy_grid)
	path = f'./{output_folder}/f{i}'
	if DISPLAY:
		show_frontiers(occupancy_grid, frontiers, show=SHOW, store=STORE, output_path=path)
	
	occupancy_grid = remove_isolated_points(occupancy_grid)
	path = f'./{output_folder}/og_rmv_iso_p{i}'
	np_path = f'./{output_folder}/og_rmv_iso_p_{frontier_selection_method}{i}'
	np.save(np_path, occupancy_grid)
	if DISPLAY:
		display_occupancy_grid(occupancy_grid, show=SHOW, store=STORE, output_path=path)
	
	frontiers = get_frontiers(occupancy_grid)
	if DISPLAY:
		path = f'./{output_folder}/f_rmv_iso_p{i}'
		show_frontiers(occupancy_grid, frontiers, show= SHOW, store=STORE, output_path=path)
	#print("OG shape", occupancy_grid.shape)
	
	'''
	if frontier_selection_method == 'closest':
		closest_frontier, shortest_path, min_frontier = get_closest_frontier(frontiers, current_point, visited_frontiers) 
	elif frontier_selection_method == 'farthest': 
		closest_frontier, shortest_path, min_frontier = get_farthest_frontier(frontiers, current_point, visited_frontiers)   
	elif frontier_selection_method == 'largest':
		closest_frontier, shortest_path, min_frontier = get_largest_frontier(frontiers, current_point, visited_frontiers) 
	#closest_frontier, shortest_path, min_frontier = get_anticipated_frontier(frontiers, current_point, visited_frontiers) 
	
	if np.any(closest_frontier) == False:
		break
	print("current_point", current_point)
	print("closest frontier", closest_frontier)
	print("centroid", min_frontier.centroid)
	print("shortest_path", shortest_path)
	'''
	

	'''
	xy_vis_points = convert_points_to_topdown(
		 sim.pathfinder, np.array([closest_frontier]), meters_per_pixel=meters_per_pixel
		)
	
	path = f'./{output_folder}/{frontier_selection_method}_frontier{i}'
	display_map(top_down_map,xy_vis_points, show=False, store=True, output_path=path)
	'''
	#current_point = closest_frontier 
	
	if trajectory_points == []:
		if frontier_selection_method == 'closest':
			selected_frontier_point, shortest_path, selected_frontier = get_closest_frontier(frontiers, current_point, visited_frontiers) 
		elif frontier_selection_method == 'farthest': 
			selected_frontier_point, shortest_path, selected_frontier = get_farthest_frontier(frontiers, current_point, visited_frontiers)   
		elif frontier_selection_method == 'largest':
			selected_frontier_point, shortest_path, selected_frontier = get_largest_frontier(frontiers, current_point, visited_frontiers) 
		elif frontier_selection_method == 'smallest':
			selected_frontier_point, shortest_path, selected_frontier = get_smallest_frontier(frontiers, current_point, visited_frontiers)
		elif frontier_selection_method == 'anticipation':
			selected_frontier_point, shortest_path, selected_frontier, anticipated_rec = get_anticipated_frontier(frontiers, current_point, visited_frontiers)	
			if DISPLAY:
				xy_vis_points = convert_points_to_topdown(
					sim.pathfinder, np.array(anticipated_rec), meters_per_pixel=meters_per_pixel)
				path = f'./{output_folder}/{frontier_selection_method}_anticipated_rec{i}'
				display_map(top_down_map,xy_vis_points, show=False, store=True, output_path=path)	
		elif frontier_selection_method == 'potential':
			selected_frontier_point, shortest_path, selected_frontier = get_potential_frontier(occupancy_grid, frontiers, current_point, visited_frontiers)
			#sys.exit()
			
			

		elif frontier_selection_method == 'TSP_dynamic':
			valid_frontiers, perm = TSP_dynamic(frontiers, current_point, visited_frontiers, path_length)
			selected_frontier = valid_frontiers[perm[0]][0]
			selected_frontier_point = valid_frontiers[perm[0]][1]
			shortest_path = get_shortest_path(current_point, selected_frontier_point)
			print("PERMUTATION", perm)
			
		if np.any(selected_frontier_point) == False:
			break
		print("current_point", current_point)
		print("closest frontier", selected_frontier_point)
		print("centroid", selected_frontier.centroid)
		print("shortest_path", shortest_path)

		visited_frontiers.append(selected_frontier)
		#path_length += shortest_path.geodesic_distance
		trajectory_points = [point for point in shortest_path.points]
		previous_point = trajectory_points.pop(0)
		current_point = trajectory_points.pop(0)

		if DISPLAY:
			xy_vis_points = convert_points_to_topdown(
			sim.pathfinder, np.array([selected_frontier_point]), meters_per_pixel=meters_per_pixel
			)   
			path = f'./{output_folder}/{frontier_selection_method}_frontier{i}'
			display_map(top_down_map,xy_vis_points, show=False, store=True, output_path=path)

			draw_trajectory(trajectory_map, shortest_path.points, index=i)
			path = f'./{output_folder}/{frontier_selection_method}_frontier_og{i}'
			show_frontiers(occupancy_grid, [selected_frontier], show= SHOW, store=STORE, output_path=path)
	else:
		previous_point = current_point
		current_point = trajectory_points.pop(0)
	#p = len(shortest_path.points)
	#halfway_point = shortest_path.points[int(p/2)]
	#current_point = halfway_point
	#path_length += shortest_path.geodesic_distance
	next_point_shortest_path = get_shortest_path(previous_point, current_point)
	path_length += next_point_shortest_path.geodesic_distance
	print(path_length)
	if path_length >= target_path_length:
		break
	
	if DISPLAY:
		xy_vis_points = convert_points_to_topdown(
			sim.pathfinder, np.array([current_point]), meters_per_pixel=meters_per_pixel
			)
		path = f'./{output_folder}/path{i}'
		display_map(top_down_map,xy_vis_points, show=SHOW, store=STORE, output_path=path)
	
	#half_x = ((halfway_point[0] - min_x) / RESOLUTION).astype(int)
	#half_z = ((halfway_point[2] - min_z) / RESOLUTION).astype(int)
	#occupancy_grid[half_z][half_x] = 20
	#display_occupancy_grid(occupancy_grid, show=True, store=False, output_path=path)
	#min_frontier.centroid[1] = half_x
	#min_frontier.centroid[0] = half_z
	#show_frontiers(occupancy_grid, [min_frontier], show= True, store=True, output_path=path)
	#visited_frontiers.append(min_frontier)
	i+=1
	
	#break
	'''
	if i==20:
		path = f'./example10/topdwn_map'
		plt.figure(figsize=(12, 8))
		ax = plt.subplot(1, 1, 1)
		plt.imshow(top_down_map)
		plt.savefig(output_path)

		print("path_length", path_length)
		
		break
	'''
	#print()
	#break
	

