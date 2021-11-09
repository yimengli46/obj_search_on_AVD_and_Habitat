import collections
import copy
import json
import os
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2
import math
from math import cos, sin, acos, atan2, pi, floor
from io import StringIO

def minus_theta_fn(previous_theta, current_theta):
  result = current_theta - previous_theta
  if result < -math.pi:
    result += 2*math.pi
  if result > math.pi:
    result -= 2*math.pi
  return result

def project_pixels_to_camera_coords (sseg_img, current_depth, current_pose, gap=2, focal_length=128, resolution=256, ignored_classes=[]):
  ## camera intrinsic matrix
  K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
  inv_K = LA.inv(K)
  ## first compute the rotation and translation from current frame to goal frame
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  
  # build the point matrix
  coords_range = range(0, resolution, gap)
  xv, yv = np.meshgrid(np.array(coords_range), np.array(coords_range))
  Z = current_depth[yv.flatten(), xv.flatten()].reshape(len(coords_range), len(coords_range))
  points_4d = np.ones((len(coords_range), len(coords_range), 4), np.float32)
  points_4d[:, :, 0] = xv
  points_4d[:, :, 1] = yv
  points_4d[:, :, 2] = Z
  points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1)) # 4 x N

  # apply intrinsic matrix
  points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
  points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
  points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  print('points_4d.shape = {}'.format(points_4d.shape))
  points_3d = points_4d[:3, :]
  print('points_3d.shape = {}'.format(points_3d.shape))

  ## pick x-row and z-row
  sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

  # ignore some classes points
  #print('sseg_points.shape = {}'.format(sseg_points.shape))
  for c in ignored_classes:
    good = (sseg_points != c)
    sseg_points = sseg_points[good]
    points_3d = points_3d[:, good]
  #print('after: sseg_points.shape = {}'.format(sseg_points.shape))
  #print('after: points_3d.shape = {}'.format(points_3d.shape))

  return points_3d, sseg_points.astype(int)


def project_pixels_to_world_coords (sseg_img, current_depth, current_pose, gap=2, focal_length=128, resolution=256, ignored_classes=[]):
  ## camera intrinsic matrix
  K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
  inv_K = LA.inv(K)
  ## first compute the rotation and translation from current frame to goal frame
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  theta = -(theta + 1.5 * pi)
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  
  # build the point matrix
  coords_range = range(0, resolution, gap)
  xv, yv = np.meshgrid(np.array(coords_range), np.array(coords_range))
  Z = current_depth[yv.flatten(), xv.flatten()].reshape(len(coords_range), len(coords_range))
  points_4d = np.ones((len(coords_range), len(coords_range), 4), np.float32)
  points_4d[:, :, 0] = xv
  points_4d[:, :, 1] = yv
  points_4d[:, :, 2] = Z
  points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1)) # 4 x N

  # apply intrinsic matrix
  points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
  points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
  points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  points_3d = transformation_matrix.dot(points_4d)

  ## pick x-row and z-row
  sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

  # ignore some classes points
  #print('sseg_points.shape = {}'.format(sseg_points.shape))
  for c in ignored_classes:
    good = (sseg_points != c)
    sseg_points = sseg_points[good]
    points_3d = points_3d[:, good]
  #print('after: sseg_points.shape = {}'.format(sseg_points.shape))
  #print('after: points_3d.shape = {}'.format(points_3d.shape))

  return points_3d, sseg_points.astype(int)

def convertInsSegToSSeg (InsSeg, ins2cat_dict):
  ins_id_list = list(ins2cat_dict.keys())
  SSeg = np.zeros(InsSeg.shape, dtype=np.int32)
  for ins_id in ins_id_list:
    SSeg = np.where(InsSeg==ins_id, ins2cat_dict[ins_id], SSeg)

  return SSeg


d3_41_colors_rgb: np.ndarray = np.array(
    [
        [0, 0, 0],
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

def apply_color_to_map (semantic_map, num_classes=41):
  assert len(semantic_map.shape) == 2
  H, W = semantic_map.shape
  color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
  for i in range(num_classes):
    color_semantic_map[semantic_map==i] = d3_41_colors_rgb[i]
  return color_semantic_map

def apply_color_to_pointCloud (sseg_points, num_classes=41):
  assert len(sseg_points.shape) == 1
  N = sseg_points.shape[0]
  color_sseg_points = np.zeros((N, 3), dtype='uint8')
  for i in range(num_classes):
    color_sseg_points[sseg_points==i] = d3_41_colors_rgb[i]
  return color_sseg_points

def create_folder (folder_name, clean_up=False):
  flag_exist = os.path.isdir(folder_name)
  if not flag_exist:
    print('{} folder does not exist, so create one.'.format(folder_name))
    os.makedirs(folder_name)
    #os.makedirs(os.path.join(test_case_folder, 'observations'))
  else:
    print('{} folder already exists, so do nothing.'.format(folder_name))
    if clean_up:
      os.system('rm {}/*.png'.format(folder_name))
      os.system('rm {}/*.npy'.format(folder_name))
      os.system('rm {}/*.jpg'.format(folder_name))

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

def semanticMap_to_binary(sem_map):
  sem_map.astype('uint8')
  sem_map[sem_map != 2] = 0
  sem_map[sem_map == 2] = 255
  return sem_map

def get_class_mapper():
  class_dict = {}
  categories = ['void', 'wall', 'floor','chair','door','table','picture','cabinet','cushion','window','sofa','bed', \
    'curtain','chest_of_drawers','plant','sink','stairs','ceiling','toilet','stool','towel','mirror','tv_monitor', \
    'shower','column','bathtub','counter','fireplace','lighting','beam','railing','shelving','blinds','gym_equipment', \
    'seating','board_panel','furniture','appliances','clothes','objects','misc']
  class_dict = {v: k for k, v in enumerate(categories)}
  return class_dict

def pxl_coords_to_pose(coords, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  x, y = coords
  min_X, min_Z, max_X, max_Z = pose_range
  min_x, min_z, max_x, max_z = coords_range

  if flag_cropped:
    X = (x + min_x) * cell_size + min_X
    Z = (y + min_z) * cell_size + min_Z
  else:
    X = (x) * cell_size + min_X
    Z = (y) * cell_size + min_Z
  return (X, Z)

def pxl_coords_to_pose_numpy(coords, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  min_X, min_Z, max_X, max_Z = pose_range
  min_x, min_z, max_x, max_z = coords_range

  pose = np.zeros(coords.shape)
  if flag_cropped:
    pose[:, 0] = (coords[:, 0] + min_x) * cell_size + min_X
    pose[:, 1] = (coords[:, 1] + min_z) * cell_size + min_Z
  else:
    pose[:, 0] = (coords[:, 0]) * cell_size + min_X
    pose[:, 1] = (coords[:, 1]) * cell_size + min_Z
  return pose


def pose_to_coords(cur_pose, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  tx, tz = cur_pose[:2]
    
  if flag_cropped:
    x_coord = int(floor((tx - pose_range[0]) / cell_size) - coords_range[0])
    z_coord = int(floor((tz - pose_range[1]) / cell_size) - coords_range[1])
  else:
    x_coord = int(floor((tx - pose_range[0]) / cell_size))
    z_coord = int(floor((tz - pose_range[1]) / cell_size))

  return (x_coord, z_coord)

def pose_to_coords_numpy(cur_pose, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  tx, tz = cur_pose[:2]
    
  coords = np.zeros(cur_pose.shape)
  if flag_cropped:
    coords[:, 0] = (np.floor((cur_pose[:, 0] - pose_range[0]) / cell_size) - coords_range[0]).astype(int)
    coords[:, 1] = (np.floor((cur_pose[:, 1] - pose_range[1]) / cell_size) - coords_range[1]).astype(int)
  else:
    coords[:, 0] = np.floor((cur_pose[:, 0] - pose_range[0]) / cell_size)
    coords[:, 1] = np.floor((cur_pose[:, 1] - pose_range[1]) / cell_size)

  coords = coords.astype(int)
  return coords
