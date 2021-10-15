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
from math import cos, sin, acos, atan2, pi
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