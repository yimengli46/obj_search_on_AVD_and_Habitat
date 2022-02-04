import numpy as np
import matplotlib.pyplot as plt 
from baseline_utils import pose_to_coords_frame, pose_to_coords, pxl_coords_to_pose
import math
import heapq as hq

class Node():
	def __init__(self, loc, parent, cost):
		self.loc = loc
		self.parent = parent # parent node
		#self.action = action # action leads from parent to current node
		self.cost = cost

class TreeList():
	def __init__(self):
		self.list = []

	def searchNode(self, searched_node):
		# check if node loc existed
		flag = False
		for node in self.list:
			if node.loc == searched_node.loc:
				flag = True
		return flag

	def insertNode(self, node):
		self.list.append(node)

	def efficientSearch(self, loc):
		start_idx = len(self.list) - 1
		idx = -1
		for i in range(start_idx, -1, -1):
			if self.list[i].loc == loc:
				idx = i
				break
		assert idx > -1
		node = self.list[idx]
		for i in range(idx-1, -1, -1):
			cur_node = self.list[i]
			if cur_node.loc == node.loc and cur_node.cost <= node.cost:
				node = cur_node
		#print(f'node.loc = {node.loc}, node.cost = {node.cost}')
		return node

	def getNode(self, loc):
		return self.efficientSearch(loc)

	def formPath(self, goal_loc):
		# the last node in the list must be a goal node
		locs = []
		'''
		for i in range(len(self.list)):
			print(f'node {i}, loc = {self.list[i].loc}, cost = {self.list[i].cost}')
		'''
		node = self.efficientSearch(goal_loc)
		
		while True:
			if node.parent is None:
				break
			locs.append(node.loc)
			parent_node = node.parent
			node = parent_node

		return locs[::-1]

class PriorityQueue:
	"""
	  Implements a priority queue data structure. Each inserted item
	  has a priority associated with it and the client is usually interested
	  in quick retrieval of the lowest-priority item in the queue. This
	  data structure allows O(1) access to the lowest-priority item.
	"""
	def  __init__(self):
		self.heap = []
		self.count = 0

	def push(self, item, priority):
		entry = (priority, self.count, item)
		hq.heappush(self.heap, entry)
		self.count += 1

	def pop(self):
		(_, _, item) = hq.heappop(self.heap)
		return item

	def isEmpty(self):
		return len(self.heap) == 0

	def update(self, item, priority):
		# If item already in priority queue with higher priority, update its priority and rebuild the heap.
		# If item already in priority queue with equal or lower priority, do nothing.
		# If item not in priority queue, do the same thing as self.push.
		for index, (p, c, i) in enumerate(self.heap):
			if i == item:
				if p <= priority:
					break
				del self.heap[index]
				self.heap.append((priority, c, item))
				hq.heapify(self.heap)
				break
		else:
			self.push(item, priority)

def AStarSearch(start_coords, goal_coords, graph):
	tree = TreeList()
	visited = []
	Q = PriorityQueue()

	start_node = Node(start_coords, None, 0.)
	goal_node = Node(goal_coords, None, 0.)
	tree.insertNode(start_node)
	Q.push(start_node.loc, 0)

	while True:
		if Q.isEmpty():
			print(f'failed to find the path ...')
			return [] # fail the search

		node_loc = Q.pop()
		node = tree.getNode(node_loc)
		if node.loc == goal_coords:
			path = tree.formPath(node_loc)
			return path
		else:
			for nei in graph[node_loc]:
				new_node = Node(nei, node, 1 + node.cost)
				tree.insertNode(new_node)
				if nei not in visited:
					heur = abs(nei[0] - goal_coords[0]) + abs(nei[1] - goal_coords[1])
					# update Q
					Q.update(nei, new_node.cost + heur)
			# add node to visited
			visited.append(node_loc)

class localNav_Astar:
	def __init__(self, pose_range, coords_range):
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.local_map_margin = 10
		self.path_poses = None
		self.path_idx = -1 # record the index of the agent in the path

	def plan(self, agent_pose, subgoal_coords, occupancy_map):
		agent_coords = pose_to_coords(agent_pose, self.pose_range, self.coords_range)

		# get a local map of the occupancy map
		H, W = occupancy_map.shape
		(xmin, zmin, xmax, zmax), agent_local_coords, subgoal_local_coords = \
			self._decide_local_map_size(agent_coords, subgoal_coords, H, W)

		local_occupancy_map = occupancy_map[zmin:zmax, xmin:xmax]

		#'''
		plt.imshow(local_occupancy_map)
		plt.show()
		#'''
		H, W = local_occupancy_map.shape
		x = np.linspace(0, W-1, W)
		y = np.linspace(0, H-1, H)
		xv, yv = np.meshgrid(x, y)
		map_coords = np.stack((xv, yv), axis=2).astype(np.int16)

		# take the non-obj pixels
		mask_free = (local_occupancy_map != 1)
		free_map_coords = map_coords[mask_free]

		#===================== build the graph ======================
		roadmap = {}
		num_nodes = free_map_coords.shape[0]
		for i in range(num_nodes):
			neighbors = []
			x, y = free_map_coords[i]
			if y+1 < H and mask_free[y+1, x]:
				neighbors.append((x, y+1))
			if y-1 >= 0 and mask_free[y-1, x]:
				neighbors.append((x, y-1))
			if x-1 >= 0 and mask_free[y, x-1]:
				neighbors.append((x-1, y))
			if x+1 < W and mask_free[y, x+1]:
				neighbors.append((x+1, y))

			roadmap[tuple(free_map_coords[i])] = neighbors

		#print(f'roadmap = {roadmap}')
		path = AStarSearch(agent_local_coords, subgoal_local_coords, roadmap)
		#print(f'path = {path}')

		#========================== visualize the path ==========================
		#'''
		mask_new = mask_free.astype(np.int16)
		for loc in path:
			mask_new[loc[1], loc[0]] = 2
		
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
		# visualize gt semantic map
		ax[0].imshow(local_occupancy_map)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('local_occupancy_map')
		# visualize built semantic map
		ax[1].imshow(mask_new)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('planned path')
		plt.show()
		#'''

		#============================== convert path to poses ===================
		self.path_poses = []
		self.path_idx = 1
		for loc in path:
			pose = pxl_coords_to_pose((loc[0]+xmin, loc[1]+zmin), self.pose_range, self.coords_range)
			self.path_poses.append(pose)
		print(f'path_idx = {self.path_idx}, path_poses = {self.path_poses}')

	def next_action(self, agent_pose, occupancy_map):
		'''
		# visualize on occupancy map
		path_pose = self.path_poses[-1]
		agent_coords = pose_to_coords(agent_pose, self.pose_range, self.coords_range)

		for i in range(0, len(self.path_poses)):
			path_coords = pose_to_coords(self.path_poses[i], self.pose_range, self.coords_range)
			occupancy_map[path_coords[1], path_coords[0]] = 6

		occupancy_map[agent_coords[1], agent_coords[0]] = 6
		
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
		# visualize gt semantic map
		ax.imshow(occupancy_map)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.set_title('occupancy_map')
		plt.show()
		'''

		if self.path_idx >= len(self.path_poses):
			return ""

		# decide path_idx
		self.stg_x, self.stg_z = self.path_poses[self.path_idx]
		angle_st_goal = math.degrees(math.atan2(self.stg_x - agent_pose[0],
												self.stg_z - agent_pose[1]))
		angle_agent = (math.degrees(agent_pose[2])) % 360.0
		if angle_agent > 180:
			angle_agent -= 360

		relative_angle = (angle_agent - angle_st_goal) % 360.0
		if relative_angle > 180:
			relative_angle -= 360

		if relative_angle > 30 / 2.:
			action = "TURN_LEFT"  # Right
		elif relative_angle < -30 / 2.:
			action = "TURN_RIGHT"  # Left
		else:
			action = "MOVE_FORWARD"  # Forward

		if action == "MOVE_FORWARD":
			self.path_idx += 1

		return action

	def _decide_local_map_size(self, agent_coords, subgoal_coords, H, W):
		x1, z1 = agent_coords
		x2, z2 = subgoal_coords

		xmin = min(x1, x2)
		xmax = max(x1, x2)
		zmin = min(z1, z2)
		zmax = max(z1, z2)

		xmin = max(0, xmin - self.local_map_margin)
		xmax = min(W, xmax + self.local_map_margin)
		zmin = max(0, zmin - self.local_map_margin)
		zmax = min(H, zmax + self.local_map_margin)

		agent_local_coords = (agent_coords[0] - xmin, agent_coords[1] - zmin)
		subgoal_local_coords = (subgoal_coords[0] - xmin, subgoal_coords[1] - zmin)

		return (xmin, zmin, xmax, zmax), agent_local_coords, subgoal_local_coords



''' test
LN = localNav_Astar((-10.6, -17.5, 18.4, 10.6), (91, 159, 198, 258))
agent_map_pose = (6.6, 6.9, 2.36)
subgoal = np.array([84, 44], dtype=np.int16)
occupancy_map = np.load('local_occupancy_map.npy')
LN.plan(agent_map_pose, subgoal, occupancy_map)
'''
