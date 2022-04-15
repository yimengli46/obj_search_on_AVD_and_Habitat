import numpy as np 
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage

class Frontier(object):
	def __init__(self, points):
		"""Initialized with a 2xN numpy array of points (the grid cell
		coordinates of all points on frontier boundary)."""
		inds = np.lexsort((points[0, :], points[1, :]))
		sorted_points = points[:, inds]
		self.props_set = False
		self.is_from_last_chosen = False
		self.is_obstructed = False
		self.prob_feasible = 1.0
		self.delta_success_cost = 0.0
		self.exploration_cost = 0.0
		self.negative_weighting = 0.0
		self.positive_weighting = 0.0

		self.counter = 0
		self.last_observed_pose = None

		# Any duplicate points should be eliminated (would interfere with
		# equality checking).
		dupes = []
		for ii in range(1, sorted_points.shape[1]):
			if (sorted_points[:, ii - 1] == sorted_points[:, ii]).all():
				dupes += [ii]
		self.points = np.delete(sorted_points, dupes, axis=1)

		# Compute and cache the hash
		self.hash = hash(self.points.tobytes())

	def set_props(self,
				  prob_feasible,
				  is_obstructed=False,
				  delta_success_cost=0,
				  exploration_cost=0,
				  positive_weighting=0,
				  negative_weighting=0,
				  counter=0,
				  last_observed_pose=None,
				  did_set=True):
		self.props_set = did_set
		self.just_set = did_set
		self.prob_feasible = prob_feasible
		self.is_obstructed = is_obstructed
		self.delta_success_cost = delta_success_cost
		self.exploration_cost = exploration_cost
		self.positive_weighting = positive_weighting
		self.negative_weighting = negative_weighting
		self.counter = counter
		self.last_observed_pose = last_observed_pose

	@property
	def centroid(self):
		return self.get_centroid()

	def get_centroid(self):
		"""Returns the point that is the centroid of the frontier"""
		centroid = np.mean(self.points, axis=1)
		return centroid

	def get_frontier_point(self):
		"""Returns the point that is on the frontier that is closest to the
		actual centroid"""
		center_point = np.mean(self.points, axis=1)
		norm = np.linalg.norm(self.points - center_point[:, None], axis=0)
		ind = np.argmin(norm)
		return self.points[:, ind]

	def get_distance_to_point(self, point):
		norm = np.linalg.norm(self.points - point[:, None], axis=0)
		return norm.min()

	def __hash__(self):
		return self.hash

	def __eq__(self, other):
		return hash(self) == hash(other)

def mask_grid_with_frontiers(occupancy_grid, frontiers, do_not_mask=None):
	"""Mask grid cells in the provided occupancy_grid with the frontier points
	contained with the set of 'frontiers'. If 'do_not_mask' is provided, and
	set to either a single frontier or a set of frontiers, those frontiers are
	not masked."""

	if do_not_mask is not None:
		# Ensure that 'do_not_mask' is a set
		if isinstance(do_not_mask, Frontier):
			do_not_mask = set([do_not_mask])
		elif not isinstance(do_not_mask, set):
			raise TypeError("do_not_mask must be either a set or a Frontier")
		masking_frontiers = frontiers - do_not_mask
	else:
		masking_frontiers = frontiers

	masked_grid = occupancy_grid.copy()
	for frontier in masking_frontiers:
		masked_grid[frontier.points[0, :],
					frontier.points[1, :]] = 2

	return masked_grid

scene_name = 'Allensville_0'
occ_map_path = f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{scene_name}'
occupancy_map = np.load(f'{occ_map_path}/BEV_occupancy_map.npy')

occupancy_grid = np.where(occupancy_map==1, 0, occupancy_map) # free cell
occupancy_grid = np.where(occupancy_map==0, 1, occupancy_grid) # occupied cell
occupancy_grid[0:50, 0:50] = -1

COLLISION_VAL = 1
FREE_VAL = 0
UNOBSERVED_VAL = -1
OBSTACLE_THRESHOLD = 0.5 * (COLLISION_VAL + FREE_VAL) 
group_inflation_radius=0

filtered_grid = scipy.ndimage.maximum_filter(np.logical_and(
		occupancy_grid < OBSTACLE_THRESHOLD, occupancy_grid == FREE_VAL), size=3)
frontier_point_mask = np.logical_and(filtered_grid, occupancy_grid == UNOBSERVED_VAL)

if group_inflation_radius < 1:
	inflated_frontier_mask = frontier_point_mask
else:
	inflated_frontier_mask = gridmap.utils.inflate_grid(frontier_point_mask,
		inflation_radius=group_inflation_radius, obstacle_threshold=0.5,
		collision_val=1.0) > 0.5

# Group the frontier points into connected components
labels, nb = scipy.ndimage.label(inflated_frontier_mask)

# Extract the frontiers
frontiers = set()
for ii in range(nb):
	raw_frontier_indices = np.where(
		np.logical_and(labels == (ii + 1), frontier_point_mask))
	frontiers.add(
		Frontier(
			np.concatenate((raw_frontier_indices[0][None, :],
							raw_frontier_indices[1][None, :]),
						   axis=0)))

masked_grid = mask_grid_with_frontiers(occupancy_grid, frontiers)