import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from baseline_utils import pxl_coords_to_pose_numpy, pose_to_coords, get_class_mapper, pxl_coords_to_pose, pose_to_coords_numpy, apply_color_to_map
import skimage.measure
import cv2
from math import floor
from sklearn.mixture import GaussianMixture
from navigation_utils import change_brightness

mode = 'semantic_prior'
flag_visualize_ins_weights = True


cat2idx_dict = get_class_mapper()
idx2cat_dict = {v: k for k, v in cat2idx_dict.items()}
print(f'idx2cat = {idx2cat_dict}')

#================================= load the weight prior =================
weight_prior = np.load(f'output/semantic_prior/weight_prior.npy', allow_pickle=True).item()


def get_cooccurred_object_weight(target_obj, relevant_obj):
	if target_obj in weight_prior:
		weights = weight_prior[target_obj]
		for a, b in weights:
			if a == relevant_obj:
				return b
	return 0.

'''
	Get initial legal positions from the grid.
	Currently this function is never used.
''' 
def getLegalPositions(semantic_map, pose_range, coords_range):
	H, W = semantic_map.shape[:2]
	W_coords_range = range(0, W)
	H_coords_range = range(0, H)
	xv, yv = np.meshgrid(np.array(W_coords_range), np.array(H_coords_range))
	xv = xv.flatten()
	yv = yv.flatten()
	coords = np.stack((xv, yv), axis=1)
	#print(f'xv.shape = {xv.shape}')
	#print(f'yv.shape = {yv.shape}')
	#print(f'coords.shape = {coords.shape}')
	poses = pxl_coords_to_pose_numpy(coords, pose_range, coords_range, cell_size=.1, flag_cropped=True)
	#print(f'poses.shape = {poses.shape}')
	poses = list(map(tuple, poses))
	return poses

def compute_centers(observed_semantic_map):
	H, W = observed_semantic_map.shape
	observed_semantic_map = cv2.resize(observed_semantic_map, (int(W*10), int(H*10)), interpolation=cv2.INTER_NEAREST)
	H, W = observed_semantic_map.shape
	x = np.linspace(0, W-1, W)
	y = np.linspace(0, H-1, H)
	xv, yv = np.meshgrid(x, y)
	#====================================== compute centers of semantic classes =====================================
	IGNORED_CLASS = [0, 40]
	cat_binary_map = observed_semantic_map.copy()
	for cat in IGNORED_CLASS:
		cat_binary_map = np.where(cat_binary_map==cat, -1, cat_binary_map)
	# run skimage to find the number of objects belong to this class
	instance_label, num_ins = skimage.measure.label(cat_binary_map, background=-1, connectivity=1, return_num=True)

	list_instances = []
	for idx_ins in range(1, num_ins+1):
		mask_ins = (instance_label == idx_ins)
		if np.sum(mask_ins) > 100: # should have at least 50 pixels
			#print(f'idx_ins = {idx_ins}')
			x_coords = xv[mask_ins]
			y_coords = yv[mask_ins]
			ins_center = (floor(np.median(x_coords)*.1), floor(np.median(y_coords)*.1))
			ins_cat = observed_semantic_map[int(y_coords[0]), int(x_coords[0])]
			ins = {}
			ins['center'] = ins_center # in coordinates, rather than size
			ins['cat'] = ins_cat

			# compute object radius size
			dist = np.sqrt((x_coords * .1 - ins_center[0])**2 + (y_coords * .1 - ins_center[1])**2)
			size = np.max(dist)
			ins['size'] = size * .1 # in meters, not coordinates

			print(f'ins_center = {ins_center}, cat = {ins_cat}, class = {idx2cat_dict[ins_cat]}, size = {size}')
			list_instances.append(ins)

	return list_instances

'''
size is the distance from the center to the boundary of the object
weight is the frequency weight between obj1 and target obj
'''
def visualize_GMM_dist(weight=1., size=1.):
	radius = size + 1.
	min_X = -radius
	max_X = radius
	min_Z = -radius
	max_Z = radius
	x_grid = np.arange(min_X, max_X, 0.1)
	z_grid = np.arange(min_Z, max_Z, 0.1)
	xv, yv = np.meshgrid(x_grid, z_grid)
	h, w = xv.shape
	xv = xv.flatten()
	yv = yv.flatten()
	locs = np.stack((yv, xv), axis=1)
	dists = np.sqrt(locs[:, 0]**2 + locs[:, 1]**2)
	dists = dists.reshape(-1, 1)
	pdf = np.ones(dists.shape) #/ dists.shape[0]
	pdf[dists <= size] = 0.
	pdf[dists > radius] = 0.
	#pdf = pdf / np.sum(pdf) #normalize it
	# prob_dist
	if False:
		prob_dist = pdf.reshape((h, w))
		plt.imshow(prob_dist)#, vmin=.0, vmax=.2)
		plt.show()
	locs_XZ = np.zeros(locs.shape)
	locs_XZ[:, 0] = locs[:, 1] # x
	locs_XZ[:, 1] = locs[:, 0] # z
	return locs_XZ, pdf

class DiscreteDistribution_grid(object):
	def __init__(self, H, W):
		self.H = H
		self.W = W
		self.grid = np.ones((self.H, self.W))

	def __getitem__(self, key):
		return self.grid[key[1], key[0]]

	def copy(self):
		"""
		Return a copy of the distribution.
		"""
		new_DD = DiscreteDistribution_grid(self.H, self.W)
		new_DD.grid = self.grid.copy()
		return new_DD

	def argMax(self):
		"""
		Return the key with the highest value.
		"""
		max_idx = np.unravel_index(self.grid.argmax(), self.grid.shape)
		return max_idx

	def total(self):
		"""
		Return the sum of values for all keys.
		"""
		return np.sum(self.grid)

	def normalize(self):
		"""
		Normalize the distribution such that the total value of all keys sums
		to 1. The ratio of values for all keys will remain the same. In the case
		where the total value of the distribution is 0, do nothing.
		"""
		total = self.total()
		#print(f'values = {self.values()}')
		if total > 0:
			self.grid = self.grid / total

	def sample(self, num_samples=100):
		"""
		Draw a random sample from the distribution and return the key, weighted
		by the values associated with each key.
		"""
		self.normalize()
		x_grid = np.arange(0, self.W, 1)
		z_grid = np.arange(0, self.H, 1)
		xv, yv = np.meshgrid(x_grid, z_grid)
		xv = xv.flatten()
		yv = yv.flatten()
		population = np.stack((xv, yv), axis=1)
		# 2d array to list of tuple
		#population = list(zip(population[:, 0], population[:, 1]))
		weights = self.grid.flatten()
		#coords = random.choices(population, weights=weights, k=num_samples)
		#return np.array([*coords])
		coords_idx = np.random.choice(list(range(population.shape[0])), size=num_samples, p=weights)
		coords = population[coords_idx]
		return coords
		#population = list(self.keys())
		#weights = list(self.values())
		#return random.choices(population, weights=weights)[0]



class ParticleFilter():
	"""
	A particle filter for approximately tracking a single ghost.
	"""
	def __init__(self, numParticles, semantic_map, pose_range, coords_range):
		self.k2 = 'refrigerator'
		self.H, self.W = semantic_map.shape[:2]
		self.setNumParticles(numParticles)
		self.semantic_map = semantic_map
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.legalPositions = getLegalPositions(self.semantic_map, self.pose_range, self.coords_range)
		self.initializeUniformly()

	def setNumParticles(self, numParticles):
		self.numParticles = numParticles
		assert numParticles == self.H * self.W

	def initializeUniformly(self):
		"""
		Initialize a list of particles. Use self.numParticles for the number of
		particles. Use self.legalPositions for the legal board positions where
		a particle could be located. Particles should be evenly (not randomly)
		distributed across positions in order to ensure a uniform prior. Use
		self.particles for the list of particles.
		"""
		self.particles = np.ones((self.H, self.W))

	def observeUpdate(self, observed_area_flag):
		"""
		Update beliefs based on the distance observation and Pacman's position.

		The observation is the noisy Manhattan distance to the ghost you are
		tracking.

		There is one special case that a correct implementation must handle.
		When all particles receive zero weight, the list of particles should
		be reinitialized by calling initializeUniformly. The total method of
		the DiscreteDistribution may be useful.
		"""
		
		weights = DiscreteDistribution_grid(self.H, self.W)

		#=================================== observe =================================
		semantic_map = self.semantic_map.copy()
		semantic_map[observed_area_flag == False] = 0
		mask_observed_and_non_obj = np.logical_and(observed_area_flag, semantic_map == 0)
		semantic_map[mask_observed_and_non_obj] = 40
		color_semantic_map = apply_color_to_map(semantic_map)
		#plt.imshow(color_semantic_map)
		#plt.show()

		list_instances = compute_centers(semantic_map)
		print(f'num_instances = {len(list_instances)}')

		#=============================== visualize detected instance centers ======================
		#'''
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 120))
		ax.imshow(color_semantic_map)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		x_coord_lst = []
		z_coord_lst = []
		for idx, inst in enumerate(list_instances):
			inst_coords = inst['center']
			x_coord_lst.append(inst_coords[0])
			z_coord_lst.append(inst_coords[1])
		ax.scatter(x_coord_lst, z_coord_lst, s=30, c='white', zorder=2)
		fig.tight_layout()
		plt.title('visualize detected instance centers')
		plt.show()
		#assert 1==2
		#'''

		#========================================= Compute Priors ===========================================
		for idx, inst in enumerate(list_instances):
			inst_pose = pxl_coords_to_pose(inst['center'], self.pose_range, self.coords_range, flag_cropped=True)
			k1 = idx2cat_dict[inst['cat']]
			if k1 == self.k2: # target object is detected
				locs, prob_dist = visualize_GMM_dist(weight_k1, inst['size'])
				prob_dist *= 10000
				locs[:, 1] += inst_pose[1]
				locs[:, 0] += inst_pose[0]
				coords = pose_to_coords_numpy(locs, self.pose_range, self.coords_range, flag_cropped=True)
				# find coords in the range
				mask_z = np.logical_and(coords[:, 1] >= 0, coords[:, 1] < self.H)
				mask_x = np.logical_and(coords[:, 0] >= 0, coords[:, 0] < self.W)
				mask_xz = np.logical_and.reduce((mask_z, mask_x))
				locs = locs[mask_xz, :]
				prob_dist = prob_dist[mask_xz]
				coords = coords[mask_xz, :]
				#print(f'later, coords.shape = {coords.shape}')
				#print(f'weight.grid.shape = {weights.grid.shape}')
				
				for j in range(coords.shape[0]):
					#print(f'coords[{j}]={coords[j]}')
					weights.grid[coords[j, 1], coords[j, 0]] += prob_dist[j]
			else:
				weight_k1 = get_cooccurred_object_weight(self.k2, k1)
				# load GMM
				if weight_k1 > 0:
					locs, prob_dist = visualize_GMM_dist(weight_k1, inst['size'])
					#print(f'locs.shape = {locs.shape}')
					#=================== shift the probability grid centered at the object center ===============
					locs[:, 1] += inst_pose[1]
					locs[:, 0] += inst_pose[0]
					coords = pose_to_coords_numpy(locs, self.pose_range, self.coords_range, flag_cropped=True)
					# find coords in the range
					mask_z = np.logical_and(coords[:, 1] >= 0, coords[:, 1] < self.H)
					mask_x = np.logical_and(coords[:, 0] >= 0, coords[:, 0] < self.W)
					mask_xz = np.logical_and.reduce((mask_z, mask_x))
					locs = locs[mask_xz, :]
					prob_dist = prob_dist[mask_xz]
					coords = coords[mask_xz, :]
					#print(f'later, coords.shape = {coords.shape}')
					#print(f'weight.grid.shape = {weights.grid.shape}')
					
					for j in range(coords.shape[0]):
						#print(f'coords[{j}]={coords[j]}')
						weights.grid[coords[j, 1], coords[j, 0]] += prob_dist[j]
				
		#==================================== visualization ====================================
		if True:
			color_semantic_map = apply_color_to_map(semantic_map)
			color_semantic_map = change_brightness(color_semantic_map, observed_area_flag, value=100)

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 90))
			ax[0].imshow(color_semantic_map)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].scatter(x_coord_lst, z_coord_lst, s=30, c='white', zorder=2)
			
			dist_map = weights.grid
			ax[1].imshow(dist_map, vmin=0.)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			fig.tight_layout()
			plt.title('dist_map distribution before ignoring explored area')
			plt.show()

		#================================== zero out weights on explored areas================================
		mask_explored = np.logical_and(observed_area_flag, self.semantic_map != cat2idx_dict[self.k2])
		#mask_unexplored = (self.semantic_map == 0)
		#mask_zero_out = np.logical_or(mask_explored, mask_outside)
		mask_zero_out = mask_explored
		weights.grid[mask_explored] = 0.

		weights.normalize()
		#=================================== resample ================================
		if flag_visualize_ins_weights:
			color_semantic_map = apply_color_to_map(semantic_map)
			color_semantic_map = change_brightness(color_semantic_map, observed_area_flag, value=100)

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 90))
			ax[0].imshow(color_semantic_map)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].scatter(x_coord_lst, z_coord_lst, s=30, c='white', zorder=2)
			
			dist_map = weights.grid
			ax[1].imshow(dist_map, vmin=0.)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			fig.tight_layout()
			plt.title('probability map after weight normalization ...')
			plt.show()

		if weights.total() == 0: # corner case
			self.initializeUniformly()
		else:
			coords = weights.sample(self.numParticles)
			new_particles = np.ones((self.H, self.W))
			for j in range(coords.shape[0]):
				new_particles[coords[j, 1], coords[j, 0]] += 1
			
			self.particles = new_particles
			plt.imshow(self.particles, vmin=0.0)
			plt.title('particles')
			plt.show()
			#plt.close()

	'''
	def getBeliefDistribution(self):
		"""
		Return the agent's current belief state, a distribution over ghost
		locations conditioned on all evidence and time passage. This method
		essentially converts a list of particles into a belief distribution.
		
		This function should return a normalized distribution.
		"""

		particle_distribution = DiscreteDistribution_grid(self.H, self.W)
		particle_distribution.grid = self.particles.copy()
		particle_distribution.normalize()
		
		return particle_distribution
	'''

	def visualizeBelief(self):
		dist_map = np.zeros((self.H, self.W))

		particle_distribution = self.particles
		#total = np.sum(particle_distribution)
		#particle_distribution = particle_distribution / total
		assert dist_map.shape == particle_distribution.shape
		dist_map = particle_distribution

		return dist_map
