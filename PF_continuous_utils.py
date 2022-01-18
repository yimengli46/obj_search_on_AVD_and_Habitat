import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from baseline_utils import pxl_coords_to_pose_numpy, pose_to_coords, get_class_mapper, pxl_coords_to_pose, pose_to_coords_numpy, apply_color_to_map
import skimage.measure
import cv2
from math import floor, sqrt
from sklearn.mixture import GaussianMixture
from navigation_utils import change_brightness

mode = 'semantic_prior'
flag_visualize_ins_weights = True
flag_visualize_peaks = True

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
def visualize_GMM_dist(weight, size, inst_pose, particles, particle_weights, flag_target=False):
	radius = size + 1.

	for particle in particles:
		particle_x, particle_z = particle
		dist = sqrt((particle_x - inst_pose[0])**2 + (particle_z - inst_pose[1])**2)
		P_e_X = 1.

		if not flag_target:
			if dist <= radius and dist >= size:
				P_e_X *= 0.5
			else:
				P_e_X *= .1
		else:
			if dist <= size:
				P_e_X *= 10.
			else:
				P_e_X *= .1

		if particle in particle_weights:
			particle_weights[particle] += P_e_X
		else:
			particle_weights[particle] = P_e_X

def confirm_nComponents(X):
	bics = []
	min_bic = 0
	counter = 1
	maximum_nComponents = min(len(X), 10)
	for i in range (1, maximum_nComponents): # test the AIC/BIC metric between 1 and 10 components
		gmm = GaussianMixture(n_components=counter, max_iter=1000, random_state=0, covariance_type = 'full').fit(X)
		bic = gmm.bic(X)
		bics.append(bic)
		if bic < min_bic or min_bic == 0:
			min_bic = bic
			opt_bic = counter

		counter += 1
	return opt_bic

class DiscreteDistribution(dict):
	"""
	A DiscreteDistribution models belief distributions and weight distributions
	over a finite set of discrete keys.
	"""
	def __getitem__(self, key):
		self.setdefault(key, 0)
		return dict.__getitem__(self, key)

	def copy(self):
		"""
		Return a copy of the distribution.
		"""
		return DiscreteDistribution(dict.copy(self))

	def argMax(self):
		"""
		Return the key with the highest value.
		"""
		if len(self.keys()) == 0:
			return None
		all = list(self.items())
		values = [x[1] for x in all]
		maxIndex = values.index(max(values))
		return all[maxIndex][0]

	def total(self):
		"""
		Return the sum of values for all keys.
		"""
		return float(sum(self.values()))

	def normalize(self):
		"""
		Normalize the distribution such that the total value of all keys sums
		to 1. The ratio of values for all keys will remain the same. In the case
		where the total value of the distribution is 0, do nothing.

		>>> dist = DiscreteDistribution()
		>>> dist['a'] = 1
		>>> dist['b'] = 2
		>>> dist['c'] = 2
		>>> dist['d'] = 0
		>>> dist.normalize()
		>>> list(sorted(dist.items()))
		[('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
		>>> dist['e'] = 4
		>>> list(sorted(dist.items()))
		[('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
		>>> empty = DiscreteDistribution()
		>>> empty.normalize()
		>>> empty
		{}
		"""
		"*** YOUR CODE HERE ***"
		total = self.total() * 1.
		#print(f'values = {self.values()}')
		if total > 0:
			for k, v in self.items():
				self[k] = v / total

	def sample(self, num_samples=1):
		"""
		Draw a random sample from the distribution and return the key, weighted
		by the values associated with each key.

		>>> dist = DiscreteDistribution()
		>>> dist['a'] = 1
		>>> dist['b'] = 2
		>>> dist['c'] = 2
		>>> dist['d'] = 0
		>>> N = 100000.0
		>>> samples = [dist.sample() for _ in range(int(N))]
		>>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
		0.2
		>>> round(samples.count('b') * 1.0/N, 1)
		0.4
		>>> round(samples.count('c') * 1.0/N, 1)
		0.4
		>>> round(samples.count('d') * 1.0/N, 1)
		0.0
		"""
		"*** YOUR CODE HERE ***"
		population = list(self.keys())
		weights = list(self.values())
		return random.choices(population, weights=weights, k=num_samples)



class ParticleFilter():
	"""
	A particle filter for approximately tracking a single ghost.
	"""
	def __init__(self, numParticles, semantic_map, pose_range, coords_range):
		self.k2 = 'refrigerator'
		self.H, self.W = semantic_map.shape[:2]
		self.numParticles = numParticles
		self.semantic_map = semantic_map
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.initializeUniformly()

	def initializeUniformly(self):
		"""
		Initialize a list of particles. Use self.numParticles for the number of
		particles. Use self.legalPositions for the legal board positions where
		a particle could be located. Particles should be evenly (not randomly)
		distributed across positions in order to ensure a uniform prior. Use
		self.particles for the list of particles.
		"""
		self.particles = []
		min_X, min_Z = pxl_coords_to_pose((self.coords_range[0], self.coords_range[1]), self.pose_range, self.coords_range, flag_cropped=False)
		max_X, max_Z = pxl_coords_to_pose((self.coords_range[2], self.coords_range[3]), self.pose_range, self.coords_range, flag_cropped=False)
		X = np.random.uniform(min_X, max_X, size=self.numParticles)
		Z = np.random.uniform(min_Z, max_Z, size=self.numParticles)
		X = np.round(X, decimals=2).reshape(-1, 1) # 10000 x 1
		Z = np.round(Z, decimals=2).reshape(-1, 1) # 10000 x 1
		particles = np.hstack((X, Z)) # 10000 x 2
		#print(f'particles.shape = {particles.shape}')
		self.particles = list(map(tuple, particles))

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
		
		weights = DiscreteDistribution()

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
		#fig.tight_layout()
		plt.title('visualize detected instance centers')
		plt.show()
		#assert 1==2
		#'''

		#========================================= Compute Priors ===========================================
		for idx, inst in enumerate(list_instances):
			inst_pose = pxl_coords_to_pose(inst['center'], self.pose_range, self.coords_range, flag_cropped=True)
			k1 = idx2cat_dict[inst['cat']]
			if k1 == self.k2: # target object is detected
				visualize_GMM_dist(weight_k1, inst['size'], inst_pose, self.particles, weights, flag_target=True)
			else:
				weight_k1 = get_cooccurred_object_weight(self.k2, k1)
				# load GMM
				if weight_k1 > 0:
					visualize_GMM_dist(weight_k1, inst['size'], inst_pose, self.particles, weights)

		#==================================== visualization ====================================
		if True:
			color_semantic_map = apply_color_to_map(semantic_map)
			color_semantic_map = change_brightness(color_semantic_map, observed_area_flag, value=60)

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 90))
			ax[0].imshow(color_semantic_map)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].scatter(x_coord_lst, z_coord_lst, s=30, c='white', zorder=2)
			
			dist_map = self.visualizeWeights(weights)
			ax[1].imshow(dist_map, vmin=0.)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			#fig.tight_layout()
			plt.title('particle weights distribution before ignoring explored area')
			plt.show()

		#================================== zero out weights on explored areas================================
		mask_explored = np.logical_and(observed_area_flag, self.semantic_map != cat2idx_dict[self.k2])
		mask_zero_out = mask_explored
		for k in weights:
			coords = pose_to_coords(k, self.pose_range, self.coords_range, flag_cropped=True)
			if mask_zero_out[coords[1], coords[0]] == 1:
				weights[k] = 0.

		weights.normalize()

		#=================================== resample ================================
		if flag_visualize_ins_weights:
			color_semantic_map = apply_color_to_map(semantic_map)
			color_semantic_map = change_brightness(color_semantic_map, observed_area_flag, value=60)

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 90))
			ax[0].imshow(color_semantic_map)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].scatter(x_coord_lst, z_coord_lst, s=30, c='white', zorder=2)
			
			dist_map = self.visualizeWeights(weights)
			ax[1].imshow(dist_map, vmin=0.)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			#fig.tight_layout()
			plt.title('particle weights distribution after weight normalization')
			plt.show()

		if weights.total() == 0: # corner case
			self.initializeUniformly()
		else:
			poses = weights.sample(self.numParticles)
			print(f'len(poses) = {len(poses)}')
			self.particles = poses

		#===================================== finding the peak ================================
		gm = self.find_peak()
		peaks = gm.means_
		peaks_coords = pose_to_coords_numpy(peaks, self.pose_range, self.coords_range)
		#===================================== visualize particles =============================
		dist_map = self.visualizeBelief()
		print(f'sum dist_map = {np.sum(dist_map)}')

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 120))
		ax.imshow(dist_map)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.scatter(peaks_coords[:, 0], peaks_coords[:, 1], s=30, c='white', zorder=2)
		#fig.tight_layout()
		plt.title('particles (white nodes are peaks of Gaussian component)')
		plt.show()

	def visualizeBelief(self):
		dist_map = np.zeros((self.H, self.W))
		particles = np.array(self.particles)
		coords = pose_to_coords_numpy(particles, self.pose_range, self.coords_range)
		tuple_coords = list(map(tuple, coords))
		for coord in tuple_coords:
			dist_map[coord[1], coord[0]] += 1
		return dist_map

	def visualizeWeights(self, weights):
		dist_map = np.zeros((self.H, self.W))
		for k in weights:
			coords = pose_to_coords(k, self.pose_range, self.coords_range)
			dist_map[coords[1], coords[0]] = weights[k]
		return dist_map

	def find_peak(self):
		np_particles = np.array(self.particles)
		num_GMM_components = confirm_nComponents(np_particles)
		gm = GaussianMixture(n_components=num_GMM_components).fit(np_particles)
		print(f'gm.weights = {gm.weights_}')
		#print(f'gm.means = {gm.means_}')
		#print(f'gm.covariances = {gm.covariances_}')
		return gm