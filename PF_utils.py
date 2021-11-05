import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
from baseline_utils import pxl_coords_to_pose_numpy, pose_to_coords, get_class_mapper, pxl_coords_to_pose
import skimage.measure
import cv2
from math import floor
from sklearn.mixture import GaussianMixture

cat2idx_dict = get_class_mapper()
idx2cat_dict = {v: k for k, v in cat2idx_dict.items()}

# load the GMM params
GMM_dict = {}
for k1 in list(cat2idx_dict.keys()):
    for k2 in list(cat2idx_dict.keys()):
        try:
            npy_file = np.load(f'output/GMM_obj_obj_params/GMM_params_{k1}_{k2}.npy', allow_pickle=True)
            params = npy_file.item()
        except:
            continue
        gm = GaussianMixture(n_components=3)
        gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(params['covariances']))
        gm.weights_ = params['weights']
        gm.means_ = params['means']
        gm.covariances_ = params['covariances']
        GMM_dict[(k1, k2)] = gm


def getLegalPositions(semantic_map, pose_range, coords_range):
    H, W = semantic_map.shape[:2]
    W_coords_range = range(0, W)
    H_coords_range = range(0, H)
    xv, yv = np.meshgrid(np.array(W_coords_range), np.array(H_coords_range))
    xv = xv.flatten()
    yv = yv.flatten()
    coords = np.stack((xv, yv), axis=1)
    print(f'xv.shape = {xv.shape}')
    print(f'yv.shape = {yv.shape}')
    print(f'coords.shape = {coords.shape}')
    poses = pxl_coords_to_pose_numpy(coords, pose_range, coords_range, cell_size=.1, flag_cropped=False)
    print(f'poses.shape = {poses.shape}')
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
    IGNORED_CLASS = [0, 1, 2, 17]
    cat_binary_map = observed_semantic_map.copy()
    for cat in IGNORED_CLASS:
        cat_binary_map = np.where(cat_binary_map==cat, -1, cat_binary_map)
    # run skimage to find the number of objects belong to this class
    instance_label, num_ins = skimage.measure.label(cat_binary_map, background=-1, connectivity=1, return_num=True)

    list_instances = []
    for idx_ins in range(1, num_ins+1):
        mask_ins = (instance_label==idx_ins)
        if np.sum(mask_ins) > 10: # should have at least 50 pixels
            #print(f'idx_ins = {idx_ins}')
            x_coords = xv[mask_ins]
            y_coords = yv[mask_ins]
            ins_center = (floor(np.median(x_coords)*.1), floor(np.median(y_coords)*.1))
            ins_cat = observed_semantic_map[int(y_coords[0]), int(x_coords[0])]
            ins = {}
            ins['center'] = ins_center
            ins['cat'] = ins_cat
            list_instances.append(ins)

    return list_instances

def visualize_GMM_dist(gm, h=200, w=200):
    min_X = -w/2 * .1
    max_X = w/2 * .1
    min_Z = -h/2 * .1
    max_Z = w/2 * .1
    x_grid = np.arange(min_X, max_X, 0.1)
    z_grid = np.arange(min_Z, max_Z, 0.1)
    locs = np.array(list(itertools.product(z_grid.tolist(), x_grid.tolist())))
    logprob = gm.score_samples(locs)
    pdf = np.exp(logprob)
    #prob_dist = pdf.reshape((h, w))
    return locs, pdf


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
        """
        total = self.total() * 1.
        #print(f'values = {self.values()}')
        if total > 0:
            for k, v in self.items():
                self[k] = v / total

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        """
        population = list(self.keys())
        weights = list(self.values())
        return random.choices(population, weights=weights)[0]

class ParticleFilter():
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, numParticles, semantic_map, pose_range, coords_range):
        self.setNumParticles(numParticles)
        self.k2 = 'table'
        self.H, self.W = semantic_map.shape[:2]
        self.semantic_map = semantic_map
        self.pose_range = pose_range
        self.coords_range = coords_range
        self.legalPositions = getLegalPositions(self.semantic_map, self.pose_range, self.coords_range)
        self.initializeUniformly()

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        
        num_particles = self.numParticles
        legal = self.legalPositions

        while num_particles > 0:
            if len(legal) >= num_particles:
                self.particles += legal
                num_particles = 0
            else:
                self.particles += legal[:num_particles]
                num_particles -= len(legal)

    def observeUpdate(self, observed_map):
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
        flag_observed_map = (observed_map > 0)
        semantic_map = self.semantic_map.copy()
        semantic_map[flag_observed_map == 0] = 0
        #plt.imshow(semantic_map)
        #plt.show()

        list_instances = compute_centers(semantic_map)
        for inst in list_instances: 
            inst_pose = pxl_coords_to_pose(inst['center'], self.pose_range, self.coords_range)
            k1 = idx2cat_dict[inst['cat']]
            # load GMM
            if (k1, self.k2) in GMM_dict:
                gm = GMM_dict[(k1, self.k2)]
                locs, prob_dist = visualize_GMM_dist(gm)
                n_locs = locs.shape[0]
                for j in range(n_locs):
                    locs_Z, locs_X = locs[j]
                    locs_Z += inst_pose[1]
                    locs_X += inst_pose[0]
                    if locs_X > self.pose_range[0] and locs_X < self.pose_range[2] and locs_Z > self.pose_range[1] and locs_Z < self.pose_range[3]:
                        particle_pos = (locs_X, locs_Z)
                        P_e_X = prob_dist[j]
                        if particle_pos in weights:
                            weights[particle_pos] += P_e_X
                        else:
                            weights[particle_pos] = P_e_X

        #=================================== resample ================================
        if weights.total() == 0: # corner case
            self.initializeUniformly()
        else:
            new_particles = []
            for _ in range(self.numParticles):
                new_particles.append(weights.sample())

            self.particles = new_particles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        
        particle_distribution = DiscreteDistribution()
        for loc in self.particles:
            if loc in particle_distribution:
                particle_distribution[loc] += 1
            else:
                particle_distribution[loc] = 1
        particle_distribution.normalize()
        return particle_distribution

    def visualizeBelief(self):
        dist_map = np.zeros((self.H, self.W))

        particle_distribution = self.getBeliefDistribution()
        for k, v in particle_distribution.items():
            x, z = pose_to_coords(k, self.pose_range, self.coords_range, cell_size=.1, flag_cropped=False)
            dist_map[z, x] = v

        return dist_map
