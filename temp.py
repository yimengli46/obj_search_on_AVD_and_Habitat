import numpy as np
import matplotlib.pyplot as plt

weight=.5
size=1.

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
pdf = np.ones(dists.shape) / dists.shape[0]
pdf[dists <= size] = 0.
pdf[dists > radius] = 0.
pdf = pdf / np.sum(pdf) #normalize it
pdf = weight * pdf
# prob_dist
if True:
	prob_dist = pdf.reshape((h, w))
	plt.imshow(prob_dist)#, vmin=.0, vmax=.2)
	plt.show()
locs_XZ = np.zeros(locs.shape)
locs_XZ[:, 0] = locs[:, 1] # x
locs_XZ[:, 1] = locs[:, 0] # z