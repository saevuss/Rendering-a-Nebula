
'''
rough obj has arbitrary coordinates, in particular, in the file we have:
    - X goes from -100 to +69 
    - Y goes from -88 to 101
    - Z goes from -105 to 72

and the geometric center is [-15.5, 6.3, -16.5]. 

On the other side, the FITS, already in range [-1.5, 1-5]. 

So before any operation, it is necessary to bring the dataset in the same reference system.

center is the average point of the bounding box, so it is substracted to move the object
in the origin. 
'''

import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter

# axe of the jet
obj_jet_axis = np.array([-0.504, -0.656, 0.562])
obj_jet_axis /= np.linalg.norm(obj_jet_axis)

target_axis = np.array([-0.653, -0.342, 0.676])
target_axis /= np.linalg.norm(target_axis)

# Rotazione che porta obj_jet_axis → target_axis
v = np.cross(obj_jet_axis, target_axis)
s = np.linalg.norm(v)
c = np.dot(obj_jet_axis, target_axis)
vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

resolution = 128
sigma      = 0.8
SYNCH_FRAC = 0.14 #occupies only 0.25% of the cube where is phusicaly

mesh = trimesh.load("objs/crabNebula.obj", force='mesh')
verts = mesh.vertices.astype(np.float32)

center = (verts.max(axis=0) + verts.min(axis=0)) / 2
scale  = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
verts  = (verts - center) / scale

verts = (verts @ R.T).astype(np.float32)
verts *= SYNCH_FRAC

mesh_rot = trimesh.Trimesh(vertices=verts, faces=mesh.faces)

# the mesh is only a surface (triangles), not a filled volume, so this function
# distributes points uniformly on triangles, proportionally to their area
points, _ = trimesh.sample.sample_surface(mesh_rot, 3_000_000)

pts = np.clip(points, -1.0, 1.0)
idx = ((pts + 1.0) / 2.0 * (resolution - 1)).astype(int)
idx = np.clip(idx, 0, resolution - 1)

grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
np.add.at(grid, (idx[:,2], idx[:,1], idx[:,0]), 1.0)

# to 
grid = gaussian_filter(grid, sigma=sigma)
grid /= grid.max()
grid = np.log1p(grid * 9.0) / np.log1p(9.0)
grid.astype('<f4').tofile("bin_128/synchrotron.bin")