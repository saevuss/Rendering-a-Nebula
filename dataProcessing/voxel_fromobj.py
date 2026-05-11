import trimesh

import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise3
import openvdb as vdb
import nanovdb
import nanovdb.io
import nanovdb.tools

resolution = 512
SYNCH_FRAC = 0.25 #the synchrotrone occupies about 25/30% of nebula

# ── orientamento jet ──────────────────────────────────────────────────────────
inc   = 27.0 * np.pi / 180.0
PA_NW = (124.0 - 180.0) * np.pi / 180.0

target_axis = np.array([
    -np.sin(PA_NW),
    +np.cos(PA_NW),
    -np.sin(inc)
])
target_axis /= np.linalg.norm(target_axis)

obj_jet_axis = np.array([-0.50354443, -0.6559642, 0.56227571])
obj_jet_axis /= np.linalg.norm(obj_jet_axis)

v  = np.cross(obj_jet_axis, target_axis)
s  = np.linalg.norm(v)
c  = np.dot(obj_jet_axis, target_axis)
vx = np.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]])
R  = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-10))

# ── carica, ruota e scala il mesh ─────────────────────────────────────────────
mesh  = trimesh.load("objs/crabNebula.obj", force='mesh')
verts = mesh.vertices.astype(np.float64)

center = (verts.max(axis=0) + verts.min(axis=0)) / 2
scale  = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
verts  = (verts - center) / scale   # normalizza in [-1, 1]
verts  = verts @ R.T                # ruota
verts *= SYNCH_FRAC                 # scala al 30% della nebulosa

mesh_rot = trimesh.Trimesh(vertices=verts.astype(np.float32),
                            faces=mesh.faces, process=False)

# ── voxelizza: campiona la superficie e riempi l'interno ──────────────────────
pts, _ = trimesh.sample.sample_surface(mesh_rot, 3_000_000)

# stessa pipeline del voxelizer FITS: normalizza → indici → grid
idx = ((pts + 1.0) / 2.0 * (resolution - 1)).astype(int)
idx = np.clip(idx, 0, resolution - 1)

grid   = np.zeros((resolution,) * 3, dtype=np.float32)
counts = np.zeros_like(grid)
np.add.at(grid,   (idx[:,2], idx[:,1], idx[:,0]), 1.0)
np.add.at(counts, (idx[:,2], idx[:,1], idx[:,0]), 1.0)


grid = gaussian_filter(grid, sigma=3.0)
mask = grid > grid.max() * 0.04

#applies noise so that the internal volume of the synchrotrone has variations of density
coords = np.mgrid[0:resolution, 0:resolution, 0:resolution].astype(np.float32)
coords /= resolution

#for each voxel it is calculated a value of perlin noise. The result is a volume of 
#casual but organic and contiguous values
noise_vol = np.vectorize(lambda z,y,x: pnoise3(x*4, y*4, z*4, octaves=4))(
    coords[0], coords[1], coords[2])
noise_vol = (noise_vol - noise_vol.min()) / (noise_vol.max() - noise_vol.min())

#normalization
grid_noisy = np.zeros_like(grid)
grid_noisy[mask] = grid[mask] * (0.6 + 0.4 * noise_vol[mask])

# gaussian filter to smooth the borders of the mask
grid_final = gaussian_filter(grid_noisy, sigma=1.2)
grid_final /= grid_final.max()
#log scretch to compress high values and increase lower ones
grid_final = np.log1p(grid_final * 4.0) / np.log1p(4.0)

vdb_grid = vdb.FloatGrid()
vdb_grid.name = "synchrotron"
vdb_grid.copyFromArray(grid_final)

# stessa transform degli altri nvdb: voxelSize = 60/512, translation = (-30, -30, -30)
scale = 60.0 / resolution
vdb_grid.transform = vdb.createLinearTransform(voxelSize=scale)
vdb_grid.transform.preTranslate((-30.0, -30.0, -30.0))
vdb_grid.prune(1e-4)

# converti in NanoVDB e salva
handle = nanovdb.tools.openToNanoVDB(vdb_grid)
nanovdb.io.writeGrid("nvdb_512_bound_shared/syn.nvdb", handle)
print("Salvato: syn.nvdb")


grid_final.astype('<f4').tofile("bin_512/synchrotron.bin")
print("Salvato: bin_512/synchrotron.bin")