import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter

# --- Asse jet fisico nel world space del ray marcher ---
# PA = 124° (Nord→Est), inclinazione = 27° verso osservatore
# flip X perché FITS ha X=Ovest, ray marcher ha X=Est
inc = 27.0  * np.pi / 180.0
PA_NW = (124.0 - 180.0) * np.pi / 180.0  # = -56°

target_axis = np.array([
    +np.sin(PA_NW),   # RA
    +np.cos(PA_NW),   # Dec
    -np.sin(inc)      # LoS
])
target_axis /= np.linalg.norm(target_axis)
print("target_axis:", target_axis)

# --- Asse principale del mesh ---
obj_jet_axis = np.array([-0.50354443, -0.6559642, 0.56227571])
obj_jet_axis /= np.linalg.norm(obj_jet_axis)

# --- Rotazione obj_jet_axis → target_axis ---
v = np.cross(obj_jet_axis, target_axis)
s = np.linalg.norm(v)
c = np.dot(obj_jet_axis, target_axis)
vx = np.array([[0, -v[2], v[1]],
               [v[2], 0, -v[0]],
               [-v[1], v[0], 0]])
R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2 + 1e-10))

# --- Carica e trasforma il mesh ---
resolution = 512
sigma      = 0.8
SYNCH_FRAC = 0.30  # era 0.14 — corretto: PWN ≈ 1/3 raggio nebulosa

mesh = trimesh.load("objs/crabNebula.obj", force='mesh')
verts = mesh.vertices.astype(np.float32)

center = (verts.max(axis=0) + verts.min(axis=0)) / 2
scale  = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
verts  = (verts - center) / scale   # normalizza in [-1, 1]
verts  = (verts @ R.T).astype(np.float32)  # ruota
verts[:, 0] *= -1  # flip RA per coerenza con voxelizer
verts  *= SYNCH_FRAC                # scala alla dimensione fisica corretta

mesh_rot = trimesh.Trimesh(vertices=verts, faces=mesh.faces)

points, _ = trimesh.sample.sample_surface(mesh_rot, 3_000_000)
pts = np.clip(points, -1.0, 1.0)
idx = ((pts + 1.0) / 2.0 * (resolution - 1)).astype(int)
idx = np.clip(idx, 0, resolution - 1)

grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
np.add.at(grid, (idx[:, 2], idx[:, 1], idx[:, 0]), 1.0)

grid = gaussian_filter(grid, sigma=sigma)
grid /= grid.max()
grid = np.log1p(grid * 9.0) / np.log1p(9.0)
grid.astype('<f4').tofile("bin_512/synchrotron.bin")

# print("YNCH_FRAC =", SYNCH_FRAC)
# print("Verts range dopo trasformazione:")
# print("X:", verts[:, 0].min(), "→", verts[:, 0].max())
# print("Y:", verts[:, 1].min(), "→", verts[:, 1].max())
# print("Z:", verts[:, 2].min(), "→", verts[:, 2].max())






