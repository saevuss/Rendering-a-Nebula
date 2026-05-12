# step1_voxelize_synch.py voxelizes the mesh of the synchrotrone, the mathematic explanation
# can be found in the report of the project
import numpy as np
import os
import argparse
import trimesh # type: ignore
from scipy.ndimage import gaussian_filter
from noise import pnoise3 # type: ignore
from common_fun import RESOLUTION, save_nvdb, save_bin, log_stretch, norm01 # type: ignore

SYNCH_FRAC = 0.25

def build_rotation():
    inc   = 27.0 * np.pi / 180.0
    PA_NW = (124.0 - 180.0) * np.pi / 180.0
    target_axis = np.array([-np.sin(PA_NW), +np.cos(PA_NW), -np.sin(inc)])
    target_axis /= np.linalg.norm(target_axis)
    obj_jet_axis = np.array([-0.50354443, -0.6559642, 0.56227571])
    obj_jet_axis /= np.linalg.norm(obj_jet_axis)
    v  = np.cross(obj_jet_axis, target_axis)
    s  = np.linalg.norm(v)
    c  = np.dot(obj_jet_axis, target_axis)
    vx = np.array([[0,-v[2],v[1]], [v[2],0,-v[0]], [-v[1],v[0],0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-10))

def load_and_rotate_mesh(path, R):
    mesh  = trimesh.load(path, force='mesh')
    verts = mesh.vertices.astype(np.float64)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    scale  = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
    verts  = (verts - center) / scale   # [-1, 1]
    verts  = verts @ R.T
    verts *= SYNCH_FRAC
    return trimesh.Trimesh(vertices=verts.astype(np.float32),
                           faces=mesh.faces, process=False)

def build_synchrotron_volume(mesh, resolution):
    pts, _ = trimesh.sample.sample_surface(mesh, 3_000_000)
    idx = ((pts + 1.0) / 2.0 * (resolution - 1)).astype(int)
    idx = np.clip(idx, 0, resolution - 1)

    grid = np.zeros((resolution,) * 3, dtype=np.float32)
    np.add.at(grid, (idx[:,2], idx[:,1], idx[:,0]), 1.0)
    grid = gaussian_filter(grid, sigma=3.0)

    mask = grid > grid.max() * 0.04
    coords = np.mgrid[0:resolution, 0:resolution, 0:resolution].astype(np.float32)
    coords /= resolution
    noise_vol = np.vectorize(
        lambda z, y, x: pnoise3(x*4, y*4, z*4, octaves=4)
    )(coords[0], coords[1], coords[2])
    noise_vol = norm01(noise_vol)

    grid_noisy = np.zeros_like(grid)
    grid_noisy[mask] = grid[mask] * (0.6 + 0.4 * noise_vol[mask])
    grid_final = gaussian_filter(grid_noisy, sigma=1.2)
    grid_final = log_stretch(norm01(grid_final), factor=4.0)
    return grid_final.astype('<f4')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="voxelization of the OBJ mesh of the synchrotrone")
    parser.add_argument("--resolution", type=int, default=512,
                        metavar="N", help=" dim of volume (default: 512)")
    parser.add_argument("--obj", default="input_data/crabNebula.obj",
                        metavar="FILE", help="path to file .obj (default: input_data/crabNebula.obj)")
    parser.add_argument("--output-dir", default="output",
                        metavar="DIR", help="path to output delivery (default: output/)")
    args = parser.parse_args()

    RES     = args.resolution
    OUT_DIR = args.output_dir

    R    = build_rotation()
    mesh = load_and_rotate_mesh(args.obj, R)
    vol  = build_synchrotron_volume(mesh, RES)

    bin_dir  = os.path.join(OUT_DIR, f"bin_{RES}")
    nvdb_dir = os.path.join(OUT_DIR, f"nvdb_{RES}")

    os.makedirs(bin_dir,  exist_ok=True)
    os.makedirs(nvdb_dir, exist_ok=True)

    save_bin (vol,               os.path.join(bin_dir,  "synchrotron.bin"))
    save_nvdb(vol, "synchrotron", os.path.join(nvdb_dir, "syn.nvdb"))
    print("  [synchrotron] ok")