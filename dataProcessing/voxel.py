'''
voxel.py
Voxelization is the process of converting a point cloud into a regular 3D grid, where each cell
 (voxel) aggregates the values of the points that fall within it — analogous to pixelating an 
 image, but in 3D. It is often required by volume renderers that expect a uniform grid rather 
 than scattered coordinates.
'''

import numpy as np
from astropy.io import fits
import os
import openvdb as vdb
import nanovdb
import nanovdb.io
import nanovdb.tools
import os

FILES = [
    "fits/3dmap_XYZflux.fits",
    "fits/3dmap_XYZnii_ha.fits",
    "fits/3dmap_XYZsii_ha.fits",
    "fits/3dmap_XYZsii_sii.fits",
    "fits/3dmap_XYZvel.fits",
]

def compute_shared_bounds(fits_paths):
    'reads coords XYZ of all the files and returns'
    'center and shared scales'
    xyz_min = np.full(3,  np.inf, dtype=np.float32)
    xyz_max = np.full(3, -np.inf, dtype=np.float32)
 
    for path in fits_paths:
        if not os.path.exists(path):
            print(f"[SKIP] {path} non trovato")
            continue
        with fits.open(path) as hdul:
            xyz = hdul[0].data[:, :3].astype(np.float32)
        xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
        xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
 
    center = (xyz_max + xyz_min) / 2.0
    scale  = (xyz_max - xyz_min).max() / 2.0   # ← isotropo: un solo scalare
 
    print(f"global bounding box :")
    print(f"  min = {xyz_min}")
    print(f"  max = {xyz_max}")
    print(f"  center = {center}  scale = {scale:.4f}")
    return center, scale


def voxelize(fits_path, resolution, center, scale, percentile_clip=99.5):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data #rough array of points
    
    xyz = data[:, :3].astype(np.float32)
    val = data[:, 3].astype(np.float32)

    #clip before voxelising, this is necessary because if there is one single point
    #which is far more luminous than others, it would result in a single pixel and the rest black
    vmax = np.percentile(val[val > 0], percentile_clip)
    val = np.clip(val, 0, vmax)

    # (1) isotropic normaliation
    xyz_norm = (xyz - center) / scale  # now center and scale is taken from input (calculated with function of shared bounding_box)

    # (2) index mapping
    idx_raw = (xyz_norm + 1.0) / 2.0 * (resolution - 1)
    idx = idx_raw.astype(int)
    idx = np.clip(idx, 0, resolution - 1)

    # creation of the empty grid
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)


    # (3) Scatter and average
    # if more than 1 point is in the same voxel, it takes an average
    counts = np.zeros_like(grid)
    np.add.at(grid, (idx[:,2], idx[:,1], idx[:,0]), val)
    np.add.at(counts, (idx[:,2], idx[:,1], idx[:,0]), 1)
    mask = counts > 0
    grid[mask] /= counts[mask] #average in those cells where there is more than one point
    
    # (4) Dynamic range compression
    # normalize [0, 1]
    if grid.max() > 0:
        grid /= grid.max()
    # log1p(x) = log(1+x)
    grid = np.log1p(grid*9.0)/np.log1p(9.0) #necessary to have a more realistic distribution, because
    #without the line above, even after clip p99.5, if there was a single voxel with very high value, all the other would have
    #resulted to be near to 0
    return grid.astype('<f4') #to resolve endianness

def voxelize_vel(fits_path, resolution, center, scale):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    xyz = data[:, :3].astype(np.float32)
    val = data[:, 3].astype(np.float32)  # km/s, range [-1814, +1737]

    idx_raw = (xyz - center) / scale
    idx_raw = (idx_raw + 1.0) / 2.0 * (resolution - 1)
    idx = idx_raw.astype(int)
    idx[:, 0] = (resolution - 1) - idx[:, 0]  # flip RA
    idx = np.clip(idx, 0, resolution - 1)

    grid   = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    counts = np.zeros_like(grid)
    np.add.at(grid,   (idx[:,2], idx[:,1], idx[:,0]), val)
    np.add.at(counts, (idx[:,2], idx[:,1], idx[:,0]), 1)
    mask = counts > 0
    grid[mask] /= counts[mask]

    # Normalizza in [0, 1] preservando il segno
    # 0.0 = max blueshift, 0.5 = gas fermo, 1.0 = max redshift
    vmin, vmax = grid[mask].min(), grid[mask].max()
    grid[mask] = (grid[mask] - vmin) / (vmax - vmin)
    # voxel vuoti restano 0.0 — nel C++ già gestiti con `if (vel > 0.f)`

    return grid.astype('<f4')

def save_as_nvdb(volume_np, grid_name, output_path, voxel_size=60.0/512):
    grid = vdb.FloatGrid()
    grid.name = grid_name
    grid.copyFromArray(volume_np)
    
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)
    grid.transform.preTranslate((-30.0, -30.0, -30.0))
    grid.prune(0.0)
    
    handle = nanovdb.tools.openToNanoVDB(grid)
    handle.floatGrid().setGridName(grid_name)
    nanovdb.io.writeGrid(output_path, handle)

if __name__ == "__main__":
    RESOLUTION = 512
 
    # ① shared bounding box, calculated one time
    center, scale = compute_shared_bounds(FILES)
 
    # ② voxelization
    density = voxelize("fits/3dmap_XYZflux.fits",   RESOLUTION, center, scale)
    nii_ha  = voxelize("fits/3dmap_XYZnii_ha.fits",  RESOLUTION, center, scale)
    sii_ha  = voxelize("fits/3dmap_XYZsii_ha.fits",  RESOLUTION, center, scale)
    sii_sii = voxelize("fits/3dmap_XYZsii_sii.fits", RESOLUTION, center, scale)
    vel     = voxelize_vel("fits/3dmap_XYZvel.fits",  RESOLUTION, center, scale)
 
    # ③ saving on .bin
    os.makedirs("bin_512_bound_shared/", exist_ok=True)
    density.tofile("bin_512_bound_shared/density.bin")
    nii_ha .tofile("bin_512_bound_shared/nii_ha.bin")
    sii_ha .tofile("bin_512_bound_shared/sii_ha.bin")
    sii_sii.tofile("bin_512_bound_shared/sii_sii.bin")
    vel    .tofile("bin_512_bound_shared/vel.bin")
 
    # ④ saving on .nvdb
    grids = {
        "density": density,
        "nii_ha":  nii_ha,
        "sii_ha":  sii_ha,
        "sii_sii": sii_sii,
        "vel":     vel,
    }
 
    os.makedirs("nvdb_512_bound_shared/", exist_ok=True)
    for name, volume in grids.items():
        out = f"nvdb_512_bound_shared/{name}.nvdb"
        save_as_nvdb(volume, name, out)
        print(f"Saved {out}")
