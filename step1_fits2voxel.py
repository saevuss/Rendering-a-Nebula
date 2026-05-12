# step1_voxelize_fits.py is a python script that voxelizes the fits received
import numpy as np
from astropy.io import fits # type: ignore
import os
from step1_common_fun import RESOLUTION, log_stretch_natural, save_nvdb, save_bin, log_stretch, norm01 # type: ignore

FILES = [
    "fits/3dmap_XYZflux.fits",
    "fits/3dmap_XYZnii_ha.fits",
    "fits/3dmap_XYZsii_ha.fits",
    "fits/3dmap_XYZsii_sii.fits",
    "fits/3dmap_XYZvel.fits",
]

def compute_shared_bounds(fits_paths):
    xyz_min = np.full(3,  np.inf, dtype=np.float32)
    xyz_max = np.full(3, -np.inf, dtype=np.float32)
    for path in fits_paths:
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        with fits.open(path) as hdul:
            xyz = hdul[0].data[:, :3].astype(np.float32)
        xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
        xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
    center = (xyz_max + xyz_min) / 2.0
    scale  = (xyz_max - xyz_min).max() / 2.0
    print(f"global bounding box: min={xyz_min} max={xyz_max}")
    print(f"  center={center}  scale={scale:.4f}")
    return center, scale

def _scatter_average(idx, val, resolution):
    """Scatter-add + average of overlapping voxels (better explained in the report)"""
    grid   = np.zeros((resolution,) * 3, dtype=np.float32)
    counts = np.zeros_like(grid)
    np.add.at(grid,   (idx[:,2], idx[:,1], idx[:,0]), val)
    np.add.at(counts, (idx[:,2], idx[:,1], idx[:,0]), 1)
    mask = counts > 0
    grid[mask] /= counts[mask]
    return grid

def _xyz_to_idx(xyz, center, scale, resolution):
    """ from coordinate to index """
    xyz_norm = (xyz - center) / scale
    idx = ((xyz_norm + 1.0) / 2.0 * (resolution - 1)).astype(int)
    return np.clip(idx, 0, resolution - 1)

def voxelize(fits_path, resolution, center, scale, percentile_clip=99.5):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    xyz = data[:, :3].astype(np.float32)
    val = data[:, 3].astype(np.float32)

    vmax = np.percentile(val[val > 0], percentile_clip)
    val  = np.clip(val, 0, vmax)

    idx  = _xyz_to_idx(xyz, center, scale, resolution)
    grid = _scatter_average(idx, val, resolution)
    grid = log_stretch_natural(grid)
    grid = np.ascontiguousarray(grid.transpose(2, 1, 0))  # ZYX → XYZ
    return grid.astype('<f4')

def voxelize_vel(fits_path, resolution, center, scale):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    xyz = data[:, :3].astype(np.float32)
    val = data[:, 3].astype(np.float32)

    idx = _xyz_to_idx(xyz, center, scale, resolution)
    # idx[:, 0] = (resolution - 1) - idx[:, 0]   # flip RA

    grid = _scatter_average(idx, val, resolution)

    mask = grid != 0
    v_min = grid[mask].min()
    v_max = grid[mask].max()
    abs_max = max(abs(v_min), abs(v_max))
    grid[mask] = (grid[mask] / (abs_max + 1e-9)) * 0.5 + 0.5
    grid = np.ascontiguousarray(grid.transpose(2, 1, 0))  # ZYX → XYZ
    return grid.astype('<f4')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voxelizza file FITS in .bin e .nvdb")
    parser.add_argument("--resolution", type=int, default=512,
                        metavar="N", help="dim of volume (default: 512)")
    parser.add_argument("--fits-dir", default="input_data",
                        metavar="DIR", help="input delivery with .fits (default: input_data/)")
    parser.add_argument("--output-dir", default="output",
                        metavar="DIR", help="path to output delivery (default: output/)")
    args = parser.parse_args()

    RES      = args.resolution
    FITS_DIR = args.fits_dir
    OUT_DIR  = args.output_dir

    FITS_FILES = [
        f"{FITS_DIR}/3dmap_XYZflux.fits",
        f"{FITS_DIR}/3dmap_XYZnii_ha.fits",
        f"{FITS_DIR}/3dmap_XYZsii_ha.fits",
        f"{FITS_DIR}/3dmap_XYZsii_sii.fits",
        f"{FITS_DIR}/3dmap_XYZvel.fits",
    ]

    center, scale = compute_shared_bounds(FITS_FILES)

    grids = {
        "density": voxelize    (f"{FITS_DIR}/3dmap_XYZflux.fits",   RES, center, scale),
        "nii_ha":  voxelize    (f"{FITS_DIR}/3dmap_XYZnii_ha.fits",  RES, center, scale),
        "sii_ha":  voxelize    (f"{FITS_DIR}/3dmap_XYZsii_ha.fits",  RES, center, scale),
        "sii_sii": voxelize    (f"{FITS_DIR}/3dmap_XYZsii_sii.fits", RES, center, scale),
        "vel":     voxelize_vel(f"{FITS_DIR}/3dmap_XYZvel.fits",     RES, center, scale),
    }

    tag = f"{RES}"

    bin_dir  = os.path.join(OUT_DIR, f"bin_{tag}")
    nvdb_dir = os.path.join(OUT_DIR, f"nvdb_{tag}")

    os.makedirs(bin_dir,  exist_ok=True)
    os.makedirs(nvdb_dir, exist_ok=True)

    for name, vol in grids.items():
        #save_bin (vol, os.path.join(bin_dir,  f"{name}.bin"))
        save_nvdb(vol,  name, os.path.join(nvdb_dir, f"{name}.nvdb"))