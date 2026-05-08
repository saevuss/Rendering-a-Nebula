"""
fits_to_vdb.py
Converte i file FITS della Crab Nebula (point cloud sparsa) in un file .vdb.


I file FITS hanno shape (N, 4) con colonne [X, Y, Z, valore].
NON sono volumi 3D regolari — sono una point cloud sparsa che va prima
voxelizzata su una griglia regolare, poi salvata come VDB sparso.


Dipendenze:
    conda install -c conda-forge openvdb astropy numpy
    (il modulo Python si importa come 'openvdb', non 'pyopenvdb')
"""

import numpy as np
from astropy.io import fits
import os
import openvdb as vdb
import nanovdb
import nanovdb.io
import nanovdb.tools
import io as _io

RESOLUTION = 512
PRUNE_THRESHOLD = 0.0
OUTPUT_FILE = "crab_nebula.nvdb"

FILES = [
    ("fits/3dmap_XYZflux.fits",    "density"),
    ("fits/3dmap_XYZnii_ha.fits",  "nii_ha"),
    ("fits/3dmap_XYZsii_ha.fits",  "sii_ha"),
    ("fits/3dmap_XYZsii_sii.fits", "sii_sii"),
    ("fits/3dmap_XYZvel.fits",     "vel"),
]

def load_point_cloud(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data[:,0], data[:,1], data[:,2], data[:,3]

def compute_shared_bounds(files):
    x_min = y_min = z_min =  np.inf
    x_max = y_max = z_max = -np.inf
    for fits_path, _ in files:
        if not os.path.exists(fits_path):
            print(f"  [SKIP] {fits_path} non trovato")
            continue
        x, y, z, _ = load_point_cloud(fits_path)
        x_min = min(x_min, x.min()); x_max = max(x_max, x.max())
        y_min = min(y_min, y.min()); y_max = max(y_max, y.max())
        z_min = min(z_min, z.min()); z_max = max(z_max, z.max())
    return (x_min, y_min, z_min), (x_max, y_max, z_max)

def voxelize(x, y, z, val, bounds_min, bounds_max, resolution):
    bmin = np.array(bounds_min, dtype=np.float32)
    bmax = np.array(bounds_max, dtype=np.float32)
    extent = bmax - bmin
    ix = np.clip(np.floor((x - bmin[0]) / extent[0] * resolution).astype(np.int32), 0, resolution-1)
    iy = np.clip(np.floor((y - bmin[1]) / extent[1] * resolution).astype(np.int32), 0, resolution-1)
    iz = np.clip(np.floor((z - bmin[2]) / extent[2] * resolution).astype(np.int32), 0, resolution-1)
    flat_idx = iz * resolution * resolution + iy * resolution + ix
    grid_sum   = np.zeros(resolution**3, dtype=np.float64)
    grid_count = np.zeros(resolution**3, dtype=np.int32)
    np.add.at(grid_sum,   flat_idx, val)
    np.add.at(grid_count, flat_idx, 1)
    with np.errstate(invalid='ignore'):
        grid_avg = np.where(grid_count > 0, grid_sum / grid_count, 0.0).astype(np.float32)
    volume        = grid_avg.reshape((resolution, resolution, resolution))
    mask_occupied = (grid_count > 0).reshape((resolution, resolution, resolution))
    return volume, mask_occupied

def normalize_grid(volume, mask_occupied, grid_name):
    valid = volume[mask_occupied]
    if len(valid) == 0:
        print(f"  [WARN] {grid_name}: nessun valore positivo, griglia vuota")
        return volume
    if grid_name == "vel":
        v_min = valid.min()
        v_max = valid.max()
        print(f"  vel range fisico: {v_min:.1f} .. {v_max:.1f} km/s")
        normed = np.zeros_like(volume)
        normed[mask_occupied] = (valid - v_min) / (v_max - v_min + 1e-9)
        return normed.astype(np.float32)
    else:
        vmin = valid.min()
        vmax = valid.max()
        print(f"  range fisico: {vmin:.4f} .. {vmax:.4f}")
        log_vol = np.log1p(volume)
        log_max = np.log1p(vmax)
        normed  = log_vol / log_max
        normed[~mask_occupied] = 0.0
        return normed.astype(np.float32)

def convert_to_vdb_grid(fits_path, grid_name, bounds_min, bounds_max, resolution):
    print(f"\n[{grid_name}] Elaborazione {fits_path}...")
    if not os.path.exists(fits_path):
        print(f"  ERRORE: file non trovato")
        return None
    x, y, z, val = load_point_cloud(fits_path)
    print(f"  Punti letti: {len(val)}")
    print(f"  Valore: min={val.min():.4f}  max={val.max():.4f}")
    print(f"  Voxelizzazione su griglia {resolution}^3...")
    volume, mask_occupied = voxelize(x, y, z, val, bounds_min, bounds_max, resolution)
    occupied = np.count_nonzero(mask_occupied)
    total    = resolution**3
    print(f"  Voxel occupati: {occupied}/{total} ({100*occupied/total:.2f}%)")
    print(f"  Normalizzazione...")
    volume_norm = normalize_grid(volume, mask_occupied, grid_name)
    grid      = vdb.FloatGrid()
    grid.name = grid_name
    grid.copyFromArray(volume_norm)
    scale       = 60.0 / resolution
    grid.transform = vdb.createLinearTransform(voxelSize=scale)
    grid.transform.preTranslate((-30.0, -30.0, -30.0))
    grid.prune(PRUNE_THRESHOLD)
    print(f"  Voxel attivi dopo pruning: {grid.activeVoxelCount()}")
    return grid

if __name__ == "__main__":
    print("=== fits_to_vdb.py ===")
    print(f"Risoluzione: {RESOLUTION}^3")
    print(f"Output: {OUTPUT_FILE}\n")
    print("Calcolo bounding box comune...")
    bounds_min, bounds_max = compute_shared_bounds(FILES)
    print(f"  X: {bounds_min[0]:.3f} .. {bounds_max[0]:.3f}")
    print(f"  Y: {bounds_min[1]:.3f} .. {bounds_max[1]:.3f}")
    print(f"  Z: {bounds_min[2]:.3f} .. {bounds_max[2]:.3f}")

    vdb_grids = []
    for fits_path, name in FILES:
        g = convert_to_vdb_grid(fits_path, name, bounds_min, bounds_max, RESOLUTION)
        if g is not None:
            vdb_grids.append(g)

    if vdb_grids:
        print(f"\nConversione in NanoVDB → {OUTPUT_FILE}...")
              
        with open(OUTPUT_FILE, 'wb') as outf:
            for g in vdb_grids:
                handle = nanovdb.tools.openToNanoVDB(g)
                handle.floatGrid().setGridName(g.name)
                
                tmp = f"_tmp_{g.name}.nvdb"
                nanovdb.io.writeGrid(tmp, handle)
                with open(tmp, 'rb') as f:
                    outf.write(f.read())
                os.remove(tmp)
                print(f"  Scritto '{g.name}'")
        
        size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
        print(f"Successo! {OUTPUT_FILE} — {size_mb:.1f} MB")
  
    else:
        print("\nNessuna griglia prodotta.")
