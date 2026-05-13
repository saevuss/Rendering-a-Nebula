# step1_common.py is a file containing all the functions that are 
# used both by step1_fits2voxel.py and step1_mesh2voxel.py

import numpy as np
import openvdb as vdb
import nanovdb, nanovdb.io, nanovdb.tools # type: ignore
import os

RESOLUTION  = 512
VOXEL_SIZE  = 60.0 / RESOLUTION      # 60 ly / 512 voxel
TRANSLATION = (-30.0, -30.0, -30.0)  # centers volume in world space

def save_nvdb(volume_np: np.ndarray, grid_name: str, output_path: str) -> None:
    """ Converts a numpy array float32 in a NanoVDB and saves it"""
    grid = vdb.FloatGrid()
    grid.name = grid_name
    grid.copyFromArray(volume_np)
    grid.transform = vdb.createLinearTransform(voxelSize=VOXEL_SIZE)
    grid.transform.postTranslate(TRANSLATION)
    grid.prune(0.0)
    handle = nanovdb.tools.openToNanoVDB(grid)
    handle.floatGrid().setGridName(grid_name)
    nanovdb.io.writeGrid(output_path, handle)

def save_bin(volume_np: np.ndarray, output_path: str) -> None:
    """ Saves the volume as a file .bin """
    volume_np.astype('<f4').tofile(output_path)

def log_stretch(grid: np.ndarray, factor: float = 9.0) -> np.ndarray:
    """ dynamic range compression (better explained in the report): log1p(x * factor) / log1p(factor)."""
    return np.log1p(grid * factor) / np.log1p(factor)

def log_stretch_natural(grid: np.ndarray) -> np.ndarray:
    """ Logaritmic natural auto-consistent compression. Maps the raw data 
    in [0, 1] rescaling it with the logarithm of the maximum real peak. """
    m = grid.max()
    if m <= 0:
        return grid
    return np.log1p(grid) / np.log1p(m)

def norm01(grid: np.ndarray) -> np.ndarray:
    """ Normalization in [0, 1]"""
    m = grid.max()
    return grid / m if m > 0 else grid