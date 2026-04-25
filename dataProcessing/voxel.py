'''
Voxelization is the process of converting a point cloud into a regular 3D grid, where each cell
 (voxel) aggregates the values of the points that fall within it — analogous to pixelating an 
 image, but in 3D. It is often required by volume renderers that expect a uniform grid rather 
 than scattered coordinates.
'''

import numpy as np
from astropy.io import fits
import os

def voxelize(fits_path, resolution,  percentile_clip=99.5):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data #rough array of points
    
    xyz = data[:, :3].astype(np.float32)
    val = data[:, 3].astype(np.float32)

    #clip before voxelising, this is necessary because if there is one single point
    #which is far more luminous than others, it would result in a single pixel and the rest black
    vmax = np.percentile(val[val > 0], percentile_clip)
    val = np.clip(val, 0, vmax)

    #normalization of xyz from their original range [-1.5, 1.5] to [0, resolution-1]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    #every point is now an index in the 3d grid
    idx = ((xyz - mins) / (maxs - mins)*(resolution -1)).astype(int)
    idx = np.clip(idx, 0, resolution -1)

    # creation of the empty grid
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)


    # if more than 1 point is in the same voxel, it takes an average
    counts = np.zeros_like(grid)
    np.add.at(grid, (idx[:,2], idx[:,1], idx[:,0]), val)
    np.add.at(counts, (idx[:,2], idx[:,1], idx[:,0]), 1)
    mask = counts > 0
    grid[mask] /= counts[mask] #average in those cells where there is more than one point
    
    # normalize [0, 1]
    if grid.max() > 0:
        grid /= grid.max()
    grid = np.log1p(grid*9.0)/np.log1p(9.0) #necessary to have a more realistic distribution, because
    #without the line above, even after clip p99.5, if there was a single voxel with very high value, all the other would have
    #resulted to be near to 0
    return grid.astype('<f4') #to resolve endianness

density = voxelize("fits/3dmap_XYZflux.fits", resolution=128) #calling the function we have defined
nii_ha    = voxelize("fits/3dmap_XYZnii_ha.fits", resolution=128) #calling the function we have defined
sii_ha    = voxelize("fits/3dmap_XYZsii_ha.fits", resolution=128) #calling the function we have defined
sii_sii    = voxelize("fits/3dmap_XYZsii_sii.fits", resolution=128) #calling the function we have defined
vel    = voxelize("fits/3dmap_XYZnii_ha.fits", resolution=128) #calling the function we have defined

os.makedirs("bin", exist_ok=True)
density.tofile("bin/density.bin")
nii_ha.tofile("bin/nii_ha.bin")
sii_ha.tofile("bin/sii_ha.bin")
sii_sii.tofile("bin/sii_sii.bin")
vel.tofile("bin/vel.bin")