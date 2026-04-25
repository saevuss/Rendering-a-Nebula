# Rendering-a-Nebulae
A nebula renderer built in C++ that uses raymarching to simulate physically-grounded light transport through procedurally generated volumetric density fields. Designed for CPU execution, the system applies algorithmic optimizations to achieve high-quality images of procedurally generated nebulae without relying on dedicated GPU hardware.

### Data - Crab Nebula (M1)

3D point clouds of the Crab Nebula from [Martin et al. (2021)](https://arxiv.org/pdf/2101.02709), available at [thomasorb/M1_paper](https://github.com/thomasorb/M1_paper/tree/master). Each FITS file contains **416,573 points** with 4 columns: `X`, `Y`, `Z` (spatial coordinates, normalized in galactic units, range ≈ −1.5 → +1.5) and a scalar value.

| File | Description |
|------|-------------|
| `3dmap_XYZflux.fits` | Total emission flux — the main volume used for 3D rendering |
| `3dmap_XYZvel.fits` | Radial velocity (km/s) — maps gas depth to reconstruct 3D structure |
| `3dmap_XYZnii_ha.fits` | [N II]/Hα ratio — highlights nitrogen-rich filaments |
| `3dmap_XYZsii_ha.fits` | [S II]/Hα ratio — highlights sulfur-rich filaments |
| `3dmap_XYZsii_sii.fits` | [S II] doublet ratio — traces electron density variations |


### Data processing
Inspiration from 
- [How To voxelize meshes and pointclouds in python](https://towardsdatascience.com/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d/)
- [Point‑Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/pdf/1907.03739.pdf)
- [Scalable desktop visualization of very large radio data cubes](https://arxiv.org/pdf/1510.03589)
