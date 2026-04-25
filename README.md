# Rendering-a-Nebulae
A nebula renderer built in C++ that uses raymarching to simulate physically-grounded light transport through procedurally generated volumetric density fields. Designed for CPU execution, the system applies algorithmic optimizations to achieve high-quality images of procedurally generated nebulae without relying on dedicated GPU hardware.

### Data - Crab Nebula (M1)

3D maps of the Crab Nebula from [Martin et al. (2021)](https://arxiv.org/pdf/2101.02709), available at [thomasorb/M1_paper](https://github.com/thomasorb/M1_paper/tree/master). Each FITS file is a 3D cube where two axes are spatial (X, Y) and the third is line-of-sight velocity (Z), used for volumetric visualization of the nebula.

| File | Description |
|------|-------------|
| `3dmap_XYZflux.fits` | Total emission flux — the main volume used for 3D rendering |
| `3dmap_XYZvel.fits` | Radial velocity (km/s) — maps gas depth to reconstruct 3D structure |
| `3dmap_XYZnii_ha.fits` | [N II]/Hα ratio — highlights nitrogen-rich filaments |
| `3dmap_XYZsii_ha.fits` | [S II]/Hα ratio — highlights sulfur-rich filaments |
| `3dmap_XYZsii_sii.fits` | [S II] doublet ratio — traces electron density variations |
