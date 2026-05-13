# Crab Nebula Volumetric Renderer
Physically-based volumetric rendering of the Crab Nebula from real astronomical data.
This project implements a custom volumetric renderer capable of reconstructing and rendering the Crab Nebula directly from observational spectroscopic datasets using ray marching and physically-based emission models.
Developed for the course **DH2323 — Computer Graphics** at KTH Royal Institute of Technology.

---

## Features

| Feature | Status |
|---|---|
| Volumetric ray marcher | ✅ |
| Color mapping | ✅ |
| Background star integration (Gaia DR3) | ✅ |
| Synchrotron emission rendering | ✅ |
| NanoVDB sparse renderer | ✅ |
| Camera orbit animation | ✅ |

---

## Results

### Nebula without stars
The first stage renders only the thermal filament structure reconstructed from the observational data.
<p align="center">
  <img src="results/no_star_no_syn.png" width="350">
</p>

---

### Nebula with synchrotron emission
The synchrotron component of the pulsar wind nebula is rendered separately and integrated as an independent emission channel.
<p align="center">
  <img src="results/no_star_yes_syn.png" width="350">
</p>

---

### Nebula with stars
Background stars from the Gaia DR3 catalogue are projected into world space and composited after volume integration.
<p align="center">
  <img src="results/yes_star_yes_syn.png" width="350">
</p>

---

## Rotating Animations

### Rotation without synchrotron
<p align="center">
  <img src="results/anim_noSyn.gif" width="350">
</p>

---

### Rotation with synchrotron
<p align="center">
  <img src="results/anim_syn.gif" width="350">
</p>

---
### Requirements

**Python** (Step 1):

```bash
pip install numpy astropy trimesh scipy noise
```

To save in NanoVDB format OpenVDB is necessary to be installed.

macOS (Homebrew):
```bash
brew install openvdb
pip install pyopenvdb
```

Linux (Ubuntu/Debian):
```bash
sudo apt install libopenvdb-dev
pip install pyopenvdb
```

**C++** (Step 2) — compiling C++17 with OpenMP and NanoVDB (included in OpenVDB).

macOS:
```bash
brew install gcc libomp openvdb
```

Linux:
```bash
sudo apt install g++ libomp-dev libopenvdb-dev
```

---

## Download dei dati
 
Before starting the rendering, downloads all necessary files and put them in `input_data/`.
 
| File | Source | Note |
|---|---|---|
| `3dmap_XYZ*.fits` | [thomasorb/M1_paper](https://github.com/thomasorb/M1_paper/tree/master) | All the `.fits` files from the repository |
| `crabNebula.obj` | [Chandra — Crab X-Ray 3D model](https://chandra.harvard.edu/deadstar/crab.html) | Download the file "Crab X-Ray 3D model" (OBJ) |
| `gaia_stars.csv` | [Gaia DR3](https://gea.esac.esa.int/archive/) | Query ADQL in the section below |
 
**Query Gaia DR3** — execute in the [Gaia ESA Archive](https://gea.esac.esa.int/archive/) to generate `gaia_stars.csv`:
 
```sql
SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < 7
ORDER BY phot_g_mean_mag ASC;
```
 
Download the result in CSV format and rename it with `gaia_stars.csv`.

### Step 1 — Voxelization

Put the files `.fits` and the mesh `crabNebula.obj` in the directory `input_data/`.

```bash
python step1_voxelize_fits.py --fits-dir input_data --output-dir output --resolution 512

python step1_voxelize_synch.py --obj input_data/crabNebula.obj --output-dir output --resolution 512
```

The output is saved in:
- `output/bin_512/` — binary raw grids 
- `output/nvdb_512/` — NanoVDB grids 

---

### Step 2 — Rendering

Choose the **binary** (more easy to compile) o **NanoVDB** (empty-space skipping, faster on sparse grids) version.

> **Note:** `gaia_stars.csv` needs to be in the working directory for both versions.

#### Binary version

```bash

g++-15 -fopenmp -std=c++17 step2_main_bin.cpp -o step2_main_bin

./step2_main_bin                              # 10 frame (default)
./step2_main_bin --frames 24                  # 24 frame
./step2_main_bin -f 1                         # single frame, no GIF
./step2_main_bin --bin-dir output/bin_256     # different resolution
```

#### Versione NanoVDB

```bash

g++-15 -fopenmp -std=c++17 \
    -I/opt/homebrew/Cellar/openvdb/13.0.0_1/include \
    step2_main_nvdb.cpp -o step2_main_nvdb


./step2_main_nvdb                              # 10 frame (default)
./step2_main_nvdb --frames 24                  # 24 frame
./step2_main_nvdb -f 1                         # single frame, no GIF
./step2_main_nvdb --nvdb-dir output/nvdb_256   # different resolution
```

The frames PPM and the animated GIF are saved respectively in `nebula_bin/` o `nebula_nvdb/`.

---

## Technologies

### Languages
- C++
- Python

### Libraries
- NanoVDB
- OpenVDB
- NumPy
- SciPy
- trimesh
- noise

---

## Data Sources
- SITELLE spectroscopic reconstruction of the Crab Nebula
- Gaia DR3 star catalogue
- NASA Crab Nebula 3D mesh

---

## Authors
- Giorgia Savo
- Francesco Filippo Stefanutti
