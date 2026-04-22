from astropy.io import fits
import numpy as np
import glob
from scipy.ndimage import zoom

files = glob.glob('./data/**/*.fits', recursive=True)
data = fits.open(files[0])[1].data  # shape: (632, 29, 29)

#Nan is substituted with 0
data = np.nan_to_num(data, nan=0.0)

data = np.clip(data, 0, None)

#reshape to 128x128x128
nz, ny, nx = data.shape
zoom_factors = (128 / nz, 128 / ny, 128 / nx)
data_128 = zoom(data, zoom_factors, order=3)  # cubic interpolation

print("Shape finale:", data_128.shape)

# 3. Normalization in [0, 1]
data_128 = (data_128 - data_128.min()) / (data_128.max() - data_128.min())


# 4. Save as binary file
data_128 = data_128.astype(np.float32)
data_128.tofile('./data/nebula_density.raw')

# saves dimension
dims = np.array([128, 128, 128], dtype=np.int32)
dims.tofile('./data/nebula_dims.raw')

print("\Saved in ./data/nebula_density.raw")
print("Dimensioni (nz, ny, nx):", dims)