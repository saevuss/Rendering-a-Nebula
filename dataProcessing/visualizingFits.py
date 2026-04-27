from astropy.io import fits
import numpy as np

file = "fits/3dmap_XYZflux.fits"
with fits.open(file) as hdul:
    data = hdul[0].data          # shape: (4, 416573)
    X   = data[0]
    Y   = data[1]
    Z   = data[2]
    val = data[3]

print("\nPrinting INFO for: ", file)
print(f"Shape (rows, columns): {data.shape} \n") #each row is a point
print("column: minval → max val")
print(f"X: {X.min():.3f} → {X.max():.3f}")
print(f"Y: {Y.min():.3f} → {Y.max():.3f}")
print(f"Z: {Z.min():.3f} → {Z.max():.3f}")
print(f"val (flux): {val.min():.3f} → {val.max():.3f}\n")