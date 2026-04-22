from astroquery.mast import Observations
from astropy.io import fits

# --- STEP 1: Download NGC 6720 (Ring Nebula) JWST s3d ---
obs = Observations.query_object("NGC 6720", radius="0.1 deg")

mask = (obs['dataproduct_type'] == 'cube') & (obs['obs_collection'] == 'JWST')
jwst = obs[mask]


products = Observations.get_product_list(jwst[:5])  # takes all products of the first 5 observations

s3d = products[
    ['s3d.fits' in str(f).lower() for f in products['productFilename']]
]

smallest = s3d[s3d['size'] == s3d['size'].min()]
Observations.download_products(smallest, download_dir='./data')
print("Download completed:", smallest['productFilename'][0])