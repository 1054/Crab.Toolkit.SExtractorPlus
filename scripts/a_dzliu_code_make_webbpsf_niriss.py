#!/usr/bin/env python
# 20221202
# 20250405
#   WebbPSF Is Now Space Telescope PSF (STPSF)
#   January 28, 2025
#   https://www.stsci.edu/contents/news/jwst/2025/webbpsf-is-now-space-telescope-psf-stpsf
# 
# stpsf 2.0.0
#   pip install -e git+https://github.com/spacetelescope/poppy.git#egg=poppy \
#               -e git+https://github.com/spacetelescope/stpsf.git#egg=stpsf
# 
import os, copy, shutil
import numpy as np
#os.environ['WEBBPSF_PATH'] = os.path.expanduser('~/Data/JWST-WebbPSF/webbpsf-data')
os.environ['STPSF_PATH'] = os.path.expanduser('~/Data/JWST-WebbPSF/stpsf-data') # 2.0.0 https://stpsf.readthedocs.io/en/stable/installation.html#data-install
import stpsf
webbpsf = stpsf
from astropy.io import fits
from astropy.table import Table
oversample = 1
overwrite = False
fov_pixels = 51
pixsc_arcsec = None
#pixsc_arcsec = 0.030
#for filter_name in ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F300M', 'F335M', 'F356W', 'F360M', 'F410M', 'F444W']:
for filter_name in ['F150W']:
    suffix = ''
    nc = webbpsf.NIRISS()
    nc.filter = filter_name
    if pixsc_arcsec is not None:
        nc.pixelscale = pixsc_arcsec
        suffix += '_pixsc{}'.format(pixsc_arcsec).replace('.','p')
    nc.options['parity'] = 'odd' # default
    output_file = f"webbpsf_NIRISS_{filter_name}_tiny{suffix}.fits"
    if not os.path.isfile(output_file) or overwrite:
        psf = nc.calc_psf(output_file, fov_pixels=fov_pixels, oversample=oversample)
        print(f"Output to {output_file}")
    
    output_psf_file = f"webbpsf_NIRISS_{filter_name}_tiny{suffix}.psf"
    if not os.path.isfile(output_psf_file) or overwrite:
        with fits.open(output_file) as hdul:
            #psf_primary_hdu = fits.PrimaryHDU(header=hdul[0].header)
            #psf_primary_hdu = fits.PrimaryHDU()
            psf_data = hdul[1].data
        psf_data = psf_data.reshape((1, 1, fov_pixels, fov_pixels)).astype(np.float32)
        psf_table = Table(dict(PSF_MASK=psf_data))
        psf_bintable = fits.BinTableHDU(psf_table, name='PSF_DATA')
        #psf_coldef = fits.ColDefs([fits.Column(name='PSF_MASK' , format='961E' , dim='(31, 31, 1)' , array=psf_data)])
        #psf_bintable = fits.BinTableHDU.from_columns(psf_coldef)
        psf_bintable.header['LOADED'] = (99, 'PSFEx')
        psf_bintable.header['ACCEPTED'] = (99, 'PSFEx')
        psf_bintable.header['CHI2'] = (1.0, 'PSFEx')
        psf_bintable.header['POLNAXIS'] = (0, 'PSFEx')
        psf_bintable.header['POLNGRP'] = (0, 'PSFEx')
        psf_bintable.header['PSF_FWHM'] = (0.05, 'PSF FWHM')
        psf_bintable.header['PSF_SAMP'] = (0.5, 'Sampling step of the PSF data')
        psf_bintable.header['PSFNAXIS'] = 3
        psf_bintable.header['PSFAXIS1'] = fov_pixels
        psf_bintable.header['PSFAXIS2'] = fov_pixels
        psf_bintable.header['PSFAXIS3'] = 1
        psf_hdul = fits.HDUList()
        psf_hdul.append(psf_bintable)
        psf_hdul.writeto(output_psf_file, overwrite=True)
        print(f"Output to {output_psf_file}")


