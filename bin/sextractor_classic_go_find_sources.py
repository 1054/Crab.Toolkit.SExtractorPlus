#!/usr/bin/env python
# 

# binary: sextractor_classic

import os, sys, re, copy, glob, shutil
import click
import numpy as np
import subprocess
from astropy.io import fits
from astropy.table import Table




@click.command()
@click.argument('image_file')
def main(
        image_file,
    ):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(os.path.dirname(script_dir), 'data', 'sextractor_classic_data')
    data_dir = os.path.dirname(os.path.abspath(image_file))
    data_name = os.path.splitext(os.path.basename(image_file))[0]
    
    sci_data = None
    rms_data = None
    wht_data = None
    sci_header = None
    rms_header = None
    wht_header = None
    main_header = None
    with fits.open(image_file) as hdul:
        main_header = copy.deepcopy(hdul[0].header)
        for hdu in hdul:
            # read first valid hdu as sci data
            if sci_data is None and hdu.data is not None and len(hdu.data.shape) >= 2:
                sci_data = copy.copy(hdu.data)
                sci_header = copy.deepcopy(hdu.header)
            # if the hdu has an extension name SCI, use it as the sci data
            if 'EXTNAME' in hdu.header:
                if hdu.header['EXTNAME'] == 'SCI':
                    sci_data = copy.copy(hdu.data)
                    sci_header = copy.deepcopy(hdu.header)
                elif hdu.header['EXTNAME'] == 'ERR':
                    rms_data = copy.copy(hdu.data)
                    rms_header = copy.deepcopy(hdu.header)
                elif hdu.header['EXTNAME'] == 'WHT':
                    wht_data = copy.copy(hdu.data)
                    wht_header = copy.deepcopy(hdu.header)
    
    # 
    if sci_data is None:
        raise Exception('Could not read SCI data from the input fits file {!r}'.format(image_file))
    
    # mask out zero-weight pixels
    if wht_data is not None:
        mask_zero_weight = (wht_data==0)
        if np.count_nonzero(mask_zero_weight) > 0:
            sci_data[mask_zero_weight] = np.nan
            rms_data[mask_zero_weight] = np.nan
    
    # create working directory
    working_dir = os.path.join(data_dir, data_name+'_run_sextractor_classic_dir')
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    
    # prepare images
    sci_file = os.path.join(working_dir, 'sci_data.fits')
    if not os.path.isfile(sci_file):
        hdul = fits.HDUList([fits.PrimaryHDU(header=main_header), fits.ImageHDU(data=sci_data, header=sci_header, name='SCI')])
        hdul.writeto(sci_file, overwrite=True)
    
    if wht_data is not None:
        wht_file = os.path.join(working_dir, 'wht_data.fits')
        if not os.path.isfile(wht_file):
            hdul = fits.HDUList([fits.PrimaryHDU(header=main_header), fits.ImageHDU(data=wht_data, header=wht_header, name='WHT')])
            hdul.writeto(wht_file, overwrite=True)
    
    if rms_data is not None:
        rms_file = os.path.join(working_dir, 'rms_data.fits')
        if not os.path.isfile(rms_file):
            hdul = fits.HDUList([fits.PrimaryHDU(header=main_header), fits.ImageHDU(data=rms_data, header=rms_header, name='RMS')])
            hdul.writeto(rms_file, overwrite=True)
    
    # copy param files
    for default_filename in ['default.conv', 'default.param', 'default.sex']:
        default_file = os.path.join(working_dir, default_filename)
        default_template_file = os.path.join(default_dir, default_filename)
        if not os.path.isfile(default_file):
            shutil.copy(default_template_file, default_file)
    
    # check output file
    out_catalog_file = os.path.join(working_dir, 'SExtractor_OutputCatalog.fits')
    
    if not os.path.isfile(out_catalog_file):
        
        # prepare run_args
        run_args = ['sex', 'sci_data.fits', '-c', 'default.sex']
        
        # find instrument and filter names
        instrument = None
        filter_name = None
        if 'INSTRUME' in main_header:
            instrument = main_header['INSTRUME']
        if 'FILTER' in main_header:
            filter_name = main_header['FILTER']
        if instrument is None:
            raise Exception('Error! Could not find INSTRUME key in the input fits file {!r}'.format(image_file))
        if filter_name is None:
            raise Exception('Error! Could not find FILTER key in the input fits file {!r}'.format(image_file))
        
        # find psf file
        psf_template_pattern = f'{default_dir}/*_{instrument}_{filter_name}_*.psf'
        psf_templates = glob.glob(psf_template_pattern)
        psf_template_file = None
        if len(psf_templates) > 0:
            psf_template_file = psf_templates[0]
            psf_template_filename = os.path.basename(psf_template_file)
            psf_file = os.path.join(working_dir, psf_template_filename)
            if not os.path.isfile(psf_file):
                shutil.copy(psf_template_file, psf_file)
            default_psf_file = os.path.join(working_dir, 'default.psf')
            if not os.path.exists(default_psf_file):
                os.symlink(psf_template_filename, default_psf_file)
            run_args.append('-PSF_NAME')
            run_args.append('default.psf')
        else:
            print('Warning! PSF template not found {!r}. Will not set a PSF for SExtractor.'.format(psf_template_pattern))
        
        
        # run SExtractor
        run_args_str = ' '.join(run_args)
        run_args_file = os.path.join(working_dir, 'run_args.sh')
        with open(run_args_file, 'w') as fp:
            fp.write(run_args_str + '\n')
        print('Running args: {}'.format(run_args_str))
        subprocess.run(
            run_args,
            cwd = working_dir,
            check = True,
        )
        
        if not os.path.isfile(out_catalog_file):
            raise Exception('Error! Failed to run SExtractor and output catalog file {!r}?'.format(out_catalog_file))
        
        print('Output to catalog file: {!r}.'.format(out_catalog_file))
    
    else:
        
        print('Found catalog: {!r}. Will not re-run SExtractor.'.format(out_catalog_file))
        
    
    # reformat out_catalog_file
    # output a ds9 region and a catfile
    out_region_file = os.path.join(working_dir, 'SExtractor_OutputCatalog.ds9.reg')
    out_catfile = os.path.join(working_dir, 'SExtractor_OutputCatalog.cat')
    if not os.path.isfile(out_region_file) or not os.path.isfile(out_catfile):
        out_catalog = Table.read(out_catalog_file)
        with open(out_region_file, 'w') as fp:
            fp.write('# DS9 Region file\n')
            fp.write('image\n')
            for i in range(len(out_catalog)):
                fp.write('circle({:.3f},{:.3f},{:.3f})\n'.format(
                    out_catalog['XWIN_IMAGE'][i], 
                    out_catalog['YWIN_IMAGE'][i], 
                    out_catalog['KRON_RADIUS'][i]
                ))
            print('Output to region file: {!r}.'.format(out_region_file))
        with open(out_catfile, 'w') as fp:
            fp.write('image\n')
            for i in range(len(out_catalog)):
                fp.write('{:15.3f} {:15.3f}\n'.format(
                    out_catalog['XWIN_IMAGE'][i], 
                    out_catalog['YWIN_IMAGE'][i]
                ))
            print('Output to catfile for tweakreg: {!r}.'.format(out_catfile))
    
    
    # end of main()


if __name__ == '__main__':
    
    main()


