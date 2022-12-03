#!/usr/bin/env python
"""
Run SExtractor classic, output to a subdirectory '{input_filename}_run_sextractor_classic_dir'
"""

import os, sys, re, copy, glob, shutil
import click
import numpy as np
import subprocess
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import logging



@click.command()
@click.argument('image_file')
@click.option('--detect-thresh', type=float, default=5.0, help='DETECT_THRESH in sigma.')
@click.option('--analysis-thresh', type=float, default=2.0, help='ANALYSIS_THRESH in sigma.')
@click.option('--detect-minradius', type=float, default=0.2, help='Set a min radius in arcsec to convert it to DETECT_MINAREA with pi * r^2.')
@click.option('--detect-maxradius', type=float, default=3.0, help='Set a max radius in arcsec to convert it to DETECT_MAXAREA with pi * r^2.')
@click.option('--deblend-mincont', type=float, default=0.1, help='Deblending min contrast fraction, DEBLEND_MINCONT. The higher the harder to deblend clumps.')
@click.option('--phot-apertures', type=float, default=1.5, help='PHOT_APERTURES in arcsec')
@click.option('--make-plot', is_flag=True, default=True, help='make plot')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite')
def main(
        image_file,
        detect_thresh,
        analysis_thresh,
        detect_minradius,
        detect_maxradius,
        deblend_mincont,
        phot_apertures,
        make_plot,
        overwrite,
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
    
    # mask out zero-rms pixels
    if rms_data is not None:
        mask_zero_rms = ~np.isfinite(rms_data)
        if np.count_nonzero(mask_zero_rms) > 0:
            sci_data[mask_zero_rms] = np.nan
    
    # get wcs
    wcs = WCS(sci_header, naxis=2)
    pixsc = np.sqrt(proj_plane_pixel_area(wcs))*3600.0 # arcsec
    
    # create working directory
    working_dir = os.path.join(data_dir, data_name+'_run_sextractor_classic_dir')
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    
    # prepare images
    sci_file = os.path.join(working_dir, 'sci_data.fits')
    if not os.path.isfile(sci_file):
        hdul = fits.HDUList([fits.PrimaryHDU(header=main_header), fits.ImageHDU(data=sci_data, header=sci_header, name='SCI')])
        hdul.writeto(sci_file, overwrite=True)
    else:
        print('Found existing file {!r}. Skipping.'.format(sci_file))
    
    if wht_data is not None:
        wht_file = os.path.join(working_dir, 'wht_data.fits')
        if not os.path.isfile(wht_file):
            hdul = fits.HDUList([fits.PrimaryHDU(header=main_header), fits.ImageHDU(data=wht_data, header=wht_header, name='WHT')])
            hdul.writeto(wht_file, overwrite=True)
        else:
            print('Found existing file {!r}. Skipping.'.format(wht_file))
    
    if rms_data is not None:
        rms_file = os.path.join(working_dir, 'rms_data.fits')
        if not os.path.isfile(rms_file):
            hdul = fits.HDUList([fits.PrimaryHDU(header=main_header), fits.ImageHDU(data=rms_data, header=rms_header, name='RMS')])
            hdul.writeto(rms_file, overwrite=True)
        else:
            print('Found existing file {!r}. Skipping.'.format(rms_file))
    
    # copy param files
    for default_filename in ['default.conv', 'default.param', 'default.sex']:
        default_file = os.path.join(working_dir, default_filename)
        default_template_file = os.path.join(default_dir, default_filename)
        if not os.path.isfile(default_file):
            shutil.copy(default_template_file, default_file)
    
    # check output file
    out_catalog_file = os.path.join(working_dir, 'SExtractor_OutputCatalog.fits')
    
    if overwrite or not os.path.isfile(out_catalog_file):
        
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
        
        # zeropoint
        if 'PHOTMJSR' in main_header and 'PIXAR_SR' in main_header and 'BUNIT' in main_header and main_header['BUNIT']:
            photmjsr = main_header['PHOTMJSR']
            pixar_sr = main_header['PIXAR_SR']
            ABMAG = ((1.0 * u.MJy/u.sr) * (pixar_sr * u.sr)).to(u.ABmag)
            run_args.append('-MAG_ZEROPOINT')
            run_args.append(str(ABMAG))
        
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
            run_args.append(psf_template_filename) # 'default.psf'
        else:
            print('Warning! PSF template not found {!r}. Will not set a PSF for SExtractor.'.format(psf_template_pattern))
        
        # SET DETECT_THRESH
        if detect_thresh > 0.0:
            run_args.append('-DETECT_THRESH')
            run_args.append('{:.3f}'.format(detect_thresh))
        
        # SET ANALYSIS_THRESH
        if analysis_thresh > 0.0:
            run_args.append('-ANALYSIS_THRESH')
            run_args.append('{:.3f}'.format(analysis_thresh))
        
        # SET DETECT_MINAREA
        if detect_minradius > 0.0:
            run_args.append('-DETECT_MINAREA')
            run_args.append('{:.3f}'.format(np.pi * (detect_minradius / pixsc)**2))
        
        # SET DETECT_MAXAREA
        if detect_maxradius > 0.0:
            run_args.append('-DETECT_MAXAREA')
            run_args.append('{:.3f}'.format(np.pi * (detect_maxradius / pixsc)**2))
        
        # SET DEBLEND_MINCONT
        if deblend_mincont > 0.0:
            run_args.append('-DEBLEND_MINCONT')
            run_args.append('{:.5g}'.format(deblend_mincont))
        
        # SET PHOT_APERTURES to 50pix*0.030arcsec=1.5arcsec
        if phot_apertures > 0.0:
            run_args.append('-PHOT_APERTURES')
            run_args.append('{:.3f}'.format(phot_apertures / pixsc))
        
        # set BACK_SIZE 1/10 image size
        if True:
            ny, nx = sci_data.shape
            run_args.append('-BACK_SIZE')
            run_args.append('{:.3f}'.format(int(max(nx, ny)/10.)))
        
        # run SExtractor
        run_args_str = ' '.join(run_args)
        run_args_file = os.path.join(working_dir, 'run_args.sh')
        with open(run_args_file, 'w') as fp:
            fp.write(run_args_str + '\n')
        print('Running args: {} (saved to script {!r})'.format(run_args_str, run_args_file))
        subprocess.run(
            run_args,
            cwd = working_dir,
            check = True,
        )
        
        if not os.path.isfile(out_catalog_file):
            raise Exception('Error! Failed to run SExtractor and output catalog file {!r}?'.format(out_catalog_file))
        
        print('Output to catalog file: {!r}.'.format(out_catalog_file))
    
    else:
        
        print('Found existing file: {!r} and overwrite is False. Will not re-run SExtractor.'.format(out_catalog_file))
        
    
    # Read output catalog
    out_catalog = Table.read(out_catalog_file)
    
    
    # reformat out_catalog_file
    # output a ds9 region and a catfile
    out_region_file = os.path.join(working_dir, 'SExtractor_OutputCatalog.ds9.reg')
    out_catfile = os.path.join(working_dir, 'SExtractor_OutputCatalogXY.csv')
    if overwrite or not os.path.isfile(out_region_file) or not os.path.isfile(out_catfile):
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
            fp.write('x,y\n')
            for i in range(len(out_catalog)):
                fp.write('{:.3f},{:.3f}\n'.format(
                    out_catalog['XWIN_IMAGE'][i], 
                    out_catalog['YWIN_IMAGE'][i]
                ))
            print('Output to catfile for tweakreg: {!r}.'.format(out_catfile))
    
    else:
        
        print('Found existing file: {!r} and overwrite is False. Skipping.'.format(out_region_file))
    
    
    # make plot
    out_figure_file = os.path.join(working_dir, 'SExtractor_OutputCatalog.quicklook.png')
    if make_plot and overwrite or not os.path.isfile(out_figure_file):
        import matplotlib as mpl
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        mpl.rcParams['axes.labelsize'] = '9' # https://matplotlib.org/users/customizing.html
        mpl.rcParams['axes.grid'] = False
        mpl.rcParams['axes.axisbelow'] = True
        mpl.rcParams['axes.labelpad'] = 10
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['font.family'] = 'Monaco'
        mpl.rcParams['grid.color'] = 'white'
        mpl.rcParams['grid.linewidth'] = 0.8
        mpl.rcParams['grid.alpha'] = 0.8
        mpl.rcParams['legend.fontsize'] = '10'
        mpl.rcParams['legend.borderaxespad'] = 0.2 # space between legend border and axis
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['savefig.dpi'] = 300
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.gridspec as gridspec
        import matplotlib.patheffects as path_effects
        import matplotlib.cm as mplcm
        import matplotlib.colors as mplcolors
        from matplotlib.patches import Ellipse, Circle
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from astropy.visualization import simple_norm, ZScaleInterval
        image = sci_data
        ny, nx = sci_data.shape
        aspect = float(ny)/float(nx)
        fig = plt.figure(figsize=(10., 10./nx*ny))
        ax = fig.add_subplot(1,1,1)
        zscale_interval = ZScaleInterval()
        min_cut, max_cut = zscale_interval.get_limits(image)
        norm = simple_norm(image, 'linear', min_cut=min_cut, max_cut=max_cut)
        ax.imshow(image, origin='lower', interpolation='nearest', 
                  norm=norm, cmap='gray', aspect=1)
        
        # plot regions
        for i in range(len(out_catalog)):
            x = out_catalog['XWIN_IMAGE'][i]
            y = out_catalog['YWIN_IMAGE'][i]
            rad = out_catalog['KRON_RADIUS'][i]
            circ = Circle([x, y], rad*2.0, edgecolor='cyan', facecolor='none')
            ax.add_artist(circ)
        
        # draw
        fig.canvas.draw()
        
        # savefig
        fig.savefig(out_figure_file)
    
    elif make_plot:
        
        print('Found existing file: {!r} and overwrite is False. Skipping.'.format(out_figure_file))
    
    
    # end of main()


if __name__ == '__main__':
    
    main()


