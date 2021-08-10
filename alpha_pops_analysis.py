"""
    Main file for analysis of MUSE Milky Way analogue datacubes
    
    Functionality (to be) included here:
    
    1) Divide up MUSE cube into regions of interest, combine and extract spectra then store
        as 1D spectra
    2) Lick-index based analysis of extracted 1D MUSE spectra
    3) ppxf-based full spectral fitting analysis of 1D MUSE spectra
        a) ignoring kinematics
        b) specifically fitting for two kinematic components
        
    Utility functionality:
    
    Fit MUSE spectra for stellar kinematics
    Fit and subtract emission lines/mask emission lines from fitting
    Convolve spectra to a desired (Lick) resolution
    Correct Lick indices for high dispersion (where necessary)
    
"""

import glob
import astropy.io.fits as fits
import numpy as np
from scipy.ndimage.interpolation import rotate as rotate
from scipy.ndimage.filters import gaussian_filter1d
import os,psutil
import ppxf_muse
import code
from astropy.table import Table
import measure_ind
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
from time import perf_counter as clock
import multiprocessing
import matplotlib

###############################################################################

def reconstruct_summary_table(weights_dir,out_dir):

    files = glob.glob(weights_dir+'/*_weights.dat')
    names = ['nbin','f_a0','age_a0','z_a0','f_a04','age_a04','z_a04']
    out_tab = Table(names=names)

    for f in files:
        tab = Table.read(f,format='ascii')
        ind = f.split('/')[-1].split('_')[0]
        
        a0 = np.where(tab['alpha'] == 0.0)[0]
        a04 = np.where(tab['alpha'] == 0.4)[0]

        f_a0 = np.sum(tab['weight'][a0])
        f_a04 = np.sum(tab['weight'][a04])

        age0 = 10.0**(np.sum(np.log10(tab['age'][a0])*tab['weight'][a0])/np.sum(tab['weight'][a0]))
        z0 = np.sum(tab['z'][a0]*tab['weight'][a0])/np.sum(tab['weight'][a0])
        age04 = 10.0**(np.sum(np.log10(tab['age'][a04])*tab['weight'][a04])/np.sum(tab['weight'][a04]))
        z04 = np.sum(tab['z'][a04]*tab['weight'][a04])/np.sum(tab['weight'][a04])

        row = [ind,f_a0,age0,z0,f_a04,age04,z04]

        out_tab.add_row(row)

    out_tab.sort('nbin')

    out_tab.write(out_dir+'full_spectral_pops.dat',format='ascii.commented_header',overwrite=True)



###############################################################################

def interpolate_alpha_spectra():

# Routine to produce an [alpha/Fe] = 0.2 set of "MILES" models
# by averaging the 0.0 and 0.4 model sets

    path_0 = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v11/MILES_BASTI_UN_Ep0.00/'
    path_04 = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v11/MILES_BASTI_UN_Ep0.40/'
    path_02 = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v11/MILES_BASTI_UN_Ep0.20_interpolated/'

    files_0 = glob.glob(path_0+'Mun1.3*.fits')
    files_04 = glob.glob(path_04+'Mun1.3*.fits')

    for file_0,file_04 in zip(files_0,files_04):
        hdu = fits.open(file_0)
        hdu_04 = fits.open(file_04)

        hdu[0].data = (hdu[0].data + hdu_04[0].data)/2
        file_new = file_0.split('/')[-1]
        file_new = file_new.replace('iTp0.00_Ep0.0','iTp0.20_Ep0.2')
        hdu.writeto(path_02+file_new)

###############################################################################

def crude_image_combine(cubes_path = ''):

# This is a routine to do a very crude combine of the MUSE collapsed images - NOT
# the cubes as they're too large. This will likely be superseded by Jesse's proper
# work on reducing/combining the cubes

# The routine takes in cubes, collapses them to images, then places them in a larger
# image using the WCS information, aligning to the nearest pixel. It will provide
# mapping from the mosaic coordinates to the input cube coordinates to allow extraction
# of regions from those cubes

# List of input cubes

    if cubes_path == '':
        cubes_path = '/import/opus1/nscott/Stellar_Populations/MUSE/MW_analogues/UGC10738_fits/'
    cubes = glob.glob(cubes_path+'*.fits')

    
    im_array = []
    head_array = []
    exp_name_array = []
    
    for cube in cubes:
        hdu = fits.open(cube)
        head = hdu[1].header
        exp_name = cube.split('/')[-1][:-5]
        print(exp_name)
        cube = hdu[1].data
        im = np.nanmedian(cube,axis=0)
        im[np.isfinite(im) == False] = 0.0
        hdu.close()
        del hdu
        
        im_array.append(im)
        head_array.append(head)
        exp_name_array.append(exp_name)
        
        del im,head,exp_name
        
    big_im = np.zeros((840,900))
    weight_im = np.copy(big_im)
    
    # Ordering [.748,.655,.727,.731,.713]
    x0s = [0,1,388,190,191]
    y0s = [448,438,0,228,218]
    
    for i in range(len(exp_name_array)):
        im = im_array[i]
        big_im[x0s[i]:x0s[i]+np.shape(im)[0],y0s[i]:y0s[i]+np.shape(im)[1]] = \
            (big_im[x0s[i]:x0s[i]+np.shape(im)[0],y0s[i]:y0s[i]+np.shape(im)[1]] + im)
        wim = np.ones(np.shape(im))
        wim[im == 0.0] = 0 
        weight_im[x0s[i]:x0s[i]+np.shape(im)[0],y0s[i]:y0s[i]+np.shape(im)[1]] = \
            (weight_im[x0s[i]:x0s[i]+np.shape(im)[0],y0s[i]:y0s[i]+np.shape(im)[1]] + wim)
    
    rot_im = rotate(big_im,-41.5)
    rot_im2 = rot_im[420:760,160:1080]
    
    return rot_im2
    
###############################################################################
    
def crude_cube_combine(cubes_path = ''):

# This is a routine to do a very crude combine of the MUSE collapsed images - NOT
# the cubes as they're too large. This will likely be superseded by Jesse's proper
# work on reducing/combining the cubes

# The routine takes in cubes, collapses them to images, then places them in a larger
# image using the WCS information, aligning to the nearest pixel. It will provide
# mapping from the mosaic coordinates to the input cube coordinates to allow extraction
# of regions from those cubes

# List of input cubes

    if cubes_path == '':
        cubes_path = '/import/opus1/nscott/Stellar_Populations/MUSE/MW_analogues/UGC10738_fits/'
    cubes = glob.glob(cubes_path+'ADP*.fits')
    
    big_cube_path = '/import/opus1/nscott/Stellar_Populations/MUSE/MW_analogues/UGC10738_fits/UGC10738_mosaic_cube.fits'
    rot_cube_path = '/import/opus1/nscott/Stellar_Populations/MUSE/MW_analogues/UGC10738_fits/UGC10738_mosaic_rot_cube.fits'
    
    # Ordering [.748,.655,.727,.731,.713]
    x0s = [0,1,388,190,191]
    y0s = [448,438,0,228,218]
    process = psutil.Process(os.getpid())
    
    for i in range(len(cubes)):
        mem = process.memory_info().rss
        print('Iteration: {}, beginning, usage: {} Gb'.format(i+1,mem/(1024**3)))
        
        with fits.open(cubes[i]) as hdu:
            if i == 0:
                head = hdu[1].header
            exp_name = cubes[i].split('/')[-1][:-5]
            print('Adding: {}'.format(exp_name))
            cube = hdu[1].data
            cube[np.isfinite(cube) == False] = 0.0
            hdu.close()
            del hdu
        
        mem = process.memory_info().rss
        print('Iteration: {}, opened cube, usage: {} Gb'.format(i+1,mem/(1024**3)))
        
        if os.path.exists(big_cube_path):
            with fits.open(big_cube_path) as hdu:
                big_cube = hdu[0].data
                hdu.close()
                del hdu
        else:
            big_cube = np.zeros((3681,840,900))
            
        #weight_cube = np.copy(big_cube)
            
        mem = process.memory_info().rss
        print('Iteration: {}, opened big cube, usage: {} Gb'.format(i+1,mem/(1024**3)))
                
        big_cube[:,x0s[i]:x0s[i]+np.shape(cube)[1],y0s[i]:y0s[i]+np.shape(cube)[2]] = \
            (big_cube[:,x0s[i]:x0s[i]+np.shape(cube)[1],y0s[i]:y0s[i]+np.shape(cube)[2]] + cube)
        #wim = np.ones(np.shape(cube))
        #wim[im == 0.0] = 0 
        #weight_cube[:,x0s[i]:x0s[i]+np.shape(cube)[1],y0s[i]:y0s[i]+np.shape(cube)[2]] = \
        #   (weight_cube[:,x0s[i]:x0s[i]+np.shape(cube)[1],y0s[i]:y0s[i]+np.shape(cube)[2]] + cube)
            
        del cube

        mem = process.memory_info().rss
        print('Iteration: {}, deleted cube, usage: {} Gb'.format(i+1,mem/(1024**3)))

        if os.path.exists(big_cube_path):
            with fits.open(big_cube_path,mode='update') as hdu:
                hdu[0].data = big_cube
                hdu.flush()
                hdu.close()
                del hdu
        else:
            hdu = fits.PrimaryHDU(big_cube)
            hdu.header = head
            hdu.writeto(big_cube_path,output_verify='fix')
            del hdu
            
        if i != len(cubes)-1:
            del big_cube
            
        mem = process.memory_info().rss
        print('Iteration: {}, written and closed big cube, usage: {} Gb'.format(i+1,mem/(1024**3)))
    
    rot_cube = rotate(big_cube,-41.5,axes=(2,1))
    del big_cube
    rot_cube = rot_cube[:,420:760,160:1080]
    
    hdu = fits.PrimaryHDU(rot_cube)
    hdu.header = head
    hdu.writeto(rot_cube_path,output_verify='fix')

    del hdu
    
###############################################################################

def alpha_full_spectral_from_voronoi_bins(n_threads=1,dummy=False,full_weights=False,
                                               three=False,do_regul=False,quadrants=False):

    matplotlib.use('agg')

    inspec = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v4/UGC10738_default_snrfw4005_BinSpectra_linear.fits'
    outdir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v4/full_spectral_fits_v2/'
    
    z_gal = 0.022402

    hdu = fits.open(inspec)
    n_spec = len(hdu[1].data)
    wav = np.asarray([i[0] for i in hdu[2].data])
    
    miles = ppxf_muse.import_alpha_templates(three=three,velscale=60.0)
    
    fwhm_gal = 2.7/(1+z_gal)
    z_gal = 0
    
        
    names = ['nbin','f_a0','age_a0','z_a0','f_a04','age_a04','z_a04']
    out_tab = Table(names=names)
    
    if n_threads == 1:
        print('Running in series')
    
        for i in range(n_spec):

            inputs = {'spec':hdu[1].data[i][0],'wav':wav,
                      'miles':miles,'z_gal':z_gal,'fwhm_gal':fwhm_gal,
                      'outdir':outdir,'full_weights':full_weights,
                      'do_regul':do_regul,'index':i*3}

            row = alpha_full_spectral_from_voronoi_bins_one(inputs)
            out_tab.add_row(row)
        
            break
        
    else:
        pool = multiprocessing.Pool(n_threads)
        inputs_list = []
        for i in range(n_spec):
            inputs = {'spec':hdu[1].data[i][0],'wav':wav,
                      'miles':miles,'z_gal':z_gal,'fwhm_gal':fwhm_gal,
                      'outdir':outdir,'full_weights':full_weights,
                      'do_regul':do_regul,'index':i*3}
            inputs_list.append(inputs)
        
        print('Processing {} spectra'.format(str(len(inputs_list))))
        
        outputs = pool.map(alpha_full_spectral_from_voronoi_bins_one,inputs_list,chunksize=1)
        pool.close()
        pool.join()
        
        for row in outputs:
            out_tab.add_row(row)

    out_tab.write(outdir+'full_spectral_pops.dat',format='ascii.commented_header',overwrite=True)

 
def alpha_full_spectral_from_voronoi_bins_one(inputs):

    spec = inputs['spec']
    wav = inputs['wav']
    miles = inputs['miles']
    z_gal = inputs['z_gal']
    fwhm_gal = inputs['fwhm_gal']
    outdir = inputs['outdir']
    full_weights = inputs['full_weights']
    do_regul = inputs['do_regul']
    k = inputs['index']

    t = clock()
    print('Beginning {}'.format((k+3)//3))
    fig = plt.figure(k,figsize=(12,6))
    pp = ppxf_muse.ppxf_muse_alpha(spec,wav,fwhm_gal,miles,z_gal,
                                       plot=False,do_regul=do_regul)
    print('Finished {}. Elapsed time: {} s'.format((k+3)//3,clock()-t))
    pp.plot()
    fig.savefig(outdir+'figures/'+str((k+3)//3)+'.pdf')
    plt.close(fig)

    weights = pp.weights[~pp.gas_component]
    weights = weights.reshape(pp.reg_dim)/weights.sum()

    f_a0 = weights[:,:,0].sum()
    f_a04 = weights[:,:,1].sum()

    #print(r'[$\alpha$/Fe] = 0.0: {}'.format(f_a0))
    az0 = miles[0].mean_age_metal(weights[:,:,0])
    #print()
    #print(r'[$\alpha$/Fe] = 0.4: {}'.format(f_a04))
    az04 = miles[1].mean_age_metal(weights[:,:,1])
    #print()

    row = [k/3,f_a0,az0[0],az0[1],f_a04,az04[0],az04[1]]

    fig = plt.figure(k+1,figsize=(12,6))
    plt.subplot(121)
    miles[0].plot(weights[:,:,0],nodots=True)
    plt.title(r'[$\alpha$/Fe] = 0.0')
    plt.subplot(122)
    miles[1].plot(weights[:,:,1],nodots=True)
    plt.title(r'[$\alpha$/Fe] = 0.4')
    fig.savefig(outdir+'figures/'+str((k+3)//3)+'_age_z.pdf')
    plt.close(fig)

    fig = plt.figure(k+2,figsize=(8,6))
    alpha_weights = np.sum(weights,axis=0)
    xgrid = miles[0].metal_grid[0:2]
    ygrid = np.zeros(np.shape(xgrid))
    ygrid[1] = 0.4

    xgrid = np.transpose(xgrid)
    ygrid = np.transpose(ygrid)
    x = xgrid[:,0]
    y = ygrid[0,:]
    xb = (x[1:] + x[:-1])/2
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
    ax = plt.gca()

    pc = plt.pcolormesh(xb,yb,alpha_weights.T,edgecolors='face')
    plt.xlabel('[Z/H]',fontsize=15)
    plt.ylabel(r'[$\alpha$/Fe]',fontsize=15)
    plt.colorbar(pc)
    plt.sca(ax)
    fig.savefig(outdir+'figures/'+str((k+3)//3)+'_z_alpha.pdf')
    plt.close(fig)

    if full_weights:
        write_full_weights(miles,weights,outdir+'weights/'+str((k+3)//3)+'_weights.dat')

    return row    

def alpha_full_spectral_from_extracted_regions(n_threads=1,dummy=False,full_weights=False,
                                               three=False,do_regul=False,quadrants=False):

    matplotlib.use('agg')

    indir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/extracted_1d_spectra_v4/'
    outdir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/full_spectral_fits_v4/'
    #files = glob.glob(indir+'*.dat')
    files = ['/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/extracted_1d_spectra_v4/UGC10738_r5_7_z0.5_1.dat']
    
    n_files = len(files)
    
    miles = ppxf_muse.import_alpha_templates(three=three)
    
    fwhm_gal = 2.7/(1+0.0224)
    z_gal = 0.0
    
    if three:
        names = ['r','z','f_a0','age_a0','z_a0','f_a02','age_a02','z_a02','f_a04','age_a04','z_a04']
    else:
        names = ['r','z','f_a0','age_a0','z_a0','f_a04','age_a04','z_a04']
    out_tab = Table(names=names)
    
    if n_threads == 1:
        print('Running in series')
    
        for f in files:

            inputs = {'file':f,'miles':miles,'z_gal':z_gal,'fwhm_gal':fwhm_gal,'three':three,
                      'outdir':outdir,'index':1,'dummy':dummy,'full_weights':full_weights,
                      'do_regul':do_regul,'quadrants':quadrants}

            row = alpha_full_spectral_from_extracted_regions_one(inputs)
            out_tab.add_row(row)
        
            break
        
    else:
        pool = multiprocessing.Pool(n_threads)
        inputs_list = []
        for k,file in enumerate(files):
            inputs = {'file':file,'miles':miles,'z_gal':z_gal,'fwhm_gal':fwhm_gal,'three':three,
                      'outdir':outdir,'index':k*3,'dummy':dummy,'full_weights':full_weights,
                      'do_regul':do_regul,'quadrants':quadrants}
            inputs_list.append(inputs)
        
        print('Processing {} spectra'.format(str(len(inputs_list))))
        
        outputs = pool.map(alpha_full_spectral_from_extracted_regions_one,inputs_list,chunksize=1)
        pool.close()
        pool.join()
        
        for row in outputs:
            out_tab.add_row(row)

    if not dummy:
        out_tab.write(outdir+'full_spectral_pops.dat',format='ascii.commented_header',overwrite=True)
       
def alpha_full_spectral_from_extracted_regions_one(inputs):

    file0 = inputs['file']
    miles = inputs['miles']
    z_gal = inputs['z_gal']
    fwhm_gal = inputs['fwhm_gal']
    outdir = inputs['outdir']
    k = inputs['index']
    dummy = inputs['dummy']
    full_weights = inputs['full_weights']
    three = inputs['three']
    do_regul = inputs['do_regul']
    quadrants = inputs['quadrants']

    if dummy:
        row = [k,k,k,k,k,k,k,k]
    else:
        tab = Table.read(file0,format='ascii.commented_header')

        ff = file0.split('/')[-1]
        r_text = ff[ff.find('_r')+2:ff.find('_z')]
        r = (np.float(r_text.split('_')[0])+np.float(r_text.split('_')[1]))/2.
        z_text = ff[ff.find('_z')+2:ff.find('.dat')]
        z = (np.float(z_text.split('_')[0])+np.float(z_text.split('_')[1]))/2.
        if z == 0.25:
            z = 0.0
        ff = ff.split('.dat')[0]

        t = clock()
        print('Beginning {}'.format(ff))
        fig = plt.figure(k,figsize=(12,6))
        pp = ppxf_muse.ppxf_muse_alpha(tab['Flux'],tab['Wav'],fwhm_gal,miles,z_gal,
                                       plot=False,three=three,do_regul=do_regul)
        print('Finished {}. Elapsed time: {} s'.format(ff,clock()-t))
        pp.plot()
        fig.savefig(outdir+'figures/'+ff+'2.pdf')
        plt.close(fig)

        weights = pp.weights[~pp.gas_component]
        weights = weights.reshape(pp.reg_dim)/weights.sum()
        if three:
            f_a0 = weights[:,:,0].sum()
            f_a02 = weights[:,:,1].sum()
            f_a04 = weights[:,:,2].sum()

            print(r'[$\alpha$/Fe] = 0.0: {}'.format(f_a0))
            az0 = miles[0].mean_age_metal(weights[:,:,0])
            print()
            print(r'[$\alpha$/Fe] = 0.2: {}'.format(f_a02))
            az02 = miles[1].mean_age_metal(weights[:,:,1])
            print()
            print(r'[$\alpha$/Fe] = 0.4: {}'.format(f_a04))
            az04 = miles[2].mean_age_metal(weights[:,:,2])
            print()

            row = [r,z,f_a0,az0[0],az0[1],f_a02,az02[0],az02[1],f_a04,az04[0],az04[1]]

            fig = plt.figure(k+1,figsize=(18,6))
            plt.subplot(131)
            miles[0].plot(weights[:,:,0])
            plt.title(r'[$\alpha$/Fe] = 0.0')
            plt.subplot(132)
            miles[1].plot(weights[:,:,1])
            plt.title(r'[$\alpha$/Fe] = 0.2')
            plt.subplot(133)
            miles[2].plot(weights[:,:,2])
            plt.title(r'[$\alpha$/Fe] = 0.4')
            fig.savefig(outdir+'figures/'+ff+'_age_z.pdf')
            plt.close(fig)

            fig = plt.figure(k+2,figsize=(8,6))
            alpha_weights = np.sum(weights,axis=0)
            xgrid = miles[0].metal_grid[0:3]
            ygrid = np.zeros(np.shape(xgrid))
            ygrid[1] = 0.2
            ygrid[2] = 0.4

            xgrid = np.transpose(xgrid)
            ygrid = np.transpose(ygrid)
            x = xgrid[:,0]
            y = ygrid[0,:]
            xb = (x[1:] + x[:-1])/2
            yb = (y[1:] + y[:-1])/2
            xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
            yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
            ax = plt.gca()

            pc = plt.pcolormesh(xb,yb,alpha_weights.T,edgecolors='face')
            plt.xlabel('[Z/H]',fontsize=15)
            plt.ylabel(r'[$\alpha$/Fe]',fontsize=15)
            plt.colorbar(pc)
            plt.sca(ax)
            fig.savefig(outdir+'figures/'+ff+'_z_alpha.pdf')
            plt.close(fig)


        else:
            f_a0 = weights[:,:,0].sum()
            f_a04 = weights[:,:,1].sum()

            print(r'[$\alpha$/Fe] = 0.0: {}'.format(f_a0))
            az0 = miles[0].mean_age_metal(weights[:,:,0])
            print()
            print(r'[$\alpha$/Fe] = 0.4: {}'.format(f_a04))
            az04 = miles[1].mean_age_metal(weights[:,:,1])
            print()

            row = [r,z,f_a0,az0[0],az0[1],f_a04,az04[0],az04[1]]

            fig = plt.figure(k+1,figsize=(12,6))
            plt.subplot(121)
            miles[0].plot(weights[:,:,0],nodots=True)
            plt.title(r'[$\alpha$/Fe] = 0.0')
            plt.subplot(122)
            miles[1].plot(weights[:,:,1],nodots=True)
            plt.title(r'[$\alpha$/Fe] = 0.4')
            fig.savefig(outdir+'figures/'+ff+'_age_z2.pdf')
            plt.close(fig)

            fig = plt.figure(k+2,figsize=(8,6))
            alpha_weights = np.sum(weights,axis=0)
            xgrid = miles[0].metal_grid[0:2]
            ygrid = np.zeros(np.shape(xgrid))
            ygrid[1] = 0.4

            xgrid = np.transpose(xgrid)
            ygrid = np.transpose(ygrid)
            x = xgrid[:,0]
            y = ygrid[0,:]
            xb = (x[1:] + x[:-1])/2
            yb = (y[1:] + y[:-1])/2
            xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])
            yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])
            ax = plt.gca()

            pc = plt.pcolormesh(xb,yb,alpha_weights.T,edgecolors='face')
            plt.xlabel('[Z/H]',fontsize=15)
            plt.ylabel(r'[$\alpha$/Fe]',fontsize=15)
            plt.colorbar(pc)
            plt.sca(ax)
            fig.savefig(outdir+'figures/'+ff+'_z_alpha2.pdf')
            plt.close(fig)

            if full_weights:
                write_full_weights(miles,weights,outdir+'weights/'+ff+'_weights.dat')

            if quadrants:
                pp1 = ppxf_muse.ppxf_muse_alpha(tab['Spec1'],tab['Wav'],fwhm_gal,miles,z_gal,
                                       plot=False,do_regul=do_regul)
                weights1 = pp1.weights[~pp.gas_component]
                weights1 = weights1.reshape(pp.reg_dim)/weights1.sum()
                write_full_weights(miles,weights1,outdir+'weights/'+ff+'_weights_a.dat')
                pp2 = ppxf_muse.ppxf_muse_alpha(tab['Spec2'],tab['Wav'],fwhm_gal,miles,z_gal,
                                       plot=False,do_regul=do_regul)
                weights2 = pp2.weights[~pp.gas_component]
                weights2 = weights2.reshape(pp.reg_dim)/weights2.sum()
                write_full_weights(miles,weights2,outdir+'weights/'+ff+'_weights_b.dat')
                pp3 = ppxf_muse.ppxf_muse_alpha(tab['Spec3'],tab['Wav'],fwhm_gal,miles,z_gal,
                                       plot=False,do_regul=do_regul)
                weights3 = pp3.weights[~pp.gas_component]
                weights3 = weights3.reshape(pp.reg_dim)/weights3.sum()
                write_full_weights(miles,weights3,outdir+'weights/'+ff+'_weights_c.dat')
                pp4 = ppxf_muse.ppxf_muse_alpha(tab['Spec4'],tab['Wav'],fwhm_gal,miles,z_gal,
                                       plot=False,do_regul=do_regul)
                weights4 = pp4.weights[~pp.gas_component]
                weights4 = weights4.reshape(pp.reg_dim)/weights4.sum()
                write_full_weights(miles,weights4,outdir+'weights/'+ff+'_weights_d.dat')

    return row    
            

###############################################################################
def write_full_weights(miles,weights,outfile):

    weights_tab = Table()
    z_alpha_low = np.ravel(miles[0].metal_grid)
    z_alpha_high = np.ravel(miles[1].metal_grid)
    age_alpha_low = np.ravel(miles[0].age_grid)
    age_alpha_high = np.ravel(miles[1].age_grid)
    alpha_low = np.zeros(len(z_alpha_low))
    alpha_high = np.ones(len(z_alpha_high))*0.4
    weights_low = np.ravel(weights[:,:,0])
    weights_high = np.ravel(weights[:,:,1])
    weights_tab['alpha'] = np.hstack((alpha_low,alpha_high))
    weights_tab['age'] = np.hstack((age_alpha_low,age_alpha_high))
    weights_tab['z'] = np.hstack((z_alpha_low,z_alpha_high))
    weights_tab['weight'] = np.hstack((weights_low,weights_high))
    weights_tab.write(outfile,format='ascii')


def hbeta_from_voronoi_test():

    cubedir = '/import/milo1/sande/muse/analysis/UGC10738/results/UGC10738_P01P02SNRBLUE4005/'
    specin = cubedir+'UGC10738_VorSpectra_linear.fits'
    hduspec = fits.open(specin)
    binin = cubedir+'UGC10738_table.fits'
    hdubin = fits.open(binin)
    kinin = cubedir+'UGC10738_ppxf.fits'
    hdukin = fits.open(kinin)
    bfin = cubedir+'UGC10738_ppxf-bestfit.fits'
    hdubf = fits.open(bfin)
    outdir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/'

    
    templates,templates_wav = ppxf_muse.import_empirical_templates(testing=True)
    z_gal = 0.022402
    fwhm_gal = 2.7/(1+z_gal)
    fwhm_tmp = 2.5
    c = 299792.458

    nbins = np.shape(hdukin[1].data)[0]
    
    index_definition_file = os.path.expanduser('/import/opus1/nscott/Stellar_Populations/misc/pseudocontinuum_absorption_line_indices.dat')
    index_definitions = ascii.read(index_definition_file)

    for i in range(nbins):
        
        binid = hdukin[1].data['BIN_ID'][i]
        ww = np.where(hdubin[1].data['BIN_ID'] == binid)[0][0]
        x,y = hdubin[1].data['XBIN'][ww],hdubin[1].data['YBIN'][ww]

        spectrum = hduspec[1].data[i][0]
        var = hduspec[1].data[i][1]
        wav = hduspec[2].data
        wav = np.asarray([wv[0] for wv in wav])
        plt.figure(1)
        plt.clf()
        pp,z_gal = ppxf_muse.ppxf_muse(spectrum,wav,fwhm_gal,0.0,#z_gal,
                                                        templates,templates_wav,fwhm_tmp,
                                                        v0=False,plot=True,derive_noise=True,
                                                        subtract_emission=True,noise=np.sqrt(var))
        sol = pp.sol
        weights = pp.weights
        bestfit = pp.bestfit
                                                                
        gas_spectrum = np.interp(wav,pp.lam,pp.gas_bestfit)
        bestfit = (np.interp(wav,pp.lam,pp.bestfit-pp.gas_bestfit)*np.nanmedian(spectrum[np.where(wav < np.max(templates_wav))]))
        spectrum = spectrum - gas_spectrum*np.nanmedian(spectrum)
        noise = np.interp(wav,pp.lam,pp.new_noise)
        variance = (spectrum*noise)**2
        sn = np.nanmedian(spectrum/np.sqrt(variance))
                                                        
        dlambda = wav[1001]-wav[1000]
        
        sig_res = fwhm_gal/np.sqrt(8*np.log(2))
        sig_gal = sol[0][1]/c*np.nanmean(wav)
        sig_tot = np.sqrt(sig_res**2+sig_gal**2)
            
        row = [x,y]
        
        res = index_definitions[np.where('Hbeta' == index_definitions['Index'])[0]]['Res'].data[0]
        sig_ind = res/np.sqrt(8*np.log(2))


        if sig_tot < sig_ind:
            sig_dif = np.sqrt(sig_ind**2 - sig_tot**2)
            lick_res_spectrum = gaussian_filter1d(spectrum,sig_dif/dlambda,mode='nearest')
            lick_res_var = gaussian_filter1d(variance,sig_dif/dlambda,mode='nearest')
            lick_res_bestfit = gaussian_filter1d(bestfit,sig_dif/dlambda,mode='nearest')
        else:
            lick_res_spectrum = spectrum
            lick_res_bestfit = bestfit
            lick_res_var = variance

        plt.figure(4)
        plt.clf()
        width0,err = measure_ind.mc_ind_errors(wav,lick_res_bestfit,lick_res_var,'Hbeta',obswav=wav*(1+0.0224))
        width = measure_ind.measure_ind(wav,lick_res_spectrum,'Hbeta',obswav=wav*(1+0.0224),plot=True)

        plt.show()

        if sig_tot > sig_ind:
            veldisp_eff = np.sqrt(sig_tot**2 - sig_ind**2)*c/np.nanmean(wav)
            if veldisp_eff > 300: veldisp_eff = 300.
            width = measure_ind.sigma_correct(veldisp_eff,width,'Hbeta')

        print('Width:{} Error:{} S/N:{}, sigma:{} x:{} y:{}'.format(width,err,pp.sol[1],sn,x,y))
        code.interact(local=dict(globals(),**locals()))

###############################################################################
def lick_indices_from_voronoi_bins():

    cubedir = '/import/milo1/sande/muse/analysis/UGC10738/results/UGC10738_P01P02SNRBLUE4005/'
    specin = cubedir+'UGC10738_VorSpectra_linear.fits'
    hduspec = fits.open(specin)
    binin = cubedir+'UGC10738_table.fits'
    hdubin = fits.open(binin)
    kinin = cubedir+'UGC10738_ppxf.fits'
    hdukin = fits.open(kinin)
    bfin = cubedir+'UGC10738_ppxf-bestfit.fits'
    hdubf = fits.open(bfin)
    outdir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/'

    
    templates,templates_wav = ppxf_muse.import_empirical_templates(testing=True)
    z_gal = 0.022402
    fwhm_gal = 2.7/(1+z_gal)
    fwhm_tmp = 2.5
    c = 299792.458
    
    index_definition_file = os.path.expanduser('/import/opus1/nscott/Stellar_Populations/misc/pseudocontinuum_absorption_line_indices.dat')
    index_definitions = ascii.read(index_definition_file)

    indices = ['Hbeta','Mgb','Fe5015','Fe5270','Fe5335','Fe5406','Fe5709','Fe5782','NaD','TiO1','TiO2','Mg1','Mg2']
    res = np.empty(len(indices))

    nbins = np.shape(hdukin[1].data)[0]

    measured_indices = np.zeros((len(indices),nbins))
    measured_errs = np.zeros((len(indices),nbins))
    
    names = ['XBIN','YBIN'] + [item for ind in indices for item in  [ind,ind+'_err']] + ['SN']  
    out_tab = Table(names=names)

    for i in range(nbins):
        
        binid = hdukin[1].data['BIN_ID'][i]
        ww = np.where(hdubin[1].data['BIN_ID'] == binid)[0][0]
        x,y = hdubin[1].data['XBIN'][ww],hdubin[1].data['YBIN'][ww]

        spectrum = hduspec[1].data[i][0]
        var = hduspec[1].data[i][1]
        wav = hduspec[2].data
        wav = np.asarray([wv[0] for wv in wav])
        pp,z_gal = ppxf_muse.ppxf_muse(spectrum,wav,fwhm_gal,0.0,#z_gal,
                                                        templates,templates_wav,fwhm_tmp,
                                                        v0=False,plot=False,derive_noise=True,
                                                        subtract_emission=True,noise=np.sqrt(var))
        sol = pp.sol
        weights = pp.weights
        bestfit = pp.bestfit
                                                                
        gas_spectrum = np.interp(wav,pp.lam,pp.gas_bestfit)
        bestfit = (np.interp(wav,pp.lam,pp.bestfit-pp.gas_bestfit)*np.nanmedian(spectrum[np.where(wav < np.max(templates_wav))]))
        spectrum = spectrum - gas_spectrum*np.nanmedian(spectrum)
        noise = np.interp(wav,pp.lam,pp.new_noise)
        variance = (spectrum*noise)**2
        sn = np.nanmedian(spectrum/np.sqrt(variance))
                                                        
        dlambda = wav[1001]-wav[1000]
        
        sig_res = fwhm_gal/np.sqrt(8*np.log(2))
        sig_gal = sol[0][1]/c*np.nanmean(wav)
        sig_tot = np.sqrt(sig_res**2+sig_gal**2)
            
        row = [x,y]
        
        for j in range(len(indices)):
            res = index_definitions[np.where(indices[j] == index_definitions['Index'])[0]]['Res'].data[0]
            sig_ind = res/np.sqrt(8*np.log(2))
    
            
            if sig_tot < sig_ind:
                sig_dif = np.sqrt(sig_ind**2 - sig_tot**2)
                lick_res_spectrum = gaussian_filter1d(spectrum,sig_dif/dlambda,mode='nearest')
                lick_res_var = gaussian_filter1d(variance,sig_dif/dlambda,mode='nearest')
                lick_res_bestfit = gaussian_filter1d(bestfit,sig_dif/dlambda,mode='nearest')
            else:
                lick_res_spectrum = spectrum
                lick_res_bestfit = bestfit
                lick_res_var = variance
            
            width0,err = measure_ind.mc_ind_errors(wav,lick_res_bestfit,lick_res_var,indices[j],obswav=wav*(1+0.0224))
            #if indices[j] == 'Hbeta':
            #    pl = True
            #else:
            pl = False
            width = measure_ind.measure_ind(wav,lick_res_spectrum,indices[j],obswav=wav*(1+0.0224),plot=pl)
            
            #if indices[j] == 'Hbeta':
            #    plt.show()
            #    plt.savefig()
            
            if sig_tot > sig_ind:
                veldisp_eff = np.sqrt(sig_tot**2 - sig_ind**2)*c/np.nanmean(wav)
                if veldisp_eff > 300: veldisp_eff = 300.
                width = measure_ind.sigma_correct(veldisp_eff,width,indices[j])

            
            measured_indices[j,i] = width
            measured_errs[j,i] = err
            row.extend([width,err])
        row.extend([sn])
            
        out_tab.add_row(row)
        
    out_tab.write(outdir+'measured_lick_indices_voronoi_v3.dat',format='ascii.commented_header',overwrite=True)

###############################################################################

def lick_indices_from_extracted_regions():

    # From the .dat extracted spectra files, measure all the available Lick
    # indices in the MUSE spectral range
    
    indir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/extracted_1d_spectra_v2/'
    outdir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/'
    files = glob.glob(indir+'*.dat')
    
    n_files = len(files)
    
    templates,templates_wav = ppxf_muse.import_empirical_templates(testing=True)
    fwhm_gal = 2.7/(1+0.0224)
    fwhm_tmp = 2.5
    z_gal = 0.0
    c = 299792.458
    
    index_definition_file = os.path.expanduser('/import/opus1/nscott/Stellar_Populations/misc/pseudocontinuum_absorption_line_indices.dat')
    index_definitions = ascii.read(index_definition_file)

    indices = ['Hbeta','Mgb','Fe5015','Fe5270','Fe5335','Fe5406','Fe5709','Fe5782','NaD','TiO1','TiO2','Mg1','Mg2']
    res = np.empty(len(indices))

    measured_indices = np.zeros((len(indices),n_files))
    measured_errs = np.zeros((len(indices),n_files))
    
    names = ['r','z']+ [item for ind in indices for item in  [ind,ind+'_err']] + ['SN']  
    out_tab = Table(names=names)

    for i,file in enumerate(files):
        
        tab = Table.read(file,format='ascii.commented_header')
        spectrum = tab['Flux']
        var = tab['Var']
        if i == 0: plot = True
        else: plot = False
        pp,z_gal = ppxf_muse.ppxf_muse(spectrum,tab['Wav'],fwhm_gal,z_gal,
                                                        templates,templates_wav,fwhm_tmp,
                                                        v0=False,plot=plot,derive_noise=True,
                                                        subtract_emission=True,noise=np.sqrt(var))
        sol = pp.sol
        weights = pp.weights
        bestfit = pp.bestfit
                                                        
        if True:
            gas_spectrum = np.interp(tab['Wav'],pp.lam,pp.gas_bestfit)
            bestfit = (np.interp(tab['Wav'],pp.lam,pp.bestfit-pp.gas_bestfit)
                        *np.nanmedian(spectrum[np.where(tab['Wav'] < np.max(templates_wav))]))
            spectrum = spectrum - gas_spectrum*np.nanmedian(spectrum)
            noise = np.interp(tab['Wav'],pp.lam,pp.new_noise)
            variance = (spectrum*noise)**2
            sn = np.nanmedian(spectrum/np.sqrt(variance))
                                                        
        dlambda = tab['Wav'][1001]-tab['Wav'][1000]
        
        sig_res = fwhm_gal/np.sqrt(8*np.log(2))
        if True:
            sig_gal = sol[0][1]/c*np.nanmean(tab['Wav'])
        else:
            sig_gal = sol[1]/c*np.nanmean(tab['Wav'])
        sig_tot = np.sqrt(sig_res**2+sig_gal**2)
        
        ff = file.split('/')[-1]
        r_text = ff[ff.find('_r')+2:ff.find('_z')]
        r = (np.float(r_text.split('_')[0])+np.float(r_text.split('_')[1]))/2.
        z_text = ff[ff.find('_z')+2:ff.find('.dat')]
        z = (np.float(z_text.split('_')[0])+np.float(z_text.split('_')[1]))/2.
        if z == 0.25:
            z = 0.0
            
        row = [r,z]
        
        for j in range(len(indices)):
            res = index_definitions[np.where(indices[j] == index_definitions['Index'])[0]]['Res'].data[0]
            sig_ind = res/np.sqrt(8*np.log(2))
    
            
            if sig_tot < sig_ind:
                sig_dif = np.sqrt(sig_ind**2 - sig_tot**2)
                lick_res_spectrum = gaussian_filter1d(spectrum,sig_dif/dlambda,mode='nearest')
                lick_res_var = gaussian_filter1d(variance,sig_dif/dlambda,mode='nearest')
                lick_res_bestfit = gaussian_filter1d(bestfit,sig_dif/dlambda,mode='nearest')
            else:
                lick_res_spectrum = spectrum#[np.where(tab['Wav'] < 6500)]
                lick_res_bestfit = bestfit#[np.where(tab['Wav'] < 6500)]
                lick_res_var = variance#[np.where(tab['Wav'] < 6500)]
            
            width0,err = measure_ind.mc_ind_errors(tab['Wav'],lick_res_bestfit,lick_res_var,indices[j],obswav=tab['Wav']*(1+0.0224))
            width = measure_ind.measure_ind(tab['Wav'],lick_res_spectrum,indices[j],obswav=tab['Wav']*(1+0.0224))
            
            if sig_tot > sig_ind:
                veldisp_eff = np.sqrt(sig_tot**2 - sig_ind**2)*c/np.nanmean(tab['Wav'])
                if veldisp_eff > 300: veldisp_eff = 300.
                width = measure_ind.sigma_correct(veldisp_eff,width,indices[j])

            
            measured_indices[j,i] = width
            measured_errs[j,i] = err
            row.extend([width,err])
        row.extend([sn])
            
        out_tab.add_row(row)
        
    out_tab.write(outdir+'measured_lick_indices.dat',format='ascii.commented_header',overwrite=True)
            
            
###############################################################################    


def extract_regions_from_cube(pc_scale = 91.0,xc=461,yc=161):

    # Extract the regions from the cube corresponding to the same regions as in 
    # Hayden et al. (2015). Write these to separate fits files
        
    outdir = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/extracted_1d_spectra_v4/'
    cube_path = '/import/milo1/sande/muse/data/P101/UGC10738/Combined/Cubes/P01_P02/UGC10738_DATACUBE_FINAL_WCS_Pall_mad_crp_mskd.fits'
    kin_path = '/import/milo1/sande/muse/analysis/UGC10738/results/UGC10738_default_snrfw10025/UGC10738_default_snrfw10025_kin.fits'
    bin_path = '/import/milo1/sande/muse/analysis/UGC10738/results/UGC10738_default_snrfw10025/UGC10738_default_snrfw10025_table.fits'

    
    with fits.open(cube_path) as hdu:
        cube_orig = hdu[1].data
        var_orig = hdu[2].data
        h = hdu[1].header

    s = np.shape(cube_orig)

    with fits.open(kin_path) as hdu:
        kin_tab = hdu[1].data

    with fits.open(bin_path) as hdu:
        bin_tab = hdu[1].data
        
    vel,sig = [],[]
    #for bin_id in bin_tab['BIN_ID']:
    #    ww = np.where(kin_tab['BIN_ID'] == bin_id)[0]
    #    if len(ww) > 0:
    #        vel.append(kin_tab['V'][ww])
    #        sig.append(kin_tab['SIGMA'][ww])
    #    else:
    #        vel.append(np.nan)
    #        sig.append(np.nan)

    for b in bin_tab:
        if b['BIN_ID'] < 0:
            vel.append(np.nan)
            sig.append(np.nan)
        else:
            vel.append(kin_tab['V'][b['BIN_ID']])
            sig.append(kin_tab['SIGMA'][b['BIN_ID']])

    vel = np.asarray(np.reshape(vel,(s[1],s[2])),dtype=np.float)
    sig = np.asarray(np.reshape(sig,(s[1],s[2])),dtype=np.float)
    
    regions_x = [0,3,5,7,9,11,13,15,17,19]
    regions_y = [0.0,0.5,1,2,4]
    
    tmp = (cube_orig[:,yc-round(regions_y[1]*1e3/pc_scale):yc+round(regions_y[1]*1e3/pc_scale),xc-round(regions_x[1]*1e3/pc_scale):xc+round(regions_x[1]*1e3/pc_scale)])
    central_spec = np.nansum(np.nansum(tmp,axis=1),axis=1)
    tmp = (var_orig[:,yc-round(regions_y[1]*1e3/pc_scale):yc+round(regions_y[1]*1e3/pc_scale),xc-round(regions_x[1]*1e3/pc_scale):xc+round(regions_x[1]*1e3/pc_scale)])
    central_var = np.nansum(np.nansum(tmp,axis=1),axis=1)
    central_var[np.isfinite(central_var) == False] = np.nanmedian(central_var[np.isfinite(central_var)])
    
    wav = np.arange(h['NAXIS3'])*h['CD3_3']+h['CRVAL3']
    
    #templates,templates_wav = ppxf_muse.import_empirical_templates()
    
    z_gal = 0.022402
    fwhm_gal = 2.7
    fwhm_tmp = 2.5
    c = 299792.458
    
    wav0 = wav/(1+z_gal)
    
    cube = np.ones(s)*np.nan
    var = np.ones(s)*np.nan

    for i in range(s[1]):
        for j in range(s[2]):
            wav_spec = wav/(1 + z_gal + vel[i,j]/c)
            spec_new = np.interp(wav0,wav_spec,cube_orig[:,i,j])
            var_new = np.interp(wav0,wav_spec,var_orig[:,i,j])
            cube[:,i,j] = spec_new
            var[:,i,j] = var_new
    
    for i in np.array(range(len(regions_x)-1)):
        for j in np.array(range(len(regions_y)-1)):
            tmp1 = cube[:,yc+round(regions_y[j]*1e3/pc_scale):yc+round(regions_y[j+1]*1e3/pc_scale),xc+round(regions_x[i]*1e3/pc_scale):xc+round(regions_x[i+1]*1e3/pc_scale)]
            tmp2 = cube[:,yc-round(regions_y[j+1]*1e3/pc_scale):yc-round(regions_y[j]*1e3/pc_scale),xc+round(regions_x[i]*1e3/pc_scale):xc+round(regions_x[i+1]*1e3/pc_scale)] 
            tmp3 = cube[:,yc+round(regions_y[j]*1e3/pc_scale):yc+round(regions_y[j+1]*1e3/pc_scale),xc-round(regions_x[i+1]*1e3/pc_scale):xc-round(regions_x[i]*1e3/pc_scale)] 
            tmp4 = cube[:,yc-round(regions_y[j+1]*1e3/pc_scale):yc-round(regions_y[j]*1e3/pc_scale),xc-round(regions_x[i+1]*1e3/pc_scale):xc-round(regions_x[i]*1e3/pc_scale)]
            spec1 = np.nansum(np.nansum(tmp1,axis=1),axis=1)
            spec2 = np.nansum(np.nansum(tmp2,axis=1),axis=1)
            spec3 = np.nansum(np.nansum(tmp3,axis=1),axis=1)
            spec4 = np.nansum(np.nansum(tmp4,axis=1),axis=1)

            tmp1 = var[:,yc+round(regions_y[j]*1e3/pc_scale):yc+round(regions_y[j+1]*1e3/pc_scale),xc+round(regions_x[i]*1e3/pc_scale):xc+round(regions_x[i+1]*1e3/pc_scale)]
            tmp2 = var[:,yc-round(regions_y[j+1]*1e3/pc_scale):yc-round(regions_y[j]*1e3/pc_scale),xc+round(regions_x[i]*1e3/pc_scale):xc+round(regions_x[i+1]*1e3/pc_scale)] 
            tmp3 = var[:,yc+round(regions_y[j]*1e3/pc_scale):yc+round(regions_y[j+1]*1e3/pc_scale),xc-round(regions_x[i+1]*1e3/pc_scale):xc-round(regions_x[i]*1e3/pc_scale)] 
            tmp4 = var[:,yc-round(regions_y[j+1]*1e3/pc_scale):yc-round(regions_y[j]*1e3/pc_scale),xc-round(regions_x[i+1]*1e3/pc_scale):xc-round(regions_x[i]*1e3/pc_scale)]
            var1 = np.nansum(np.nansum(tmp1,axis=1),axis=1)
            var2 = np.nansum(np.nansum(tmp2,axis=1),axis=1)
            var3 = np.nansum(np.nansum(tmp3,axis=1),axis=1)
            var4 = np.nansum(np.nansum(tmp4,axis=1),axis=1)

            var1[np.isfinite(var1) == False] == np.nanmedian(var1)
            var2[np.isfinite(var2) == False] == np.nanmedian(var2)
            var3[np.isfinite(var3) == False] == np.nanmedian(var3)
            var4[np.isfinite(var4) == False] == np.nanmedian(var4)

            #var4_0 = np.interp(wav0,wav4,var4)
            
            spec = spec1 + spec2 + spec3 + spec4
            var0 = var1 + var2 + var3 + var4
            tab = Table([wav0,spec,var0,spec1,var1,spec2,var2,spec3,var3,spec4,var4],
                        names=['Wav','Flux','Var','Spec1','Var1','Spec2','Var2','Spec3','Var3','Spec4','Var4'])   
            fname = 'UGC10738_r{}_{}_z{}_{}.dat'.format(regions_x[i],regions_x[i+1],regions_y[j],regions_y[j+1])
            tab.write(outdir+fname,format='ascii.commented_header',overwrite=True)


###############################################################################
            
def determine_goodpixels(logLam, lamRangeTemp, z, width=800):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    :param logLam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lamRangeTemp: Two elements vectors [lamMin2, lamMax2] with the minimum
        and maximum wavelength in Angstrom in the stellar *template* used in PPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodPixels to be used as input for pPXF

    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
    dv = np.full_like(lines, width)  # width/2 of masked gas emission region in km/s
    c = 299792.458 # speed of light in km/s

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (np.exp(logLam) > line*(1 + z)*(1 - dvj/c)) \
              & (np.exp(logLam) < line*(1 + z)*(1 + dvj/c))

    flag |= np.exp(logLam) > lamRangeTemp[1]*(1 + z)*(1 - 900/c)   # Mask edges of
    flag |= np.exp(logLam) < lamRangeTemp[0]*(1 + z)*(1 + 900/c)   # stellar library

    return np.flatnonzero(~flag)
    
###############################################################################
    
def ind_to_pop(index_set='',errs=False,chi2_plot=False):


    # Set input and output locations and identify input files
    infile = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/measured_lick_indices_voronoi_v3.dat'
    outfile = '/import/opus1/nscott/Stellar_Populations/MW_analogues/MUSE/UGC10738_v3/measured_ssps_voronoi_schi_v3.dat'

    # Read in the Index -> Pop conversion table, either TMJ or Schiavon
    ssp_table_file = '/import/opus1/nscott/Stellar_Populations/misc/alpha-models-ti-schiavon_reinterpolated_log_age_nolick.dat'
    #ssp_table_file = '/import/opus1/nscott/Stellar_Populations/misc/tmj_interpolated_extended_log_age.dat'
    ssp_table = ascii.read(ssp_table_file)
    
    # Set initial variables
    tab = ascii.read(infile)
    
    # Define different sets of indices. Non-specified indices are not taken into account in the fit.
    # NB Can add custom sets here easily
    all_muse = ['Hbeta','Mgb','Fe5015','Fe5270','Fe5335','Fe5406','Fe5709','Fe5782','NaD','TiO1','TiO2','Mg1','Mg2']
    min_muse = ['Hbeta','Mgb','Fe5015','Fe5270','Fe5335','Fe5406','Fe5709','Fe5782']
    schi_muse = ['Hbeta','Mgb','Fe5015','Mg2','Fe5270','Fe5335']
    
    all = ['Mgb','Hbeta','Fe5270','Fe5335','Fe5015','Fe4383','Fe4668','Ca4227','CN1','CN2',
           'HdeltaA','G4300','HgammaA','HgammaF','HdeltaF','Mg1','Mg2','Ca4455','Fe5406','Fe4531']
    all_schi = ['Mgb','Hbeta','Fe5270','Fe5335','Fe5015','Fe4383','Fe4668','Ca4227','CN1','CN2',
                       'HdeltaA','G4300','HgammaA','HgammaF','HdeltaF','Mg2']
    sauron = ['Mgb','Hbeta','Fe5015']
    red = ['Mgb','Hbeta','Fe5015','Fe5270','Fe5335']
    robust = ['HdeltaA','HgammaA','Hbeta','Mgb','Fe5270','Fe5335']
    good = ['Mgb','Fe5270','Fe5335','Fe4383','Fe4668','Ca4227','CN1','CN2','HdeltaA','G4300','HgammaA','Mg1','Mg2','HdeltaF','HdeltaA','Fe5406','Fe4531','Ca4455']

    
    # Select which index set to use
    if (index_set == '') | (index_set == 'MUSE'):
        indices = all_muse
        index_set = 'MUSE'
    elif index_set == 'MIN_MUSE':
        indices = min_muse
    elif index_set == 'ALL':
        indices = all
    elif index_set == 'SAURON':
        indices = sauron
    elif index_set == 'RED':
        indices = red
    elif index_set == 'ROBUST':
        indices = robust
    elif index_set == 'GOOD':
        indices = good
    elif index_set == 'ALL_SCHI':
        indices = all_schi
    elif index_set == 'SCHI_MUSE':
        indices = schi_muse

    # Set output format depending on inputs i.e. is this apertures, radial profiles or maps
    if errs:
        #new_columns = ['r','z','Age','-Age','+Age','[Z/H]','-Z','+Z','[alpha/Fe]','-a','+a','Chi2','SN']
        new_columns = ['x','y','Age','-Age','+Age','[Z/H]','-Z','+Z','[alpha/Fe]','-a','+a','Chi2','SN']
        dtype = ('f4','f4','f8','f8','f8','f4','f4','f4','f4','f4','f4','f8','f4')
    else:
        #new_columns = ['r','z','Age','[Z/H]','[alpha/Fe]','Chi2','SN']
        new_columns = ['x','y','Age','[Z/H]','[alpha/Fe]','Chi2','SN']
        dtype = ('f4','f4','f8','f4','f4','f8','f4')

    out_table = Table(names=new_columns,dtype=dtype)

    # Transfer some columns direct from the input files, depending on input format
    n_points = len(tab['Mgb'])

    for i in range(n_points):
        good_inds_counter = 0

        # Add only indices of interest to the array to compare to models
        ind_array = []
        ind_err_array = []
        for ind in indices:
        
            ind_array.append(tab[ind][i])
            ind_err_array.append(tab[ind+'_err'][i])
        

        ind_array = np.array(ind_array)
        ind_err_array = np.array(ind_err_array)

        # "Mask" bad values
        ind_array[np.where(np.isfinite(ind_array) == False)] = np.nan
        ind_err_array[np.where(np.isfinite(ind_err_array) == False)] = np.nan

        ww = np.where(np.isfinite(ind_array) & np.isfinite(ind_err_array))
        # Here is a loop that calculates chi^2 for all SSP models, identifies outlying index measurements,
        # masks them, then loops until no more outliers exist
        while len(ww[0]) > 0:
            chi2_array = []
            for j in range(len(ind_array)):
                # Calculate the chi^2 for each individual Lick index value for every single SSP model
                chi2_array.append(((ind_array[j] - ssp_table[indices[j]])/ind_err_array[j])**2)

            # Sum the chi^2 for all indices
            chi2 = np.nansum(chi2_array,axis=0)
            # Identify the minimum chi^2 value and which SSP model it corresponds to
            w = np.where(chi2 == np.min(chi2))
            chi2_array = np.array(chi2_array)
            #ww = np.where(np.squeeze(np.abs(chi2_array[:,w] - np.nanmean(chi2_array[:,w])) > 2*np.nanstd(chi2_array[:,w])))
            
            # Identify and mask outlying indices
            ww = np.where(np.squeeze(chi2_array[:,w]) > 3.**2)
            if len(ww[0]) > 0:
                ind_array[ww] = np.nan
        
        w = np.where(chi2 == np.min(chi2))
        # Set output format based on input format
        if (len(w[0]) != 0) & (len(np.where(np.isfinite(ind_array) == True)[0]) >= 3):
            if len(w[0]) > 1:
                w = np.array([w[0][0]])
            if errs == False:
                # Just output the best-fitting values if we didn't ask for uncertainties
                data[2,i] = np.median(ssp_table['age'][w])
                data[3,i] = np.median(ssp_table['[Z/H]'][w])
                data[4,i] = np.median(ssp_table['[alpha/Fe]'][w])
                data[-1,i] = chi2[w]/(len(np.where(np.isfinite(ind_array) == True)[0])-2)

            
            else:
                # If uncertainties are required, identify the range of models who have chi^2 within
                # 1-sigma of the best fitting model, then find the max/min values of the SSP params
                # that lie within this 1-sigma chi^2 contour
                v = np.where(chi2 < np.min(chi2)+3.5)
                tmax = np.max(ssp_table['age'][v])
                tmin = np.min(ssp_table['age'][v])
                zmax = np.max(ssp_table['[Z/H]'][v])
                zmin = np.min(ssp_table['[Z/H]'][v])
                amax = np.max(ssp_table['[alpha/Fe]'][v])
                amin = np.min(ssp_table['[alpha/Fe]'][v])

            if (chi2_plot == True) and (i == 0):
                # Option to pause code to check intermediate outputs
                code.interact(local=dict(globals(),**locals()))

    # Write data to file
    #ascii.write(np.transpose(data),outfile,names=new_columns)

            row = [tab['XBIN'][i],tab['YBIN'][i],np.median(ssp_table['age'][w]),tmin,tmax,
               np.median(ssp_table['[Z/H]'][w]),zmin,zmax,np.median(ssp_table['[alpha/Fe]'][w]),
               amin,amax,chi2[w][0]/(len(np.where(np.isfinite(ind_array) == True)[0])-2),tab['SN'][i]]

        out_table.add_row(row)

    out_table.write(outfile,format='ascii.commented_header',overwrite=True)
