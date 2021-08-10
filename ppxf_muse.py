from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from scipy import ndimage
from time import perf_counter as clock
import astropy.io.ascii as ascii
import glob,copy
import numpy as np
import matplotlib.pyplot as plt
import code
from scipy.ndimage.filters import gaussian_filter1d
from ppxf import miles_util as lib

def ppxf_muse_alpha(gal_lin,wav_lin,fwhm_gal,miles,z_gal,plot=False,three=False,do_regul=False):

    c = 299792.458
    
    # Set up input spectrum and stellar template library for non-regularized fits

    mask = ((wav_lin > np.exp(miles[0].log_lam_temp[0])) 
                & (wav_lin < np.exp(miles[0].log_lam_temp[-11])))

    gal_lin = gal_lin[mask]
    wav_lin = wav_lin[mask]

    lamRange1 = np.array([wav_lin[0],wav_lin[-1]])
    lamRange1 = lamRange1/(1+z_gal)
    fwhm_gal = fwhm_gal/(1+z_gal)
    
    galaxy,logLam1,velscale = util.log_rebin(lamRange1,gal_lin,velscale=60.0)
    norm_val = np.median(galaxy)
    galaxy = galaxy/np.median(galaxy)
    noise = np.full_like(galaxy,0.0047)
    
    s = miles[0].templates.shape
    templates_unpacked = np.append(np.reshape(miles[0].templates,(s[0],s[1]*s[2])),
                                    np.reshape(miles[1].templates,(s[0],s[1]*s[2])),axis=1)
    if np.shape(miles)[0] == 3:
        templates_unpacked = np.append(templates_unpacked,
                                    np.reshape(miles[2].templates,(s[0],s[1]*s[2])),axis=1)

    logLam2 = miles[0].log_lam_temp
    lamRange2 = [np.exp(logLam2[0]),np.exp(logLam2[-1])]
    dv = c*(miles[0].log_lam_temp[0] - logLam1[0])
    z = 0.0
    
    good_pixels = util.determine_goodpixels(logLam1,lamRange2,z)  
    start = [0., 100.,0,0] 
    fixed = [0,0,0,0] 
    
    # Perform initial fit to estimate typical noise

    pp = ppxf(templates_unpacked,galaxy,noise,velscale,start,
        goodpixels=good_pixels,plot=False,moments=4,
        degree=10,vsyst=dv,lam=np.exp(logLam1),fixed=fixed,quiet=True)
                
    
    noise_local = np.sqrt(np.median((galaxy-pp.bestfit)**2))
    noise = np.full_like(galaxy,noise_local)
        
    #code.interact(local=dict(globals(),**locals()))

    # Perform second fit to estimate the noise vector        

    pp = ppxf(templates_unpacked,galaxy,noise,velscale,start,
            goodpixels=good_pixels,plot=False,moments=4,
            degree=10,vsyst=dv,lam=np.exp(logLam1),
            fixed=fixed,clean=True,quiet=True)
 
    noise_local = np.abs(galaxy-pp.bestfit)
    noise_local[np.where(noise_local[:-1]-noise_local[1:] > 0.05)] = np.nanmedian(noise_local[good_pixels])
    noise_local[np.where(noise_local[:-1]-noise_local[1:] > 0.05)] = np.nanmedian(noise_local[good_pixels])
    noise_local[~good_pixels] = np.nanmedian(noise_local[good_pixels])
    noise_local = gaussian_filter1d(noise_local,20)
    pp.new_noise = noise_local

    noise = np.copy(noise_local)
    
    # Perform third fit to determine final noise for population fitting
    
    pp = ppxf(templates_unpacked,galaxy,noise,velscale,start,
            goodpixels=pp.goodpixels,plot=False,moments=4,
            degree=10,vsyst=dv,lam=np.exp(logLam1),
            fixed=fixed,clean=False,quiet=True)
            
    noise = noise*np.sqrt(pp.chi2)
    
    # Mask edges of wavelength range, 5577 skyline and NaD
    
    flag = False
    flag |= np.exp(logLam1) > lamRange2[1]*(1-900/c)
    flag |= np.exp(logLam1) < lamRange2[0]*(1+900/c)
    flag |= ((np.exp(logLam1) < 5587/(1+0.0224)) & (np.exp(logLam1) > 5567/(1+0.0224)))
    flag |= ((np.exp(logLam1) < 5905) & (np.exp(logLam1) > 5880)) # NaD
    flag |= ((np.exp(logLam1) < 5135) & (np.exp(logLam1) > 5069)) # Mg1
    good_pixels = np.flatnonzero(~flag)
    
    # Set up gas templates and stellar templates in preparation for population fit
    
    gas_templates, gas_names, line_wave = util.emission_lines(
    logLam2, lamRange1, fwhm_gal, tie_balmer=False, limit_doublets=True)
    
    if three:
        stellar_templates = np.zeros((list(s)+[3]))
        stellar_templates[:,:,:,0] = miles[0].templates
        stellar_templates[:,:,:,1] = miles[1].templates  
        stellar_templates[:,:,:,2] = miles[2].templates
    else:
        stellar_templates = np.zeros((list(s)+[2]))
        stellar_templates[:,:,:,0] = miles[0].templates
        stellar_templates[:,:,:,1] = miles[1].templates
    reg_dim = stellar_templates.shape[1:]
    stellar_templates = stellar_templates.reshape(stellar_templates.shape[0],-1)

    stellar_templates_norm = np.median(stellar_templates)
    stellar_templates = stellar_templates/stellar_templates_norm
        
    all_templates = np.column_stack([stellar_templates,gas_templates])
    n_temps = stellar_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])
    n_balmer = len(gas_names) - n_forbidden
    component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
    gas_component = np.array(component) > 0
        
    moments = [2,2,2]
    start = [pp.sol[0],pp.sol[1]]
    start = [start,start,start]
    fixed = [[0,0],[0,0],[0,0]]
    
    # Perform and return results of gas+population fit
    
    pp_gas = ppxf(all_templates, galaxy, noise, velscale, start,
              plot=plot, moments=moments, degree=-1,mdegree=10, vsyst=dv,
              lam=np.exp(logLam1), clean=False,goodpixels=good_pixels,
              component=component, gas_component=gas_component,
              reg_dim=reg_dim,fixed=fixed,regul=0,
              gas_names=gas_names,gas_reddening=None,quiet=True)
    pp_gas.new_noise = noise_local

    if do_regul:
        noise = noise*np.sqrt(pp_gas.chi2)
        pp_gas = ppxf(all_templates, galaxy, noise, velscale, start,
              plot=plot, moments=moments, degree=-1,mdegree=10, vsyst=dv,
              lam=np.exp(logLam1), clean=False,goodpixels=good_pixels,
              component=component, gas_component=gas_component,
              reg_dim=reg_dim,fixed=fixed,regul=0,
              gas_names=gas_names,gas_reddening=None,quiet=True)
        chi2_unregul = pp_gas.chi2
        delta_regul = 0.1
        
        while ((pp_gas.chi2*good_pixels.size < 
               chi2_unregul*good_pixels.size+np.sqrt(2*good_pixels.size)) & 
               (delta_regul > 0.0005)):

            pp_gas_old = copy.deepcopy(pp_gas)
            pp_gas = ppxf(all_templates, galaxy, noise, velscale, start,
                          plot=plot, moments=moments, degree=-1,mdegree=10, vsyst=dv,
                          lam=np.exp(logLam1), clean=False,goodpixels=good_pixels,
                          component=component, gas_component=gas_component,
                          reg_dim=reg_dim,fixed=fixed,reg_ord=2,regul=1/delta_regul,
                          gas_names=gas_names,gas_reddening=None,quiet=True)
            delta_regul = delta_regul/2
            
        pp_gas = pp_gas_old

    return pp_gas

def ppxf_muse(gal_lin,wav_lin,fwhm_gal,z_gal,templates,templates_wav,fwhm_temp,v0=False,
                plot=False,derive_noise=False,subtract_emission=False,noise=[]):

    c = 299792.458
    velscale_ratio = 1

    lamRange1 = np.array([wav_lin[0],wav_lin[-1]])
    lamRange1 = lamRange1/(1+z_gal)
    fwhm_gal = fwhm_gal/(1+z_gal)

    galaxy,logLam1,velscale = util.log_rebin(lamRange1,gal_lin)
    norm_val = np.median(galaxy)
    galaxy = galaxy/np.median(galaxy)
    if noise == []:
        noise = np.full_like(galaxy,0.0047)
    
    lamRange2 = np.array([templates_wav[0],templates_wav[-1]])
    ssp_new, logLam2, velscale_temp = util.log_rebin(lamRange2,templates[:,0],velscale=velscale/velscale_ratio)
    templates_new = np.empty((ssp_new.size,templates.shape[1]))
    
    fwhm_dif = np.sqrt(fwhm_gal**2 - fwhm_temp**2)
    sigma = fwhm_dif/2.355/(templates_wav[1]-templates_wav[0])
    
    for j in range(templates.shape[1]):
        ssp = templates[:,j]
        if fwhm_gal > fwhm_temp:
            ssp = ndimage.gaussian_filter1d(ssp,sigma)
        ssp_new, logLam2, velscape_temp = util.log_rebin(lamRange2,ssp,velscale=velscale/velscale_ratio)
        templates_new[:,j] = ssp_new/np.median(ssp_new)

    dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c
    z = 0.0
    
    good_pixels = util.determine_goodpixels(logLam1,lamRange2,z)
    
    if v0 == True:
        vel = 0
    else:
        vel = c*np.log(1+z)
    start = [vel, 100.]
    t = clock()
    
    if v0 == True:
        fixed = [1,0]
    else:
        fixed = [0,0]
    
    pp = ppxf(templates_new,galaxy,noise,velscale,start,
        goodpixels=good_pixels,plot=False,moments=2,
        degree=10,vsyst=dv,velscale_ratio=velscale_ratio,
        lam=np.exp(logLam1),fixed=fixed,quiet=True)
                
    if derive_noise:
        
        noise_local = np.sqrt(np.median((galaxy-pp.bestfit)**2))
        noise = np.full_like(galaxy,noise_local)
        
        if plot == True:
            plt.figure(1)
            plt.clf()
        
        pp = ppxf(templates_new,galaxy,noise,velscale,start,
            goodpixels=good_pixels,plot=False,moments=2,
            degree=10,vsyst=dv,velscale_ratio=velscale_ratio,
            lam=np.exp(logLam1),fixed=fixed,clean=True,quiet=True)
        noise_local = np.abs(galaxy-pp.bestfit)
        noise_local[np.where(noise_local[:-1]-noise_local[1:] > 0.05)] = np.nanmedian(noise_local[good_pixels])
        noise_local[np.where(noise_local[:-1]-noise_local[1:] > 0.05)] = np.nanmedian(noise_local[good_pixels])
        noise_local[~good_pixels] = np.nanmedian(noise_local[good_pixels])
        noise_local = gaussian_filter1d(noise_local,20)
        pp.new_noise = noise_local
        
        noise = np.copy(noise_local)
            
    if subtract_emission:
        gas_templates, gas_names, line_wave = util.emission_lines(
        logLam2, lamRange1, fwhm_gal, tie_balmer=False, limit_doublets=True)
        
        all_templates = np.column_stack([templates_new,gas_templates])
        n_temps = templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])
        n_balmer = len(gas_names) - n_forbidden
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        gas_component = np.array(component) > 0
        
        moments = [2,2,2]
        start = [start,start,start]
        
        if plot == True:
            plt.figure(1)
            plt.clf()
        
        flag = False
        flag |= np.exp(logLam1) > lamRange2[1]*(1-900/c)
        flag |= np.exp(logLam1) < lamRange2[0]*(1+900/c)
        good_pixels = np.flatnonzero(~flag)

        pp_gas = ppxf(all_templates, galaxy, noise, velscale, start,
              plot=plot, moments=moments, degree=10, vsyst=dv,
              velscale_ratio=velscale_ratio,
              lam=np.exp(logLam1), clean=True,goodpixels=good_pixels,
              component=component, gas_component=gas_component,
              gas_names=gas_names,gas_reddening=None,quiet=True)
        pp_gas.new_noise = noise_local
    
    if subtract_emission:
        return pp_gas,(z_gal + 1)*(1 + pp.sol[0]/c) - 1,
    else:
        return pp,(z_gal + 1)*(1 + pp.sol[0]/c) - 1
    
def import_empirical_templates(testing=False):

    lib_dir = '/import/opus1/nscott/Stellar_Populations/misc/MILES_library_v9.1/'
    template_files = glob.glob(lib_dir+'*')
    FWHM_temp = 2.50
    ssp_col_names = ['wav','flux']

    ssp = ascii.read(template_files[0],names=ssp_col_names)

    wav = ssp['wav']

    if testing:
        n_templates = 30
    else:
        n_templates = len(template_files)
    inc = len(template_files)//n_templates
        
    templates = np.empty((ssp['flux'].size,n_templates))
    templates[:,0] = ssp['flux']

    for i in np.array(range(n_templates-1))+1:
        ssp = ascii.read(template_files[(i*inc)],names=ssp_col_names)
        templates[:,i] = ssp['flux']/np.nanmedian(ssp['flux'])

    print('Empirical templates loaded')

    return [templates,wav]

def import_synthetic_templates(mode='',template_min_age=1.,template_max_age=14.5):

    lib_dir = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v9.1/'
    template_files = glob.glob(lib_dir+'*.fits')
    n_temps = len(template_files)
    
    template_ages = np.zeros((n_temps))
    template_zs = np.zeros((n_temps))
    template_z_names = np.empty((n_temps),'S5')

    
    for i in range(n_temps):
        template_ages[i] = np.float(template_files[i].split('T')[1][:7])
        template_z_names[i] = template_files[i].split('Z')[1][:5]
        sign_char = template_z_names[i][0]
        if sign_char == 'm':
            sign = -1.0
        elif sign_char == 'p':
            sign = 1.0
        else:
            sign = 1.0
        template_zs[i] = np.float(template_files[i].split('Z')[1][1:5])*sign
    
    
    reorder = np.array([3,2,1,0,4,5])
    template_ages = np.unique(template_ages)
    template_zs = np.unique(template_zs)
    template_z_names = np.unique(template_z_names)[reorder]
    if mode == 'AGN':
        template_min_age = 0.08
    template_ages = template_ages[np.where((template_ages > template_min_age) & (template_ages < template_max_age))]
    nAge = len(template_ages)
    nZ = len(template_zs)
    
    hdu_ssp = fits.open(template_files[0])
    ssp = hdu_ssp[0].data
    ssp_header = hdu_ssp[0].header
    wav = np.arange(ssp_header['NAXIS1'])*ssp_header['CDELT1'] + ssp_header['CRVAL1']
    
    templates = np.zeros((ssp.size,nAge,nZ))
    template_file = np.empty((nAge,nZ),dtype='|S100')
    
    for i in range(n_temps):
        hdu_ssp = fits.open(template_files[i])
        ssp = hdu_ssp[0].data
        template_age = np.float(template_files[i].split('T')[1][:7])
        if template_age > template_min_age:
            template_z_name = template_files[i].split('Z')[1][:5]
            Zind = np.array(np.where(template_z_name == template_z_names)[0])
            Tind = np.array(np.where(template_age == template_ages)[0])
            templates[:,Tind,Zind] = ssp.reshape(ssp.size,1)
            template_file[Tind,Zind] = template_files[i]

    print('Synthetic templates loaded')

    return [templates,wav,template_ages,template_zs,template_z_names]
    
def import_alpha_templates(velscale=61.94592691,three=False):
    # Some constants, either for physics or for the specific templates
    
    fwhm_miles = 2.5
    c = 299792.458
    
    # Path to the solar-alpha model directory - edit to match where they are on your machine
    path_0 = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v11/MILES_BASTI_UN_Ep0.00/Mun1.3*.fits'
    
    # Path to the super-solar-alpha model directory - edit to match where they are on your machine
    path_04 = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v11/MILES_BASTI_UN_Ep0.40/Mun1.3*.fits'
    
    # Load template libraries
    miles_0 = lib.miles(path_0,velscale,fwhm_miles)
    miles_04 = lib.miles(path_04,velscale,fwhm_miles)
    
    ssp_template_sets = [miles_0,miles_04]

    if three:
        path_02 = '/import/opus1/nscott/Stellar_Populations/misc/MILES_ssps_v11/MILES_BASTI_UN_Ep0.20_interpolated/Mun1.3*.fits'
        miles_02 = lib.miles(path_02,velscale,fwhm_miles)
        ssp_template_sets = [miles_0,miles_02,miles_04]
    
    return ssp_template_sets
    
