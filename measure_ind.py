import os
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.pylab as plt
from numpy import random as random

import code

def sigma_correct(veldisp,width,index):
    """
        Correct Lick index measurements for observed
        velocity dispersion if larger than Lick resolution
    """

    sigcorr_slope_file = '/import/opus1/nscott/Stellar_Populations/misc/sigcorr_slope.dat'
    sigcorr_icpt_file = '/import/opus1/nscott/Stellar_Populations/misc/sigcorr_icpt.dat'

    tab1 = ascii.read(sigcorr_slope_file)
    tab2 = ascii.read(sigcorr_icpt_file)

    slope_arr = tab1[index]
    icpt_arr = tab2[index]

    sigma = np.arange(17)*25.
    if (np.ceil(veldisp) % 25) == 0:
        width = width * slope_arr[np.int(np.ceil(veldisp)//25)] + icpt_arr[np.int(np.ceil(veldisp)//25)]
    else:
        try:
            high = width * slope_arr[np.int(np.ceil(veldisp)//25+1)] + icpt_arr[np.int(np.ceil(veldisp)//25+1)]
            low = width * slope_arr[np.int(np.ceil(veldisp)//25)] + icpt_arr[np.int(np.ceil(veldisp)//25)]
            width = ((25. - (veldisp - sigma[np.int(np.ceil(veldisp)//25)]))*low +
                 (25. - (sigma[np.int(np.ceil(veldisp)//25+1)] - veldisp))*high)/25.
        except:
            code.interact(local=dict(globals(),**locals()))
    return width

def measure_ind(rest_wavelength,spectrum,index,plot=False,obswav=''):

    # Measure the requested Lick index

    index_definition_file = os.path.expanduser('/import/opus1/nscott/Stellar_Populations/misc/pseudocontinuum_absorption_line_indices.dat')
    #names = ['Index','Index_Lower','Index_Upper','BC_Lower','BC_Upper','RC_Lower','RC_Upper','Source']
    index_definitions = ascii.read(index_definition_file)#,names=names)

    #Check requested index exists in the table
    
    index_names = index_definitions['Index']
    aa = np.where(index == index_names)[0]
    if len(aa) == 0:
        raise Exception('Invalid index entered')

    #Need to check for D4000 or R94 defined indices

    ps_blue_inds = np.where((rest_wavelength > float(index_definitions[aa]['BC_lower'].data[0])) &
                             (rest_wavelength < float(index_definitions[aa]['BC_upper'].data[0])))[0]
    ps_red_inds = np.where((rest_wavelength > float(index_definitions[aa]['RC_lower'].data[0])) &
                           (rest_wavelength < float(index_definitions[aa]['RC_upper'].data[0])))[0]
    ps_blue_centre = np.mean(rest_wavelength[ps_blue_inds])
    ps_red_centre = np.mean(rest_wavelength[ps_red_inds])
    ps_blue = np.median(spectrum[ps_blue_inds])
    ps_red = np.median(spectrum[ps_red_inds])


    ps_gradient = (ps_red - ps_blue)/(ps_red_centre - ps_blue_centre)
    ps_zeropoint = ps_red - ps_gradient*ps_red_centre

    index_inds = np.where((rest_wavelength > float(index_definitions[aa]['I_lower'].data[0])) &
                          (rest_wavelength < float(index_definitions[aa]['I_upper'].data[0])))[0]
    index_centre = np.mean(rest_wavelength[index_inds])

    ps_flux = 0.0
    obs_flux = 0.0
    for ind in index_inds:
        ps_flux += ps_zeropoint + ps_gradient*rest_wavelength[ind]
        obs_flux += spectrum[ind]

    missing_flux = ps_flux - obs_flux
    ps_ind_flux = ps_zeropoint + ps_gradient*index_centre
    equivalent_width = missing_flux/ps_ind_flux

    # Handle bad columns - just return NaN where present

    if plot:
        plt.figure(4)
        plt.clf()
        plt.plot(rest_wavelength,spectrum)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Flux')
        plt.title(index)
        plt.plot([ps_blue_centre,ps_red_centre],[ps_blue,ps_red],'ks-',markersize=10,linewidth=2.5)
        plt.plot([rest_wavelength[ps_blue_inds[0]],rest_wavelength[ps_blue_inds[-1]]],[ps_blue*1.05,ps_blue*1.05],'k--',linewidth=2.0)
        plt.text(ps_blue_centre,ps_blue*1.1,'Blue Pseudo-\ncontinuum Band',ha='center')
        plt.plot([rest_wavelength[ps_red_inds[0]],rest_wavelength[ps_red_inds[-1]]],[ps_red*1.05,ps_red*1.05],'k--',linewidth=2.0)
        plt.text(ps_red_centre,ps_red*1.1,'Red Pseudo-\ncontinuum Band',ha='center')
        plt.plot([rest_wavelength[index_inds[0]],rest_wavelength[index_inds[-1]]],[ps_ind_flux*1.1,ps_ind_flux*1.1],'k--',linewidth=2.0)
        plt.text(index_centre,ps_ind_flux*1.15,'Index Band',ha='center')
        plt.plot([rest_wavelength[index_inds[0]],rest_wavelength[index_inds[0]]],[0.0,spectrum[index_inds[0]]],'k-')
        plt.plot([rest_wavelength[index_inds[-1]],rest_wavelength[index_inds[-1]]],[0.0,spectrum[index_inds[-1]]],'k-')
        ps_line = ps_zeropoint + ps_gradient*rest_wavelength[index_inds]
        plt.fill_between(rest_wavelength[index_inds],ps_line,spectrum[index_inds],facecolor='green',alpha=0.5)
        plt.axis([min(rest_wavelength[ps_blue_inds])-10,max(rest_wavelength[ps_red_inds])+10,
          min(spectrum[index_inds])*0.9,max([ps_blue,ps_red])*1.1])
        plt.draw()
        plt.show(block=False)

    if np.abs(np.mean(obswav[index_inds]) - 5577.0) < 15.:
        equivalent_width = np.nan

    if index_definitions['Molecular'][aa] == 1:
        equivalent_width = -2.5*np.log10(1 - equivalent_width/(index_definitions[aa]['I_upper'].data-index_definitions[aa]['I_lower'].data))
        equivalent_width = equivalent_width[0]
        
    return equivalent_width

def mc_ind_errors(rest_wavelength,spectrum,variance,index,n=100,obswav=''):

    # Determine uncertainty on a Lick index measurement by randomly drawing
    # noise from the variance spectrum and re-measuring the index.

    ind_vals = np.zeros((n))
    
    spectrum[np.isfinite(spectrum) == False] = np.nanmedian(spectrum)
    variance[np.isfinite(variance) == False] = np.nanmedian(variance)
    
    for i in range(n):
        spec_new = np.copy(spectrum) + random.randn(len(spectrum))*np.sqrt(variance)
        ind_vals[i] = measure_ind(rest_wavelength,spec_new,index,obswav=obswav)

    mean_ind = np.mean(ind_vals)
    err_ind = np.std(ind_vals)

    return mean_ind,err_ind
