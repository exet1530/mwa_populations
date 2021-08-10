from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
import code
import matplotlib
import glob,os
import scipy.stats as st
from astropy.visualization import make_lupton_rgb
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def plot_weights_2d(xgrid, ygrid, weights, ylabel=r"[$\alpha$/Fe]",
                    xlabel="[Z/H]", title="", nodots=True,
                    colorbar=False, ax = '', log=True, **kwargs):
    """
    Plot an image of the 2-dim weights, as a function of xgrid and ygrid.
    This function allows for non-uniform spacing in x or y.

    """
    assert weights.ndim == 2, "`weights` must be 2-dim"
    assert xgrid.shape == ygrid.shape == weights.shape, \
        'Input arrays (xgrid, ygrid, weights) must have the same shape'

    x = xgrid[:, 0]  # Grid centers
    y = ygrid[0, :]
    xb = (x[1:] + x[:-1])/2  # internal grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])  # 1st/last border
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)

    if ax == '':
        ax = plt.gca()
    else:
        plt.sca(ax)
    cmap = cm.YlOrRd
    cmap.set_bad('white')
    if log:
        pc = plt.pcolormesh(xb, yb, np.log10(weights.T),vmin=-2,vmax=np.log10(0.32),
         edgecolors='face', cmap = cmap, **kwargs)
    else:
        pc = plt.pcolormesh(xb, yb, weights.T,
         edgecolors='face', cmap = cmap, **kwargs)        
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #plt.title(title)
    if not nodots:
        plt.plot(xgrid, ygrid, 'w,')
    if colorbar:
        plt.colorbar(pc)
        plt.sca(ax)  # Activate main plot before returning

    return pc

def quadrants_diagnostic_summary():

    weights_folder_path = ('/Users/nscott/Data/MW_analogues/MUSE/'+
        'UGC10738_v3/weights_two_ord2_quadrants/')
    base_files = glob.glob('*weights.dat')
    base_files.sort()
    
    names = ('Rad','Height','Z','<Z>','rms(Z)','Delta_Z_dust',
        'alpha','<alpha>','rms(alpha)','Delta_alpha_dust',
        'age','<age>','rms(age)','Delta_age_dust')
    dtypes = (np.float,np.float,np.float,np.float,np.float,np.float,np.float,
        np.float,np.float,np.float,np.float,np.float,np.float,np.float)
    outtab = Table(names=names,dtype=dtypes)
    
    for base_file in base_files:
        tab = Table.read(base_file,format='ascii')
        tab1 = Table.read(base_file[:-4]+'_a'+base_file[-4:],format='ascii')
        tab2 = Table.read(base_file[:-4]+'_b'+base_file[-4:],format='ascii')
        tab3 = Table.read(base_file[:-4]+'_c'+base_file[-4:],format='ascii')
        tab4 = Table.read(base_file[:-4]+'_d'+base_file[-4:],format='ascii')
        
        alphas = tab['alpha'].data.reshape(2,53,12)
        ages = tab['age'].data.reshape(2,53,12)
        metals = tab['z'].data.reshape(2,53,12)
        weights = tab['weight'].data.reshape(2,53,12)
        weights1 = tab1['weight'].data.reshape(2,53,12)
        weights2 = tab2['weight'].data.reshape(2,53,12)
        weights3 = tab3['weight'].data.reshape(2,53,12)
        weights4 = tab4['weight'].data.reshape(2,53,12)        
        
        rad = base_file.split('_',1)[1].split('_z')[0]
        radnum = np.float(rad[1])
        height = 'z'+base_file.split('z')[1].split('_wei')[0]
        hnum = np.float(height.split('_')[0][1:])
        
        zs = [np.sum(weights*metals),np.sum(weights1*metals),np.sum(weights2*metals),
            np.sum(weights3*metals),np.sum(weights4*metals)]
        ts = [10.0**np.sum(weights*np.log10(ages)),10.0**np.sum(weights1*np.log10(ages)),
            10.0**np.sum(weights2*np.log10(ages)),10.0**np.sum(weights3*np.log10(ages)),
            10.0**np.sum(weights4*np.log10(ages))]
        als = [np.sum(weights*alphas),np.sum(weights1*alphas),np.sum(weights2*alphas),
            np.sum(weights3*alphas),np.sum(weights4*alphas)] 
        
        print('###########################################################')
        print(rad,height)
        print('[Z/H]: ',zs[0],np.mean(zs[1:4]),np.std(zs[1:4]),(zs[1]+zs[3]-zs[2]-zs[4])/2)
        print(r'[$\alpha$/Fe]: ',als[0],np.mean(als[1:4]),np.std(als[1:4]),(als[1]+als[3]-als[2]-als[4])/2)
        print('Age: ',ts[0],np.mean(ts[1:4]),np.std(ts[1:4]),(ts[1]+ts[3]-ts[2]-ts[4])/2)
        print('###########################################################')
        print()
        
        row = [radnum,hnum,zs[0],np.mean(zs[1:4]),np.std(zs[1:4]),(zs[1]+zs[3]-zs[2]-zs[4])/2,
            als[0],np.mean(als[1:4]),np.std(als[1:4]),(als[1]+als[3]-als[2]-als[4])/2,
            ts[0],np.mean(ts[1:4]),np.std(ts[1:4]),(ts[1]+ts[3]-ts[2]-ts[4])/2]
            
        outtab.add_row(row)
        
    return outtab
        
def quadrants_diagnostic_plot(r='r0_3',z='z0.0_0.5'):

    weights_folder_path = ('/Users/nscott/Data/MW_analogues/MUSE/'+
        'UGC10738_v3/weights_two_ord2_quadrants/')
    
    fig,axes = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True,figsize=(12,8))

    tab = Table.read(weights_folder_path+'UGC10738_'+r+'_'+z+'_weights.dat',format='ascii')
    tab1 = Table.read(weights_folder_path+'UGC10738_'+r+'_'+z+'_weights_a.dat',format='ascii')
    tab2 = Table.read(weights_folder_path+'UGC10738_'+r+'_'+z+'_weights_b.dat',format='ascii')
    tab3 = Table.read(weights_folder_path+'UGC10738_'+r+'_'+z+'_weights_c.dat',format='ascii')
    tab4 = Table.read(weights_folder_path+'UGC10738_'+r+'_'+z+'_weights_d.dat',format='ascii')

    alphas = tab['alpha'].data.reshape(2,53,12)
    ages = tab['age'].data.reshape(2,53,12)
    metals = tab['z'].data.reshape(2,53,12)
    weights = tab['weight'].data.reshape(2,53,12)
    weights1 = tab1['weight'].data.reshape(2,53,12)
    weights2 = tab2['weight'].data.reshape(2,53,12)
    weights3 = tab3['weight'].data.reshape(2,53,12)
    weights4 = tab4['weight'].data.reshape(2,53,12)
    
    plt.subplots_adjust(left=0.075,right=0.85)

    cnorm = colors.Normalize(vmin=-2,vmax=np.log10(0.32))
    cbaxes = fig.add_axes([0.875,0.2,0.025,0.6])
    cbar = plt.colorbar(cm.ScalarMappable(norm=cnorm,cmap='YlOrRd'),cax=cbaxes)
    cbar.set_label('Mass Fraction',fontsize=15)
    cbar_ticks = cbar.get_ticks()
    cbar_ticks = [np.round(10.0**tick,decimals=3) for tick in cbar_ticks]
    cbar.ax.set_yticklabels(cbar_ticks)
    
    pc = plot_weights_2d(metals[:,0,:].T,alphas[:,0,:].T,
                np.sum(weights,axis=1).T,ax=axes[0,0])
    pc = plot_weights_2d(metals[:,0,:].T,alphas[:,0,:].T,
                np.sum(weights1,axis=1).T,ax=axes[0,1])
    pc = plot_weights_2d(metals[:,0,:].T,alphas[:,0,:].T,
                np.sum(weights2,axis=1).T,ax=axes[1,1])
    pc = plot_weights_2d(metals[:,0,:].T,alphas[:,0,:].T,
                np.sum(weights3,axis=1).T,ax=axes[0,2])
    pc = plot_weights_2d(metals[:,0,:].T,alphas[:,0,:].T,
                np.sum(weights4,axis=1).T,ax=axes[1,2])
                    
    plt.draw()
    plt.show()    

def figure_alpha_z_regions(contours=True):

    weights_folder_path = ('/Users/nscott/Data/MW_analogues/MUSE/'+
        'UGC10738_v3/weights_mg1/')
        
    height_labels = ['1_2','0.5_1','0.0_0.5']
    width_labels = ['3_5','5_7','7_9','9_11','11_13','13_15']
    width_text_labels = ['3 < R < 5 kpc',
                        '5 < R < 7 kpc',
                        '7 < R < 9 kpc',
                        '9 < R < 11 kpc',
                        '11 < R < 13 kpc',
                        '13< R < 15 kpc']
                        
    if contours:
        tab_mw = Table.read('/Users/nscott/Data/MW_analogues/MUSE/hayden2015dr12.txt'
            ,format='ascii')
        zh = tab_mw['feh'].data+np.log10(0.694*10.0**tab_mw['alpha/fe'].data+0.306)
        afe = tab_mw['alpha/fe'].data
        
        xx,yy = np.mgrid[-2.25:0.55:150j,-0.2:0.6:150j]
        positions = np.vstack([xx.ravel(),yy.ravel()])
    
    fig,axes = plt.subplots(nrows=3,ncols=6,sharex=True,sharey=True,figsize=(10,6))
    fig.subplots_adjust(wspace=0)

    for i,hl in enumerate(height_labels):
        for j,wl in enumerate(width_labels):
            tab = Table.read(weights_folder_path+'UGC10738_r'+wl+'_z'+hl+'_weights.dat',
                format='ascii')
            alphas = tab['alpha'].data.reshape(2,53,12)
            ages = tab['age'].data.reshape(2,53,12)
            metals = tab['z'].data.reshape(2,53,12)
            weights = tab['weight'].data.reshape(2,53,12)
            
            summed_weights = np.sum(weights,axis=1)
            #summed_weights[summed_weights < np.max(summed_weights)/100.] = 0.0
            summed_weights[summed_weights < 1e-2] = np.nan
            
            pc = plot_weights_2d(metals[:,0,:].T,alphas[:,0,:].T,
                summed_weights.T,ax=axes[i,j],log=False)
                
            if i == 0:
                plt.text(-2.25,0.5,width_text_labels[j],fontsize=10)
            if contours:
                h = hl.split('_')
                h = [np.float(h[0]),np.float(h[1])]
                w = wl.split('_')
                w = [np.float(w[0]),np.float(w[1])]
                ww = np.where((tab_mw['dr (kpc)'] >= w[0]) & (tab_mw['dr (kpc)'] < w[1]) &
                     (tab_mw['d|z| (kpc)'] >= h[0]) & (tab_mw['d|z| (kpc)'] < h[1]))
                values = np.vstack([zh[ww],afe[ww]])
                kernel = st.gaussian_kde(values)
                f = np.reshape(kernel(positions).T,xx.shape)  
                values = np.sort(f.ravel())[::-1]
                vlevels = np.cumsum(values)
                vlevels = vlevels/vlevels[-1]
                levels = [np.argmin(np.abs(vlevels-0.99)),
                    np.argmin(np.abs(vlevels-0.9)),np.argmin(np.abs(vlevels-0.75)),
                    np.argmin(np.abs(vlevels-0.5))]
                levels = values[levels]
                plt.contour(xx,yy,f,colors='k',levels=levels)                

    plt.subplots_adjust(left=0.075,right=0.85)

    cnorm = colors.Normalize(vmin=-2,vmax=np.log10(0.32))
    cbaxes = fig.add_axes([0.875,0.2,0.025,0.6])#,frameon=False)
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    cbar = plt.colorbar(cm.ScalarMappable(norm=cnorm,cmap='YlOrRd'),cax=cbaxes)
    cbar.set_label('Mass Fraction',fontsize=15)
    cbar_ticks = cbar.get_ticks()
    cbar_ticks = [np.round(10.0**tick,decimals=3) for tick in cbar_ticks]
    cbar.ax.set_yticklabels(cbar_ticks)
      
    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("[Z/H]",fontsize=15)
    plt.ylabel(r"[$\alpha$/Fe]",fontsize=15)
    plt.text(0.4425,1.02,'1 < |z| < 2 kpc',fontsize=12)
    plt.text(0.42,0.665,'0.5 < |z| < 1 kpc',fontsize=12)
    plt.text(0.4425,0.3075,'0 < |z| < 0.5 kpc',fontsize=12)
            
    plt.draw()
    plt.show()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/figures/alpha_z_regions_v7.pdf')
    plt.close(fig)

def figure_mw_maps():

    import ebf
    xgrid = ebf.read('/Users/nscott/Data/MW_analogues/MUSE/Feh_alpha_age_given_R_z.ebf','/xgrid/') 
    median_Rz = ebf.read('/Users/nscott/Data/MW_analogues/MUSE/Feh_alpha_age_given_R_z.ebf','/median_Rz/')
    median_Rz['fe_h'][0,:] = median_Rz['fe_h'][1,:]
    median_Rz['alpha_fe'][0,:] = median_Rz['alpha_fe'][1,:]
    median_Rz['tau'][0,:] = median_Rz['tau'][1,:]
    
    #xz = np.append(xgrid['z'],4.0)-0.2
    xz = xgrid['z']-0.2
    #xr = np.append(xgrid['r'],20.5)-0.5
    xr = xgrid['r']-0.5
    
    xs = [r for r in xr for k in range(len(xz))]
    ys = list(xz)*len(xr)
    widths = np.ones(len(xs))
    heights = np.ones(len(ys))*0.4
    
    cmap = cm.YlOrRd
    
    zh = median_Rz['fe_h']+np.log10(0.694*10.0**median_Rz['alpha_fe']+0.306)
    
    vals = np.ravel(zh)
    #cnorm = colors.Normalize(np.min(vals),np.max(vals))
    cnorm = colors.Normalize(-0.5,0.2)
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=cmap)
    
    fig,ax = plt.subplots(figsize=(24/2.54,10/2.54))
       
    for x,y,w,h,val in zip(xs,ys,widths,heights,vals):
        rect = patches.Rectangle((x,y),w,h,color=scalarmap.to_rgba(val))
        ax.add_patch(rect)
        
    plt.xlabel('Radius (kpc)',fontsize=15)
    plt.ylabel('Height (kpc)',fontsize=15)
    ax.axes.set_xlim(0,19)
    ax.axes.set_ylim(0,4)
    
    scalarmap.set_array(vals)
    cbar = plt.colorbar(scalarmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='[Z/H]',size=15)
    
    plt.tight_layout()
    
    plt.show()
    plt.draw()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/z_map_mw.pdf')
    plt.clf()
    plt.close('all')
    
    vals = np.ravel(median_Rz['alpha_fe'])
    #cnorm = colors.Normalize(np.min(vals),np.max(vals))
    cnorm = colors.Normalize(0.0,0.25)
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=cmap)
    
    fig,ax = plt.subplots(figsize=(24/2.54,10/2.54))
       
    for x,y,w,h,val in zip(xs,ys,widths,heights,vals):
        rect = patches.Rectangle((x,y),w,h,color=scalarmap.to_rgba(val))
        ax.add_patch(rect)
        
    plt.xlabel('Radius (kpc)',fontsize=15)
    plt.ylabel('Height (kpc)',fontsize=15)
    ax.axes.set_xlim(0,19)
    ax.axes.set_ylim(0,4)
    
    scalarmap.set_array(vals)
    cbar = plt.colorbar(scalarmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=r'[$\alpha$/Fe]',size=15)
    
    plt.tight_layout()
    
    plt.show()
    plt.draw()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/alpha_map_mw.pdf')
    plt.clf()
    plt.close('all')

    vals = np.ravel(median_Rz['tau'])
    #cnorm = colors.Normalize(np.min(vals),np.max(vals))
    cnorm = colors.Normalize(4,12)
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=cmap)
    
    fig,ax = plt.subplots(figsize=(24/2.54,10/2.54))
       
    for x,y,w,h,val in zip(xs,ys,widths,heights,vals):
        rect = patches.Rectangle((x,y),w,h,color=scalarmap.to_rgba(val))
        ax.add_patch(rect)
        
    plt.xlabel('Radius (kpc)',fontsize=15)
    plt.ylabel('Height (kpc)',fontsize=15)
    ax.axes.set_xlim(0,19)
    ax.axes.set_ylim(0,4)
    
    scalarmap.set_array(vals)
    cbar = plt.colorbar(scalarmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='Age (Gyrs)',size=15)
    
    plt.tight_layout()
    
    plt.show()
    plt.draw()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/age_map_mw.pdf')
    plt.clf()
    plt.close('all')

def figure_pops_maps():

    pops_table_path = ('/Users/nscott/Data/MW_analogues/MUSE/'+
        'UGC10738_v3/full_spectral_pops_two_ord2.dat')
    tab = Table.read(pops_table_path,format='ascii')

    xs = [ 0,  0,  0,  0,  3,  3,  3,  3,  5,  5,  5,  5,  7,  7,  7,  7,  9,
        9,  9,  9, 11, 11, 11, 11, 13, 13, 13, 13, 15, 15, 15, 15, 17, 17,
       17, 17]
    ys = [0,0.5,1,2,0,0.5,1,2,0,0.5,1,2,0,0.5,1,2,0,0.5,1,2,0,0.5,1,2,0,0.5,
        1,2,0,0.5,1,2,0,0.5,1,2]
    heights = [0.5, 0.5, 1. , 2. , 0.5, 0.5, 1. , 2. , 0.5, 0.5, 1. , 2. , 0.5,
       0.5, 1. , 2. , 0.5, 0.5, 1. , 2. , 0.5, 0.5, 1. , 2. , 0.5, 0.5,
       1. , 2. , 0.5, 0.5, 1. , 2. , 0.5, 0.5, 1. , 2. ]
    widths = [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    cmap = cm.YlOrRd
    
    ##### Metallicity #####
    

    vals = tab['f_a0']*tab['z_a0']+tab['f_a04']*tab['z_a04']
    #cnorm = colors.Normalize(np.min(vals),np.max(vals))
    cnorm = colors.Normalize(-0.5,0.2)
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=cmap)
    
    fig,ax = plt.subplots(figsize=(24/2.54,10/2.54))
       
    for x,y,w,h,val in zip(xs,ys,widths,heights,vals):
        rect = patches.Rectangle((x,y),w,h,color=scalarmap.to_rgba(val))
        ax.add_patch(rect)
        
    plt.xlabel('Projected radius (kpc)',fontsize=15)
    plt.ylabel('Projected height (kpc)',fontsize=15)
    ax.axes.set_xlim(0,19)
    ax.axes.set_ylim(0,4)
    
    scalarmap.set_array(vals)
    cbar = plt.colorbar(scalarmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='[Z/H]',size=15)
    
    plt.plot([13,19],[2,2],'k--',lw=5)
    plt.plot([13,13],[2,4],'k--',lw=5)
    
    plt.tight_layout()
    
    plt.show()
    plt.draw()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/z_map_v2.pdf')
    plt.clf()
    plt.close('all')
    
        
    ##### Alpha #####

    vals = tab['f_a04']*0.4
    #cnorm = colors.Normalize(np.min(vals),np.max(vals))
    cnorm = colors.Normalize(0.0,0.25)
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=cmap)
    
    fig,ax = plt.subplots(figsize=(24/2.54,10/2.54))
       
    for x,y,w,h,val in zip(xs,ys,widths,heights,vals):
        rect = patches.Rectangle((x,y),w,h,color=scalarmap.to_rgba(val))
        ax.add_patch(rect)
        
    plt.xlabel('Projected radius (kpc)',fontsize=15)
    plt.ylabel('Projected height (kpc)',fontsize=15)
    ax.axes.set_xlim(0,19)
    ax.axes.set_ylim(0,4)
    
    scalarmap.set_array(vals)
    cbar = plt.colorbar(scalarmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label=r'[$\alpha$/Fe]',size=15)
    
    plt.plot([13,19],[2,2],'k--',lw=5)
    plt.plot([13,13],[2,4],'k--',lw=5)
    
    plt.tight_layout()
    
    plt.show()
    plt.draw()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/alpha_map_v2.pdf')
    plt.clf()
    plt.close('all')

    ##### Age #####
    
    vals = 10.0**(tab['f_a0']*tab['age_a0']+tab['f_a04']*tab['age_a04'])/1e9
    #cnorm = colors.Normalize(np.min(vals),np.max(vals))
    cnorm = colors.Normalize(4,12)
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=cmap)
    
    fig,ax = plt.subplots(figsize=(24/2.54,10/2.54))
       
    for x,y,w,h,val in zip(xs,ys,widths,heights,vals):
        rect = patches.Rectangle((x,y),w,h,color=scalarmap.to_rgba(val))
        ax.add_patch(rect)
        
    plt.xlabel('Projected radius (kpc)',fontsize=15)
    plt.ylabel('Projected height (kpc)',fontsize=15)
    ax.axes.set_xlim(0,19)
    ax.axes.set_ylim(0,4)
    
    scalarmap.set_array(vals)
    cbar = plt.colorbar(scalarmap)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='Age (Gyrs)',size=15)

    plt.plot([13,19],[2,2],'k--',lw=5)
    plt.plot([13,13],[2,4],'k--',lw=5)

    plt.tight_layout()
    
    plt.show()
    plt.draw()
    plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/age_map_v2.pdf')
    plt.clf()
    plt.close('all')
    
def colour_image(pr_image=False):

    from mpdaf.obj import Cube,Image
    from mpdaf.drs import PixTable

    infile = '/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/' + \
        'UGC10738_DATACUBE_FINAL_WCS_Pall_mad_crop.fits'
    cube = Cube(infile)
 
    conti = 0
    if (conti==True):
          
        imWhite = cube.sum(axis=0)
        imWhite.write(data_dir+infile.replace("DATACUBE_FINAL", "IMAGE_FOV_white"))      

        #plt.figure()
        #imWhite.plot()
      
    colimg = 1
    if (colimg==True):
      
        if not os.path.exists('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_g.fits'):
      
            imV = cube.get_band_image('SDSS_g')
            imR = cube.get_band_image('SDSS_r')
            imI = cube.get_band_image('SDSS_i')
            imV.write('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_g.fits')      
            imR.write('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_r.fits')      
            imI.write('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_i.fits')      
        else:
            imV = Image('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_g.fits')
            imR = Image('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_r.fits')
            imI = Image('/Users/nscott/Data/MW_analogues/MUSE/UGC10738_v3/UGC10738_IMAGE_FOV_SDSS_i.fits')
        if pr_image:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 20))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        imV.plot(ax=ax1, title="SDSS_g")
        imR.plot(ax=ax2, title="SDSS_r")
        imI.plot(ax=ax3, title="SDSS_i")

        #img = np.zeros([imV.shape[0], imV.shape[1], 3], dtype=float)      
        #img[:,:,0] = img_scale.asinh(imI.data, scale_min=0)
        #img[:,:,1] = img_scale.asinh(imR.data, scale_min=0)
        #img[:,:,2] = img_scale.asinh(imV.data, scale_min=0)

        #code.interact(local=dict(globals(),**locals()))

        img = make_lupton_rgb(imI.data, imR.data, imV.data,stretch=8,minimum=[0.5,0.5,0.5],Q=10)
        s = np.shape(img)

        xc,yc = 458,159
        ext = [(-1*xc)*0.2,(s[1]-xc)*0.2,(-1*yc)*0.2,(s[0]-yc)*0.2]
        plt.clf()
        ax = plt.imshow(img, aspect='equal',origin='lower',extent=ext,interpolation='nearest')
        if not pr_image:
            plt.xlabel('X (arcsec)')
            plt.ylabel('Y (arcsec)')
            plt.plot([60,60+(5/0.459)],[-25,-25],'-w',lw=2)
            plt.plot([60,60],[-24.25,-25.75],'-w',lw=2)
            plt.plot([60+(5/0.459),60+(5/0.459)],[-24.25,-25.75],'-w',lw=2)
            plt.annotate('5 kpc', (60+2.5/0.459,-25),color='w',ha='center',va='bottom')
        else:
            plt.axis('off')
            plt.tight_layout()
        
        #py.title('Blue = V, Green = R, Red = I')
        if pr_image:
            plt.savefig('/Users/nscott/Documents/My Papers/Alpha-enhanced disks in MWAs letter/UGC10738_image3.png',dpi=1200,bbox_inches='tight',pad_inches=0)
            #plt.imsave('/Users/nscott/Documents/My Papers/Alpha-enhanced disks in MWAs letter/UGC10738_image4.png',img,dpi=1200)
        else:
            plt.savefig('/Users/nscott//Dropbox (Sydney Uni)/Apps/Overleaf/'+
        'Alpha-Enhanced Disks in MW analogues/figures/gri_image2.pdf')