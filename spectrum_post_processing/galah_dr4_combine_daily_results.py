#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Preamble 
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

from astropy.table import Table,join,hstack,vstack
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import corner
import time


# In[ ]:


dates = glob.glob('daily/*.fits')
dates = [date[-11:-5] for date in dates]
dates.sort()

print(dates)


# In[ ]:


def combine_dates(dates):
    
    data = Table.read('daily/galah_dr4_allspec_not_validated_'+str(dates[0])+'.fits')
    
    for date in dates[1:]:
        data_next = Table.read('daily/galah_dr4_allspec_not_validated_'+str(date)+'.fits')
        data = vstack([data, data_next])
    data['tmass_id'] = np.array(data['tmass_id'],dtype=str)

    return(data)

data = combine_dates(dates)


# In[ ]:


data.write('galah_dr4_allspec_not_validated.fits',overwrite=True)


# In[ ]:


data


# In[ ]:


# Plot HRD & abundances 

flag0 = data['flag_sp'] == 0
flag1 = data['flag_sp'] > 0

if len(data) < 2000:
    plt.figure()
    plt.scatter(
        data['teff'][flag0],
        data['logg'][flag0],
        s=1
    )
    plt.scatter(
        data['teff'][flag1],
        data['logg'][flag1],
        marker='v',
        s=0.5
    )
    plt.xlabel('Teff / K')
    plt.ylabel('log (g / cm s-2)')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
else:
    plt.figure()
    plt.hist2d(
        data['teff'][flag0],
        data['logg'][flag0],
        bins=100,cmin=1,
        norm=LogNorm()
    )
#     plt.scatter(
#         data['teff'][flag1],
#         data['logg'][flag1],
#         marker='v',
#         s=0.5
#     )
    plt.xlabel('Teff / K')
    plt.ylabel('log (g / cm s-2)')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()

for label in [
    'Li',
    'C',
    'N',
    'O',
    'Na',
    'Mg',
    'Al',
    'Si',
    'K',
    'Ca',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Rb',
    'Sr',
    'Y',
    'Zr',
    'Mo',
    'Ru',
    'Ba',
    'La',
    'Ce',
    'Nd',
    'Sm',
    'Eu'
    ]:
    
    flag0 = (data['flag_'+label.lower()+'_fe'] == 0) #& (data['fe_h'] > -1)
    flag1 = (data['flag_'+label.lower()+'_fe'] == 1) #& (data['fe_h'] > -1)
    
    if len(data) < 2000:
        plt.figure()
        plt.scatter(
            data['fe_h'][flag0],
            data[label.lower()+'_fe'][flag0],
            s=1,label='Detection'
        )
        plt.scatter(
            data['fe_h'][flag1],
            data[label.lower()+'_fe'][flag1],
            marker='v',label='Upper limit',
            s=0.5
        )
        plt.legend()
        plt.xlabel('[Fe/H]')
        plt.ylabel('['+label+'/Fe]')
        plt.savefig('overview_'+label+'.png',dpi=200,bbox_inches='tight')
        plt.show()
        plt.close()
    
    else:
        f, gs = plt.subplots(1,2,figsize=(8,3),sharex=True,sharey=True)
        ax = gs[0]
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('['+label+'/Fe]')
        corner.hist2d(
            data['fe_h'][flag0],
            data[label.lower()+'_fe'][flag0],
            ax = ax,bins=50
        )
        ax = gs[1]
        ax.set_xlabel('[Fe/H]')
        ax.set_ylabel('['+label+'/Fe]')
        corner.hist2d(
            data['fe_h'][flag1],
            data[label.lower()+'_fe'][flag1],
            ax = ax,bins=50,
            range = [(-3,1),(-1,2)]
        )
        ax.text
        plt.savefig('figures/overview_'+label+'.png',dpi=200,bbox_inches='tight')
        plt.show()
        plt.close()


# In[ ]:


f, gs = plt.subplots(1,4,sharex=True,sharey=True,figsize=(2.5*4,2.5))

x_low  = -2.50
x_high =  0.75
y_low  = -1.00
y_high =  1.75

panels = ['a)','b)','c)','d)']

for i,label in enumerate(['C','N','CN','O']):
    ax = gs[i]

    if label in ['C','N','O']:
        flag0 = (data['flag_'+label.lower()+'_fe'] == 0) #& (data['fe_h'] > -1)
        flag1 = (data['flag_'+label.lower()+'_fe'] == 1) #& (data['fe_h'] > -1)
        corner.hist2d(
            data['fe_h'][flag0],
            data[label.lower()+'_fe'][flag0],
            ax = ax,bins=(np.linspace(x_low,x_high,50),np.linspace(y_low,y_high,50))
        )
#         ax.hist2d(
#             data['fe_h'][flag0],
#             data[label.lower()+'_fe'][flag0],
#             cmin = 1, bins=50
#             #s=1,label='Detection'
#         )
#         ax.scatter(
#             data['fe_h'][flag1],
#             data[label.lower()+'_fe'][flag1],
#             marker='v',label='Upper limit',
#             s=0.5
#         )
        
    if label == 'CN':
        flag0 = (data['flag_c_fe'] == 0) & (data['flag_n_fe'] == 0) #& (data['fe_h'] > -1)
        flag1 = ((data['flag_c_fe'] == 1) | (data['flag_n_fe'] == 1)) #& (data['fe_h'] > -1)
        corner.hist2d(
            data['fe_h'][flag0],
            data['c_fe'][flag0]-data['n_fe'][flag0],
            ax = ax,bins=(np.linspace(x_low,x_high,50),np.linspace(y_low,y_high,50))
        )
        #         ax.hist2d(
#             data['fe_h'][flag0],
#             data['c_fe'][flag0]-data['n_fe'][flag0],
#             cmin = 1, bins=50
# #             s=1,label='Detection'
#         )
#         ax.scatter(
#             data['fe_h'][flag1],
#             data['c_fe'][flag1]-data['n_fe'][flag1],
#             marker='v',label='Upper limit',
#             s=0.5
#         )
    
    ax.set_xlim(x_low,x_high)
    ax.set_ylim(y_low,y_high)
    
#     if i==0:
#         ax.legend()
    ax.set_xlabel('[Fe/H]')
    if label in ['C','N','O']:
        ax.set_ylabel('['+label+'/Fe]')
    if label == 'CN':
        ax.set_ylabel('[C/N]')
plt.tight_layout()
plt.savefig('figures/overview_CNO.png',dpi=200,bbox_inches='tight')
plt.show()
plt.close()


# In[ ]:


# [X/Fe] = [X/H] - [M/H]
# [X/H] + (A_X - 12) = log(N_X/N_H) 
# [X/Fe] = log(N_X / N_H) - log(N_X / N_H)_Sun - [Fe/H]
# [C + N / Fe] = log((N_C + N_N) / N_H) - log((N_C_Sun + (N_N_Sun / N_H)_Sun - [Fe/H]
# [C + N / Fe] = log(N_C/N_H + N_N/N_H) - log(N_C_Sun/H_Sun + N_N_Sun/N_H_Sun) - [Fe/H]

A_N_Sun = 7.78+0.15
A_C_Sun = 8.39+0.037

A_C = data['c_fe'] + data['fe_h'] + (A_C_Sun)
A_N = data['n_fe'] + data['fe_h'] + (A_N_Sun)

N_C_N_H = 10**(A_C + 12)
N_N_N_H = 10**(A_N + 12)
N_C_N_H_sun = 10**(A_C_Sun + 12)
N_N_N_H_sun = 10**(A_N_Sun + 12)

data['cn_fe'] = np.log10(N_C_N_H + N_N_N_H) - np.log10(N_C_N_H_sun + N_N_N_H_sun) - data['fe_h']


# In[ ]:


def cn_masses(fe_h, c_fe, n_fe, cn_fe):
    return(
        1.08 - 0.18 * fe_h + 4.30 * c_fe +1.43 * n_fe - 7.55 * cn_fe
        - 1.05 * (fe_h)**2 - 1.12 * (fe_h * c_fe) - 0.67 * (fe_h * n_fe) - 1.30 * (fe_h * cn_fe)
        - 49.92 * (c_fe)**2 - 41.04 * (c_fe * n_fe) + 139.92 * (c_fe * cn_fe)
        - 0.63 * (n_fe)**2 + 47.33 * (n_fe * cn_fe)
        - 86.62 * (cn_fe)**2
    )
data['mass'] = cn_masses(fe_h=data['fe_h'], c_fe=data['c_fe'], n_fe=data['n_fe'], cn_fe=(data['cn_fe']-0.1)/2.)


# In[ ]:


def cn_ages(fe_h, c_fe, n_fe, cn_fe, teff, logg):
    return(
        -54.35 + 6.53*fe_h -19.02 *c_fe -12.18*n_fe +37.22*cn_fe +59.58*teff +16.14*logg
        +0.74*fe_h*fe_h +4.04*fe_h*c_fe +0.76*fe_h*n_fe -4.94*fe_h*cn_fe -1.46*fe_h*teff -1.56*fe_h*logg
        +26.90*c_fe*c_fe +13.33*c_fe*n_fe -77.84*c_fe*cn_fe +48.29*c_fe*teff -13.12*c_fe*logg
        -1.04*n_fe*n_fe -17.60*n_fe*cn_fe +13.99*n_fe*teff -1.77*n_fe*logg
        +51.24*cn_fe*cn_fe -65.67*cn_fe*teff +14.24*cn_fe*logg
        +15.54*teff*teff -34.68*teff*logg
        +4.17*logg*logg
    )
data['age'] = 10**(cn_ages(fe_h=data['fe_h'], c_fe=data['c_fe'], n_fe=data['n_fe'], cn_fe=data['cn_fe'], teff=data['teff']/4000., logg=data['logg']))


# In[ ]:


def hist2d_bin_colored(X,Y,Z,X_label='X\_label',Y_label='Y\_label',Z_label='Z\_label',bins=30,bin_function='median',ax=None,cmap='seismic_r',minimum_bin_entries = 5,**kwargs):
    """
    INPUT:
    X : x-axis parameter
    Y : y-axis parameter
    Z : parameter that will be used for coloring the bins
    X/Y/Z_label : label names
    bins = 30, but you can also give it bins = (np.linspace(x_min,x_max,30),np.linspace(y_min,y_max,30))
    bin_function : median/average/sum
    ax : if you plot it as part of an f,ax = plt.subplots()
    minimum_bin_entries : how many entries do we expect, before we even consider throwing some fancy function at them
    
    OUTPUT:
    plt.imshow
    """
    
    # First make sure we only work with finite values
    finite = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    if len(X[finite])!=len(X):
        print('Not all values were finite! Continuing with only finite ones')
    X=X[finite];Y=Y[finite];Z=Z[finite]
    
    # Now create the matrix of bins and its bin-edges
    H,xedges,yedges = np.histogram2d(X,Y,bins=bins)

    # Create the matrix that we want to store color-values in
    color_matrix = np.zeros_like(H)
    color_matrix[:] = np.nan
    
    # Loop through the x- and y-bins
    for x_bin in range(len(xedges)-1):
        for y_bin in range(len(yedges)-1):
            in_xy_bin = (X>=xedges[x_bin])&(X<xedges[x_bin+1])&(Y>=yedges[y_bin])&(Y<yedges[y_bin+1])
            
            # We only add a value if there are more than *minimum_bin_entries* in the bin
            if len(Z[in_xy_bin]) >= minimum_bin_entries:
                if bin_function=='median':
                    color_matrix[x_bin,y_bin]=np.median(Z[in_xy_bin])
                elif bin_function=='average':
                    color_matrix[x_bin,y_bin]=np.average(Z[in_xy_bin])
                elif bin_function=='sum':
                    color_matrix[x_bin,y_bin]=np.sum(Z[in_xy_bin])
                elif bin_function=='std':
                    color_matrix[x_bin,y_bin]=np.std(Z[in_xy_bin])
                else:
                    raise NameError('Only bin_function = median/average/sum available')

    # Create an axis if not given
    if ax==None:
        ax = plt.gca()
    else:
        ax=ax
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)

    # Populate the keyword arguments for the imshow
    imshow_kwargs = dict(
        cmap = cmap,aspect='auto',origin='lower'
    )
    # Update by any arguments given through **kwargs
    imshow_kwargs.update(kwargs)

    # Plot!
    s = ax.imshow(color_matrix.T,extent=(xedges[0],xedges[-1],yedges[0],yedges[-1]),**imshow_kwargs)
    c = plt.colorbar(s, ax=ax)
    c.set_label(Z_label)


# In[ ]:


giants = (data['logg'] < 3.5) & (data['flag_c_fe'] == 0) & (data['flag_n_fe'] == 0)

f, gs = plt.subplots(2,2,figsize=(7,5))
hist2d_bin_colored(
    data['fe_h'][giants],
    data['mg_fe'][giants],
    data['mass'][giants],
    X_label='[Fe/H]',
    Y_label='[Mg/Fe]',
    Z_label=r'Mass / $\mathrm{M}_\odot$',
    ax = gs[0,0],
    bins = (np.linspace(-0.8,0.3,30),np.linspace(-0.1,0.4,30)),
    vmin = 0.8, vmax = 2.0,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
hist2d_bin_colored(
    data['fe_h'][giants],
    data['c_fe'][giants]-data['n_fe'][giants],
    data['mass'][giants],
    X_label='[Fe/H]',
    Y_label='[C/N]',
    Z_label=r'Mass / $\mathrm{M}_\odot$',
    ax = gs[0,1],
    bins = (np.linspace(-0.8,0.3,30),np.linspace(-0.8,0.4,30)),
    vmin = 0.8, vmax = 2.0,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
hist2d_bin_colored(
    data['teff'][giants],
    data['logg'][giants],
    data['mass'][giants],
    X_label='Teff / K',
    Y_label='logg',
    Z_label=r'Mass / $\mathrm{M}_\odot$',
    ax = gs[1,0],
    bins = (np.linspace(3000,5500,30),np.linspace(0,4,30)),
    vmin = 0.8, vmax = 2.0,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
gs[1,0].set_xlim(5500,3750)
gs[1,0].set_ylim(4,0)
hist2d_bin_colored(
    data['c_fe'][giants]-data['n_fe'][giants],
    data['mass'][giants],
    data['fe_h'][giants],
    X_label='[C/N]',
    Y_label=r'Mass / $\mathrm{M}_\odot$',
    Z_label='[Fe/H]',
    ax = gs[1,1],
    bins = (np.linspace(-0.8,0.4,30),np.linspace(0.5,3.0,30)),
    vmin = -0.8, vmax = 0.3,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
plt.tight_layout()


# In[ ]:


giants = (data['logg'] > 1.8) & (data['logg'] < 3.3) & (data['flag_c_fe'] == 0) & (data['flag_n_fe'] == 0)

f, gs = plt.subplots(2,2,figsize=(7,5))
hist2d_bin_colored(
    data['fe_h'][giants],
    data['mg_fe'][giants],
    data['age'][giants],
    X_label='[Fe/H]',
    Y_label='[Mg/Fe]',
    Z_label=r'Age / Gyr',
    ax = gs[0,0],
    bins = (np.linspace(-0.8,0.3,30),np.linspace(-0.1,0.4,30)),
    vmin = 0, vmax = 14,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
hist2d_bin_colored(
    data['fe_h'][giants],
    data['c_fe'][giants]-data['n_fe'][giants],
    data['age'][giants],
    X_label='[Fe/H]',
    Y_label='[C/N]',
    Z_label=r'Age / Gyr',
    ax = gs[0,1],
    bins = (np.linspace(-0.8,0.3,30),np.linspace(-0.8,0.4,30)),
    vmin = 0, vmax = 14,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
hist2d_bin_colored(
    data['teff'][giants],
    data['logg'][giants],
    data['age'][giants],
    X_label='Teff / K',
    Y_label='logg',
    Z_label=r'Age / Gyr',
    ax = gs[1,0],
    bins = (np.linspace(3000,5500,30),np.linspace(0,4,30)),
    vmin = 0, vmax = 14,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
gs[1,0].set_xlim(5500,3750)
gs[1,0].set_ylim(4,0)
hist2d_bin_colored(
    data['c_fe'][giants]-data['n_fe'][giants],
    data['age'][giants],
    data['fe_h'][giants],
    X_label='[C/N]',
    Y_label=r'Age / Gyr',
    Z_label='[Fe/H]',
    ax = gs[1,1],
    bins = (np.linspace(-0.8,0.4,30),np.linspace(0,14,30)),
    vmin = -0.8, vmax = 0.3,
    cmap = 'RdYlBu_r',
    minimum_bin_entries = 1
)
plt.tight_layout()


# In[ ]:




