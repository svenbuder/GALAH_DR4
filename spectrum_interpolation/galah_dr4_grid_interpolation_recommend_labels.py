#!/usr/bin/env python
# coding: utf-8

# # galah_dr4_grid_interpolation_recommend_labels

# In[1]:


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

import numpy as np
import os
import pickle
from astropy.table import Table
from scipy.io import readsav
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time


# In[2]:


# Read in all available grids
grids = Table.read('../spectrum_grids/galah_dr4_model_trainingset_gridpoints.fits')


# In[4]:


# choose one grid_index
try:
    grid_index = int(sys.argv[1])
    print('Using Grid index ',grid_index)
except:
    # Done:
    """
    1259,
    1832,
    1931,1932
    2013,2014,2015
    
    currently: 1930
    
    next: 
    
    1834,
    1844,1845,1846,
    1918,1919,1920,
    2001,2002,2003,
    
    to be calculated
    1933
    
    """
    grid_index = 1931 # 5750_4.50_0.00
    #grid_index = 1259 # 4250_1.50_-0.50
    #grid_index = 1919 # 5750_4.00_0.00
    
    print('Using default grid index ',grid_index)

try:
    grids = Table.read('../spectrum_grids/galah_dr4_model_trainingset_gridpoints.fits')
    teff_logg_feh_name = str(int(grids['teff_subgrid'][grid_index]))+'_'+"{:.2f}".format(grids['logg_subgrid'][grid_index])+'_'+"{:.2f}".format(grids['fe_h_subgrid'][grid_index])
    gradient_spectra_up = Table.read('gradient_spectra/'+teff_logg_feh_name+'/'+teff_logg_feh_name+'_gradient_spectra_up.fits')
    gradient_spectra_down = Table.read('gradient_spectra/'+teff_logg_feh_name+'/'+teff_logg_feh_name+'_gradient_spectra_down.fits')

        # We will definitely always fit the stellar parameters
    labels = ['teff','logg','fe_h','vmic','vsini']

    # Now let's loop over all the possible elements to figure out, 
    # which ones we can actually fit for this Teff/logg/[Fe/H] range
    # For the others, we simply assume the GALAH+ DR3 or initial value
    possible_elements = [
        'Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Co','Ni','Cu','Zn',
        'Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu'
    ]

    spectrum_flux_change_threshold_max = 0.07
    spectrum_flux_change_threshold_min = 0.005
    percentage_thresshold = 50

    print(
        'Element,',
        'dFlux,',
        '> '+str(spectrum_flux_change_threshold_max)+'?',
        '(dFlux > '+str(spectrum_flux_change_threshold_min)+')/%',
        '> '+str(percentage_thresshold)+'%?'
    )

    # For the test, we will only rely on the main GALAH wavelength range from deSilva et al. (2015)
    # This will avoid for example the strong Na doublet that is almost always outside of our reach
    in_main_galah_wavelength_range = (
        ((gradient_spectra_up['wave'] > 4713) & (gradient_spectra_up['wave'] < 4903)) |
        ((gradient_spectra_up['wave'] > 5648) & (gradient_spectra_up['wave'] < 5873)) |
        ((gradient_spectra_up['wave'] > 6478) & (gradient_spectra_up['wave'] < 6737)) |
        ((gradient_spectra_up['wave'] > 7585) & (gradient_spectra_up['wave'] < 7887))
    )

    for label in possible_elements:

        # We will apply 2 tests
        # Test 1: Does the spectrum at any pixel change more than a certain threshold *spectrum_flux_change_threshold_max*

        maximum_flux_change = np.round(np.max(np.abs(gradient_spectra_up[label.lower()+'_fe'][in_main_galah_wavelength_range])),decimals=3)
        test1 = spectrum_flux_change_threshold_max <= maximum_flux_change

        # Test 2: At how many pixels does the spectrum actually change more than the minimum threshold *spectrum_flux_change_threshold_min*
        percentage_above_threshold = np.round(100*len(np.where(np.max([np.abs(gradient_spectra_up[label.lower()+'_fe'][in_main_galah_wavelength_range]),np.abs(gradient_spectra_down[label.lower()+'_fe'][in_main_galah_wavelength_range])],axis=0) >= spectrum_flux_change_threshold_min)[0])/len(gradient_spectra_up['wave'][in_main_galah_wavelength_range]))
        test2 = percentage_above_threshold >= percentage_thresshold

        print(
            (test1 | test2),
            label,
            maximum_flux_change,
            test1,
            percentage_above_threshold,
            test2
        )
        if (test1 | test2):
            labels.append(label.lower()+'_fe')
    np.savetxt('gradient_spectra/'+teff_logg_feh_name+'/recommended_fit_labels_'+teff_logg_feh_name+'.txt',np.array(labels),fmt='%s')
    print('Recommended labels:')
    print(labels)
    
except:
    print('Could not recommend labels')

