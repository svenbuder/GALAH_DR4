#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Compatibility with Python 3
from __future__ import (absolute_import, division, print_function)

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

# Basic Tools
import numpy as np
from astropy.table import Table,vstack
import pickle
import time
import matplotlib.pyplot as plt
import scipy.optimize as op
import sys
import time
from sklearn.model_selection import train_test_split

# The Payne, see https://github.com/tingyuansen/The_Payne for more details
from The_Payne import training
from The_Payne import utils
from The_Payne import spectral_model

# # That's how we would do it:
# # training_labels, training_spectra, validation_labels, validation_spectra = utils.load_training_data()
# """
# Changes that need to be made to training.py in The_Payne if no CUDA is available

# if torch.cuda.is_available():
#     dtype = torch.cuda.FloatTensor
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     dtype = torch.FloatTensor
#     torch.set_default_tensor_type('torch.FloatTensor')
    
# if torch.cuda.is_available():
#     model.cuda()
        
# if torch.cuda.is_available():
#     perm = perm.cuda()
# """
# # That's how we do it:


# In[ ]:


# INITIALISATION
try:
    grid_index = int(sys.argv[1])
    print('Using Grid index ',grid_index)
    
    try:
        number_grid_points = int(sys.argv[2])
    except:
        number_grid_points = 3*3*3
    print('Using '+str(number_grid_points)+' grid points')

except:
    grid_index = 2002
    print('Using default grid index ',grid_index)

    # 3 dimensional points == 27
    number_grid_points = 3*3*3

    # middle point + surrounding +- ones == 7
    # number_grid_points = 1+2+2+2
    
    print('Using '+str(number_grid_points)+' grid points')
    
if number_grid_points not in [7,27]:
    raise ValueError('number_grid_points not valid (needs to be 7 or 27)')
    
grids = Table.read('../../spectrum_grids/galah_dr4_model_trainingset_gridpoints.fits')
teff_logg_feh_name = str(int(grids['teff_subgrid'][grid_index]))+'_'+"{:.2f}".format(grids['logg_subgrid'][grid_index])+'_'+"{:.2f}".format(grids['fe_h_subgrid'][grid_index])


# In[ ]:


# Find upper and lower Teff points
teff_middle = int(grids['teff_subgrid'][grid_index])
if teff_middle <= 4000:
    teff_lower = teff_middle - 100
else:
    teff_lower = teff_middle - 250
if teff_middle <= 3900:
    teff_higher = teff_middle + 100
else:
    teff_higher = teff_middle + 250

# Find upper and lower logg points
logg_middle = grids['logg_subgrid'][grid_index]
logg_lower = logg_middle - 0.5
logg_higher = logg_middle + 0.5

# Find upper and lower fe_h points
fe_h_middle = grids['fe_h_subgrid'][grid_index]
if fe_h_middle <= -0.75:
    fe_h_lower = fe_h_middle - 0.5
else:
    fe_h_lower = fe_h_middle - 0.25
if fe_h_middle <= -1.5:
    fe_h_higher = fe_h_middle + 0.5
else:
    fe_h_higher = fe_h_middle + 0.25


# In[ ]:


print(teff_middle, teff_lower, teff_higher)
print(logg_middle, logg_lower, logg_higher)
print(fe_h_middle, fe_h_lower, fe_h_higher)


# In[ ]:


def find_3x3x3_indices(grid_index):
    
    grid_indices_3x3x3 = []

    for teff in [teff_middle,teff_lower,teff_higher]:
        for logg in [logg_middle,logg_lower,logg_higher]:
            for fe_h in [fe_h_middle,fe_h_lower,fe_h_higher]:
                grid_indices_3x3x3.append(str(int(teff))+'_'+"{:.2f}".format(logg)+'_'+"{:.2f}".format(fe_h))
                
    return(grid_indices_3x3x3)


# In[ ]:


def find_1_2_2_2_indices(grid_index):
    
    grid_indices_1_2_2_2 = []

    grid_indices_1_2_2_2.append(str(int(teff_middle))+'_'+"{:.2f}".format(logg_middle)+'_'+"{:.2f}".format(fe_h_middle))
    grid_indices_1_2_2_2.append(str(int(teff_middle))+'_'+"{:.2f}".format(logg_middle)+'_'+"{:.2f}".format(fe_h_lower))
    grid_indices_1_2_2_2.append(str(int(teff_middle))+'_'+"{:.2f}".format(logg_middle)+'_'+"{:.2f}".format(fe_h_higher))
    grid_indices_1_2_2_2.append(str(int(teff_middle))+'_'+"{:.2f}".format(logg_lower)+'_'+"{:.2f}".format(fe_h_middle))
    grid_indices_1_2_2_2.append(str(int(teff_middle))+'_'+"{:.2f}".format(logg_higher)+'_'+"{:.2f}".format(fe_h_middle))
    grid_indices_1_2_2_2.append(str(int(teff_lower))+'_'+"{:.2f}".format(logg_middle)+'_'+"{:.2f}".format(fe_h_middle))
    grid_indices_1_2_2_2.append(str(int(teff_higher))+'_'+"{:.2f}".format(logg_middle)+'_'+"{:.2f}".format(fe_h_middle))

    return(grid_indices_1_2_2_2)


# In[ ]:


# Call the appropriate function to find the subsets
if number_grid_points == 27:
    subset_names = find_3x3x3_indices(grid_index)
elif number_grid_points == 7:
    subset_names = find_1_2_2_2_indices(grid_index)


# In[ ]:


available = []
not_available = []
for subset_index, subset_name in enumerate(subset_names):
    try:
        training_set = Table.read('../training_input/'+subset_name+'/galah_dr4_trainingset_'+subset_name+'_incl_vsini.fits')
        print(subset_name)
        available.append(subset_name)
    except:
        print(subset_name+' n/a')
        # available.append(teff_logg_feh_name)
subset_names = available


# In[ ]:


training_labels = []
training_set_flux = []

wavelength_file = '../training_input/galah_dr4_3dbin_wavelength_array.pickle'
wavelength_file_opener = open(wavelength_file,'rb')
wavelength_array = pickle.load(wavelength_file_opener)
wavelength_file_opener.close()

for subset_index, subset_name in enumerate(subset_names):
    
    training_labels_subset_index = Table.read('../training_input/'+subset_name+'/galah_dr4_trainingset_'+subset_name+'_incl_vsini.fits')
    flux_ivar_file = '../training_input/'+subset_name+'/galah_dr4_trainingset_'+subset_name+'_incl_vsini_flux_ivar.pickle'
    flux_ivar_file_opener = open(flux_ivar_file,'rb')

    if subset_index == 0:
        labels = tuple(training_labels_subset_index.keys()[2:-1])
        
        print(subset_index,subset_name,'Using '+str(len(training_labels_subset_index['fe_h']))+' of '+str(len(training_labels_subset_index['fe_h']))+' spectra')
        training_labels.append(np.array([training_labels_subset_index[label] for label in labels]).T)
        training_set_flux.append(pickle.load(flux_ivar_file_opener))

    elif number_grid_points == 27:
        # If we go with the 3x3x3 version, we will only use those with teff_lower <= teff <= teff_higher

        # If we use the 3x3x3 subsets, but some subset was not available,
        # we have to make sure to only include the stars that were not sampled.
        within_teff_logg_fe_h_limits = (
            (training_labels_subset_index['teff'] > teff_lower) &
            (training_labels_subset_index['teff'] < teff_higher) &
            (training_labels_subset_index['logg'] > logg_lower) &
            (training_labels_subset_index['logg'] < logg_higher) &
            (training_labels_subset_index['fe_h'] > fe_h_lower) &
            (training_labels_subset_index['fe_h'] < fe_h_higher)
        )

        print(subset_index,subset_name,'Using '+str(len(training_labels_subset_index['fe_h'][within_teff_logg_fe_h_limits]))+' of '+str(len(training_labels_subset_index['fe_h']))+' spectra')
        training_labels.append(np.array([training_labels_subset_index[label][within_teff_logg_fe_h_limits] for label in labels]).T)
        training_set_flux.append(pickle.load(flux_ivar_file_opener)[within_teff_logg_fe_h_limits])

    else:
        print(subset_index,subset_name,'Using '+str(len(training_labels_subset_index['fe_h']))+' of '+str(len(training_labels_subset_index['fe_h']))+' spectra')
        training_labels.append(np.array([training_labels_subset_index[label] for label in labels]).T)
        training_set_flux.append(pickle.load(flux_ivar_file_opener))
    
    flux_ivar_file_opener.close()

training_labels = np.concatenate((training_labels))
training_set_flux = np.concatenate((training_set_flux))


# In[ ]:


print('Shapes of training set labels and flxues:')
print(np.shape(training_labels),np.shape(training_set_flux))


# In[ ]:


labels = tuple(training_set.keys()[2:-1])

print('Labels to be fitted: ',len(labels))
print(labels)


# In[ ]:


# Call the appropriate function to find the subsets
if number_grid_points == 27:
    model_file = 'galah_dr4_neutral_network_3x3x3_'+teff_logg_feh_name+'_'+str(len(labels))+'labels'
elif number_grid_points == 7:
    model_file = 'galah_dr4_neutral_network_1plus6_'+teff_logg_feh_name+'_'+str(len(labels))+'labels'
else:
    raise ValueError('number_grid_points not valid')
    
print('Will create neural network to be stored at ')
print('models/'+model_file+'.model')


# In[ ]:


if number_grid_points == 27:
    print('Randomly sampling to get 90% training and 10% test sample')
    train, test = train_test_split(np.arange(np.shape(training_set_flux)[0]), test_size=0.10, random_state=int(teff_middle)+int(10*logg_middle)+int(100*fe_h_middle))

elif number_grid_points == 7:
    # We will split the training set into 252 x N spectra to train (90%) and 28 x N spectra to test (10%).
    # To have a representative test set, we use the last 14 xN spectra of the narrowly and broadly random samples

    train_indices = []
    test_indices = []

    # loop through the size-280 sets
    for index in range(int(np.shape(training_labels)[0] / 280)):
        # we train the 72 spectra testing the boundaries and the next 104-14 narrowly sampled values
        train_indices.append(np.arange(280*index,280*(index+1)-104-14))
        # we test with the last 14 entries of the narrowly sampled ones
        test_indices.append(np.arange(280*(index+1)-104-14,280*(index+1)-104))
        # we train on the next broadly sampled ones up until the last 14 entries
        train_indices.append(np.arange(280*(index+1)-104,280*(index+1)-14))
        # we test with the last 14 entries of the broadly sampled ones
        test_indices.append(np.arange(280*(index+1)-14,280*(index+1)))

    train = np.concatenate((train_indices))
    test = np.concatenate((test_indices))

print('Training with '+str(len(train))+' spectra ('+str(int(100*len(train)/(len(train)+len(test))))+'%)')
print('Testing with '+str(len(test))+' spectra ('+str(int(100*len(test)/(len(train)+len(test))))+'%)')


# In[ ]:


# Plot distribution of training set, if interactive
if sys.argv[1] == '-f':
    plt.show()
    
    for each in range(36):
        f, gs = plt.subplots(1,3,figsize=(10,3))
        ax = gs[0]
        ax.scatter(
            training_labels[train,0],
            training_labels[train,each],
            s=1
        )
        ax.set_xlabel(labels[0])
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylabel(labels[each])
        if each == 1:
            ax.set_ylim(ax.get_ylim()[::-1])

        ax = gs[1]
        ax.scatter(
            training_labels[train,1],
            training_labels[train,each],
            s=1
        )
        ax.set_xlabel(labels[1])
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylabel(labels[each])
        if each == 1:
            ax.set_ylim(ax.get_ylim()[::-1])

        ax = gs[2]
        ax.scatter(
            training_labels[train,2],
            training_labels[train,each],
            s=1
        )
        ax.set_xlabel(labels[2])
        ax.set_ylabel(labels[each])
        if each == 1:
            ax.set_ylim(ax.get_ylim()[::-1])

        plt.tight_layout()
        plt.show()
        plt.close()


# In[ ]:


training.neural_net(
    training_labels = training_labels[train,:], 
    training_spectra = training_set_flux[train,:],
    validation_labels = training_labels[test,:], 
    validation_spectra = training_set_flux[test,:],
    num_neurons=300,
    learning_rate=1e-4,
    num_steps=1e4,
    batch_size=128,
    num_pixel=np.shape(training_set_flux[0])[0],
    training_loss_name = 'loss_functions/'+model_file+'_loss.npz',
    payne_model_name = 'models/'+model_file+'.npz'
    )


# In[ ]:


tmp = np.load('loss_functions/'+model_file+'_loss.npz') # the output array also stores the training and validation loss
training_loss = tmp["training_loss"]
validation_loss = tmp["validation_loss"]

plt.figure(figsize=(14, 4))
plt.plot(np.arange(training_loss.size)*100, training_loss, 'k', lw=0.5, label = 'Training set')
plt.plot(np.arange(training_loss.size)*100, validation_loss, 'r', lw=0.5, label = 'Validation set')
plt.legend(loc = 'best', frameon = False, fontsize= 18)
plt.yscale('log')
#plt.ylim([5,100])
plt.xlabel("Step", size=20)
plt.ylabel("Loss", size=20)
plt.savefig('loss_functions/'+model_file+'_loss.png',dpi=200,bbox_inches='tight')
if sys.argv[1] == '-f':
    plt.show()
plt.close()

