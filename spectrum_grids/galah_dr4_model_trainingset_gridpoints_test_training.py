from astropy.table import Table
import glob
import numpy as np

grids = Table.read('galah_dr4_model_trainingset_gridpoints.fits')
grids['model_name'] = np.array([str(int(grid['teff_subgrid']))+'_'+str("{:.2f}".format(grid['logg_subgrid']))+'_'+str("{:.2f}".format(grid['fe_h_subgrid'])) for grid in grids])

names_3x3x3 = np.array([x[81:-13] for x in glob.glob('../spectrum_interpolation/neural_networks/models/galah_dr4_neutral_network_3x3x3_*.npz')])

names_old_extra6 = np.array([x[74:-13] for x in glob.glob('../spectrum_interpolation/ThePayne/models/galah_dr4_thepayne_model_extra6_*.npz')])

has_3x3x3 = []
has_old_extra6 = []

for model_name in grids['model_name']:
    if model_name in names_3x3x3:
        has_3x3x3.append(True)
    else:
        has_3x3x3.append(False)
    if model_name in names_old_extra6:
        has_old_extra6.append(True)
    else:
        has_old_extra6.append(False)

grids['has_model_3x3x3'] = np.array(has_3x3x3,dtype=bool)
grids['has_model_extra6'] = np.array(has_old_extra6,dtype=bool)

grids.write('galah_dr4_model_trainingset_gridpoints_trained.fits',overwrite=True)
