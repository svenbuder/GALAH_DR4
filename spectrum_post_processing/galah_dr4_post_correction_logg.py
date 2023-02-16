#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Preamble 
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
except:
    pass

import numpy as np
import pandas as pd
from astropy.table import Table, join
import astropy.units as u
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.neighbors import KDTree
import sys


# In[ ]:


if sys.argv[1] != '-f':
    date = sys.argv[1]
else:
    date = '131216'
    date = '131220' # 47Tuc Globular Cluster
#     date = '140824' # Melotte 25 Open Cluster

print('Post-Processing '+date)


# In[ ]:


galah_raw = Table.read('daily/galah_dr4_allspec_not_validated_'+date+'_single.fits')


# In[ ]:


galah_extra = Table.read('../auxiliary_information/dr60_230101_ebv_wise_tmass_gaiadr3corr_xmatch.fits')


# In[ ]:


galah_raw_extra = join(galah_raw,galah_extra,keys='sobject_id',join_type='left')


# In[ ]:


print(len(galah_raw_extra['sobject_id']))


# # Adjust distances and calculate A(Ks) as well as E(B-V)

# In[ ]:


extra_info = Table()
for key in [
        'sobject_id','gaiadr3_source_id','teff','logg','fe_h','phot_g_mean_mag','phot_bp_mean_mag','bp_rp',
        'h_m','h_msigcom', 'ks_m', 'ks_msigcom', 'W2mag', 'e_W2mag',
        'ebv'
    ]:
        extra_info[key] = galah_raw_extra[key]
        
key = 'parallax'
extra_info[key] = galah_raw_extra[key]
extra_info['e_'+key] = galah_raw_extra[key+'_error']
extra_info['parallax_gaia_edr3'] = galah_raw_extra[key]
extra_info['e_parallax_gaia_edr3'] = galah_raw_extra[key+'_error']

extra_info['rv_gaia_dr3'] = galah_raw_extra['radial_velocity']
extra_info['e_rv_gaia_dr3'] = galah_raw_extra['radial_velocity_error']
extra_info['ruwe_gaia_dr3'] = galah_raw_extra['ruwe']

extra_info['r_med'] = 1000. /extra_info['parallax']
extra_info['r_lo'] = 1000. /(extra_info['parallax']+extra_info['e_parallax'])
extra_info['r_hi'] = 1000. /(extra_info['parallax']-extra_info['e_parallax'])

has_r_med_photogeo = np.isfinite(galah_raw_extra['r_med_photogeo'])
extra_info['r_med'][has_r_med_photogeo] = galah_raw_extra['r_med_photogeo'][has_r_med_photogeo]
extra_info['r_lo'][has_r_med_photogeo] = galah_raw_extra['r_lo_photogeo'][has_r_med_photogeo]
extra_info['r_hi'][has_r_med_photogeo] = galah_raw_extra['r_hi_photogeo'][has_r_med_photogeo]

has_r_med_geo = np.isnan(galah_raw_extra['r_med_photogeo']) & np.isfinite(galah_raw_extra['r_med_geo'])
extra_info['r_med'][has_r_med_geo] = galah_raw_extra['r_med_geo'][has_r_med_geo]
extra_info['r_lo'][has_r_med_geo] = galah_raw_extra['r_lo_geo'][has_r_med_geo]
extra_info['r_hi'][has_r_med_geo] = galah_raw_extra['r_hi_geo'][has_r_med_geo]


# In[ ]:


galah_raw_extra['tmass_ph_qual'][galah_raw_extra['tmass_ph_qual'].mask] = 'UUU'
galah_raw_extra['tmass_ph_qual'][galah_raw_extra['tmass_ph_qual'] == ''] = 'UUU'

galah_raw_extra['qph'][galah_raw_extra['qph'].mask] = 'UUUU'
galah_raw_extra['qph'][galah_raw_extra['qph'] == ''] = 'UUUU'

extra_info['a_ks'] = np.zeros(len(extra_info['sobject_id']))

good_h_w2 = np.all([
    [x[1] == 'A' for x in galah_raw_extra['tmass_ph_qual']],
    [x[1] == 'A' for x in galah_raw_extra['qph']]
],axis=0)

extra_info['a_ks'][good_h_w2] = (0.918 * (extra_info['h_m'][good_h_w2] - extra_info['W2mag'][good_h_w2] - 0.08)).clip(min=0.00,max=0.50)
extra_info['a_ks'][~good_h_w2] = (0.36 * galah_raw_extra['ebv'][~good_h_w2]).clip(min = 0.00, max = 0.50)

for index, sobject_id in enumerate(extra_info['sobject_id']):
    if sobject_id in [140710008301032,131220004401099,140207004801201]:
        if sobject_id == 140710008301032:
            extra_info['ks_m'][index] = 1.43 # * u.mag
            extra_info['ks_msigcom'][index] = 0.02 # * u.mag
        if sobject_id == 131220004401099:
            extra_info['ks_m'][index] = 1.46 # * u.mag
            extra_info['ks_msigcom'][index] = 0.03 # * u.mag
        if sobject_id == 140207004801201:
            extra_info['ks_m'][index] = 2.20 # * u.mag
            extra_info['ks_msigcom'][index] = 0.01 # * u.mag

    if sobject_id in [210115002201239,150210005801171,140710006601104,140709004401117,140708005801203,141102003801353,140710000801284,140709001901194]:

        if sobject_id == 210115002201239:
            extra_info['ks_m'][index] = 3.28 # * u.mag
            extra_info['ks_msigcom'][index] = 0.02 # * u.mag
            extra_info['parallax'][index] = 100.0 # * u.mas
            extra_info['e_parallax'][index] = 0.1 # * u.mas
        if sobject_id == 150210005801171:
            extra_info['ks_m'][index] = -3.00 # * u.mag
            extra_info['ks_msigcom'][index] = 0.03 # * u.mag
            extra_info['parallax'][index] = 88.83 # * u.mas
            extra_info['e_parallax'][index] = 0.54 # * u.mas
        if sobject_id == 140710006601104:
            extra_info['ks_m'][index] = -1.68 # * u.mag
            extra_info['ks_msigcom'][index] = 0.05 # * u.mag
            extra_info['parallax'][index] = 13.09 # * u.mas
            extra_info['e_parallax'][index] = 0.44 # * u.mas
        if sobject_id == 140709004401117:
            extra_info['ks_m'][index] = -0.16 # * u.mag
            extra_info['ks_msigcom'][index] = 0.04 # * u.mag
            extra_info['parallax'][index] = 12.62 # * u.mas
            extra_info['e_parallax'][index] = 0.18 # * u.mas
        if sobject_id == 140708005801203:
            extra_info['parallax'][index] = 134.07 # * u.mas
            extra_info['e_parallax'][index] = 0.11 # * u.mas
        if sobject_id == 141102003801353:
            extra_info['ks_m'][index] = -2.84 # * u.mag
            extra_info['ks_msigcom'][index] = 0.06 # * u.mag
            extra_info['parallax'][index] = 48.94 # * u.mas
            extra_info['e_parallax'][index] = 0.77 # * u.mas
        if sobject_id == 140710000801284:
            extra_info['ks_m'][index] = 2.20 # * u.mag
            extra_info['ks_msigcom'][index] = 0.01 # * u.mag
            extra_info['parallax'][index] = 9.705958463334975 # * u.mas
            extra_info['e_parallax'][index] = 0.15301941 # * u.mas
        if sobject_id == 140709001901194:
            extra_info['ks_m'][index] = 1.36 # * u.mag
            extra_info['ks_msigcom'][index] = 0.02 # * u.mag
            extra_info['parallax'][index] = 87.75 # * u.mas
            extra_info['e_parallax'][index] = 1.24 # * u.mas  

        extra_info['r_med'][index] = 1000. /extra_info['parallax'][index]
        extra_info['r_lo'][index] = 1000. /(extra_info['parallax'][index]+extra_info['e_parallax'][index])
        extra_info['r_hi'][index] = 1000. /(extra_info['parallax'][index]-extra_info['e_parallax'][index])

        extra_info['ebv'][index] = 0.0
        extra_info['a_ks'][index] = 0.0

    if extra_info['parallax'][index] > 10.:
        extra_info['ebv'][index] = 0.0
        extra_info['a_ks'][index] = 0.0

    if extra_info['a_ks'][index] > 2 * 0.36 * extra_info['ebv'][index]:
        extra_info['a_ks'][index] = 0.36 * extra_info['ebv'][index]
        
    if  extra_info['ebv'][index] > 2 * extra_info['a_ks'][index] / 0.36:
        extra_info['ebv'][index] = 2.78 * extra_info['a_ks'][index]


# In[ ]:


# Check entries in open cluster catalog by Cantat-Gaudin et al., 2020, A&A 640, 1
cantatgaudin2020_parallaxes = Table.read('../auxiliary_information/CantatGaudin_2020_AandA_640_1.fits')

for index, gaiadr3_source_id in enumerate(extra_info['gaiadr3_source_id']):
    cantatgaudin2020_match = np.where(gaiadr3_source_id == cantatgaudin2020_parallaxes['GaiaDR2'])[0]
    # If there is an entry in this catalog
    if len(cantatgaudin2020_match) > 0:
        print(cantatgaudin2020_parallaxes['Cluster'][cantatgaudin2020_match[0]])
        # replace parallax to be used, if Cantat-Gaudin et al. parallax has smaller uncertainty
        if cantatgaudin2020_parallaxes['e_plx'][cantatgaudin2020_match[0]] < extra_info['e_parallax'][index]:
            extra_info['parallax'][index] = cantatgaudin2020_parallaxes['plx'][cantatgaudin2020_match[0]]
            extra_info['e_parallax'][index] = cantatgaudin2020_parallaxes['e_plx'][cantatgaudin2020_match[0]]
            extra_info['r_med'][index] = 1000. /extra_info['parallax'][index]
            extra_info['r_lo'][index] = 1000. /(extra_info['parallax'][index]+extra_info['e_parallax'][index])
            extra_info['r_hi'][index] = 1000. /(extra_info['parallax'][index]-extra_info['e_parallax'][index])

# Check entries in open cluster catalog by Vasiliev & Baumgardt (2021), MNRAS, 505, 5978
vasiliev2021_parallaxes = Table.read('../auxiliary_information/VasilievBaumgardt_2021_MNRAS_505_5978_cluster_source_id_memberprob0p7.fits')
globular_clusters = Table.read('../auxiliary_information/GlobularClustersGALAHDR4.fits')

for index, gaiadr3_source_id in enumerate(extra_info['gaiadr3_source_id']):
    vas = dict()
    vasiliev2021_match = np.where(gaiadr3_source_id == vasiliev2021_parallaxes['source_id'])[0]
    if len(vasiliev2021_match) > 0:
        correct_cluster = np.where(globular_clusters['Cluster'] == vasiliev2021_parallaxes['cluster'][vasiliev2021_match[0]])[0]
        if len(correct_cluster) > 0:
            correct_cluster = globular_clusters[correct_cluster[0]]
            vas['parallax_vb21'] = correct_cluster['parallax']
            vas['e_parallax_vb21'] = correct_cluster['e_parallax']
            vas['r_med_vb21'] = correct_cluster['r_med']
            vas['r_lo_vb21'] = correct_cluster['r_lo']
            vas['r_hi_vb21'] = correct_cluster['r_hi']
        else:
            raise ValueError('No extra information for Globular Cluster in auxiliary_information/GlobularClustersGALAHDR4.fits')

        if vas['e_parallax_vb21'] < extra_info['e_parallax'][index]:
            extra_info['parallax'][index] = vas['parallax_vb21']
            extra_info['e_parallax'][index] = vas['e_parallax_vb21']
            extra_info['r_med'][index] = vas['r_med_vb21']
            extra_info['r_lo'][index] = vas['r_lo_vb21']
            extra_info['r_hi'][index] = vas['r_hi_vb21']


# In[ ]:


# Read in isochrone grid and trained nearest neighbor search machinery 'kdtree'
parsec = Table.read('../auxiliary_information/parsec_isochrones/parsec_isochrones_logt_6p19_0p01_10p17_mh_m2p75_0p25_m0p75_mh_m0p60_0p10_0p70_GaiaEDR3_2MASS.fits')
# parsec = Table.read('../auxiliary_information/parsec_isochrones/parsec_isochrones_logt_6p19_0p01_10p17_mh_m2p75_0p25_1p00_mh_m0p75_0p05_0p75_GaiaEDR3_2MASS.fits')
file = open('../auxiliary_information/parsec_isochrones/isochrone_kdtree_Teff_logg_M_H.pickle','rb')
parsec_kdtree = pickle.load(file)
file.close()


# In[ ]:


def calculate_age_mass(teff, logg, loglum, m_h, e_teff = 100, e_logg = 0.5, e_loglum = 0.1, e_m_h = 0.2):

    e_loglum = e_loglum * loglum
    
    # Make sure that [Fe/H] stays within parsec grid limits
    unique_m_h = np.unique(parsec['m_h'])
    if m_h < unique_m_h[0]:
        m_h = unique_m_h[0] + 0.001
        print('adjust m_h input to ',m_h)
    if m_h > unique_m_h[-1]:
        m_h = unique_m_h[-1] - 0.001
        print('adjust m_h input to ',m_h)
        
    # Make sure we have at least 2 [Fe/H] dimensions to integrate over
    lower_boundary_m_h = np.argmin(np.abs(unique_m_h - (m_h - e_m_h)))
    upper_boundary_m_h = np.argmin(np.abs(unique_m_h - (m_h + e_m_h)))
    if lower_boundary_m_h == upper_boundary_m_h:
        if lower_boundary_m_h == 0:
            upper_boundary_m_h = 1
        if lower_boundary_m_h == len(unique_m_h)-1:
            lower_boundary_m_h = len(unique_m_h)-2
    
    # find all relevant isochrones points
    relevant_isochrone_points = (
        (parsec['logT'] > np.log10(teff - e_teff)) & 
        (parsec['logT'] < np.log10(teff + e_teff)) &
        (parsec['logg'] > logg - e_logg) & 
        (parsec['logg'] < logg + e_logg) &
        (parsec['logL'] > loglum - e_loglum) & 
        (parsec['logL'] < loglum + e_loglum) &
        (parsec['m_h']  >= unique_m_h[lower_boundary_m_h]) & 
        (parsec['m_h']  <= unique_m_h[upper_boundary_m_h])
    )
    # if len(parsec['logT'][relevant_isochrone_points]) < 10:
    #     print('Only '+str(len(parsec['logT'][relevant_isochrone_points]))+' isochrones points available')
    
    # 
    model_points = np.array([
        10**parsec['logT'][relevant_isochrone_points],
        parsec['logg'][relevant_isochrone_points],
        parsec['logL'][relevant_isochrone_points],
        parsec['m_h'][relevant_isochrone_points]
    ]).T
    
    # find normalising factor
    norm = np.log(np.sqrt((2.*np.pi)**4.*np.prod(np.array([e_teff, e_logg, e_loglum ,e_m_h])**2)))
    
    # sum up lnProb and weight ages/masses by 
    lnProb = - np.sum(((model_points - [teff, logg, loglum, m_h])/[e_teff, e_logg, e_loglum, e_m_h])**2, axis=1) - norm    
    age = np.sum(10**parsec['logAge'][relevant_isochrone_points] * np.exp(lnProb)/10**9)
    mass = np.sum(parsec['mass'][relevant_isochrone_points] * np.exp(lnProb))
    
    # Normalise by probability
    Prob_sum = np.sum(np.exp(lnProb))
    age /= Prob_sum
    mass /= Prob_sum
    
    return(age, mass)


# In[ ]:


def calculate_bc(teff, logg, fe_h, alpha_fe):

    bc_distance_matches, bc_closest_matches = bc_kdtree.query(np.array([np.log10(teff),logg,fe_h,alpha_fe]).T,k=8)
    bc_weights = 1/bc_distance_matches**2
    
    bc_ks = np.average(bc_grid['mbol'][bc_closest_matches] - bc_grid['Ks'][bc_closest_matches],weights=bc_weights,axis=-1)
    
    return(bc_ks)


# In[ ]:


def calculate_logg_parallax(teff, logg_in, fe_h, ks_m, ks_msigcom, r_med, r_lo, r_hi, a_ks, e_teff = 100, e_logg = 0.25, e_m_h = 0.2):
    
    if fe_h < -1:
        alpha_fe = 0.4
    elif fe_h > 0:
        alpha_fe = 0.0
    else:
        alpha_fe = -0.4 *fe_h
    
    m_h = fe_h + np.log10(10**alpha_fe * 0.694 + 0.306)
    
    bc_ks = calculate_bc(teff, logg_in, fe_h, alpha_fe)
    
    loglbol = - 0.4 * (ks_m - 5.0*np.log10(r_med/10.) + bc_ks - a_ks - 4.75)#[0]
    # Take into account uncertainties of Ks, distance, and adds uncertainties of +- 0.05 mag for A(Ks) and BC(Ks)
    loglbol_lo = - 0.4 * (ks_m + ks_msigcom - 5.0*np.log10(r_lo/10.) + (bc_ks + 0.05) - (a_ks - 0.05) - 4.75)#[0]
    loglbol_hi = - 0.4 * (ks_m - ks_msigcom - 5.0*np.log10(r_hi/10.) + (bc_ks - 0.05) - (a_ks + 0.05) - 4.75)#[0]
    
    e_loglum = 0.5*(loglbol_hi-loglbol_lo) / loglbol
        
    age, mass = calculate_age_mass(teff, logg_in, loglbol, m_h, e_teff, e_logg, e_loglum, e_m_h)
    if np.isnan(mass):
        age, mass = calculate_age_mass(teff, logg_in, loglbol, m_h, e_teff*2, e_logg*2, e_loglum*2, e_m_h*2)
        if np.isnan(mass):
            age, mass = calculate_age_mass(teff, logg_in, loglbol, m_h, e_teff*3, e_logg*3, e_loglum*3, e_m_h*3)
            if np.isnan(mass):
                mass = 1.0
                age = np.NaN
        
    return(4.438 + np.log10(mass) + 4*np.log10(teff/5772.) - loglbol, mass, age, bc_ks, 10**loglbol, loglbol_lo, loglbol_hi)


# In[ ]:


def iterate_logg_mass_age_bc_ks_lbol(teff, logg_in, fe_h, ks_m, ks_msigcom, r_med, r_lo, r_hi, a_ks):
    logg_out, mass, age, bc_ks, lbol, loglbol_lo, loglbol_hi = calculate_logg_parallax(teff, logg_in, fe_h, ks_m, ks_msigcom, r_med, r_lo, r_hi, a_ks)        
    iteration = 0
    while (abs(logg_out - logg_in) > 0.01) & (iteration < 4):
        logg_in = logg_out
        logg_out, mass, age, bc_ks, lbol, loglbol_lo, loglbol_hi = calculate_logg_parallax(teff, logg_in, fe_h, ks_m, ks_msigcom, r_med, r_lo, r_hi, a_ks)
        iteration += 1
    return(mass, age, bc_ks, lbol, logg_out)


# In[ ]:


# Read in BC grid for 0.00
bc_grid = np.genfromtxt('../auxiliary_information/BC_Tables/grid/STcolors_2MASS_GaiaDR2_EDR3_Rv3.1_EBV_0.00.dat',names=True)
file = open('../auxiliary_information/BC_Tables/grid/bc_grid_kdtree_ebv_0.00.pickle','rb')
bc_kdtree = pickle.load(file)
file.close()


# In[ ]:


extra_info['mass'] = np.zeros(np.shape(extra_info['sobject_id'])[0])
extra_info['age'] = np.zeros(np.shape(extra_info['sobject_id'])[0])
extra_info['bc_ks'] = np.zeros(np.shape(extra_info['sobject_id'])[0])
extra_info['lbol'] = np.zeros(np.shape(extra_info['sobject_id'])[0])
extra_info['logg_plx'] = np.zeros(np.shape(extra_info['sobject_id'])[0])
for keys in ['mass','age','bc_ks','lbol','logg_plx']:
    extra_info[keys][:] = np.NaN


# In[ ]:


print('')
for i in range(len(extra_info['sobject_id'])):
    if i%250==0:
        print(i,'of',len(extra_info['sobject_id']))
    if np.all(np.isfinite([extra_info['teff'][i],
        extra_info['logg'][i], 
        extra_info['fe_h'][i], 
        extra_info['ks_m'][i], extra_info['ks_msigcom'][i], 
        extra_info['r_med'][i], extra_info['r_lo'][i], extra_info['r_hi'][i], 
        extra_info['a_ks'][i]])
    ):
        extra_info['mass'][i], extra_info['age'][i], extra_info['bc_ks'][i], extra_info['lbol'][i], extra_info['logg_plx'][i] = iterate_logg_mass_age_bc_ks_lbol(
            extra_info['teff'][i], 
            extra_info['logg'][i],
            extra_info['fe_h'][i], 
            extra_info['ks_m'][i], extra_info['ks_msigcom'][i], 
            extra_info['r_med'][i], extra_info['r_lo'][i], extra_info['r_hi'][i], 
            extra_info['a_ks'][i]
        )


# In[ ]:


extra_info.write('daily/galah_dr4_allspec_not_validated_plxlogg_'+date+'.fits',overwrite=True)

