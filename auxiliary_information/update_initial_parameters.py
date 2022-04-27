#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Preamble 
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
    get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
except:
    pass

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table, join
from scipy.interpolate import Akima1DInterpolator, interp1d
import glob
import pickle


# In[ ]:


dr60 = Table.read('dr6.0_220412.fits')
gaia_edr3 = Table.read('dr6.0_220412_gaiaedr3_calj_tmass.fits')
dr60_gaia_edr3 = join(dr60, gaia_edr3, keys='sobject_id')


# In[ ]:


# dr3_allspec = Table.read('../../GALAH_DR3/catalogs/GALAH_DR3_main_allspec_v2.fits')
# dr3 = Table()
# for label in ['sobject_id']:
#     dr3[label] = dr3_allspec[label]
# for label in ['flag_sp']:
#     dr3[label] = np.array(dr3_allspec[label],dtype=int)
# for label in ['teff','logg','fe_h','vmic','vbroad','rv_galah']:
#     dr3[label] = np.array(dr3_allspec[label],dtype=float)
# dr3_dr6 = join(dr60_gaia_edr3, dr3, keys='sobject_id')
# initial_teff = dr3_dr6['teff_r']
# initial_logg = dr3_dr6['logg_r']
# initial_fe_h = dr3_dr6['fe_h_r']
# initial_vmic = dr3_dr6['vmic_r']
# initial_vbroad = dr3_dr6['vbroad_r']
# initial_vrad = dr3_dr6['rv_com']
# plt.hist2d(
#     initial_teff,
#     initial_logg,
#     range=[(3000,8000),(0,5)],
#     bins=100,cmin=1
# );


# In[ ]:


# good_dr3 = (dr3_dr6['teff'] > 3000) & (dr3_dr6['flag_sp'] < 8)
# initial_teff[good_dr3] = dr3_dr6['teff'][good_dr3]
# initial_logg[good_dr3] = dr3_dr6['logg'][good_dr3]
# initial_fe_h[good_dr3] = dr3_dr6['fe_h'][good_dr3]
# initial_vmic[good_dr3] = dr3_dr6['vmic'][good_dr3]
# initial_vbroad[good_dr3] = dr3_dr6['vbroad'][good_dr3]


# In[ ]:


# plt.hist2d(
#     initial_teff,
#     initial_logg,
#     range=[(3000,8000),(0,5)],
#     bins=100,cmin=1
# );


# In[ ]:


# useful = (
#     np.isfinite(dr60_gaia_edr3['phot_g_mean_mag']) &
#     np.isfinite(dr60_gaia_edr3['phot_bp_mean_mag']) &
#     np.isfinite(dr60_gaia_edr3['phot_rp_mean_mag']) &
#     np.isfinite(dr60_gaia_edr3['j_m']) &
#     np.isfinite(dr60_gaia_edr3['h_m']) &
#     np.isfinite(dr60_gaia_edr3['ks_m']) &
#     np.isfinite(dr60_gaia_edr3['e_b-v']) &
#     np.isfinite(dr60_gaia_edr3['r_med_photogeo'])
# )
# dr60_gaia_edr3 = dr60_gaia_edr3[useful]

# #  Adjust missing values
# dr60_gaia_edr3['logg_r'][np.isnan(dr60_gaia_edr3['logg_r'])] = 3.0
# dr60_gaia_edr3['fe_h_r'][np.isnan(dr60_gaia_edr3['fe_h_r'])] = 0.0

# # Adjust logg and feh min and max to reasonable values
# dr60_gaia_edr3['logg_r'] = (dr60_gaia_edr3['logg_r']).clip(min=0.0,max=5.5);
# dr60_gaia_edr3['fe_h_r'] = (dr60_gaia_edr3['fe_h_r']).clip(min=-4,max=1.0);


# In[ ]:


twentyfivek_index = 0


# In[ ]:


dr60_gaia_edr3 = dr60_gaia_edr3[twentyfivek_index*25000:(twentyfivek_index+1)*25000]


# # Prepare IRFM Calculation

# In[ ]:


def calculate_irfm_teffs(logg0,feh0,gg0,bp0,rp0,j20,h20,k20,ebv0):
    """
    From Casagrande et al. (2021) on using Gaia + 2MASS photometry (+logg, [Fe/H] and E(B-V))
    to estimate IRFM Teffs
    """

    cpol=np.zeros([12,15])
    cpol[0]  = np.array([7980.8845,  -4138.3457,  1264.9366,   -130.4388,         0.,   285.8393,   -324.2196,   106.8511,    -4.9825,        0.,     4.5138,  -203.7774, 126.6981, -14.7442,    40.7376]) # BP-RP
    cpol[1]  = np.array([8172.2439,  -2508.6436,   442.6771,    -25.3120,         0.,   251.5862,   -240.7094,    86.0579,   -11.2705,        0.,   -45.9166,  -137.4645,  75.3191,  -8.7175,    21.5739]) # BP-J
    cpol[2]  = np.array([8158.9380,  -2146.1221,   368.1630,    -24.4624,         0.,   231.8680,   -170.8788,    52.9164,    -6.8455,        0.,   -45.5554,  -142.9127,  55.2465,  -4.1694,    17.6593]) # BP-H
    cpol[3]  = np.array([8265.6045,  -2124.5574,   355.5051,    -23.1719,         0.,   209.9927,   -161.4505,    50.5904,    -6.3337,        0.,   -27.2653,  -160.3595,  67.9016,  -6.5232,    16.5137]) # BP-K
    cpol[4]  = np.array([9046.6493,  -7392.3789,  2841.5464,          0.,   -85.7060,         0.,    -88.8397,    80.2959,         0.,  -15.3872,         0.,    54.6816,       0.,       0.,   -32.9499]) # RP-J
    cpol[5]  = np.array([8870.9090,  -4702.5469,  1282.3384,          0.,   -15.8164,         0.,    -30.1373,    27.9228,         0.,   -4.8012,         0.,    25.1870,       0.,       0.,   -22.3020]) # RP-H
    cpol[6]  = np.array([8910.6966,  -4305.9927,  1051.8759,          0.,    -8.6045,         0.,    -76.7984,    55.5861,         0.,   -3.9681,         0.,    35.4718,       0.,       0.,   -16.4448]) # RP-K
    cpol[7]  = np.array([8142.3539,  -3003.2988,   499.1325,     -4.8473,         0.,   244.5030,   -303.1783,   125.8628,   -18.2917,        0.,  -125.8444,    59.5183,       0.,       0.,    16.8172]) #  G-J
    cpol[8]  = np.array([8133.8090,  -2573.4998,   554.7657,    -54.0710,         0.,   229.2455,   -206.8658,    68.6489,   -10.5528,        0.,  -124.5804,    41.9630,       0.,       0.,     7.9258]) #  G-H
    cpol[9]  = np.array([8031.7804,  -1815.3523,         0.,     70.7201,    -1.7309,   252.9647,   -342.0817,   161.3031,   -26.7714,        0.,  -120.1133,    42.6723,       0.,       0.,    10.0433]) #  G-K
    cpol[10] = np.array([7346.2000,   5810.6636,         0.,  -2880.3823,   669.3810,   415.3961,   2084.4883,  3509.2200,  1849.0223,        0.,   -49.0748,     6.8032,       0.,       0.,  -100.3419]) # G-BP 
    cpol[11] = np.array([8027.1190,  -5796.4277,         0.,   1747.7036,  -308.7685,   248.1828,   -323.9569,  -120.2658,   225.9584,        0.,   -35.8856,   -16.5715,       0.,       0.,    48.5619]) # G-RP

    # Fitzpatrick/Schlafly extinction coefficients
    itbr = 0.8        
    cRg  = np.array([2.609,-0.475, 0.053])
    cRb  = np.array([2.998,-0.140,-0.175,0.062])
    cRr  = np.array([1.689,-0.059])
    cRj  =  0.719                                         
    cRh  =  0.455
    cRk  =  0.306

    #compute colour dependent extinction coefficients
    bprp0 = (bp0-rp0) - itbr*ebv0
    R_gg  = cRg[0] + cRg[1]*bprp0 + cRg[2]*bprp0*bprp0
    R_bp  = cRb[0] + cRb[1]*bprp0 + cRb[2]*bprp0*bprp0 + cRb[3]*bprp0*bprp0*bprp0
    R_rp  = cRr[0] + cRr[1]*bprp0
    R_j2  = np.zeros(1) + cRj
    R_h2  = np.zeros(1) + cRh
    R_k2  = np.zeros(1) + cRk

    # colour range for dwarfs
    d_r=np.array([2.00,3.00,4.00,4.20,1.05,1.60,1.85,2.10,2.60,2.80,-0.15,0.85])
    d_b=np.array([0.20,0.25,0.40,0.30,0.20,0.20,0.20,0.15,0.25,0.20,-1.00,0.15])

    # colour range for giants
    g_r=np.array([2.55,4.20,4.90,5.30,1.55,2.45,2.70,2.80,3.70,3.90,-0.15,1.15])
    g_b=np.array([0.20,0.90,0.40,0.30,0.60,0.20,0.20,1.00,0.25,0.20,-1.40,0.15])

    clr0       = np.zeros([12,1])
    teff_cal   = np.zeros([12,1])

    clr0[0]  = bp0-rp0 - (R_bp-R_rp)*ebv0
    clr0[1]  = bp0-j20 - (R_bp-R_j2)*ebv0
    clr0[2]  = bp0-h20 - (R_bp-R_h2)*ebv0
    clr0[3]  = bp0-k20 - (R_bp-R_k2)*ebv0
    clr0[4]  = rp0-j20 - (R_rp-R_j2)*ebv0
    clr0[5]  = rp0-h20 - (R_rp-R_h2)*ebv0
    clr0[6]  = rp0-k20 - (R_rp-R_k2)*ebv0
    clr0[7]  = gg0-j20 - (R_gg-R_j2)*ebv0
    clr0[8]  = gg0-h20 - (R_gg-R_h2)*ebv0
    clr0[9]  = gg0-k20 - (R_gg-R_k2)*ebv0
    clr0[10] = gg0-bp0 - (R_gg-R_bp)*ebv0
    clr0[11] = gg0-rp0 - (R_gg-R_rp)*ebv0


    #derive Teff in all colour indices
    for j in range(0,12):
        teff_cal[j] = cpol[j,0] + cpol[j,1]*clr0[j] + cpol[j,2]*clr0[j]*clr0[j] + cpol[j,3]*clr0[j]*clr0[j]*clr0[j] + cpol[j,4]*clr0[j]*clr0[j]*clr0[j]*clr0[j]*clr0[j] + cpol[j,5]*logg0 + cpol[j,6]*logg0*clr0[j] + cpol[j,7]*logg0*clr0[j]*clr0[j] + cpol[j,8]*logg0*clr0[j]*clr0[j]*clr0[j] + cpol[j,9]*logg0*clr0[j]*clr0[j]*clr0[j]*clr0[j]*clr0[j] + cpol[j,10]*feh0 + cpol[j,11]*feh0*clr0[j] + cpol[j,12]*feh0*clr0[j]*clr0[j] + cpol[j,13]*feh0*clr0[j]*clr0[j]*clr0[j] + cpol[j,14]*feh0*logg0*clr0[j]

    return(teff_cal)


# # Prepare BC Calculation

# In[ ]:


# linear interpolation for 2 points, Akima for more. Returns nan if 
# not possible or if extrapolated. The MARCS grid of BC used here is ordered
# such that gridt is monotonic. If not, sorting is necessary.
def mal(val,gridt,gridbc,dset):
    if len(dset[0])>2:
        mfun = Akima1DInterpolator(gridt[dset],gridbc[dset])
        itp  = mfun(val)
    if len(dset[0])==2:
        mfun = interp1d(gridt[dset],gridbc[dset],bounds_error=False) 
        itp  = mfun(val)        
    if len(dset[0])<2:
        itp = np.nan
    return(itp)

# bracket by +/-nn values over (irregular) grid. If idx True, then indices 
# are returned instead
def bracket(inval,grval,nn,idx=False):
    
    norep = np.sort(np.array(list(dict.fromkeys(list(grval)))))
    
    x1    = np.where(norep<=inval)
    x2    = np.where(norep>inval)
    
    if idx==False:
        lo = norep[x1][-nn::]
        up = norep[x2][0:nn]        
    else:
        lo = x1[0][-nn::]
        up = x2[0][0:nn]
        
    return(lo,up)


# In[ ]:


def bcstar(sid,teff,logg,feh,ebv,filters=False):

    # check input data are OK
    if np.isscalar(teff)==True:
        sid  = np.atleast_1d(sid)
        teff = np.atleast_1d(teff)
        logg = np.atleast_1d(logg)
        feh  = np.atleast_1d(feh)
        ebv  = np.atleast_1d(ebv)
    else:
        biglist = [sid,teff,logg,feh,ebv]
        it      = iter(biglist)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            print('** ERROR ** Not all required inputs have same length. Exiting now ...')
            raise SystemExit
            #raise ValueError('Not all required inputs have same length!')
            
        check_arr = isinstance(teff,np.ndarray)+isinstance(logg,np.ndarray)+isinstance(feh,np.ndarray)+isinstance(ebv,np.ndarray)
        if check_arr < 4:
            print('** ERROR **: teff,logg,feh,ebv all need to be arrays. Exiting now ...')
            raise SystemExit
            
    # Rv=3.1 by default
    el = '3.1'
    
    # keep only requested filters. By default only Gaia DR3 G,BP,RP
    flist  = ['','','','','','','','','']
    rmi    = []
    frange = np.arange(9)
    if type(filters)==str:
        if 'G2'  in filters: flist[0]='BC_G2'
        else: rmi.append(0)

        if 'BP2' in filters: flist[1]='BC_BP2'
        else: rmi.append(1)
        
        if 'RP2' in filters: flist[2]='BC_RP2'
        else: rmi.append(2)
        
        if 'G3'  in filters: flist[3]='BC_G3'
        else: rmi.append(3)
        
        if 'BP3' in filters: flist[4]='BC_BP3'
        else: rmi.append(4)
        
        if 'RP3' in filters: flist[5]='BC_RP3'
        else: rmi.append(5)

        if 'J2M' in filters: flist[6]='BC_J'
        else: rmi.append(6)
        
        if 'H2M' in filters: flist[7]='BC_H'
        else: rmi.append(7)
        
        if 'K2M' in filters: flist[8]='BC_Ks'
        else: rmi.append(8)

        frange = np.delete(frange,rmi)
        flist  = list(np.delete(flist,rmi))

        if len(rmi)==9:
            print('Not suitable filters have been selected.')
            print('Try again by choosing among the following ones:')
            print('G2 BP2 RP2 G3 BP3 RP3 J2M H2M K2M. Exiting now ...')
            raise SystemExit
    else:
        frange = np.arange(3,6) 
        flist  = ['BC_G3','BC_BP3','BC_RP3']
        rmi    = [0,1,2,6,7,8]    
        
    # read input tables of BCs for several values of E(B-V)
    files  = np.sort(glob.glob('BC_Tables/grid/STcol*Rv'+el+'*.dat'))
    gebv   = []
    gri_bc = []
    
    kk=0
    for f in files:

        gebv.append(float(f[-8:-4]))
    
        grid = Table.read(f,format='ascii')
        if kk==0:
            gteff, gfeh, glogg = grid['Teff'],grid['feh'],grid['logg']
        
        bc_g2  = grid['mbol']-grid['G2']
        bc_bp2 = grid['mbol']-grid['BP2']
        bc_rp2 = grid['mbol']-grid['RP2']

        bc_g3  = grid['mbol']-grid['G3']
        bc_bp3 = grid['mbol']-grid['BP3']
        bc_rp3 = grid['mbol']-grid['RP3']

        bc_j   = grid['mbol']-grid['J']
        bc_h   = grid['mbol']-grid['H']
        bc_k   = grid['mbol']-grid['Ks']
    
        tmp = np.transpose([bc_g2,bc_bp2,bc_rp2,bc_g3,bc_bp3,bc_rp3,bc_j,bc_h,bc_k])
        gri_bc.append(tmp)
    
        kk=kk+1
    
    gebv   = np.array(gebv)
    gri_bc = np.array(gri_bc)

    itp_bc = np.zeros(9) + np.nan

    # remove entries with E(B-V) outside grid or undetermined. Also remove a
    # few points towards the edge of the grid, where differences wrt Fortran
    # interpolation routines in BCcodes.tar are the largest.    
    # rme       = np.where((ebv>=0.72) | (ebv<0.) | (np.isfinite(ebv)==False) | ((logg<2.5) & (teff<3500)))
    # use this instead not to remove grid edges at low log(g) values
    rme       = np.where((ebv>=0.72) | (ebv<0.))

    
    fold      = feh.copy()
    bold      = ebv.copy()
    
    fold[rme] = -99.
    bold[rme] =   0.
    
    bcs = []
    
    for i in range(len(teff)):
    
        # take +/-3 steps in [Fe/H] grid
        snip = np.concatenate(bracket(fold[i],gfeh,3))
        itp1 = np.zeros((2,len(snip),9))+np.nan

        # take +/-1 step in E(B-V) grid
        eb   = np.concatenate(bracket(bold[i],gebv,1,idx=True))
        
        bc_list = []
        
        for k in range(len(snip)):
        
            x0   = np.where((gfeh==snip[k]) & (np.abs(glogg-logg[i])<1.1))
            lg0  = np.array(list(dict.fromkeys(list(glogg[x0]))))
            itp0 = np.zeros((2,len(lg0),9))+np.nan
        
            # at given logg and feh, range of Teff to interpolate across
            for j in range(len(lg0)):
                ok      = np.where((np.abs(gteff-teff[i])<1000) &                                    (gfeh==snip[k]) & (glogg==lg0[j]))
                # do it for all selected filters
                for f in frange:
                    itp0[0,j,f] = mal(teff[i],gteff,gri_bc[eb[0],:,f],ok)
                    itp0[1,j,f] = mal(teff[i],gteff,gri_bc[eb[1],:,f],ok)
                
            for f in frange:
                # remove any nan, in case. Either of itp[?,:,:] is enough
                k0 = np.where(np.isnan(itp0[0,:,f])==False)
                # interpolate in logg at correct Teff
                itp1[0,k,f] = mal(logg[i],lg0,itp0[0,:,f],k0)
                itp1[1,k,f] = mal(logg[i],lg0,itp0[1,:,f],k0)

        for f in frange:
            # remove any nan, in case
            k1  = np.where(np.isnan(itp1[0,:,f])==False)
            lor = mal(fold[i],snip,itp1[0,:,f],k1)
            upr = mal(fold[i],snip,itp1[1,:,f],k1)

            # linear interpolate in reddening
            itp_bc[f] = lor + (upr-lor)*(bold[i]-gebv[eb][0])/(gebv[eb][1]-gebv[eb][0])
            
            bc_list.append(itp_bc[f])
        bcs.append(bc_list)

    return(np.array(bcs))


# # Prepare quick mass interpolation

# In[ ]:


# Read in isochrone grid and trained nearest neighbor search machinery 'kdtree'
parsec = Table.read('parsec_isochrones/parsec_isochrones_logt_6p19_0p01_10p17_mh_m2p75_0p25_m0p75_mh_m0p60_0p10_0p70_GaiaEDR3_2MASS.fits')
# parsec = Table.read('../auxiliary_information/parsec_isochrones/parsec_isochrones_logt_6p19_0p01_10p17_mh_m2p75_0p25_1p00_mh_m0p75_0p05_0p75_GaiaEDR3_2MASS.fits')
file = open('parsec_isochrones/isochrone_kdtree_Teff_logg_M_H.pickle','rb')
parsec_kdtree = pickle.load(file)
file.close()


# In[ ]:


def update_teff_logg(sobject_id, teff, logg, fe_h, g, bp, rp, j, h, ks, ebv, distance):
    
    initial_teff = teff.clip(min=3000,max=8000)
    initial_logg = logg.clip(min=0.5,max=5.)
    initial_fe_h = fe_h.clip(min=-4.0,max=1.0)
    
    # Part 1: IRFM Teff
    
    initial_irfm_teff = []
    for index in range(len(logg)):
        irfm_teff_array = calculate_irfm_teffs(
            initial_logg[index],
            initial_fe_h[index],
            g[index],
            bp[index],
            rp[index],
            j[index],
            h[index],
            ks[index],
            ebv[index]
        ).T[0]

        p16,p50,p84 = np.nanpercentile(irfm_teff_array,q=[16,50,84])

        initial_irfm_teff.append(p50.clip(min=3050,max=7950))
    
    initial_irfm_teff = np.array(initial_irfm_teff)
    
    # Part 2: Auxiliary input for logg
    
    a_ks = 0.38 * ebv
    
    initial_mass = []
    for index in range(len(logg)):
        distance_matches, closest_matches = parsec_kdtree.query([[np.log10(initial_irfm_teff[index]),initial_logg[index],initial_fe_h[index]]],k=100)

        # Calculate weighted average values based on squared KDTree distances
        weights = 1/distance_matches**2
        initial_mass.append(np.average(parsec['mass'][closest_matches],weights=weights))
    
    initial_mass = np.array(initial_mass).clip(max=2.)
    
    initial_bcs = bcstar(
        sid = sobject_id,
        teff = initial_irfm_teff,
        logg = initial_logg,
        feh = initial_fe_h,
        ebv = ebv.clip(min=0.0,max=0.71),
        filters='G3_BP3_RP3_J2M_H2M_K2M',
    )
    
    initial_log10_lbol_g = - 0.4 * (
        g 
        - 5*np.log10(distance/10.) 
        + initial_bcs[:,0]
        - a_ks
        - 4.7554)
    initial_log10_lbol_bp = - 0.4 * (
        bp
        - 5*np.log10(distance/10.) 
        + initial_bcs[:,1]
        - a_ks
        - 4.7554)
    initial_log10_lbol_rp = - 0.4 * (
        rp 
        - 5*np.log10(distance/10.) 
        + initial_bcs[:,2]
        - a_ks
        - 4.7554)
    initial_log10_lbol_j = - 0.4 * (
        j
        - 5*np.log10(distance/10.) 
        + initial_bcs[:,3]
        - a_ks
        - 4.7554)
    initial_log10_lbol_h = - 0.4 * (
        h 
        - 5*np.log10(distance/10.) 
        + initial_bcs[:,4]
        - a_ks
        - 4.7554)
    initial_log10_lbol_ks = - 0.4 * (
        ks 
        - 5*np.log10(distance/10.) 
        + initial_bcs[:,5]
        - a_ks
        - 4.7554)

    initial_log10_lbol = np.median([initial_log10_lbol_g,initial_log10_lbol_bp,initial_log10_lbol_rp,initial_log10_lbol_j,initial_log10_lbol_h,initial_log10_lbol_ks],axis=0)
    
    initial_logg = np.array(4.438 + np.log10(initial_mass) + 4*np.log10(initial_irfm_teff/5772.) - initial_log10_lbol)
    
    return(
        initial_irfm_teff, 
        initial_logg, 
        initial_fe_h,
        initial_mass,
        initial_log10_lbol,
        initial_log10_lbol_g,
        initial_log10_lbol_bp,
        initial_log10_lbol_rp,
        initial_log10_lbol_j,
        initial_log10_lbol_h,
        initial_log10_lbol_ks,
        initial_bcs
    )

new_teff, new_logg, new_fe_h, new_mass, new_log10_lbol,new_log10_lbol_g, new_log10_lbol_bp, new_log10_lbol_rp, new_log10_lbol_j, new_log10_lbol_h, new_log10_lbol_ks, initial_bcs = update_teff_logg(
    sobject_id = np.array(dr60_gaia_edr3['sobject_id']),
    teff = np.array(dr60_gaia_edr3['teff_r']), 
    logg = np.array(dr60_gaia_edr3['logg_r']), 
    fe_h = np.array(dr60_gaia_edr3['fe_h_r']), 
    g = np.array(dr60_gaia_edr3['phot_g_mean_mag']), 
    bp = np.array(dr60_gaia_edr3['phot_bp_mean_mag']), 
    rp = np.array(dr60_gaia_edr3['phot_rp_mean_mag']), 
    j = np.array(dr60_gaia_edr3['j_m']), 
    h = np.array(dr60_gaia_edr3['h_m']), 
    ks = np.array(dr60_gaia_edr3['ks_m']), 
    ebv = np.array(dr60_gaia_edr3['e_b-v']),
    distance = np.array(dr60_gaia_edr3['r_med_photogeo'])
)


# In[ ]:


updated_starting_values = Table()
updated_starting_values['sobject_id'] = dr60_gaia_edr3['sobject_id']
updated_starting_values['teff'] = new_teff
updated_starting_values['logg'] = new_logg
updated_starting_values['fe_h'] = new_fe_h
updated_starting_values['mass'] = new_mass
updated_starting_values['log_lbol'] = new_log10_lbol
updated_starting_values['log_lbol_g'] = new_log10_lbol_g
updated_starting_values['log_lbol_bp'] = new_log10_lbol_bp
updated_starting_values['log_lbol_rp'] = new_log10_lbol_rp
updated_starting_values['log_lbol_j'] = new_log10_lbol_j
updated_starting_values['log_lbol_h'] = new_log10_lbol_h
updated_starting_values['log_lbol_ks'] = new_log10_lbol_ks
updated_starting_values['bc_g'] = initial_bcs[:,0]
updated_starting_values['bc_bp'] = initial_bcs[:,1]
updated_starting_values['bc_rp'] = initial_bcs[:,2]
updated_starting_values['bc_j'] = initial_bcs[:,3]
updated_starting_values['bc_h'] = initial_bcs[:,4]
updated_starting_values['bc_ks'] = initial_bcs[:,5]
updated_starting_values.write('updated_initial_values_220422_'+str(twentyfivek_index)+'.fits',overwrite=True)


# In[ ]:


f, (ax1, ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
p = ax1.hist2d(
    dr60_gaia_edr3['teff_r'],
    dr60_gaia_edr3['logg_r'],
    bins=(np.linspace(2000,9000,100),np.linspace(-1,6,100)),
    cmin=1,vmax=80
)
plt.colorbar(p[-1],ax=ax1)
p = ax2.hist2d(
    new_teff,new_logg,
    bins=(np.linspace(2000,9000,100),np.linspace(-1,6,100)),
    cmin=1,vmax=80
)
plt.colorbar(p[-1],ax=ax2)
ax2.set_xlim(9000,2000)
ax2.set_ylim(6,-1)
ax1.set_xlabel('Reduction Teff')
ax2.set_xlabel('IRFM Teff')
ax1.set_ylabel('Reduction logg')
ax2.set_ylabel('bolometric logg')


# In[ ]:




