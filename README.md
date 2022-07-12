# GALAH_DR4

This repository accompanies the Data Release 4 (DR4) of the Galactic Archaeology with HERMES (GALAH) Survey.

## Authors
- [Sven Buder](https://github.com/svenbuder) (ANU, ASTRO 3D)

## Where to find and how to use the data release catalogs

The catalogs will be released publicly on the datacentral website: [http://datacentral.org.au/teamdata/GALAH/public/](http://datacentral.org.au/teamdata/GALAH/public/).

The inofficial releases can be downloaded by logging into the internal GALAH website: [https://internal.galah-survey.org](https://internal.galah-survey.org)

Use flag_sp == 0 and flag_x_fe == 0 for best results!

## About Galactic Archaeology with HERMES (GALAH)

GALAH is a stellar spectroscopic survey of a million stars in the Milky Way. It's scientific motivation is described by [De Silva et al. (2015)](http://adsabs.harvard.edu/abs/2015MNRAS.449.2604D). GALAH had three previous data releases: DR1 [(Martell et al. 2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.3203M), DR2 [Buder et al. 2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.4513B), and DR3 [(Buder et al. 2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506..150B). For more information see [the GALAH survey website](https://www.galah-survey.org).

## Overview of this repository

In this repository, you find the relevant code and example spectra (excluding large files) for reproducing the GALAH data analysis. Input files are provided by the reduction pipeline of GALAH, which is also published on a [github repository](https://github.com/sheliak/galah_reduction).

GALAH operates with unique *sobject_id* as identifier of spectra for dates (first 6 digits), runs (next 4 digits), and repeats (next 2 digits), fibres (next 3 digits), and CCDs (last digit).

Code/Files are accessed in the following order:
1. observations (reduced spectra for each sobject_id in subdirectories of date/sobject_id from the reduction pipeline),
2. auxiliary_information (information to aid the spectroscopic analysis, e.g. cross matches with Gaia and other catalogs/literature),
3. spectrum_grids (code used to create the synthetic spectra with SME, including IDL scripts with SME580),
4. spectrum_interpolation (spectrum interpolation models/code used to train them with neural networks based on the synthetic spectra), 
5. spectrum_analysis (code used to analyse the reduced spectra),
6. analysis_products (directory where the analysis products are saved, including the synthetic spectra and renormalised observations),
7. spectrum_post_processing (directory where the post processing code and processed data products are stored),
8. validation (directory with validation codes and diagnostic figures), and
9. catalogs (final catalogs to be published).

## Overview of the data release products as of 11 July 2022

![Overview of the fourth data release as of 11 July 2022 with density plots of unflagged stellar parameters and abundances](spectrum_post_processing/figures/galah_dr4_overview.png)

### What's new? Fastly interpolated synthetic spectra for the whole wavelength range

To allow the simultanious fitting of stellar parameters and abundances (a shortcoming of the previous data releases affecting especially blended regions), we have changed our fitting approach. We are now producing synthetic spectra for a limited random selection and train neural networks on them. This allows to fit all 5 stellar parameters (Teff, logg, [Fe/H], vmic, vsini) and up to 31 elemental abundances at the same time.
<p align=center>
    <img src="analysis_products/210115/210115002201239/210115002201239_simple_fit_comparison.png" alt="Observed and synthetic spectrum for VESTA" width="50%"/>
</p>

### What's new? CNO Abundances

We are fitting CNO abundances now! Thanks to the enhanced creation of synthetic stellar spectra, we are now also producing synthetic spectra for regions with strong molecular absorption features, like C2 (C12-C12 Swan bands before 4738Å) and CN (beyond 7870Å) as well as an underlying CN feature throughout most of the red and infrared region (most notably in cool giants).
<p align=center>
    <img src="spectrum_post_processing/figures/overview_CNO.png" alt="CNO abundance overview" width="50%"/>
</p>

## Validation

### Gaia FGK Benchmark Stars

<p align=center>
    <img src="validation/figures/gbs_performance_lbol.png" alt="drawing" width="50%"/>
</p>

### APOGEE DR17     

#### Comparison of APOGEE DR17 stellar parameters with GALAH DR4, for all stars (top), dwarfs (middle) and giants (bottom).

<p align=center>
    <img src="validation/figures/galah_dr4_validation_apogeedr17_teffloggfeh_diff_all.png" alt="drawing" width="75%"/>
    <img src="validation/figures/galah_dr4_validation_apogeedr17_teffloggfeh_diff_dwarfs.png" alt="drawing" width="75%"/>
    <img src="validation/figures/galah_dr4_validation_apogeedr17_teffloggfeh_diff_giants.png" alt="drawing" width="75%"/>
</p>
    
#### [Fe/H] vs. [Mg/Fe] compared to GALAH DR3 and APOGEE DR17
    
<p align=center>
    <img src="validation/figures/galah_dr4_validation_galah_dr3_mg_fe_density.png" alt="drawing" width="45%"/>
    <img src="validation/figures/galah_dr4_validation_apogeedr17_mg_fe_density.png" alt="drawing" width="45%"/>
</p>

#### [Fe/H] vs. [Ni/Fe] compared to GALAH DR3 and APOGEE DR17

<p align=center>
    <img src="validation/figures/galah_dr4_validation_galah_dr3_ni_fe_density.png" alt="drawing" width="45%"/>
    <img src="validation/figures/galah_dr4_validation_apogeedr17_ni_fe_density.png" alt="drawing" width="45%"/>
</p>

Attribution
-----------

If you make use of this code, please cite the paper::

    @article{Buder2022,
      url = {https://github.com/svenbuder/GALAH_DR4},
      year = in prep.,
      author = {Sven Buder},
      title = {The GALAH Survey: Data Release 4}
    }
