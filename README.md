# GALAH_DR4

This repository accompanies the Data Release 4 (DR4) of the Galactic Archaeology with HERMES (GALAH) Survey.

## Authors
- [Sven Buder](https://github.com/svenbuder) (ANU, ASTRO 3D)

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

### Stellar Parameters

### Elemental abundances

### Comparison with literature (like APOGEE DR17)

Attribution
-----------

If you make use of this code, please cite the paper::

    @article{gala,
      url = {https://github.com/svenbuder/GALAH_DR4},
      year = in prep.,
      author = {Sven Buder},
      title = {The GALAH Survey: Data Release 4}
    }
