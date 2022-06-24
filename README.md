# GALAH_DR4

This repository accompanies the fourth data release of the Galactic Archaeology with HERMES (GALAH) Survey.

GALAH is a stellar spectroscopic survey of stars in the Milky Way. It's scientific motivation is described by [De Silva et al. (2015)](http://adsabs.harvard.edu/abs/2015MNRAS.449.2604D).

In this repository, you find the relevant code and example spectra (excluding large files) for reproducing the GALAH data analysis. Input files are provided by the reduction pipeline of GALAH, which is also published on a [github repository](https://github.com/sheliak/galah_reduction).

GALAH operates with unique *sobject_id* as identifier of spectra for dates (first 6 digits), runs (next 4 digits), and repeats (next 2 digits), fibres (next 3 digits), and CCDs (last digit).

## Overview of directories

observations: reduced spectra for each sobject_id in subdirectories of date/sobject_id

auxiliary_information: auxiliary information to be used to aid the spectroscopic analysis (e.g. cross matches with Gaia and other catalogs/literature)

spectrum_grids: code used to create the synthetic spectra with SME, including IDL scripts with SME580.

spectrum_interpolation: spectrum interpolation models and code used to train them with neural networks based on the synthetic spectra

spectrum_analysis: code used to analyse the reduced spectra

analysis_products: directory where the analysis products are saved

spectrum_post_processing: directory where the post processing code and processed data products are stored

validation: directory with validation codes and diagnostic figures

catalogs: final catalogs to be published (TBD)
