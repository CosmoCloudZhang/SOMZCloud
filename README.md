# SOMZCloud

Mock Realisations of Tomographic Ensemble Redshift Distributions for LSST Y1 & Y10

**SOMZCloud** generates simulated / mock catalogues of tomographic redshift distributions, incorporating selection effects, augmentation, and uncertainty modeling, to support cosmological analyses (e.g. weak lensing, clustering) in LSST and other surveys.

---

## 🚀 Features

- Create ensembles of redshift distributions across tomographic bins for lens and source galaxies
- Incorporate simulation-based augmentation to fill gaps in spectroscopic training sets  
- Propagate redshift uncertainties realistically via probalistic sampling 
- Easily configurable for LSST Y1/ Y4 / Y7 / Y10 settings
- Interoperable with cosmology pipelines

---

## 🏗️ Project Structure

Here’s a typical layout of the repository:
SOMZCloud/
├── DATASET/               # data inputs, training / test sets
├── MODEL/                 # latent / generative models, e.g. VAE, flows
├── scripts/               # scripts to run calibration, mock generation, evaluation
├── notebooks/             # Jupyter notebooks for examples / demos
├── tests/                 # unit tests, validation tests
├── figures/               # plots / visual outputs
├── configs/               # configuration YAML / JSON files
├── LICENSE                # BSD-3-Clause or your chosen license
└── README.md
