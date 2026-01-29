# SOMZCloud: Simulation-Informed Machine-Learning for Redshift Calibration

SOMZCloud is an end-to-end, modular pipeline for photometric redshift calibration, validation, and uncertainty propagation based on simulation-informed machine-learning methods. The project is designed to support Stage-IV cosmology analyses, including LSST, Euclid, and Roman, with an emphasis on population-level redshift distribution calibration, robustness to domain shift between simulations and observations, and reproducibility and scientific traceability for downstream cosmological inference.

The codebase adopts a component-based architecture, in which each directory corresponds to a well-defined stage of the scientific workflow and can be used independently or composed into an end-to-end analysis pipeline.

## Scientific Scope

SOMZCloud supports:

- Construction, validation, and comparison of ensemble redshift distributions
- Calibration using spectroscopic and simulation-based reference samples
- Uncertainty quantification and its propagation into cosmological observables
- Reproducible evaluation of model performance under controlled systematic variations

The pipeline is designed to integrate naturally into analyses of cosmic shear, galaxy–galaxy lensing, and galaxy clustering.

## Repository Structure

Each top-level directory corresponds to a logical component of the pipeline. Detailed documentation is provided in a dedicated `README.md` within each folder.

```
SOMZCloud/
├── ANALYZE/        # Diagnostic analysis and validation metrics for conditional redshift distributions
├── ASSESS/         # Quality assessment and performance evaluation for marginal redshift distributions
├── CALIBRATE/      # Redshift calibration methods and workflows
├── CELL/           # Investigate impact on summary statistics of angular power spectra
├── COMPARE/        # Comparison model without augmentation training and estimation 
├── CONSTRAIN/      # Constrain model with full augmentation training and estimation
├── DATASET/        # Dataset definitions, loaders, and metadata
├── FIGURE/         # Figure generation and plotting scripts
├── INFO/           # Configuration, metadata, and run information
├── MODEL/          # Fiducial machine-learning models and training logic
├── PRIOR/          # Prior assumptions and population models
├── SUMMARIZE/      # Summarisation for conditional redshift distributions
├── SYNTHESIZE/     # Synthesis of marginal redshift distributions with uncertainty quantification
├── VALUE/          # Derived quantities and summary statistics
├── LICENSE         # BSD-3-Clause license text
└── README.md       # Top-level project overview
```

## Design Principles

- **Modularity**: Each component can be used independently or composed into end-to-end analysis pipelines.
- **Reproducibility**: Deterministic configurations with explicitly documented assumptions and processing choices.
- **Scientific traceability**: Clear and auditable mapping from inputs, through assumptions and methods, to final outputs.
- **Survey agnosticism**: Designed to generalise across LSST, Euclid, Roman, and a range of simulation suites.
- **Ethical and responsible machine learning**: Explicit treatment of bias, uncertainty, and known methodological limitations.

## Getting Started

1. Clone the repository:

   ```
   bash
   git clone git@github.com:CosmoCloudZhang/SOMZCloud.git
   cd SOMZCloud
   ```
2. Data reduction, catalogue manipulation and initial processing in `DATASET`.
3. Train machine-learning models under different configurations and conduct photometric redshift point estimates using `MODEL`, `COMPARE`, and `CONSTRAIN`.
4. Generate diagnostic plots for photometric redshift point estimates using `FIGURE`.
5. Construct conditional redshift distributions using `SUMMARIZE`.
6. Derive marginal redshift distributions and systematic uncertainties using `SYNTHESIZE`.
7. Evaluate ensemble redshift distribution properties and performance metrics using `ANALYZE` and `ASSESS`.
8. Correct residual systematic biases and define priors on nuisance parameters using `CALIBRATE` and `PRIOR`.
9. Propagate redshift distribution uncertainties to angular power spectra at `CELL` using configuration and metadata defined in `INFO`.
10. Quantify the impact of redshift uncertainties on cosmological parameter inference using `VALUE`.

## Citation

If you use SOMZCloud in a publication, please cite:

```
@ARTICLE{2025MNRAS.tmp.2117Z,
       author = {{Zhang}, Yun-Hao and {Zuntz}, Joe and {Moskowitz}, Irene and {Gawiser}, Eric and {Kuijken}, Konrad and {Asgari}, Marika and {Hoekstra}, Henk and {Malz}, Alex I. and {Yan}, Ziang and {Zhang}, Tianqing},
        title = "{Improved photometric redshift estimations through self-organising map-based data augmentation}",
      journal = {\mnras},
     keywords = {Astrophysics of Galaxies, Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = dec,
          doi = {10.1093/mnras/staf2226},
archivePrefix = {arXiv},
       eprint = {2508.20903},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.tmp.2117Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License

This project is distributed under the BSD 3-Clause License.

Copyright © 2025 Yun-Hao Zhang. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions set out in the `LICENSE` file are met. The software is provided “as is”, without warranty of any kind, express or implied. See the `LICENSE` file in the root of this repository for the full license text.

## Data Availability

Data products used or generated by SOMZCloud are stored on the National Energy Research Scientific Computing Center (NERSC) Community File System (CFS).

Access conditions and permitted usage are governed by the policies of the original data providers, relevant survey collaborations, and NERSC allocation agreements. Availability of specific datasets may therefore vary depending on user affiliation and authorization.

## Contact

For questions, issues, or collaboration inquiries:

Open a GitHub issue

Or contact the repository maintainer directly
