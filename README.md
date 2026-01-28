# SOMZCloud: Simulation-Informed Machine-Learning for Redshift Calibration

SOMZCloud is an end-to-end, modular pipeline for photometric redshift calibration, validation, and uncertainty propagation using simulation-informed machine learning. The project is designed for Stage-IV cosmology analyses (e.g. LSST, Euclid), with a focus on population-level redshift distribution calibration, robustness to domain shift between simulations and observations, and reproducibility and traceability for downstream cosmological inference.

The codebase follows a component-based architecture, where each directory implements a well-defined stage of the scientific workflow.

---

## Scientific Scope

SOMZCloud supports:

- Construction and comparison of redshift distributions \(n(z)\)
- Calibration using spectroscopic and simulated reference samples
- Quantification of uncertainty and its propagation into cosmological inference
- Reproducible evaluation of model performance under controlled assumptions

The pipeline is designed to integrate naturally into cosmic shear, galaxy–galaxy lensing, and galaxy clustering analyses.

---

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
├── SYNTHESIZE/     # Synthesis of marginal redshift distributions with uncertainty quantifications
├── VALUE/          # Derived quantities and summary statistics
├── LICENSE         # BSD-3-Clause license text
└── README.md       # Top-level project overview
```
---

## Design Principles

- **Modularity**: Each component can be used independently or composed into pipelines.
- **Reproducibility**: Deterministic configurations and explicitly documented assumptions.
- **Scientific traceability**: Clear mapping from inputs to assumptions to outputs.
- **Survey agnosticism**: Designed to generalise across LSST, Euclid, and simulation suites.
- **Ethical & Responsible ML**: Explicit handling of bias, uncertainty, and limitations.

---

## Getting Started

1. Clone the repository:
   ```
   bash
   git clone git@github.com:CosmoCloudZhang/SOMZCloud.git
   cd SOMZCloud
   ```
2. See DATASET/README.md for data requirements and formats.
3. See MODEL/README.md and CALIBRATE/README.md for training and calibration workflows.
4. Each directory contains a standalone README describing:
   - purpose,
   - inputs/outputs,
   - assumptions,
   - example usage.

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

This project is released under the terms specified in LICENSE.

## Contact

For questions, issues, or collaboration inquiries:

Open a GitHub issue

Or contact the repository maintainer directly
