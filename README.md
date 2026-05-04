# SOMZCloud: Simulation-Informed Machine Learning for Redshift Calibration

SOMZCloud is a modular, end-to-end pipeline for photometric redshift calibration, validation, and uncertainty propagation using simulation-informed machine-learning methods. It targets Stage-IV cosmology use cases (LSST, Euclid, Roman), with emphasis on population-level redshift distribution calibration, robustness to simulation–data domain shift, and reproducible traceability for downstream inference.

Each top-level directory is a self-contained stage that can be run alone or composed into a full workflow. Stage-specific conventions, scripts, and SLURM wrappers are documented in that folder’s `README.md`.

## Scientific scope

- Construction, validation, and comparison of ensemble redshift distributions  
- Calibration using spectroscopic and simulation-based reference samples  
- Uncertainty quantification and propagation toward cosmological observables  
- Controlled evaluation of performance under systematic and configuration changes

The design aligns naturally with cosmic shear, galaxy–galaxy lensing, and clustering analyses that rely on well-characterised redshift distributions.

## Repository layout

```
SOMZCloud/
├── DATASET/      # Catalogue construction, observing realism, SOM, augmentation
├── MODEL/        # Fiducial ML training, estimation, and evaluation
├── COMPARE/      # Model vs observational spectroscopy (no simulation augmentation)
├── CONSTRAIN/    # Stress tests and bounds using simulation-only (incl. augmentation)
├── FIGURE/       # Publication-style figures from upstream products
├── SUMMARIZE/    # Aggregation of model/compare/constraint outputs by tracer & strategy
├── SYNTHESIZE/   # Unified marginal products across strategies and materials
├── ANALYZE/      # Diagnostics and metrics for marginal redshift distributions
├── ASSESS/       # Diagnostics and metrics for conditional redshift distributions
├── CORRECT/      # Explicit corrections: shift, scale, and shape
├── PRIOR/        # Nuisance priors from ensemble statistics
├── INFO/         # Survey, galaxy, lensing, and cosmology configuration helpers
├── LOG/          # Runtime / job logging (optional convention)
├── LICENSE
└── README.md
```

## End-to-end workflow (recommended order)

1. **DATASET** — Build LSST-like catalogues, SOM cells, selections, augmentation, and association.
2. **MODEL** — Train and evaluate photometric models; produce point estimates and intermediates.
3. **COMPARE** / **CONSTRAIN** — Benchmark against spec samples (COMPARE) or simulation-only stress cases (CONSTRAIN).
4. **FIGURE** — Visual QA and paper figures from dataset and model outputs.
5. **SUMMARIZE** — Collapse results by material (e.g. COPPER, GOLD, …), tracer (lens/source), and strategy (truth, direct, hybrid, stack).
6. **SYNTHESIZE** — Merge summaries into analysis-ready marginal ensembles.
7. **ANALYZE** / **ASSESS** — Quantify marginal vs conditional distribution quality.
8. **CORRECT** — Apply transparent **SHIFT** (additive), **SCALE** (multiplicative), and **SHAPE** (combined) corrections informed by those diagnostics.
9. **PRIOR** — Turn expectations, deviations, covariances, and ensembles (including CORRECT outputs) into nuisance priors for inference.
10. **INFO** — Cosmology and survey metadata for any downstream likelihood or forecasting code.

Each stage writes explicit, path-stable products; rerun order is flexible when upstream artefacts are frozen.

## Design principles

- **Modularity** — Stages are loosely coupled via filesystem contracts.  
- **Reproducibility** — Argparse-driven scripts, documented paths, seeds where applicable.  
- **Traceability** — Clear mapping from inputs and assumptions to figures and priors.  
- **Survey agnosticism** — Y1 and Y10 layouts mirror different depth/volume regimes.  
- **Responsible ML** — Uncertainty and limitations are first-class, not afterthoughts.

## Prerequisites

- Python 3 with standard scientific packages used across stages (`numpy`, `scipy`, `h5py`, `matplotlib`, `yaml`, `sklearn`).  
- Access to stage-specific environments and tools referenced in shell wrappers (for example, Conda envs and SLURM on HPC systems).  
- Read/write permissions for the configured data roots (code tree plus product/output trees).
- External astronomy dependencies where required by stage scripts (for example `GCRCatalogs`, `rail`, `ceci`, `pyccl`, `photerr`).

## Getting started

1. Clone the repository:

   ```bash
   git clone git@github.com:CosmoCloudZhang/SOMZCloud.git
   cd SOMZCloud
   ```
2. Follow **DATASET** for catalogue generation and augmentation.
3. Use **MODEL**, **COMPARE**, and **CONSTRAIN** for training and controlled benchmarks.
4. Use **FIGURE** for diagnostic and publication plots.
5. Run **SUMMARIZE** then **SYNTHESIZE** to build marginal ensembles.
6. Run **ANALYZE** and **ASSESS** for population-level quality metrics.
7. Run **CORRECT** (`SHIFT` → `SCALE` → `SHAPE` as needed), then **PRIOR** for nuisance priors.
8. Point external cosmology pipelines at **PRIOR** outputs and **INFO** configuration.

Environment activation, modules, and SLURM directives are stage-specific; see each `README.md` and the accompanying `*.sh` scripts.

## Minimal quickstart (Y1 example)

Use these commands as a lightweight entry point before running full index sweeps:

```bash
cd DATASET/Y1 && python OBSERVE.py --help
cd ../.. && cd MODEL/Y1 && python ESTIMATE.py --help
cd ../.. && cd SUMMARIZE/Y1/COPPER && python TRUTH_SOURCE.py --help
cd ../../.. && cd SYNTHESIZE/Y1 && python TRUTH.py --help
```

The exact required arguments are stage-specific and documented by each script's CLI help and local `README.md`.

## Stage input/output contracts
| Stage          | Primary inputs                                | Primary outputs                                                |
| -------------- | --------------------------------------------- | -------------------------------------------------------------- |
| **DATASET**    | Source catalogues and survey configuration    | Processed catalogues, SOM mappings, selected/augmented samples |
| **MODEL**      | DATASET products                              | Trained estimators, predictions, evaluation artefacts          |
| **COMPARE**    | MODEL outputs and spectroscopic references    | Observation-anchored agreement metrics                         |
| **CONSTRAIN**  | MODEL outputs and simulation-only references  | Simulation stress-test metrics and bounds                      |
| **FIGURE**     | Upstream numerical products                   | QA and publication figures                                     |
| **SUMMARIZE**  | MODEL/COMPARE/CONSTRAIN outputs               | Material/tracer/strategy summaries                             |
| **SYNTHESIZE** | SUMMARIZE outputs                             | Unified marginal ensemble products                             |
| **ANALYZE**    | SYNTHESIZE products                           | Marginal quality diagnostics                                   |
| **ASSESS**     | SYNTHESIZE products                           | Conditional quality diagnostics                                |
| **CORRECT**    | ANALYZE/ASSESS diagnostics plus distributions | SHIFT/SCALE/SHAPE-corrected products                           |
| **PRIOR**      | SYNTHESIZE/ANALYZE/ASSESS/CORRECT products    | Nuisance-parameter prior artefacts                             |
| **INFO**       | Cosmology/survey assumptions                  | Configuration helper files for downstream inference            |
## Glossary

- **Material**: Named scenario grouping (for example COPPER, GOLD, ZINC) used for controlled comparisons.
- **Strategy**: Combination pathway (`TRUTH`, `DIR`, `HYBRID`, `STACK`) used in summary and synthesis stages.
- **Tracer**: Lens or source population branch processed separately.
- **Y1 / Y10**: Parallel survey-depth regimes; maintain separate trees for reproducibility.

## Citation

If you use SOMZCloud in a publication, please cite:

```bibtex
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

This project is distributed under the BSD 3-Clause License. Copyright © 2025 Yun-Hao Zhang. See `LICENSE` for full terms.

## Data availability

Data products referenced by the pipeline are stored on the NERSC Community File System (CFS). Access depends on survey collaboration policies, data-provider terms, and your NERSC allocation.

## Contact

Open a GitHub issue or contact the repository maintainer for questions and collaboration.
