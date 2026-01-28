# CELL Pipeline

This folder defines the **CELL** stage of the pipeline, responsible for computing angular power spectra using ensembles of redshift distributions in order to propagate redshift uncertainty into two-point statistics. It provides a controlled and modular framework for translating uncertainty in redshift distributions into uncertainty in angular clustering and lensing observables.

The CELL pipeline operates strictly downstream of redshift modelling, synthesis, assessment, and calibration stages. It does not infer redshift distributions or cosmological parameters itself. Instead, it takes realizations of ensemble redshift distributions as inputs and evaluates their impact on angular power spectra, enabling uncertainty propagation, sensitivity analysis, and robustness checks for cosmological inference.

The design follows the same principles as the rest of the pipeline: **modular, configuration-driven, reproducible, and HPC-ready**, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All CELL computations are implemented as standalone Python scripts controlled via `argparse`, with corresponding shell scripts used to manage execution environments and computational resources.

## High-level Structure

```text
CELL/
├── Y1/ & Y10/            # LSST Year 1 and Year 10 configurations
│   ├── COVARIANCE/       # Covariance construction from ensemble realisations
│   │   ├── DATA.py       # Power spectrum data vector generation
│   │   ├── DATA.sh
│   │
│   ├── NN/               # Nearest-neighbour or network-based ensembles
│   ├── NS/               # Nuisance-sampling based ensembles
│   ├── SS/               # Synthetic / stress-test ensembles
│   │   ├── CORRECT.py    # Bias-corrected redshift realisations
│   │   ├── SCALE.py      # Amplitude scaling parameterisations
│   │   ├── SHIFT.py      # Mean-shift parameterisations
│   │   ├── *.sh
│
└── README.md
```

## Conceptual Role in the Pipeline

The CELL pipeline connects **redshift uncertainty** to **cosmological observables**.

Rather than evaluating angular power spectra for a single best-estimate redshift distribution, CELL computes spectra across **ensembles of redshift realisations**, allowing:

* Propagation of redshift uncertainty into angular power spectra ( C_\ell )
* Construction of redshift-induced covariance contributions
* Sensitivity testing under extreme or adversarial redshift scenarios
* Validation of calibration and marginalisation strategies

This stage provides the quantitative bridge between redshift modelling and downstream cosmological constraint pipelines.

## Ensemble-based Power Spectrum Computation

### Redshift Ensembles

The pipeline supports multiple ensemble construction strategies, including:

* **NN/**
  Ensembles derived from nearest-neighbour or learned latent representations.

* **NS/**
  Ensembles generated via nuisance-parameter sampling from derived priors.

* **SS/**
  Synthetic or stress-test ensembles designed to bracket worst-case or adversarial redshift configurations, including:

  * Mean shifts
  * Width rescalings
  * Residual bias corrections

Each ensemble produces a set of redshift distributions that are passed independently through power spectrum evaluation.

### Angular Power Spectra

For each redshift realisation, the CELL pipeline computes angular power spectra for the relevant probes, such as:

* Galaxy clustering
* Galaxy–galaxy lensing
* Cosmic shear
* Cross-correlations between tomographic bins

The resulting ensemble of ( C_\ell ) vectors enables direct estimation of:

* Mean power spectra
* Redshift-induced variance
* Cross-bin covariance contributions
* Sensitivity of observables to redshift systematics

## Covariance Construction

The `COVARIANCE/` submodule aggregates ensemble power spectra into covariance estimates that capture uncertainty induced by redshift distributions alone.

These covariance products are intended to be:

* Combined with shape noise and sample variance
* Used in likelihood analyses or Fisher forecasts
* Compared across redshift modelling strategies

## LSST Configurations

The CELL pipeline is parameterised for different survey depths and statistical regimes.

* **Y1/**
  Uses LSST Year 1–like ensemble sizes, binning schemes, and noise levels, suitable for early-data validation and robustness tests.

* **Y10/**
  Uses LSST Year 10–like ensemble statistics and resolution, enabling stringent uncertainty propagation for final survey analyses.

Configurations are fully isolated to prevent cross-contamination of results.

## Execution Model

### Python Scripts

Each Python script uses `argparse` to expose:

* Input paths to ensemble redshift distributions
* Cosmological and tracer configuration options
* Multipole ranges and binning schemes
* Numerical and performance controls

Scripts are stateless and restart-safe, allowing selective recomputation of power spectra or covariance components.

### Shell Scripts

Shell wrappers manage:

* Environment activation
* CPU/GPU resource requests
* Memory and walltime constraints
* Parallel execution across ensemble members

This separation ensures portability across laptops, clusters, and HPC systems.

## Reproducibility and Governance

The CELL pipeline is designed for full auditability:

* All power spectra are traceable to explicit redshift realisations
* Ensemble definitions are versioned and configuration-driven
* No hidden global state or implicit assumptions are used
* Outputs are written explicitly to disk for downstream use

This ensures that uncertainty propagation can be reproduced, inspected, and justified in analysis reviews.

## Intended Use

This folder provides the angular-power-spectrum backbone for:

* Propagating redshift uncertainties into cosmological observables
* Constructing redshift-induced covariance contributions
* Stress-testing cosmological analyses under worst-case scenarios
* Supporting robust marginalisation over redshift systematics

It is intended to support **controlled, analysis-driven cosmological inference** rather than exploratory or ad hoc power spectrum calculations.