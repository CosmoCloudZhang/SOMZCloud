# PRIOR Pipeline

This folder defines the prior-construction stage of the pipeline, focused on deriving statistically well-motivated priors for nuisance parameters used in downstream inference and constraint analyses. It provides a structured framework for translating empirical performance diagnostics into prior distributions that encode residual uncertainty and systematic variation.

The PRIOR pipeline operates on outputs from the ANALYZE, ASSESS, CALIBRATE, and SYNTHESIZE stages. It does not perform inference or parameter estimation itself. Instead, it constructs prior distributions that summarise uncertainty in nuisance parameters arising from redshift calibration, modelling assumptions, and systematic effects.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All prior-construction steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

PRIOR/
├── Y1/ & Y10 # LSST Year 1 and Year 10 prior construction
│ ├── EXPECTATION.py # Prior means from ensemble expectations
│ ├── DEVIATION.py # Bias and spread of nuisance parameters
│ ├── COVARIANCE.py # Covariance estimation for nuisance parameters
│ ├── ENSEMBLE.py # Ensemble-based prior construction
│ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md


## Conceptual Role in the Pipeline

The PRIOR pipeline converts empirical performance information into probabilistic priors suitable for cosmological inference. It provides a principled way to propagate calibration and modelling uncertainty into downstream likelihood analyses by encoding uncertainty in nuisance parameters.

Rather than relying on ad hoc or externally imposed priors, this stage derives priors directly from pipeline outputs, ensuring internal consistency between calibration, assessment, and inference.

## Prior Construction Components

Priors are derived using several complementary statistical components.

### Expectation Values

- EXPECTATION  
  Computes ensemble expectation values of nuisance parameters, which define the central values of prior distributions.

### Deviations and Scatter

- DEVIATION  
  Quantifies bias, scatter, and higher-order variation in nuisance parameters across configurations or realisations. These statistics determine prior widths and asymmetries.

### Covariance Structure

- COVARIANCE  
  Estimates covariance matrices between nuisance parameters, capturing correlated uncertainties that must be accounted for in joint inference.

### Ensemble-Based Priors

- ENSEMBLE  
  Combines expectation values, deviations, and covariance information into full prior distributions, potentially multivariate, suitable for direct use in likelihood analyses.

This modular decomposition allows individual components to be inspected, validated, and updated independently.

## LSST Configurations

The prior-construction pipeline is parameterised for different survey depths.

- Y1/  
  Constructs priors appropriate for LSST Year 1-like data, reflecting larger uncertainties and stronger systematic effects.

- Y10/  
  Constructs priors appropriate for LSST Year 10-like data, enabling tighter but still conservative treatment of nuisance parameters.

Each configuration uses independent input paths and output directories to ensure isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including:

- Input paths to assessment and calibration products  
- Selection of nuisance parameters and grouping schemes  
- Statistical assumptions and aggregation controls  
- Output formats for prior products  
- Parallelisation or batching options  

Scripts are designed to be stateless and restart-safe, enabling controlled regeneration of priors under updated assumptions.

### Shell Scripts

Shell wrappers manage execution details external to the prior logic. These scripts handle environment activation, computational resource requests such as CPU allocation, memory, and walltime, and parallel execution on HPC systems.

This separation ensures portability across local machines and high-performance computing environments.

## Typical Usage

The prior pipeline is executed from within a specific configuration directory, such as `Y1` or `Y10`, by running the corresponding shell scripts for expectation, deviation, covariance, and ensemble construction. Individual steps may be run independently depending on the required level of detail.

## Reproducibility and Governance

The PRIOR pipeline is designed with reproducibility and traceability as core principles. All prior products are derived from explicitly versioned upstream inputs, no hidden global state is shared between scripts, and survey configuration is cleanly separated from prior-construction logic.

This design ensures that nuisance parameter priors are transparent, auditable, and scientifically justified.

## Intended Use

This folder provides the prior backbone for:

- Construction of nuisance parameter priors for cosmological inference  
- Propagation of calibration and modelling uncertainty  
- Consistent treatment of correlated systematics  
- Inputs to downstream CONSTRAIN and inference pipelines  

It is intended to support principled, data-informed prior specification rather than ad hoc or externally imposed choices.
