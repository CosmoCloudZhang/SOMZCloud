# CALIBRATE Pipeline

This folder defines the calibration stage of the pipeline, focused on correcting residual biases in redshift distributions using multiple parameterisations. It provides a structured framework for applying controlled transformations to marginal and conditional redshift distributions in order to mitigate systematic offsets identified in upstream analysis and assessment stages.

The CALIBRATE pipeline operates on outputs from the ANALYZE and ASSESS modules and does not perform modelling, training, or inference itself. Instead, it applies explicit, parameterised calibration mappings designed to correct residual biases while preserving the statistical structure of the distributions.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All calibration steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

CALIBRATE/
├── Y1/ & Y10 # LSST Year 1 and Year 10 calibration configuration
│ ├── SHIFT.py # Additive bias correction
│ ├── SCALE.py # Multiplicative rescaling correction
│ ├── CORRECT.py # Combined or general calibration mapping
│ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md


## Conceptual Role in the Pipeline

The CALIBRATE pipeline bridges performance evaluation and downstream valuation. It takes diagnostic information about bias, dispersion, and deviation from the ANALYZE and ASSESS stages and applies controlled corrections to redshift distributions.

Calibration is performed explicitly and transparently, using simple but interpretable parameterisations. This ensures that any bias mitigation remains auditable, reproducible, and separable from modelling assumptions.

## Calibration Parameterisations

Residual biases are corrected using several complementary parameterisations.

### Additive Shift

- SHIFT  
  Applies additive corrections to redshift distributions, targeting systematic offsets in central tendency (for example, mean or median redshift bias).

### Multiplicative Scaling

- SCALE  
  Applies multiplicative rescaling to redshift distributions, correcting biases in width, dispersion, or effective uncertainty.

### General Correction Mapping

- CORRECT  
  Applies combined or more flexible correction mappings that incorporate both shift and scale effects, or other low-dimensional calibration parameters.

These parameterisations are designed to be minimal yet expressive, allowing systematic effects to be corrected without introducing unnecessary complexity.

## LSST Configurations

The calibration pipeline is parameterised for different survey depths.

- Y1/  
  Applies calibration appropriate for LSST Year 1-like data volumes and uncertainties, focusing on robust bias mitigation under limited data conditions.

- Y10/  
  Applies calibration for LSST Year 10-like data, enabling more precise correction of smaller residual systematics at full survey depth.

Each configuration uses independent input paths and output directories to ensure isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including:

- Input paths to redshift distributions and assessment diagnostics  
- Choice of calibration parameterisation  
- Calibration parameters and bounds  
- Numerical controls and binning choices  
- Parallelisation or batching options  

Scripts are designed to be stateless and restart-safe, enabling controlled experimentation with different calibration strategies.

### Shell Scripts

Shell wrappers manage execution details external to the calibration logic. These scripts handle environment activation, computational resource requests such as CPU allocation, memory, and walltime, and parallel execution on HPC systems.

This separation ensures portability across local machines and high-performance computing environments.

## Typical Usage

The calibration pipeline is executed from within a specific configuration directory, such as `Y1` or `Y10`, by running the corresponding shell scripts for the desired calibration parameterisation. Individual calibration scripts may be run independently depending on which corrections are required.

## Reproducibility and Governance

The CALIBRATE pipeline is designed with reproducibility and traceability as core principles. All calibration products are generated from explicitly versioned upstream inputs, no hidden global state is shared between scripts, and calibration logic is cleanly separated from analysis and assessment stages.

This design ensures that any applied bias corrections can be fully traced, reproduced, and justified.

## Intended Use

This folder provides the calibration backbone for:

- Mitigation of residual redshift biases  
- Controlled correction of marginal and conditional distributions  
- Sensitivity studies over calibration parameterisations  
- Preparation of calibrated inputs for downstream VALUE and constraint analyses  

It is intended to support transparent, analysis-driven bias correction rather than opaque or data-driven re-training.