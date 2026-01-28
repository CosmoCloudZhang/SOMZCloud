# ANALYZE Pipeline

This folder defines the analysis stage of the pipeline, focused on evaluating the performance of marginal redshift distributions produced by upstream modules. It provides a structured framework for computing diagnostic statistics, error measures, and performance metrics that quantify the accuracy, bias, and uncertainty properties of marginal redshift estimates.

The ANALYZE pipeline operates on outputs from the DATASET, MODEL, COMPARE, CONSTRAIN, and SYNTHESIZE stages, and does not perform modelling, training, or inference itself. Instead, it computes well-defined statistical summaries that characterise the quality of marginal redshift distributions under different assumptions and configurations.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All analysis steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

ANALYZE/
├── Y1/ & Y10 # LSST Year 1 and Year 10 analysis configuration
│ ├── CENTER.py # Central tendency diagnostics
│ ├── WIDTH.py # Distribution width and dispersion metrics
│ ├── EXPECTATION.py # Conditional expectation statistics
│ ├── DEVIATION.py # Bias and deviation measures
│ ├── MARGINAL.py # Marginal distribution-level diagnostics
│ ├── VALUE.py # Aggregated performance metrics
│ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md


## Conceptual Role in the Pipeline

The ANALYZE pipeline quantifies how well marginal redshift distributions reproduce target properties of the underlying redshift population. It translates full-sample redshift distributions into interpretable performance metrics that can be compared across methods, survey depths, and experimental scenarios.

Analysis is performed after modelling and synthesis, and in parallel to conditional assessment. The resulting metrics are intended to support validation, method comparison, and sensitivity studies focused on ensemble redshift properties rather than object-level conditional behaviour.

## Analysis Dimensions

Performance is evaluated along several complementary statistical dimensions.

### Central Tendency

- CENTER  
  Computes measures of central tendency for marginal redshift distributions, such as mean or median redshift, and compares them to reference or truth distributions.

### Dispersion and Width

- WIDTH  
  Quantifies the spread of marginal redshift distributions, including variance, effective width, or credible interval measures. These metrics assess whether the overall redshift uncertainty is under- or over-estimated.

### Expectation Values

- EXPECTATION  
  Evaluates expectation values derived from marginal redshift distributions, enabling direct comparison between predicted and reference ensemble properties.

### Bias and Deviation

- DEVIATION  
  Computes deviations between estimated and reference marginal quantities, including bias measures, residuals, and systematic offsets as a function of redshift.

### Distribution-level Diagnostics

- MARGINAL  
  Assesses properties of the full marginal redshift distributions beyond low-order moments, such as shape agreement, normalisation consistency, or coverage behaviour.

### Aggregated Performance Metrics

- VALUE  
  Combines individual analysis statistics into aggregated performance indicators that summarise overall marginal redshift quality for a given configuration.

## LSST Configurations

The analysis pipeline is parameterised for different survey depths.

- Y1/  
  Computes analysis metrics appropriate for LSST Year 1-like data volumes and uncertainties, suitable for early validation and robustness checks.

- Y10/  
  Computes analysis metrics for LSST Year 10-like data, enabling stringent evaluation of marginal redshift performance at full survey depth.

Each configuration uses independent input paths and output directories to ensure isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including:

- Input paths to marginal redshift distributions and reference quantities  
- Selection of analysis metrics and evaluation ranges  
- Numerical controls and binning choices  
- Parallelisation or batching options  

Scripts are designed to be stateless and restart-safe, enabling selective recomputation of analysis products.

### Shell Scripts

Shell wrappers manage execution details external to the analysis logic. These scripts handle environment activation, computational resource requests such as CPU allocation, memory, and walltime, and parallel execution on HPC systems.

This separation ensures portability across local machines and high-performance computing environments.

## Typical Usage

The analysis pipeline is executed from within a specific configuration directory, such as `Y1` or `Y10`, by running the corresponding shell scripts for the desired analysis metrics. Individual analysis scripts may be run independently depending on which diagnostics are required.

## Reproducibility and Governance

The ANALYZE pipeline is designed with reproducibility and traceability as core principles. All analysis products are generated from explicitly versioned upstream inputs, no hidden global state is shared between scripts, and survey configuration is cleanly separated from analysis logic.

This design ensures that marginal redshift performance metrics can always be traced back to their originating data and modelling assumptions.

## Intended Use

This folder provides the analysis backbone for:

- Performance evaluation of marginal redshift distributions  
- Bias, variance, and uncertainty diagnostics at the population level  
- Cross-method and cross-configuration comparison  
- Inputs to downstream assessment, valuation, and decision frameworks  

It is intended to support controlled, analysis-driven evaluation rather than exploratory or ad hoc diagnostics.
