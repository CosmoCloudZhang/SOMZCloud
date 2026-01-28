# ASSESS Pipeline

This folder defines the assessment stage of the pipeline, focused on evaluating the performance of conditional redshift distributions produced by upstream modules. It provides a structured framework for computing diagnostic statistics, error measures, and performance metrics that quantify the accuracy, bias, and uncertainty properties of conditional redshift estimates.

The ASSESS pipeline operates on outputs from the DATASET, MODEL, COMPARE, CONSTRAIN, and SYNTHESIZE stages, and does not perform modelling, training, or inference itself. Instead, it computes well-defined statistical summaries that characterise the quality of conditional redshift distributions under different assumptions and configurations.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All assessment steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

ASSESS/
├── Y1/ & Y10 # LSST Year 1 and Year 10 assessment configuration
│ ├── CENTER.py # Central tendency diagnostics
│ ├── WIDTH.py # Distribution width and dispersion metrics
│ ├── EXPECTATION.py # Conditional expectation statistics
│ ├── DEVIATION.py # Bias and deviation measures
│ ├── CONDITIONAL.py # Conditional distribution-level diagnostics
│ ├── VALUE.py # Aggregated performance metrics
│ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md

## Conceptual Role in the Pipeline

The ASSESS pipeline quantifies how well conditional redshift distributions reproduce target properties of the underlying redshift–observable relationship. It translates distribution-level outputs into interpretable performance metrics that can be compared across methods, survey depths, and experimental scenarios.

Assessment is performed after modelling and synthesis, and before final valuation or decision-making. The resulting metrics are intended to support method comparison, validation, and sensitivity studies.

## Assessment Dimensions

Performance is evaluated along several complementary statistical dimensions.

### Central Tendency

- CENTER  
  Computes measures of central tendency for conditional redshift distributions, such as conditional means or medians, and compares them to reference or truth values.

### Dispersion and Width

- WIDTH  
  Quantifies the spread of conditional redshift distributions, including measures of variance, credible interval width, or effective dispersion. These metrics assess whether uncertainty is under- or over-estimated.

### Conditional Expectation

- EXPECTATION  
  Evaluates conditional expectation values of redshift given observables, enabling direct comparison between predicted and reference conditional relationships.

### Bias and Deviation

- DEVIATION  
  Computes deviations between estimated and reference quantities, including bias measures, residuals, and systematic offsets as a function of observable space.

### Distribution-level Diagnostics

- CONDITIONAL  
  Assesses properties of the full conditional redshift distributions beyond low-order moments, such as shape consistency or conditional coverage behaviour.

### Aggregated Performance Metrics

- VALUE  
  Combines individual assessment statistics into aggregated performance indicators that summarise overall conditional redshift quality for a given configuration.

## LSST Configurations

The assessment pipeline is parameterised for different survey depths.

- Y1/  
  Computes assessment metrics appropriate for LSST Year 1-like data volumes and uncertainties, suitable for early validation and robustness checks.

- Y10/  
  Computes assessment metrics for LSST Year 10-like data, enabling stringent evaluation of conditional redshift performance at full survey depth.

Each configuration uses independent input paths and output directories to ensure isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including:

- Input paths to conditional redshift distributions and reference quantities  
- Selection of assessment metrics and evaluation ranges  
- Numerical controls and binning choices  
- Parallelisation or batching options  

Scripts are designed to be stateless and restart-safe, enabling selective recomputation of assessment products.

### Shell Scripts

Shell wrappers manage execution details external to the assessment logic. These scripts handle environment activation, computational resource requests such as CPU allocation, memory, and walltime, and parallel execution on HPC systems.

This separation ensures portability across local machines and high-performance computing environments.

## Typical Usage

The assessment pipeline is executed from within a specific configuration directory, such as `Y1` or `Y10`, by running the corresponding shell scripts for the desired assessment metrics. Individual assessment scripts may be run independently depending on which diagnostics are required.

## Reproducibility and Governance

The ASSESS pipeline is designed with reproducibility and traceability as core principles. All assessment products are generated from explicitly versioned upstream inputs, no hidden global state is shared between scripts, and survey configuration is cleanly separated from assessment logic.

This design ensures that conditional redshift performance metrics can always be traced back to their originating data and modelling assumptions.

## Intended Use

This folder provides the assessment backbone for:

- Performance evaluation of conditional redshift distributions  
- Bias, variance, and uncertainty diagnostics  
- Cross-method and cross-configuration comparison  
- Inputs to downstream valuation and decision frameworks  

It is intended to support controlled, analysis-driven evaluation rather than exploratory or ad hoc diagnostics.
