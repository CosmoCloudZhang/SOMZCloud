# FIGURE Pipeline

This folder defines the figure-generation and visualisation stage of the pipeline. It is responsible for producing diagnostic plots, summary figures, and presentation-ready visualisations from the outputs of the DATASET, MODEL, COMPARE, and CONSTRAIN modules.

The FIGURE pipeline does not perform modelling, comparison, or inference. Instead, it provides a structured and reproducible framework for rendering results, validating assumptions, and communicating outcomes across different LSST configurations.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All figure-generation steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

FIGURE/
├── Y1/ # LSST Year 1 figure configuration
│ ├── BASELINE.py # Baseline visualisations and reference figures
│ ├── BENCHMARK.py # Benchmark and comparison figures
│ ├── CATALOG.py # Catalogue-level diagnostic plots
│ ├── CONTRAST.py # Contrast and difference visualisations
│ ├── CONTROL.py # Control and sanity-check figures
│ ├── DIAGRAM.py # Schematic and workflow diagrams
│ ├── HISTOGRAM.py # Distribution and histogram plots
│ ├── MAP.py # Spatial or sky-projected visualisations
│ ├── METRIC.py # Metric-based diagnostic plots
│ ├── OPTICAL.py # Optical-band specific visualisations
│ ├── INFRARED.py # Infrared-band specific visualisations
│ ├── QUANTILE.py # Quantile and percentile-based plots
│ ├── REDSHIFT.py # Redshift-dependent visualisations
│ ├── REGULATE.py # Regularisation and smoothing diagnostics
│ ├── RESTRAIN.py # Constraint and bound visualisations
│ ├── SAMPLE.py # Sample-level inspection plots
│ ├── SOM.py # SOM-related visualisations
│ ├── *.sh # Execution scripts and HPC environment control
│
├── Y10/ # LSST Year 10 figure configuration
│ ├── (mirrors Y1/)
│
└── README.md


## Conceptual Role in the Pipeline

The FIGURE pipeline operates as a downstream consumer of all major pipeline products. Its role is to translate numerical outputs into interpretable visual representations that support:

- Validation and debugging  
- Method comparison and benchmarking  
- Sensitivity and robustness assessment  
- Communication of results in papers, talks, and reports  

All figures are generated deterministically from versioned inputs, ensuring consistency between numerical results and visual outputs.

## LSST Configurations

Figure generation is parameterised for different LSST survey depths and data regimes.

- Y1/  
  Produces figures corresponding to LSST Year 1-like data volumes and uncertainties, suitable for early validation, diagnostics, and method development.

- Y10/  
  Produces figures corresponding to LSST Year 10-like data, enabling high-precision visualisation and stress testing at full survey depth.

Each configuration uses independent input paths and output directories, ensuring isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script generates a specific class of figures and uses argparse to expose configuration options, including:

- Input paths to dataset, model, comparison, or constraint products  
- Plot configuration parameters and styling options  
- Random seeds where applicable  
- Parallelisation or batching controls  

Scripts are designed to be stateless and restart-safe, enabling selective regeneration of figures.

### Shell Scripts

Shell wrappers manage execution details external to plotting logic. These scripts handle environment activation, computational resource requests, and parallel execution on HPC systems.

This separation ensures portability across local machines and high-performance computing environments, even for large-scale figure generation.

## Typical Usage

The figure pipeline is executed from within a specific configuration directory, such as Y1 or Y10, by running the corresponding shell scripts for the desired figure types. Individual scripts may be run independently depending on which diagnostics or plots are required.

## Reproducibility and Governance

The FIGURE pipeline is designed with reproducibility and traceability as core principles. All figures are generated from explicitly versioned inputs, no hidden global state is shared between scripts, and survey configuration is cleanly separated from plotting logic.

This design ensures that published figures can always be regenerated from the underlying data products.

## Intended Use

This folder provides the visualisation backbone for:

- Diagnostic inspection of intermediate and final products  
- Comparison and validation across pipeline stages  
- Generation of publication- and presentation-ready figures  
- Transparent communication of modelling and inference results  

It is not intended to perform analysis or inference, but to faithfully represent the outputs of upstream pipeline components.
