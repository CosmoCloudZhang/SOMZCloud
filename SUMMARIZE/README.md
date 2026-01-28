# SUMMARIZE Pipeline

This folder defines the summarisation stage of the pipeline. It operates on outputs produced by the ANALYZE, MODEL, COMPARE, and CONSTRAIN modules, and is responsible for aggregating, condensing, and organising results into structured summary products.

The SUMMARIZE pipeline does not perform modelling, comparison, or inference. Instead, it collects results across multiple configurations, tracers, and methodologies, and produces coherent summary representations that facilitate interpretation, reporting, and downstream valuation.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All summarisation steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

SUMMARIZE/
├── Y1/ & Y10/ # LSST Year 1 & Year 10 summarisation
│ ├── COPPER/
│ ├── GOLD/
│ ├── IRON/
│ ├── SILVER/
│ ├── TITANIUM/
│ ├── ZINC/
│ │ ├── DIR_LENS.py # Direct lens summarisation
│ │ ├── DIR_SOURCE.py # Direct source summarisation
│ │ ├── HYBRID_LENS.py # Hybrid lens summarisation
│ │ ├── HYBRID_SOURCE.py # Hybrid source summarisation
│ │ ├── STACK_LENS.py # Stack-based lens summarisation
│ │ ├── STACK_SOURCE.py # Stack-based source summarisation
│ │ ├── TRUTH_LENS.py # Truth lens summarisation
│ │ ├── TRUTH_SOURCE.py # Truth source summarisation
│ │ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md

## Conceptual Role in the Pipeline

The SUMMARIZE pipeline serves as the aggregation layer between detailed analysis outputs and high-level interpretation. It collects results across:

- Different modelling or inference strategies (truth, direct, hybrid, stack)
- Different tracer types (lens and source)
- Different survey configurations (Y1 and Y10)
- Different experimental groupings or scenarios

The resulting summary products are designed to be compact, interpretable, and directly usable for reporting, comparison across scenarios, and downstream valuation or decision-making.

## Methodological Dimensions

Summarisation is organised along several explicit methodological axes.

### Strategy

- Truth  
  Summaries based on idealised or simulation-truth information.

- Direct  
  Summaries derived directly from observationally accessible quantities.

- Hybrid  
  Summaries combining simulation-informed and observational components.

- Stack  
  Summaries constructed by aggregating multiple samples or realisations.

### Tracer Type

- Lens  
  Summaries related to foreground or lens galaxy populations.

- Source  
  Summaries related to background or source galaxy populations.

This explicit separation ensures clarity when interpreting aggregated results and comparing methodological assumptions.

## LSST Configurations

The summarisation pipeline is parameterised for different survey depths.

- Y1/  
  Produces summaries corresponding to LSST Year 1-like data volumes and uncertainties, suitable for early-stage assessment and validation.

- Y10/  
  Produces summaries corresponding to LSST Year 10-like data, enabling full-depth aggregation and sensitivity studies.

Each configuration uses independent input paths and output directories, ensuring isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script aggregates and condenses a specific subset of results and uses argparse to expose configuration options, including:

- Input paths to analysis, model, comparison, or constraint products  
- Selection of tracer type and summarisation strategy  
- Numerical and aggregation controls  
- Parallelisation or batching options  

Scripts are designed to be stateless and restart-safe, enabling selective regeneration of summary products.

### Shell Scripts

Shell wrappers manage execution details external to summarisation logic. These scripts handle environment activation, computational resource requests such as CPU or GPU allocation, memory, and walltime, and parallel execution strategies on HPC systems.

This separation ensures portability across local machines and high-performance computing environments.

## Typical Usage

The summarisation pipeline is executed from within a specific configuration and material directory, such as `Y1/TITANIUM` or `Y10/GOLD`, by running the corresponding shell scripts for the desired tracer and strategy. Individual summarisation scripts may be run independently depending on the required outputs.

## Reproducibility and Governance

The SUMMARIZE pipeline is designed with reproducibility and traceability as core principles. All summary products are generated from explicitly versioned upstream inputs, no hidden global state is shared between scripts, and survey configuration is cleanly separated from summarisation logic.

This design ensures that high-level summaries can always be traced back to their originating analyses.

## Intended Use

This folder provides the summarisation backbone for:

- Aggregation of results across methods and tracers  
- Condensation of large-scale outputs into interpretable summaries  
- Cross-scenario and cross-configuration synthesis  
- Preparation of inputs for downstream VALUE and INFO modules  

It is not intended to perform analysis, modelling, or inference, but to distil existing results into structured, decision-ready representations.
