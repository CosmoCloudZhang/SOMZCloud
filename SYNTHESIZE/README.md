# SYNTHESIZE Pipeline

This folder defines the synthesis stage of the pipeline. It operates on summarised outputs produced by the SUMMARIZE module and is responsible for combining, harmonising, and assembling results into unified synthesis products suitable for downstream valuation, interpretation, and decision-making.

The SYNTHESIZE pipeline does not perform modelling, comparison, or inference. Instead, it consolidates results across different methodological strategies, tracers, and experimental groupings, producing coherent synthesis-level datasets that capture the combined behaviour of the pipeline under well-defined assumptions.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All synthesis steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

SYNTHESIZE/
├── Y1/ & Y10 # LSST Year 1 and Year 10 synthesis configuration
│ ├── DIR.py # Direct synthesis products
│ ├── HYBRID.py # Hybrid synthesis products
│ ├── STACK.py # Stack synthesis products
│ ├── TRUTH.py # Truth-based synthesis products
│ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md

## Conceptual Role in the Pipeline

The SYNTHESIZE pipeline serves as the integration layer between summarised results and final valuation or interpretation stages. It aggregates outputs across:

- Multiple methodological strategies (truth, direct, hybrid, stack)
- Multiple tracer populations and analysis channels
- Multiple experimental groupings or material categories
- Multiple survey configurations (Y1 and Y10)

The resulting synthesis products are designed to be compact, internally consistent, and directly usable for downstream VALUE, INFO, or reporting modules.

## Methodological Strategies

Synthesis is organised explicitly by strategy, with each script producing a distinct class of combined products.

- TRUTH  
  Synthesises results derived from simulation-truth information, serving as idealised or reference-level combinations.

- DIR  
  Synthesises results based on direct, observation-facing quantities without simulation augmentation.

- HYBRID  
  Synthesises results that combine observational inputs with simulation-informed components.

- STACK  
  Synthesises results obtained by stacking or aggregating multiple samples or realisations.

This separation ensures clarity when interpreting synthesis outputs and tracing them back to underlying assumptions.

## Material and Scenario Looping

Each synthesis script operates over a predefined set of experimental groupings (for example `COPPER`, `GOLD`, `IRON`, `SILVER`, `TITANIUM`, `ZINC`) and processes them in a uniform, automated manner. This enables consistent synthesis across heterogeneous result sets while maintaining strict bookkeeping.

The number of realisations or samples to synthesise is configurable via command-line arguments, allowing controlled scaling and sensitivity studies.

## LSST Configurations

The synthesis pipeline is parameterised for different survey depths.

- Y1/  
  Produces synthesis products corresponding to LSST Year 1-like data volumes and uncertainties, suitable for early-stage integration and assessment.

- Y10/  
  Produces synthesis products corresponding to LSST Year 10-like data, enabling full-depth synthesis and stress testing.

Each configuration uses independent input paths and output directories, ensuring isolation between survey scenarios.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including:

- Input paths to summarised results  
- Selection of synthesis strategy and experimental grouping  
- Number of samples or realisations to synthesise  
- Parallelisation and batching controls  

Scripts are designed to be stateless and restart-safe, enabling selective regeneration of synthesis products.

### Shell Scripts

Shell wrappers manage execution details external to synthesis logic. These scripts handle environment activation, computational resource requests such as CPU allocation, memory, and walltime, and parallel execution across experimental groupings using HPC schedulers.

Typical execution involves launching independent synthesis tasks for each grouping within a single job allocation, as shown in the provided SLURM scripts .

## Typical Usage

The synthesis pipeline is executed from within a specific configuration directory, such as `Y1` or `Y10`, by running the corresponding shell scripts for the desired synthesis strategy. Individual synthesis scripts may be run independently depending on which combined products are required.

## Reproducibility and Governance

The SYNTHESIZE pipeline is designed with reproducibility and traceability as core principles. All synthesis products are generated from explicitly versioned summarised inputs, no hidden global state is shared between scripts, and survey configuration is cleanly separated from synthesis logic.

This design ensures that synthesis-level results can always be traced back to their originating summaries and analyses.

## Intended Use

This folder provides the synthesis backbone for:

- Integration of summarised results across methods and tracers  
- Construction of unified, analysis-ready datasets  
- Preparation of inputs for downstream VALUE and INFO modules  
- Transparent aggregation of pipeline outputs under controlled assumptions  

It is not intended to perform analysis or inference, but to assemble and harmonise existing results into coherent synthesis products.
