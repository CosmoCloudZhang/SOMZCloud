# COMPARE Pipeline

This folder defines the comparison stage of the pipeline. It operates on model outputs produced by the MODEL module and observational spectroscopic samples derived from the DATASET module, with the purpose of performing controlled, quantitative comparisons between models, data, and configurations.

Unlike the DATASET and MODEL stages, the COMPARE pipeline does not introduce simulated or augmented samples. All comparisons are carried out using observational spectroscopic data, ensuring that evaluation and benchmarking are grounded in directly observed populations.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All comparison steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

COMPARE/
├── Y1/ & Y10 # LSST Year 1 and Year 10 comparison configuration
│ ├── REFERENCE.py # Define reference datasets or baseline comparisons
│ ├── TARGET.py # Define comparison targets and observables
│ ├── COMPARE.py # Perform model–data or model–model comparisons
│ ├── EVALUATE.py # Quantify agreement and discrepancy metrics
│ ├── INFORM.py # Diagnostic summaries and comparison metadata
│ ├── *.sh # Execution scripts and HPC environment control
│
└── README.md

## Conceptual Pipeline

The modelling pipeline follows an ordered sequence of steps. Each stage consumes versioned outputs from the DATASET pipeline and produces explicit intermediate or final model products that are written to disk.

1. `INFORM`  
   Defines the modelling context and bookkeeping for a run, including input dataset references, configuration metadata, and diagnostic summaries used to standardise downstream estimation and evaluation. This stage establishes consistent paths, identifiers, and reporting products for the rest of the modelling workflow.

2. `ESTIMATE`  
   Performs parameter estimation or model fitting under the specified configuration, mapping inputs and modelling assumptions to target quantities. Depending on the analysis setup, this step may involve likelihood evaluation, optimisation, sampling, or other estimation procedures, and writes fitted parameters and associated products to disk.

3. `EVALUATE`  
   Assesses model quality and robustness using validation metrics, residual diagnostics, uncertainty checks, and internal consistency tests. Outputs from this stage are intended to verify that the estimation results meet analysis requirements and to identify failure modes.

4. `TARGET`  
   Specifies the modelling targets and observables used for estimation and evaluation, such as summary statistics, redshift distributions, latent representations, or other derived quantities constructed from dataset products. This stage provides the explicit definitions and selections that the estimation step aims to match.

5. `REFERENCE`  
   Defines reference or baseline models used for comparison and calibration, such as fiducial parameter sets, simulation-derived priors, or externally defined benchmarks. The reference model provides a controlled baseline against which targets, estimates, and evaluation results are interpreted.

## LSST Configurations

The comparison pipeline is parameterised for different survey depths and data regimes.

- Y1/  
  Uses LSST Year 1-like observational spectroscopic samples and model outputs, suitable for early validation and method comparison.

- Y10/  
  Uses LSST Year 10-like data volumes and model outputs, enabling more stringent comparison and sensitivity testing.

Each configuration provides independent configuration files and output directories, dedicated shell scripts for execution and resource management, and isolation of results to avoid cross-configuration interference.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including input model paths, observational spectroscopic data paths, comparison settings, random seeds, and parallelisation or batching options.

Scripts are designed to be stateless, reproducible, and restart-safe, allowing controlled re-runs and systematic comparison studies.

### Shell Scripts

Shell wrappers manage execution details external to the comparison logic. These scripts handle environment activation, computational resource requests such as CPU or GPU allocation, memory, and walltime, and parallel execution strategies on HPC systems.

This separation between comparison logic and execution governance ensures portability across local machines and high-performance computing environments.

## Typical Usage

The comparison pipeline is executed from within a specific configuration directory, such as Y1 or Y10, by running the corresponding shell scripts in sequence. The exact ordering and subset of steps may vary depending on the comparison task and analysis configuration.

## Reproducibility and Governance

The pipeline is designed with reproducibility and auditability as core principles. Deterministic seeding is supported where applicable, all intermediate and final comparison products are written explicitly to disk, no hidden global state is shared between stages, and survey configuration is cleanly separated from comparison logic.

This design supports transparent benchmarking, controlled cross-model comparisons, and robust validation against observational spectroscopic data.

## Intended Use

This folder provides the comparison backbone for:

- Model–data consistency checks using observational spectroscopic samples  
- Cross-model and cross-configuration benchmarking  
- Sensitivity studies across LSST survey depths  
- Preparation of validated comparison products for downstream constraint analyses  

It is intended to support controlled, analysis-driven modelling rather than general-purpose machine-learning workflows.
