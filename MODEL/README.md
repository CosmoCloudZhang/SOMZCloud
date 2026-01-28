# MODEL Pipeline

This folder defines the modelling stage of the pipeline, operating on datasets produced by the DATASET module. It provides a structured framework for defining reference models, specifying modelling targets, performing parameter estimation, evaluating model performance, and generating diagnostic information for downstream comparison and inference.

The design follows the same principles as the DATASET pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All modelling steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

MODEL/
├── Y1/ & Y10 # LSST Year 1 and Y10 modelling configuration
│ ├── REFERENCE.py # Define reference or baseline models
│ ├── TARGET.py # Define modelling targets and observables
│ ├── ESTIMATE.py # Parameter estimation (model fitting)
│ ├── EVALUATE.py # Model validation and performance metrics
│ ├── INFORM.py # Diagnostic summaries and metadata products
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

The modelling pipeline is parameterised for different survey depths and data regimes.

- Y1/  
  LSST Year 1-like data volumes and uncertainties, suitable for early validation, stress testing, and methodological development.

- Y10/  
  LSST Year 10-like data, enabling high-precision modelling and sensitivity studies.

Each configuration provides independent configuration files and output directories, dedicated shell scripts for execution and resource management, and isolation of results to avoid cross-configuration interference.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including input and output paths, model and hyperparameter settings, random seeds and numerical controls, and parallelisation or batching options.

Scripts are designed to be stateless, reproducible, and restart-safe, allowing partial re-runs and controlled experimentation.

### Shell Scripts

Shell wrappers manage execution details external to the modelling logic. These scripts handle environment activation, such as Conda or module-based setups, computational resource requests including CPU or GPU allocation, memory, and walltime, and parallel execution strategies on HPC systems.

This separation between modelling logic and execution governance ensures portability across local machines and high-performance computing environments.

## Typical Usage

The modelling pipeline is executed from within a specific configuration directory, such as Y1 or Y10, by running the corresponding shell scripts in sequence. Explicit command examples are omitted here, as the precise ordering and subset of steps may vary depending on the modelling task and analysis configuration.

## Reproducibility and Governance

The pipeline is designed with reproducibility and auditability as core principles. Deterministic seeding is supported throughout, intermediate products are written explicitly to disk, no hidden global state is shared between stages, and survey configuration is cleanly separated from modelling logic.

This design supports reproducible modelling, controlled ablation studies, and transparent validation.

## Intended Use

This folder provides the modelling backbone for simulation-informed model construction, parameter estimation and validation, performance benchmarking across survey depths, and preparation of inputs for comparison and cosmological constraint stages.

It is intended to support controlled, analysis-driven modelling rather than general-purpose machine-learning workflows.

