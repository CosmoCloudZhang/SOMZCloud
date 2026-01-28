# CONSTRAIN Pipeline

This folder defines the constraint stage of the pipeline. It operates on model outputs produced by the MODEL module and simulated datasets constructed in the DATASET module, with the purpose of quantifying parameter constraints and bounding systematic effects under controlled, worst-case scenarios.

Unlike the COMPARE stage, the CONSTRAIN pipeline does not use any observational spectroscopic samples. All training and inference are performed exclusively on simulated data products, including simulation-based augmentation where applicable. This design allows the pipeline to probe limiting cases and to benchmark the maximum systematic biases that may arise in the absence of observational anchoring.

The design follows the same principles as the rest of the pipeline: modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) scenarios.

All constraint steps are implemented as standalone Python scripts controlled via argparse, with corresponding shell scripts used to manage execution environments, parallelisation, and computational resource allocation.

## High-level Structure

CONSTRAIN/
├── Y1/ # LSST Year 1 constraint configuration
│ ├── REFERENCE.py # Define reference simulations and fiducial scenarios
│ ├── TARGET.py # Define constraint targets and parameters of interest
│ ├── CONSTRAIN.py # Perform constraint estimation and bias propagation
│ ├── EVALUATE.py # Quantify bias, uncertainty, and robustness metrics
│ ├── INFORM.py # Diagnostic summaries and constraint metadata
│ ├── *.sh # Execution scripts and HPC environment control
│
├── Y10/ # LSST Year 10 constraint configuration
│ ├── REFERENCE.py
│ ├── TARGET.py
│ ├── CONSTRAIN.py
│ ├── EVALUATE.py
│ ├── INFORM.py
│ ├── *.sh
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

The constraint pipeline is parameterised for different survey depths and simulation regimes.

- Y1/  
  Uses LSST Year 1-like simulated data volumes and noise properties, enabling early-stage constraint forecasts seen as conservative or pessimistic scenarios.

- Y10/  
  Uses LSST Year 10-like simulated data, enabling high-precision constraint forecasts and stress testing of systematic effects at full survey depth.

Each configuration provides independent configuration files and output directories, dedicated shell scripts for execution and resource management, and isolation of results to avoid cross-configuration interference.

## Execution Model

### Python Scripts

Each Python script uses argparse to expose configuration options, including simulated data paths, model outputs, parameter definitions, random seeds, and parallelisation or batching options.

Scripts are designed to be stateless, reproducible, and restart-safe, enabling controlled re-runs and systematic exploration of systematic uncertainties.

### Shell Scripts

Shell wrappers manage execution details external to the constraint logic. These scripts handle environment activation, computational resource requests such as CPU or GPU allocation, memory, and walltime, and parallel execution strategies on HPC systems.

This separation between constraint logic and execution governance ensures portability across local machines and high-performance computing environments.

## Typical Usage

The constraint pipeline is executed from within a specific configuration directory, such as Y1 or Y10, by running the corresponding shell scripts in sequence. The exact ordering and subset of steps may vary depending on the constraint strategy and analysis configuration.

## Reproducibility and Governance

The pipeline is designed with reproducibility and auditability as core principles. Deterministic seeding is supported throughout, all intermediate and final constraint products are written explicitly to disk, no hidden global state is shared between stages, and survey configuration is cleanly separated from inference logic.

This design supports transparent stress testing, controlled bias propagation, and reproducible constraint forecasts.

## Intended Use

This folder provides the constraint backbone for:

- Worst-case and conservative constraint forecasts  
- Quantification of maximum systematic biases  
- Sensitivity studies using simulation-only data  
- Stress testing modelling assumptions prior to observational calibration  

It is not intended to provide validated constraints against real data, but rather to establish conservative bounds and failure envelopes that inform interpretation of observational analyses.
