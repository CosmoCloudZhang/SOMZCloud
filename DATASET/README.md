# DATASET Pipeline

This folder defines the end-to-end dataset construction and adaptive simulation-based data augmentation pipeline used to generate LSST-like photometric and spectroscopic samples for downstream calibration, SOM-based mapping, and cosmological analyses. The design is modular, configuration-driven, and HPC-ready, with explicit support for LSST Year 1 (Y1) and Year 10 (Y10) observing scenarios.

The pipeline is implemented as a sequence of clearly scoped Python scripts, each controlled via argparse, and orchestrated through corresponding shell scripts that manage environments, parallelisation, and compute-resource allocation on HPC systems.

## High-level Structure

```
DATASET/
├── CATALOG.py # Dataset configuration and catalogue bookkeeping
├── CATALOG.sh # Environment setup and execution wrapper
│
├── Y1/ & Y10 # LSST Year 1 and Year 10 configuration
│ ├── OBSERVE.py # Apply Y1 observing conditions
│ ├── SIMULATE.py # Forward-model photometry from truth catalogues
│ ├── SOM.py # Train and apply Self-Organising Maps
│ ├── APPLY.py # Apply SOM mappings and weights
│ ├── SELECT.py # Define photometric and spectroscopic selections
│ ├── RESTRICT.py # Enforce redshift, magnitude, and quality cuts
│ ├── DEGRADE.py # Degrade ideal data to Y1 survey realism
│ ├── AUGMENT.py # Adaptive simulation-based augmentation
│ ├── COMBINE.py # Merge observed and augmented samples
│ ├── ASSOCIATE.py # Photometric–spectroscopic association
│ ├── *.sh # HPC job scripts and environment control
│
└── README.md
```

## Conceptual Pipeline

The dataset construction follows an ordered, modular procedure. Each step corresponds to a single script and produces explicit, versioned intermediate data products that are written to disk and consumed by subsequent stages.

1. CATALOG  
   Defines centralised catalogue structure, path resolution, metadata, and experiment bookkeeping. This script specifies how truth catalogues, simulated data, and derived products are named, stored, and linked consistently across pipeline stages.

2. OBSERVE  
   Applies LSST observing conditions appropriate to the selected configuration (Y1 or Y10), including depth, noise, band coverage, and masking. This step introduces survey realism without modifying intrinsic galaxy properties.

3. SIMULATE  
   Forward-models observed photometry from intrinsic galaxy properties, including bandpass effects and noise realisations. The output consists of LSST-like multi-band photometric catalogues.

4. SOM  
   Trains and/or applies Self-Organising Maps to the simulated data, constructing a low-dimensional representation used to relate photometric and spectroscopic samples.

5. APPLY  
   Applies SOM-derived cell assignments, weights, or mappings to catalogues. This step operationalises the learned representation for downstream selection and augmentation.

6. SELECT  
   Defines photometric and spectroscopic samples through configurable selection functions, such as magnitude limits, colour cuts, or tomographic binning.

7. RESTRICT  
   Enforces additional redshift, quality, or completeness cuts to ensure consistency between samples and alignment with target analysis requirements.

8. DEGRADE  
   Degrades idealised or high-fidelity data products to match LSST observational realism, ensuring fair alignment between simulations and observed-like samples.

9. AUGMENT  
   Performs adaptive simulation-based data augmentation, resampling or generating synthetic objects to improve coverage in under-represented regions of colour–magnitude–redshift space.

10. COMBINE  
    Merges observed, simulated, and augmented samples into unified datasets suitable for calibration and inference.

11. ASSOCIATE  
    Associates photometric objects with spectroscopic counterparts or probabilistic labels, enabling redshift calibration and the transfer of spectroscopic information.

## LSST Configurations: Y1 and Y10

The pipeline is explicitly parameterised to support different LSST observing depths and strategies.

- Y1/  
  Represents LSST Year 1-like depth and completeness, suitable for early analyses, validation, and method development.

- Y10/  
  Represents LSST Year 10-like depth, enabling high-precision end-state forecasts and stress testing.

Each configuration provides:

- Dedicated configuration files specifying paths, depths, and noise models  
- Shell scripts for environment setup and job submission  
- Independent output directories to avoid cross-contamination between configurations  

## Execution Model

### Python scripts

Each Python script uses argparse to expose configurable parameters, including:

- Input and output paths  
- Survey configuration options  
- Random seeds and augmentation controls  
- Parallelisation and data chunking settings  

Scripts are designed to be stateless, reproducible, and restart-safe.

### Shell scripts

Shell wrappers are used to manage execution and resource governance, including:

- Environment activation (for example via Conda or module systems)  
- Resource requests such as CPU or GPU allocation, memory, and walltime  
- Parallel execution strategies, including array jobs or task splitting  

This separation between logic and execution ensures portability across local machines and HPC systems.

## Reproducibility and Governance

The pipeline is designed with reproducibility and auditability as first-class goals.

- Deterministic seeding is supported throughout  
- Intermediate data products are written explicitly to disk  
- No hidden global state is shared between stages  
- Configuration (Y1/Y10) is cleanly separated from pipeline logic  

This design supports reproducible cosmological analyses, controlled ablation studies, and transparent validation.

## Typical Usage Example

The modelling pipeline is executed from within a specific configuration directory, such as Y1 or Y10, by running the corresponding shell scripts in sequence. The exact ordering and subset of steps may vary depending on the modelling task and analysis configuration.

## Intended Use

This folder provides the dataset backbone for:

- Photometric redshift calibration  
- SOM-based population matching  
- Simulation-informed uncertainty propagation  
- LSST Y1–Y10 cosmological forecasting and analysis  

It is not intended to function as a general-purpose survey simulator, but rather as a controlled, analysis-driven data construction framework.