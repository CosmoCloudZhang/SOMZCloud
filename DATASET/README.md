# DATASET

**DATASET** builds LSST-like photometric and spectroscopic samples for SOM-based calibration and downstream ML. It is modular, `argparse`-driven, and HPC-oriented, with parallel **Y1** and **Y10** depth tracks.

## Layout

```
DATASET/
├── CATALOG.py
├── CATALOG.sh
├── Y1/
│   ├── OBSERVE.py
│   ├── SIMULATE.py
│   ├── SOM.py
│   ├── APPLY.py
│   ├── SELECT.py
│   ├── RESTRICT.py
│   ├── DEGRADE.py
│   ├── AUGMENT.py
│   ├── COMBINE.py
│   ├── ASSOCIATE.py
│   └── *.sh
├── Y10/
│   └── (same stage names)
└── README.md
```

## Stage sequence (typical)

1. **CATALOG** — Central paths, naming, and bookkeeping for truth, sims, and derivatives.  
2. **OBSERVE** — Survey realism (depth, noise, bands, masking) without changing intrinsic galaxy SEDs.  
3. **SIMULATE** — Forward-model photometry from truth tables.  
4. **SOM** — Train / apply self-organising maps for low-dimensional structure.  
5. **APPLY** — Propagate SOM cells, weights, or mappings to catalogues.  
6. **SELECT** — Photometric and spectroscopic selection functions.  
7. **RESTRICT** — Extra cuts on redshift, quality, and completeness.  
8. **DEGRADE** — Match ideal sims to target survey realism if a high-fidelity intermediate exists.  
9. **AUGMENT** — Adaptive simulation-based augmentation in colour–magnitude–redshift space.  
10. **COMBINE** — Merge observed, simulated, and augmented streams.  
11. **ASSOCIATE** — Photometric–spectroscopic matching for calibration labels.

Exact subsets depend on your science case; scripts are designed to chain via explicit on-disk products.

## Y1 vs Y10

Each epoch has its own configuration, noise model, and output root so forecasts and early-survey tests do not overwrite one another.

## Execution

- **Python** — Paths, seeds, chunking, and augmentation hyperparameters are CLI flags.  
- **Shell** — Module or Conda loads, CPUs/GPUs, memory, walltime, and array jobs.

Stages avoid hidden global state: every step reads named inputs and writes named outputs.

## Reproducibility

Deterministic seeding is supported; intermediates are materialised to disk for audit and partial reruns.

## Intended use

Controlled dataset construction for this repository’s **MODEL** / **COMPARE** / **CONSTRAIN** stages—not a general-purpose all-sky survey simulator.
