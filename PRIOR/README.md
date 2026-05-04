# PRIOR

The **PRIOR** stage builds statistically motivated priors on nuisance parameters for downstream inference. It aggregates information from **ANALYZE**, **ASSESS**, **SYNTHESIZE**, and **CORRECT** (including **SHIFT**, **SCALE**, and **SHAPE** products), but does not run likelihood sampling or cosmological MCMC itself.

Like other stages, **PRIOR** is split into **Y1** and **Y10** configurations with `argparse` scripts and SLURM `*.sh` helpers.

## Layout

```
PRIOR/
├── Y1/
│   ├── EXPECTATION.py
│   ├── DEVIATION.py
│   ├── COVARIANCE.py
│   ├── ENSEMBLE.py
│   └── *.sh
├── Y10/
│   └── (same script names)
└── README.md
```

## Role in the pipeline

**PRIOR** translates empirical pipeline variability into distributions over nuisance parameters:

- **EXPECTATION** — Central values (means / locations) of nuisance parameters across ensembles.  
- **DEVIATION** — Scatter, bias structure, and higher-moment behaviour that set prior scales.  
- **COVARIANCE** — Joint uncertainty between nuisances for multivariate priors.  
- **ENSEMBLE** — Combines the above into consolidated prior products; reads corrected distributions from **CORRECT** subtrees (`SHIFT`, `SCALE`, `SHAPE`) when building shape-related ensembles (see code comments and output names such as `SHAPE.pdf`).

This keeps inference priors aligned with the same simulations, assessments, and correction choices used in the rest of SOMZCloud.

## LSST configurations

- **Y1/** — Wider, more conservative priors reflecting early-survey systematics.  
- **Y10/** — Tighter priors appropriate to full-depth statistical errors (still conservative where needed).

Input and output roots are disjoint between Y1 and Y10 to avoid cross-talk.

## Inputs

- Diagnostic and distribution products from **ANALYZE**, **ASSESS**, and **SYNTHESIZE**.  
- Corrected branches from **CORRECT** (`SHIFT`, `SCALE`, `SHAPE`) where included.  
- Nuisance grouping and aggregation settings from CLI configuration.

## Outputs

- Prior central values, deviations, covariance matrices, and ensemble products.  
- Export-ready nuisance prior artefacts for downstream inference pipelines.

## Execution

- **Python** — Paths, grouping of nuisances, aggregation rules, and export formats are CLI flags.  
- **Shell** — Wrappers handle allocation, env activation, and batch fan-out on HPC.

Typical order: **EXPECTATION** → **DEVIATION** → **COVARIANCE** → **ENSEMBLE**, rerunning later steps when upstream metrics or **CORRECT** branches change.

## Example commands

```bash
cd PRIOR/Y1
python EXPECTATION.py --help
python COVARIANCE.py --help
python ENSEMBLE.py --help
```

Use `PRIOR/Y10` for the full-depth branch.

## Failure modes and restart guidance

- Rerun `COVARIANCE` and `ENSEMBLE` whenever upstream correction branches change.  
- Keep nuisance ordering consistent across exports to avoid downstream parameter mismatches.  
- Version output directories when modifying aggregation conventions.

## Reproducibility

All prior artefacts are derived from versioned upstream directories; there is no shared mutable state between scripts. Survey tags and scenario names are explicit in paths so priors can be reproduced or differenced across pipeline versions.

## Intended use

- Nuisance priors for **CONSTRAIN**-style or external likelihood codes.  
- Consistent propagation of calibration and correction uncertainty.  
- Documentation of systematic floors via ensemble spread.

**PRIOR** is meant for data- and pipeline-informed priors, not arbitrary hand-set hyperpriors disconnected from the SOMZCloud workflow.
