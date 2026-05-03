# MODEL

**MODEL** trains and evaluates fiducial photometric-redshift models on **DATASET** products. It separates bookkeeping, targets, fitting, evaluation, and reference baselines into small scripts orchestrated by shell wrappers (**Y1** / **Y10**).

## Layout

```
MODEL/
├── Y1/
│   ├── REFERENCE.py
│   ├── TARGET.py
│   ├── INFORM.py
│   ├── ESTIMATE.py
│   ├── EVALUATE.py
│   └── *.sh
├── Y10/
│   └── (same script names)
└── README.md
```

## Conceptual order

A typical run cycles through the following logic (exact CLI order may follow your internal convention; the dependency flow is):

1. **REFERENCE** — Fiducial parameters, baselines, or external benchmarks against which results are judged.  
2. **TARGET** — Quantities to predict (e.g. redshift summaries, latent features) and selection definitions.  
3. **INFORM** — Run metadata, resolved paths, and standardised diagnostic headers for the rest of the stage.  
4. **ESTIMATE** — Training / optimisation / sampling as implemented for your model class.  
5. **EVALUATE** — Held-out metrics, residuals, and sanity checks on **ESTIMATE** outputs.

Scripts communicate only through versioned files on disk.

## LSST configurations

**Y1** targets early-survey volumes and noise; **Y10** targets full-depth statistical error. Keep outputs in separate directory trees.

## Execution

- **Python** — Model paths, hyperparameters, seeds, and device/batch settings are exposed via `argparse`.  
- **Shell** — Environment activation and SLURM (or local) resource requests.

## Reproducibility

No cross-run mutable singletons; configuration is externalised to flags and small config files where used.

## Intended use

Analysis-specific ML for SOMZCloud—not an open-ended AutoML sandbox. Products feed **COMPARE**, **CONSTRAIN**, **SUMMARIZE**, and **FIGURE**.
