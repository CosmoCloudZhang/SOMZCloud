# CONSTRAIN

**CONSTRAIN** quantifies how tightly parameters can be recovered—or how large systematic residuals can grow—using **simulation-only** data products from **DATASET** (including augmentation where enabled) together with **MODEL** outputs. It **does not** use observational spectroscopy.

Use **COMPARE** when you need empirical anchoring to real spectra; use **CONSTRAIN** for conservative envelopes and augmentation stress paths.

## Layout

```
CONSTRAIN/
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

## Conceptual workflow

1. **INFORM** — Wire together simulation roots, augmentation flags, and model artefacts.  
2. **REFERENCE** — Fiducial cosmology or simulation scenario against which biases are measured.  
3. **TARGET** — Parameters or summary statistics whose recoverability you probe.  
4. **ESTIMATE** — Core simulation-only fitting / sampling / stress logic for constraint forecasts (see script docstrings).  
5. **EVALUATE** — Summarise offsets, degeneracies, and failure modes relative to **REFERENCE**.

Outputs are explicit tables and plots suitable for **SUMMARIZE** and **FIGURE**.

## LSST configurations

**Y1** paths emphasise noise-dominated regimes; **Y10** paths explore systematics at high \(n\). Never mix outputs between epochs in the same directory.

## Execution

Python CLIs for all science switches; shell wrappers for batch systems. Statelessness enables large parameter sweeps.

## Intended use

Pessimistic forecasts, augmentation sensitivity, and systematic ceilings—not a replacement for spectroscopic validation (**COMPARE**).
