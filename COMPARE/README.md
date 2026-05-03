# COMPARE

**COMPARE** benchmarks **MODEL** outputs against **observational spectroscopic** samples from **DATASET**. It deliberately **does not** use simulation-only augmentation streams: evaluation is anchored in real spectra where available.

For worst-case or augmentation-inclusive stress tests on pure simulations, use **CONSTRAIN** instead.

## Layout

```
COMPARE/
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

1. **INFORM** — Resolve paths to model checkpoints and spectroscopic catalogues; emit run metadata.  
2. **REFERENCE** — Choose reference subsamples, redshift quality cuts, and baseline comparisons.  
3. **TARGET** — Define observables and summary statistics for the comparison (e.g. binned metrics in colour–redshift space).  
4. **ESTIMATE** — Fit comparison-specific parameters (scatter models, offsets) if your analysis requires them—**without** introducing new ML photometric estimators.  
5. **EVALUATE** — Quantify agreement, tension metrics, and residual structure between model and data.

Each step reads/writes explicit artefacts so partial reruns stay cheap.

## LSST configurations

**Y1** uses early-depth spec samples; **Y10** assumes larger statistical leverage. Paths are isolated per epoch.

## Execution

`argparse` scripts for all numerical choices; `*.sh` for allocation and env modules. Designed for array jobs over tracers or tomographic bins when needed.

## Intended use

Observational validation prior to relying on marginal products in **SUMMARIZE** / **SYNTHESIZE**—complementing **CONSTRAIN**, which stress-tests models without spec anchoring.
