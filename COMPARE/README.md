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

## Inputs

- **MODEL** outputs to be validated.  
- Observational spectroscopic catalogues and quality cuts from **DATASET** products.  
- Run metadata/configuration resolved in `INFORM`.

## Outputs

- Agreement/tension diagnostics between model predictions and spectroscopic truth anchors.  
- Comparison-specific tables and residual summaries for later aggregation.

## Execution

`argparse` scripts for all numerical choices; `*.sh` for allocation and env modules. Designed for array jobs over tracers or tomographic bins when needed.

## Example commands

```bash
cd COMPARE/Y1
python INFORM.py --help
python ESTIMATE.py --help
python EVALUATE.py --help
```

Run the analogous scripts in `COMPARE/Y10` for the full-depth branch.

## Failure modes and restart guidance

- If spec sample filtering changes, rerun from `REFERENCE`/`TARGET` onward to keep diagnostics coherent.  
- Keep spectroscopic quality cuts versioned in job logs for traceability.  
- Avoid mixing validation products from different survey epochs in a shared output root.

## Intended use

Observational validation prior to relying on marginal products in **SUMMARIZE** / **SYNTHESIZE**—complementing **CONSTRAIN**, which stress-tests models without spec anchoring.
