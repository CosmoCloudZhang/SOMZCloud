# ANALYZE

**ANALYZE** evaluates **marginal** redshift distributions: how well ensemble-level \(n(z)\)-like quantities match references after **SYNTHESIZE**. It does not train models; it computes reproducible scalar and binned diagnostics for comparison across methods and materials.

Companion stage **ASSESS** treats **conditional** \(p(z\,|\,\mathrm{obs})\) quality; use both for a full picture. Metrics from **ANALYZE** inform **CORRECT** (especially **SCALE** and population mean biases).

## Layout

```
ANALYZE/
├── Y1/
│   ├── CENTER.py
│   ├── WIDTH.py
│   ├── EXPECTATION.py
│   ├── DEVIATION.py
│   ├── MARGINAL.py
│   ├── VALUE.py
│   └── *.sh
├── Y10/
│   └── (same script names)
└── README.md
```

## Metrics (conceptual)

| Script | Focus |
|--------|--------|
| **CENTER** | Mean / median / mode offsets vs truth or reference \(n(z)\). |
| **WIDTH** | Variance, effective width, credible-interval calibration. |
| **EXPECTATION** | Ensemble expectations of redshift or weighted summaries. |
| **DEVIATION** | Binned residuals and systematic trends in redshift. |
| **MARGINAL** | Global shape, normalisation, and coverage beyond low-order moments. |
| **VALUE** | Composite scores for quick ranking of configurations. |

## LSST configurations

**Y1** emphasises robustness at lower signal-to-noise; **Y10** supports high-statistics stress tests. Paths are disjoint between the two.

## Inputs

- Marginal products from **SYNTHESIZE**.  
- Reference distributions/labels and metric configuration settings.  
- Optional comparison branches for method ranking.

## Outputs

- Marginal diagnostic tables, score summaries, and analysis figures.  
- Quantitative inputs for **CORRECT** and uncertainty propagation into **PRIOR**.

## Execution

Python scripts take input roots, metric choices, binning, and parallelism flags. Shell scripts set up the environment and SLURM resources. Scripts are stateless and can be rerun selectively after upstream **SYNTHESIZE** updates.

## Example commands

```bash
cd ANALYZE/Y1
python CENTER.py --help
python MARGINAL.py --help
python VALUE.py --help
```

Use `ANALYZE/Y10` for full-depth runs.

## Failure modes and restart guidance

- If binning or reference choices change, rerun affected metrics to keep ranking tables consistent.  
- Recompute only modified metric branches after partial upstream updates.  
- Record metric configuration in job logs for reproducible comparisons.

## Intended use

Population-level validation, method comparison, and quantitative inputs to **CORRECT** and **PRIOR**—not exploratory object-by-object debugging (use **FIGURE** and catalogues for that).
