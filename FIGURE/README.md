# FIGURE

**FIGURE** turns numerical pipeline products into reproducible **matplotlib** figures for QA, talks, and papers. It does not modify catalogues or retrain models.

Inputs are typically **DATASET**, **MODEL**, **COMPARE**, **CONSTRAIN**, and—where plotting scripts exist—**ANALYZE** / **ASSESS** metrics. Paths and style flags are set per script via `argparse`.

## Layout

```
FIGURE/
├── Y1/
│   ├── BASELINE.py, BENCHMARK.py, CATALOG.py
│   ├── CONTRAST.py, CONTROL.py, DIAGRAM.py
│   ├── HISTOGRAM.py, MAP.py, METRIC.py
│   ├── OPTICAL.py, INFRARED.py
│   ├── QUANTILE.py, REDSHIFT.py, REGULATE.py
│   ├── RESTRAIN.py, SAMPLE.py, SOM.py
│   └── *.sh
├── Y10/
│   └── (mirrors Y1)
└── README.md
```

Each script targets one figure family so you can regenerate only what changed upstream.

## Publication-oriented defaults

Scripts favour deterministic rendering from fixed inputs. Where heavy vector graphics would bloat PDFs, artists use rasterisation (`rasterized=True` on dense elements) and high-resolution `savefig(..., dpi=512)` for vector exports—see individual scripts for exact call patterns.

## LSST configurations

**Y1** and **Y10** mirror other stages: separate roots so plots never cross-contaminate epochs.

## Inputs

- Numerical products from upstream stages (**DATASET**, **MODEL**, **COMPARE**, **CONSTRAIN**, and optionally **ANALYZE** / **ASSESS**).  
- Plot configuration via CLI flags (paths, labels, colourmaps, output names).

## Outputs

- Deterministic QA and publication-ready figure files.  
- Script-specific plot products grouped by epoch and figure family.

## Execution

- **Python** — Input directories, catalogue subsets, colourmaps, and output filenames.  
- **Shell** — HPC submission for large batches of panels or bootstrap loops.

## Example commands

```bash
cd FIGURE/Y1
python SOM.py --help
python METRIC.py --help
```

Use matching scripts in `FIGURE/Y10` for full-depth products.

## Failure modes and restart guidance

- Regenerate only the affected figure family after upstream changes.  
- Keep output naming/version tags explicit to avoid overwriting publication candidates.  
- Ensure plotting backends/fonts are available on the target environment before large batch submissions.

## Reproducibility

Figures are pure functions of upstream data and CLI parameters; regenerate after any **MODEL** or **SYNTHESIZE** change to keep paper plots in sync.

## Intended use

Visual regression testing, collaborator communication, and journal-ready exports—not interactive exploration (use notebooks locally for that).
