# ASSESS

**ASSESS** evaluates **conditional** redshift distributions \(p(z\,|\,\mathrm{obs})\): accuracy, scatter, and calibration of per-object or per-cell posteriors from **SYNTHESIZE** and related stages. It does not perform training or sampling.

Companion stage **ANALYZE** treats **marginal** ensemble distributions; the two are complementary. **ASSESS** metrics often motivate **CORRECT** choices (e.g. conditional bias patterns feeding **SHIFT**).

## Layout

```
ASSESS/
├── Y1/
│   ├── CENTER.py
│   ├── WIDTH.py
│   ├── EXPECTATION.py
│   ├── DEVIATION.py
│   ├── CONDITIONAL.py
│   ├── VALUE.py
│   └── *.sh
├── Y10/
│   └── (same script names)
└── README.md
```

## Metrics (conceptual)

| Script | Focus |
|--------|--------|
| **CENTER** | Conditional mean / median vs spectroscopic or simulation truth. |
| **WIDTH** | Uncertainty calibration (too narrow vs too wide posteriors). |
| **EXPECTATION** | \(\mathbb{E}[z\,|\,\mathrm{obs}]\) maps and analogous summaries. |
| **DEVIATION** | Residuals in observable or redshift space. |
| **CONDITIONAL** | Full shape, coverage, and PIT-style diagnostics where implemented. |
| **VALUE** | Aggregated conditional-quality scores. |

## LSST configurations

**Y1** and **Y10** mirror survey depth; keep outputs in separate trees when running both.

## Execution

Each script is `argparse`-driven with explicit input paths and numerical controls. `*.sh` wrappers handle modules, Conda, and batch schedulers.

## Intended use

Conditional validation for tomography and per-tracer science, inputs to **CORRECT**, and evidence for **PRIOR** scatter on nuisance parameters tied to per-galaxy redshift behaviour.
