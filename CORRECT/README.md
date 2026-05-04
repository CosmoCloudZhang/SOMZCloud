# CORRECT

The **CORRECT** stage applies explicit, low-dimensional transformations to redshift distributions to mitigate residual biases identified in **ANALYZE** and **ASSESS**. It does not train new models or perform inference; it implements auditable **shift**, **scale**, and **shape** mappings on existing marginal (and related) products.

The layout mirrors the rest of SOMZCloud: **Y1** and **Y10** survey configurations, Python entry points with `argparse`, and SLURM-friendly `*.sh` wrappers.

## Layout

```
CORRECT/
├── Y1/
│   ├── SHIFT.py
│   ├── SCALE.py
│   ├── SHAPE.py
│   └── *.sh
├── Y10/
│   ├── SHIFT.py
│   ├── SCALE.py
│   ├── SHAPE.py
│   └── *.sh
└── README.md
```

On disk, outputs are organised under a user-defined base path, typically with separate subtrees for each correction family, for example:

`…/CORRECT/<scenario_tag>/SHIFT/`, `…/CORRECT/<scenario_tag>/SCALE/`, `…/CORRECT/<scenario_tag>/SHAPE/`

Downstream **PRIOR** scripts (e.g. `ENSEMBLE.py`) consume these trees alongside other pipeline products; ensemble figures use the **shape** branch where applicable (e.g. `SHAPE.pdf`).

## Role in the pipeline

**CORRECT** sits between performance evaluation and prior construction:

- Inputs: diagnostics and distribution products from **ANALYZE**, **ASSESS**, and related **SYNTHESIZE** paths (see script help for exact arguments).  
- Outputs: corrected distributions and sidecar metrics suitable for sensitivity studies and for **PRIOR**.

Corrections are intentionally simple so they remain interpretable and separable from modelling choices.

## Inputs

- Diagnostic products from **ANALYZE** and **ASSESS**.  
- Distribution products from **SYNTHESIZE** (and related branches as configured).  
- Correction configuration for selected families (`SHIFT`, `SCALE`, `SHAPE`).

## Outputs

- Corrected distribution artefacts for each correction family.  
- Sidecar diagnostics and comparison products for audit and prior construction.

## Parameterisations

| Script   | Role |
|----------|------|
| **SHIFT** | Additive adjustment of redshift (centroid / offset biases). |
| **SCALE** | Multiplicative rescaling of redshift (width / effective uncertainty biases). |
| **SHAPE** | Combined or more flexible low-order mapping that can encode shift- and scale-like effects together. |

Run only the subset your analysis requires; each script is independent and restart-safe.

## Execution

- **Python** — Each script exposes paths, binning, bounds, and parallel/batch options via `argparse`.  
- **Shell / SLURM** — `*.sh` files request resources, load modules or Conda envs, and launch jobs (e.g. job names like `CORRECT_Y1_SHAPE`).

Run from `Y1/` or `Y10/` (or call scripts with absolute paths) so relative path conventions in your config stay consistent.

## Example commands

```bash
cd CORRECT/Y1
python SHIFT.py --help
python SCALE.py --help
python SHAPE.py --help
```

Repeat in `CORRECT/Y10` for the corresponding branch.

## Practical decision rubric

- Start with **SHIFT** when dominant residuals are centroid offsets.  
- Use **SCALE** when posterior widths are systematically too narrow or too broad.  
- Use **SHAPE** when residual structure cannot be captured by additive or multiplicative transforms alone.

## Reproducibility

- No hidden global state; inputs and outputs are path-addressable.  
- Corrections are orthogonal to **MODEL** training: they post-process fixed upstream artefacts.  
- Document which **SHIFT** / **SCALE** / **SHAPE** run feeds **PRIOR** so ensembles remain traceable.

## Intended use

- Mitigate residual population-level redshift bias after ML and synthesis.  
- Propagate correction choices into **PRIOR** for nuisance parameters.  
- Run ablations over correction order and strength.

**CORRECT** is not a substitute for improved training data or architecture; it is a transparent adjustment layer for already-published or frozen model outputs.
