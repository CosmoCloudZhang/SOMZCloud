# SYNTHESIZE

**SYNTHESIZE** merges **SUMMARIZE** outputs into unified, analysis-ready **marginal** products across strategies (truth, direct, hybrid, stack), materials (e.g. COPPER, GOLD, IRON, SILVER, TITANIUM, ZINC), and survey epoch (**Y1** / **Y10**). It does not fit new models; it harmonises existing summaries.

Downstream **ANALYZE** and **ASSESS** read **SYNTHESIZE** products; **CORRECT** and **PRIOR** consume the resulting distribution trees.

## Layout

```
SYNTHESIZE/
├── Y1/
│   ├── DIR.py
│   ├── HYBRID.py
│   ├── STACK.py
│   ├── TRUTH.py
│   └── *.sh
├── Y10/
│   └── (same script names)
└── README.md
```

## Strategies

| Script | Meaning |
|--------|---------|
| **TRUTH** | Simulation-truth reference combinations. |
| **DIR** | Direct, observation-facing branch without augmentation path. |
| **HYBRID** | Blends observational and simulation-informed components. |
| **STACK** | Aggregates multiple realisations or subsamples. |

Each script loops materials consistently so cross-scenario comparisons stay aligned.

## Inputs

- Material/tracer/strategy summaries from **SUMMARIZE**.  
- Strategy-specific synthesis parameters and scenario tags.

## Outputs

- Unified marginal products per strategy and epoch.  
- Harmonised artefacts for **ANALYZE**, **ASSESS**, **CORRECT**, and **PRIOR**.

## Execution

- **Python** — Inputs point at **SUMMARIZE** trees; flags control which materials, how many realisations, and batching.  
- **Shell / SLURM** — Often one allocation fan-outs independent tasks per material.

Scripts are restart-safe: you can regenerate a single strategy without touching others.

## Example commands

```bash
cd SYNTHESIZE/Y1
python TRUTH.py --help
python STACK.py --help
```

Run corresponding scripts in `SYNTHESIZE/Y10` for the full-depth branch.

## Failure modes and restart guidance

- Rebuild only impacted strategy outputs when one upstream summary branch changes.  
- Keep material lists consistent across strategy runs to preserve comparability.  
- Version output roots when changing synthesis configuration defaults.

## Reproducibility

Synthesis is a pure function of frozen **SUMMARIZE** artefacts plus CLI parameters. No shared mutable globals between jobs.

## Intended use

Feeding **ANALYZE**, **ASSESS**, **CORRECT**, and **PRIOR** with one coherent marginal layer per analysis variant—not for ad hoc hand-merging of pickles.
