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

## Execution

- **Python** — Inputs point at **SUMMARIZE** trees; flags control which materials, how many realisations, and batching.  
- **Shell / SLURM** — Often one allocation fan-outs independent tasks per material.

Scripts are restart-safe: you can regenerate a single strategy without touching others.

## Reproducibility

Synthesis is a pure function of frozen **SUMMARIZE** artefacts plus CLI parameters. No shared mutable globals between jobs.

## Intended use

Feeding **ANALYZE**, **ASSESS**, **CORRECT**, and **PRIOR** with one coherent marginal layer per analysis variant—not for ad hoc hand-merging of pickles.
