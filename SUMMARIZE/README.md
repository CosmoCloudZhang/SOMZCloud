# SUMMARIZE

**SUMMARIZE** condenses **MODEL**, **COMPARE**, and **CONSTRAIN** outputs into structured summaries indexed by **material** (e.g. COPPER, GOLD, IRON, SILVER, TITANIUM, ZINC), **tracer** (lens vs source), **strategy** (truth, direct, hybrid, stack), and survey epoch (**Y1** / **Y10**). It performs no training or inference beyond aggregation.

Outputs feed **SYNTHESIZE**, then **ANALYZE**, **ASSESS**, **CORRECT**, and **PRIOR**.

## Layout

```
SUMMARIZE/
├── Y1/
│   ├── COPPER/
│   ├── GOLD/
│   ├── IRON/
│   ├── SILVER/
│   ├── TITANIUM/
│   ├── ZINC/
│   │   ├── DIR_LENS.py, DIR_SOURCE.py
│   │   ├── HYBRID_LENS.py, HYBRID_SOURCE.py
│   │   ├── STACK_LENS.py, STACK_SOURCE.py
│   │   ├── TRUTH_LENS.py, TRUTH_SOURCE.py
│   │   └── *.sh
│   └── …
├── Y10/
│   └── (same material / tracer / strategy pattern)
└── README.md
```

## Axes of aggregation

- **Strategy** — Truth, direct, hybrid, and stack branches mirror the science variants in **SYNTHESIZE**.  
- **Tracer** — `_LENS` vs `_SOURCE` scripts keep tomographic and lensing samples separate.  
- **Material** — Named metal folders encode experimental groupings used consistently across the pipeline.

## Execution

Python scripts expose paths into **MODEL** / **COMPARE** / **CONSTRAIN** products, aggregation rules, and parallelism. Shell scripts manage HPC resources (including GPU jobs where used).

Run from the relevant `Y*/<MATERIAL>/` directory so relative paths match your `CATALOG` layout.

## Reproducibility

Summaries are deterministic given upstream hashes and CLI flags. Y1 and Y10 never write into each other’s trees.

## Intended use

Shrinking large model output collections into stable, documented tensors and tables for synthesis and publication—not a substitute for **ANALYZE** statistical tests.
