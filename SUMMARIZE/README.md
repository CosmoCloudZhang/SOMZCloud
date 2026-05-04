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

## Inputs

- Stage products from **MODEL**, **COMPARE**, and **CONSTRAIN**.  
- Material/tracer/strategy-specific script settings and path roots.  
- Frozen upstream artefacts for deterministic aggregation.

## Outputs

- Structured summary tables/tensors partitioned by material, tracer, strategy, and epoch.  
- Intermediate products consumed directly by **SYNTHESIZE**.

## Execution

Python scripts expose paths into **MODEL** / **COMPARE** / **CONSTRAIN** products, aggregation rules, and parallelism. Shell scripts manage HPC resources (including GPU jobs where used).

Run from the relevant `Y*/<MATERIAL>/` directory so relative paths match your `CATALOG` layout.

## Example commands

```bash
cd SUMMARIZE/Y1/COPPER
python TRUTH_SOURCE.py --help
python HYBRID_LENS.py --help
```

Repeat for other materials and for `SUMMARIZE/Y10/*`.

## Failure modes and restart guidance

- Regenerate only changed material/tracer/strategy branches when upstream artefacts are unchanged elsewhere.  
- Keep absolute input roots explicit if running outside material directories.  
- Avoid mixing summary versions from different upstream model runs in the same output tree.

## Reproducibility

Summaries are deterministic given upstream hashes and CLI flags. Y1 and Y10 never write into each other’s trees.

## Intended use

Shrinking large model output collections into stable, documented tensors and tables for synthesis and publication—not a substitute for **ANALYZE** statistical tests.
