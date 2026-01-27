# SOMZCloud: Simulation-Optimised Machine-Learning for Redshift Calibration

This repository contains **SOMZCloud**, an end-to-end, modular pipeline for
**photometric redshift calibration, validation, and uncertainty propagation**
using simulation-informed machine learning.

The project is designed for **Stage-IV cosmology analyses** (e.g. LSST, Euclid),
with an emphasis on:
- population-level redshift distribution calibration,
- robustness to domain shift between simulations and observations,
- reproducibility, interpretability, and downstream cosmological validity.

The codebase follows a **component-based architecture**, where each directory
implements a well-defined stage of the scientific workflow.

---

## Scientific Scope

SOMZCloud supports:
- Construction and comparison of redshift distributions \(n(z)\)
- Calibration using spectroscopic and simulated reference samples
- Quantification of uncertainty and its propagation into cosmological inference
- Reproducible evaluation of model performance under controlled assumptions

The pipeline is designed to integrate naturally into
**cosmic shear, galaxy–galaxy lensing, and clustering analyses**.

---

## Repository Structure

Each top-level directory corresponds to a **logical component** of the pipeline.
Detailed documentation is provided in a dedicated `README.md` inside each folder.

SOMZCloud/
├── ANALYZE/ # Diagnostic analysis and validation metrics
├── ASSESS/ # Quality assessment and performance evaluation
├── CALIBRATE/ # Redshift calibration methods and workflows
├── CELL/ # Atomic processing units (reusable pipeline cells)
├── COMPARE/ # Model and distribution comparison tools
├── CONSTRAIN/ # Cosmological or statistical inference interfaces
├── DATASET/ # Dataset definitions, loaders, and metadata
├── FIGURE/ # Figure generation and plotting scripts
├── INFO/ # Configuration, metadata, and run information
├── MODEL/ # Machine-learning models and training logic
├── PRIOR/ # Prior assumptions and population models
├── SUMMARIZE/ # Aggregation and reporting utilities
├── SYNTHESIZE/ # Synthetic data generation and augmentation
├── VALUE/ # Derived quantities and summary statistics
├── LICENSE # License information
└── README.md # This file

---

## Design Principles

- **Modularity**: Each component can be used independently.
- **Reproducibility**: Deterministic configurations and documented assumptions.
- **Scientific Traceability**: Clear mapping from inputs → assumptions → outputs.
- **Survey-agnosticism**: Designed to generalise across LSST, Euclid, and simulations.
- **Ethical & Responsible ML**: Explicit handling of bias, uncertainty, and limitations.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/SOMZCloud.git
   cd SOMZCloud
2. See DATASET/README.md for data requirements and formats.
3. See MODEL/README.md and CALIBRATE/README.md for training and calibration workflows.
4. Each directory contains a standalone README describing:
   - purpose,
   - inputs/outputs,
   - assumptions,
   - example usage.

## Citation

If you use SOMZCloud in a publication, please cite:

TBD – preprint / paper in preparation

A BibTeX entry will be provided upon publication.

## License

This project is released under the terms specified in LICENSE
.

## Contact

For questions, issues, or collaboration inquiries:

Open a GitHub issue

Or contact the repository maintainer directly
