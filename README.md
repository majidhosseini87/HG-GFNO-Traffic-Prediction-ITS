# HG-GFNO Traffic Forecasting (PeMS) — Research Code

This repository contains the research implementation for the paper:

**“Integrated Spatio-Temporal Modeling with Hybrid Graph Convolutions and the Graph Fourier Neural Operator for Traffic Prediction”**

HG-GFNO (**H**ybrid **G**raph Convolutions + **G**raph **F**ourier **N**eural **O**perator) is a unified spatio-temporal forecasting framework for traffic prediction on road networks. It combines:
- **Hybrid static–adaptive graph convolutions** for localized + dynamic spatial learning, and
- **Graph Fourier Neural Operator (GFNO)** for efficient long-range temporal modeling in the graph spectral domain.

> The paper reports consistent improvements on PeMS benchmarks (PEMS03/04/07/08), with notable gains (up to ~10.9% RMSE and ~11.9% MAE reported) over strong baselines, while remaining parameter-efficient and stable.

---

## Table of Contents
- [Highlights](#highlights)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Configuration Files](#configuration-files)
- [Training & Evaluation](#training--evaluation)
- [Reproducibility](#reproducibility)
- [Outputs & Logging](#outputs--logging)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Highlights
- **End-to-end** spatio-temporal modeling (no fragmented spatial/temporal pipeline).
- **Hybrid graph learning**: combines physical topology (static adjacency) + data-driven correlations (adaptive adjacency).
- **Sequence-as-Token** representation: each node’s full historical sequence is embedded as a single token.
- **GFNO spectral-temporal block**: models long-range dependencies without heavy attention/recurrent overhead.
- Evaluated on **PEMS03 / PEMS04 / PEMS07 / PEMS08**.

---

## Method Overview

HG-GFNO follows an end-to-end pipeline:

1) **Multi A-FGCN (Hybrid Static–Adaptive Spatial Encoder)**  
   Two stacked hybrid graph convolution layers with residual connections and normalization:
   - Static path: message passing on the predefined road graph adjacency
   - Adaptive path: learnable adjacency from node embeddings (dynamic correlations)
   - Fusion with gating / residuals improves stability and expressivity

2) **Sequence-as-Token Embedding**  
   Instead of tokenizing time steps, we tokenize each node’s full historical sequence:
   - Input: `Z ∈ R^{B×T×N×D}`
   - Tokens: `Z' ∈ R^{B×N×d_model}` via a learnable linear projection

3) **GFNO Block (Graph Fourier Neural Operator)**  
   GFNO models global dependencies by filtering in the **graph spectral domain**:
   - Build hybrid adjacency `A = A_static + A_adp`
   - Form Laplacian `L`, select Fourier bases (top-K eigenvectors)
   - Apply GFT → spectral filtering → inverse GFT
   - Residual/LayerNorm + final projection to the forecasting horizon

### Architecture Figure
Create an `assets/` folder and place the main architecture diagram as:

- `assets/architecture.png`

Then it will render here:

![HG-GFNO Architecture](assets/architecture.png)

> Tip: Export the main model figure from the paper and save it with the name above.

---

## Repository Structure

Current project layout:

```text
.
├── configurations/
│   ├── PEMS03_mgcn.conf
│   ├── PEMS04_mgcn.conf
│   ├── PEMS07_mgcn.conf
│   └── PEMS08_mgcn.conf
├── data/
│   ├── PEMS03/ (PEMS03.csv, PEMS03.npz)
│   ├── PEMS04/ (PEMS04.csv, PEMS04.npz)
│   ├── PEMS07/ (PEMS07.csv, PEMS07.npz)
│   ├── PEMS08/ (PEMS08.csv, PEMS08.npz)
│   └── README.md
├── data_provider/
│   ├── data_factory.py
│   ├── data_loader.py
│   └── __init__.py
├── lib/
│   ├── metrics.py
│   └── utils.py
├── model/
│   ├── layers.py
│   ├── models.py
│   └── utils.py
├── run.py
└── run_all.py
