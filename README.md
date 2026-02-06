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


![HG-GFNO Architecture](architecture.png)



---

## Installation

### Environment

Recommended: Python 3.9+ (tested with 3.10)

Example using conda:

```bash
conda create -n hggfno python=3.10 -y
conda activate hggfno
```

### Dependencies

If you provide a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Minimal starting point (adjust as needed):

```bash
pip install numpy torch matplotlib tensorboardX
```

> GPU training is recommended.

---

## Datasets

We evaluate on four standard traffic benchmarks collected from **Caltrans PeMS**:

* **PEMS03, PEMS04, PEMS07, PEMS08**

PeMS portal:

* [https://pems.dot.ca.gov/](https://pems.dot.ca.gov/)

Common protocol:

* Train/Val/Test split: **60% / 20% / 20%**
* Z-score normalization using **train statistics only**
* Missing values: forward fill + backward fill

### Data Files


This repo expects dataset folders under `data/` (see structure above).

#### Download (Google Drive mirror)
Download the dataset archive from:

- Google Drive: [https://drive.google.com/file/d/1RR0r2yGiWG8h6depT3SBiRsGB4hWb9Hr/view?usp=sharing](https://drive.google.com/file/d/1RR0r2yGiWG8h6depT3SBiRsGB4hWb9Hr/view?usp=sharing)

Then extract it so the folder structure looks like:

```text
data/
  PEMS03/
  PEMS04/
  PEMS07/
  PEMS08/
---

## Configuration Files

All experiments are controlled via `.conf` files in `configurations/`.

Each config typically includes:

* `[Data]`: dataset paths, graph adjacency path, horizon, input length, etc.
* `[Training]`: learning rate, epochs, batch size, model dims, scheduler, early stopping, etc.

> Please check the existing `.conf` templates in `configurations/` and edit paths if necessary.

---

## Training & Evaluation

### 1) Train/Evaluate one dataset configuration

`run.py` supports a `--config` argument (default: `configurations/PEMS08.conf`):

```bash
python run.py --config configurations/PEMS08_mgcn.conf
```

### 2) Run the multi-horizon suite

`run_all.py` runs multiple horizons by generating temporary configs and executing `run.py`.

```bash
python run_all.py
```

---

## Reproducibility

To reproduce paper-level results:

* Keep preprocessing/splits consistent across runs
* Fix random seeds (Python/NumPy/PyTorch)
* Log the exact config used for each run
* Use the same horizon settings reported in the paper (e.g., 12/24/48/96)

Typical (paper) settings include:

* Input sequence length: **192**
* GCN order (K): **2**
* Dropout: **0.15**
* Optimizer: **Adam**
* Loss: **MAE**
* Cosine annealing schedule + warmup
* Early stopping patience: **20**

> Map these settings to the appropriate fields in your `.conf` files.

---

## Outputs & Logging

Experiments are stored under:

```text
experiments/<DATASET>/predict<PRED_LEN>_MGCN_<YYYYMMDD_HHMMSS>/
```

Typical contents:

* training logs (text)
* TensorBoard logs
* checkpoints (`ckpt_best.params`, `epoch_*.params`, etc.)

### TensorBoard

```bash
tensorboard --logdir experiments
```

---

## Results

We report **MAE** and **RMSE** (lower is better) on **PEMS03/04/07/08** across four forecasting horizons (**12, 24, 48, 96** steps).
Table 5 compares HG-GFNO with **Transformer-based** baselines, and Table 6 compares with **Non-Transformer** baselines.

### Table 5. Performance comparison among Transformer-based models

| Dataset | Horizon | HG-GFNO RMSE | HG-GFNO MAE | Informer RMSE | Informer MAE | Autoformer RMSE | Autoformer MAE | FEDformer RMSE | FEDformer MAE | iTransformer RMSE | iTransformer MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PEMS03 | 12  | 26.1 | 15.4 | 36.4 | 22.2 | 40.8 | 26.1 | 33.4 | 22.4 | 27.3 | 17.3 |
| PEMS03 | 24  | 28.9 | 17.2 | 38.5 | 23.9 | 54.0 | 29.5 | 37.3 | 25.1 | 33.9 | 21.5 |
| PEMS03 | 48  | 32.8 | 19.7 | 43.2 | 26.5 | 60.3 | 36.9 | 44.8 | 30.3 | 44.7 | 28.7 |
| PEMS03 | 96  | 39.0 | 22.8 | 48.5 | 29.1 | 73.0 | 45.3 | 57.3 | 38.5 | 50.1 | 31.3 |
| PEMS03 | Avg | 31.7 | 18.7 | 41.7 | 25.4 | 57.0 | 34.4 | 43.2 | 29.1 | 39.0 | 24.7 |
| PEMS04 | 12  | 31.1 | 19.1 | 40.5 | 25.6 | 45.2 | 30.7 | 42.9 | 29.3 | 36.8 | 23.3 |
| PEMS04 | 24  | 33.0 | 20.4 | 41.7 | 26.5 | 50.1 | 34.8 | 50.5 | 32.0 | 40.3 | 26.3 |
| PEMS04 | 48  | 35.7 | 22.1 | 42.6 | 27.4 | 60.1 | 41.4 | 54.1 | 38.0 | 47.1 | 31.0 |
| PEMS04 | 96  | 38.1 | 23.9 | 42.8 | 27.6 | 65.9 | 49.1 | 70.1 | 50.6 | 53.1 | 35.2 |
| PEMS04 | Avg | 34.4 | 21.3 | 41.9 | 26.8 | 55.4 | 39.0 | 54.7 | 37.5 | 44.3 | 29.0 |
| PEMS07 | 12  | 34.1 | 20.7 | 57.4 | 33.8 | 56.2 | 38.1 | 43.1 | 29.1 | 38.1 | 24.8 |
| PEMS07 | 24  | 37.43 | 22.6 | 58.8 | 35.8 | 67.0 | 46.1 | 50.9 | 33.7 | 45.0 | 29.6 |
| PEMS07 | 48  | 41.5 | 25.0 | 60.1 | 36.5 | 70.0 | 48.2 | 57.3 | 38.7 | 54.8 | 36.2 |
| PEMS07 | 96  | 45.57 | 27.2 | 61.2 | 37.3 | 84.4 | 61.1 | 66.6 | 44.3 | 64.9 | 43.3 |
| PEMS07 | Avg | 39.6 | 23.8 | 59.4 | 35.7 | 69.4 | 48.4 | 55.2 | 37.0 | 50.7 | 33.6 |
| PEMS08 | 12  | 24.8 | 15.4 | 46.6 | 29.2 | 41.6 | 28.6 | 36.6 | 25.0 | 28.7 | 18.0 |
| PEMS08 | 24  | 27.7 | 17.3 | 52.4 | 32.3 | 47.2 | 36.6 | 38.2 | 27.0 | 35.7 | 22.6 |
| PEMS08 | 48  | 30.5 | 19.2 | 57.3 | 37.3 | 67.8 | 40.0 | 48.1 | 32.1 | 48.6 | 31.7 |
| PEMS08 | 96  | 33.6 | 21.1 | 60.6 | 38.1 | 79.5 | 51.3 | 62.4 | 44.2 | 57.1 | 38.7 |
| PEMS08 | Avg | 29.15 | 18.2 | 54.2 | 34.2 | 63.2 | 39.4 | 48.1 | 33.6 | 42.5 | 27.8 |

### Table 6. Performance comparison among Non-Transformer baselines (Linear, State Space, and Graph-based models)

| Dataset | Horizon | HG-GFNO RMSE | HG-GFNO MAE | DLinear RMSE | DLinear MAE | MambaTS RMSE | MambaTS MAE | Minusformer RMSE | Minusformer MAE | MGCN RMSE | MGCN MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PEMS03 | 12  | 26.1 | 15.4 | 40.9 | 26.8 | 26.2 | 16.6 | 25.9 | 16.4 | 29.5 | 16.9 |
| PEMS03 | 24  | 28.9 | 17.2 | 50.7 | 33.5 | 32.0 | 20.1 | 31.3 | 19.8 | 32.2 | 19.2 |
| PEMS03 | 48  | 32.8 | 19.7 | 64.3 | 38.3 | 41.7 | 26.4 | 41.2 | 26.0 | 38.7 | 22.9 |
| PEMS03 | 96  | 39.0 | 22.8 | 75.5 | 52.2 | 51.2 | 33.4 | 50.2 | 32.4 | 42.3 | 25.2 |
| PEMS03 | Avg | 31.7 | 18.7 | 57.9 | 38.9 | 37.8 | 24.1 | 37.2 | 23.7 | 35.7 | 21.1 |
| PEMS04 | 12  | 31.1 | 19.1 | 55.9 | 38.2 | 35.9 | 22.5 | 35.3 | 22.2 | 31.8 | 19.7 |
| PEMS04 | 24  | 33.0 | 20.4 | 64.5 | 44.7 | 43.4 | 27.6 | 42.2 | 26.8 | 34.3 | 21.6 |
| PEMS04 | 48  | 35.7 | 22.1 | 77.0 | 53.8 | 54.2 | 35.1 | 53.8 | 34.8 | 37.2 | 23.4 |
| PEMS04 | 96  | 38.1 | 23.9 | 85.8 | 61.2 | 66.7 | 44.5 | 64.7 | 41.9 | 39.7 | 25.1 |
| PEMS04 | Avg | 34.4 | 21.3 | 70.8 | 49.5 | 50.1 | 32.4 | 49.0 | 31.4 | 35.8 | 22.5 |
| PEMS07 | 12  | 34.1 | 20.7 | 55.1 | 38.6 | 37.4 | 23.3 | 36.6 | 22.6 | 35.1 | 21.8 |
| PEMS07 | 24  | 37.43 | 22.6 | 70.2 | 49.8 | 44.6 | 28.4 | 43.7 | 27.1 | 38.1 | 23.6 |
| PEMS07 | 48  | 41.5 | 25.0 | 82.8 | 56.5 | 57.1 | 38.7 | 54.8 | 34.4 | 48.2 | 26.6 |
| PEMS07 | 96  | 45.57 | 27.2 | 90.5 | 79.8 | 68.2 | 45.0 | 68.8 | 44.6 | 47.8 | 30.2 |
| PEMS07 | Avg | 39.6 | 23.8 | 74.7 | 58.7 | 51.8 | 33.9 | 51.0 | 32.1 | 41.0 | 25.6 |
| PEMS08 | 12  | 24.8 | 15.4 | 47.0 | 32.6 | 27.4 | 17.1 | 27.0 | 16.1 | 24.8 | 15.4 |
| PEMS08 | 24  | 27.7 | 17.3 | 54.1 | 38.0 | 33.2 | 20.6 | 32.6 | 20.5 | 27.6 | 17.2 |
| PEMS08 | 48  | 30.5 | 19.2 | 67.8 | 48.1 | 41.8 | 26.4 | 42.5 | 27.2 | 31.3 | 19.7 |
| PEMS08 | 96  | 33.6 | 21.1 | 77.3 | 56.3 | 51.0 | 33.3 | 52.8 | 34.1 | 34.0 | 21.4 |
| PEMS08 | Avg | 29.15 | 18.2 | 61.5 | 43.8 | 38.4 | 24.4 | 38.7 | 24.7 | 29.4 | 18.4 |


---

## Citation

If you use this repository in academic work, please cite the paper:

```bibtex
@article{hosseini_hg_gfno_traffic_2026,
  title   = {Integrated Spatio-Temporal Modeling with Hybrid Graph Convolutions and the Graph Fourier Neural Operator for Traffic Prediction},
  author  = {Hosseini, Seyed-Majid and Rahmatinia, S. Mozhgan and Hosseini-Seno, Seyed-Amin},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Accepted for publication}
}

```

---

## License

Add your preferred license (MIT / Apache-2.0) as `LICENSE`.

---

## Contact

For questions or collaborations:

* Name: **Seyed-Majid Hosseini**
* Email: **hosseini.seyedmajid@mail.um.ac.ir**

