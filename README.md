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

After running experiments, you may include:

* Main tables for MAE/RMSE on each dataset & horizon
* Training curves and stability plots
* Efficiency plots (speed/params)

Suggested place for figures:

* `assets/`

Example:

```markdown
![Validation Curves](assets/val_curves.png)
```

---

## Citation

If you use this repository in academic work, please cite the paper:

```bibtex
@article{hosseini_hg_gfno_traffic,
  title   = {Integrated Spatio-Temporal Modeling with Hybrid Graph Convolutions and the Graph Fourier Neural Operator for Traffic Prediction},
  author  = {Hosseini, Seyed-Majid and Rahmatinia, S. Mozhgan and Hosseini-Seno, Seyed-Amin},
  journal = {Under review / To appear},
  year    = {2026}
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

```

اگر دوست داری، همین الان بگو **آیا دیتا را داخل GitHub می‌گذاری یا نه**؛ اگر نه، من یک `data/README.md` انگلیسی هم می‌نویسم که دقیقاً توضیح دهد کاربر چطور دیتا را دانلود کند و کجا قرار دهد.
```
