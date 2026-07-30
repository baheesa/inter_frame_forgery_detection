# Enhanced Inter-Frame Video Forgery Detection

### Convolutional Network + Stacking Ensemble

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/License-Research-0B6E4F?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <a href="https://link.springer.com/article/10.1007/s11042-026-21684-x"><strong>Read the paper on Springer</strong></a>
  &nbsp;·&nbsp;
  <a href="https://doi.org/10.1007/s11042-026-21684-x"><strong>DOI 10.1007/s11042-026-21684-x</strong></a>
</p>

---

Official implementation accompanying:

> **Fatima, B., Bakhshi, A.D. & Ghafoor, A.**  
> *Enhanced inter-frame video forgery detection using convolutional network and stacking ensemble.*  
> **Multimedia Tools and Applications**, **85**, 497 (2026).  
> Published 06 May 2026 · National University of Sciences and Technology (NUST), Islamabad

Detects and classifies **frame-insertion**, **frame-deletion**, and **frame-duplication** forgeries in static and dynamic videos, with competitive F1-scores on three public benchmarks.

---

## Highlights

| | |
|---|---|
| **Multi-forgery** | Insertion · deletion · duplication · original |
| **Hybrid pipeline** | Classical spatial/temporal cues → TCN encoder → stacking ensemble |
| **Robust preprocessing** | NLM denoising + sharpening against post-manipulation artefacts |
| **Public benchmarks** | VFD · TDTVD · VIFFD (balanced protocol) |
| **Reported F1** | **0.994** (VFD) · **0.975** (TDTVD) · **0.940** (VIFFD) |

---

## Method Overview

The paper describes a **five-stage** passive forensic pipeline:

```text
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│  Stage I   │ → │  Stage II    │ → │  Stage III  │ → │  Stage IV    │ → │  Stage V    │
│ Preprocess  │   │ Features     │   │ TCN Encode  │   │ Stacking     │   │ Localise    │
│ NLM+sharp   │   │ D · O · H    │   │ 64-D vector │   │ RF+GB+SVM+kNN│   │ ∆w + Otsu   │
└─────────────┘   └──────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
```

| Stage | What it does |
|:-----:|---|
| **I** | Decompose video → grayscale → non-local means denoise → sharpen → normalise (+ geometric augmentation) |
| **II** | Extract **edge difference D**, Farneback **optical flow O**, and **HOG H** (Eqs. 1–4) |
| **III** | Encode `F = [D ‖ O ‖ H]` with causal Conv1D residuals + LSTM → fixed **64-D** `F_tcn` |
| **IV** | Stacking ensemble (RF, GBoost, SVM, kNN) with **logistic regression** meta-classifier |
| **V** | Wavelet histogram difference **∆w** + **Otsu** threshold to localise forged frame ranges |

> **This repository** ships Stages **0–IV** (annotation → preprocess/features → TCN → classification). Stage V localisation follows §2.2.5 of the paper.

---

## Repository Layout

```text
inter_frame_forgery_detection/
├── annotate.py              # Stage 0 — build forgery-type annotation JSON
├── preprocess_extract.py    # Stages I–II — preprocess + D/O/H features
├── tcn_feat.py              # Stage III — TCN / Conv1D+LSTM encoding
├── ensemble_class.py        # Stage IV — stacking ensemble classifier
├── requirements.txt
└── README.md
```

### Run order

```bash
python annotate.py
python preprocess_extract.py
python tcn_feat.py
python ensemble_class.py
```

Keep the `split_data` flag **consistent** across all four scripts.

---

## Installation

```bash
git clone https://github.com/<YOUR_USER>/inter_frame_forgery_detection.git
cd inter_frame_forgery_detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Datasets

Evaluated on three public inter-frame forgery corpora:

| Dataset | Role | Content |
|---------|------|---------|
| **[VFD](https://www.kaggle.com/datasets/rajshah1/video-forgery--dataset)** | Largest (6176 videos) | Static & dynamic · 960×720 / 1280×720 |
| **[TDTVD](https://doi.org/10.1007/s11042-020-09205-w)** | Temporal-domain set | Static & dynamic · multiple forgeries |
| **[VIFFD](https://data.mendeley.com/datasets)** | Inter-frame forgery set | Mostly static · 640×480 / 1920×1080 |

Imbalanced VIFFD / TDTVD splits are **balanced** in the paper via controlled synthesis from disjoint authentic seed videos (70/30 train/test seeds, 5–20% segment length, ≥10 frames), yielding 1600 (VIFFD) and 1380 (TDTVD) clips with equal class counts.

### Expected folder layout

```text
<dataset_root>/
  original/   *.mp4|avi|mov
  insert/     …
  delete/     …
  duplicate/  …
```

Update absolute paths inside each script’s `main()` before running.

---

## Configuration Cheatsheet

| Script | Key options | Primary outputs |
|--------|-------------|-----------------|
| `annotate.py` | `parent_folder`, `output_folder`, `split_data` | `output/annotations/*_annotate.json` |
| `preprocess_extract.py` | `augment_data`, `split_data` | `output/*_preprocess_feat.json` |
| `tcn_feat.py` | `split_data` | `output/*_tcn.npy` (N × 64) |
| `ensemble_class.py` | annotation + TCN paths | `output/stacking_model.pkl`, metrics |

Default train / val / test split in the ensemble script is approximately **60 / 20 / 20**.

---

## Model Details (Stage III)

| Hyperparameter | Value |
|----------------|-------|
| Conv1D layers | 3 × 64 filters, kernel 2, causal, ReLU |
| Residuals | After 2nd & 3rd Conv blocks |
| LSTM | 2 × 64 units |
| Pooling | Global average → **64-D** |
| Dropout | 0.5 |
| Optimiser / loss | Adam / MSE |
| Trainable params | **83,520** |

---

## Results (from the paper)

### Overall forgery detection (full pipeline)

| Dataset | Accuracy / F1 |
|---------|---------------|
| **VFD** | **0.994** |
| **TDTVD** | **0.975** |
| **VIFFD** | **0.940** |

Ablation (Table 3) shows that **HOG**, edge difference, optical flow, augmentation, and the TCN each contribute; removing any module reduces accuracy on at least one corpus / forgery type. The full configuration is consistently strongest overall.

---

## Citation

If you use this code or method, please cite:

```bibtex
@article{Fatima2026InterFrame,
  title   = {Enhanced inter-frame video forgery detection using convolutional network and stacking ensemble},
  author  = {Fatima, Baheesa and Bakhshi, Asim Dilawar and Ghafoor, Abdul},
  journal = {Multimedia Tools and Applications},
  volume  = {85},
  number  = {497},
  year    = {2026},
  doi     = {10.1007/s11042-026-21684-x},
  url     = {https://link.springer.com/article/10.1007/s11042-026-21684-x}
}
```

---

## Authors

| Author | Affiliation | Contact |
|--------|-------------|---------|
| **Baheesa Fatima** | NUST, Islamabad | bfatima.phdsemcs@student.nust.edu.pk |
| **Asim Dilawar Bakhshi** *(corresponding)* | NUST, Islamabad | asim.dilawar@mcs.edu.pk |
| **Abdul Ghafoor** | NUST, Islamabad | abdulghafoor-mcs@nust.edu.pk |

---

## Licence & Acknowledgements

Research code released to support reproducibility of the published article.  
Dataset licences remain with their original providers (VFD / TDTVD / VIFFD).  
© 2026 The authors · Springer Nature publishing agreement applies to the article text.
