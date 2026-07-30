<div align="center">

# Enhanced Inter-Frame Video Forgery Detection
### Convolutional network · Stacking ensemble

Python code from our Springer paper (*Multimedia Tools and Applications*, 2026)

<br/>

[![Paper](https://img.shields.io/badge/Paper-Springer-0F4C81?style=flat-square)](https://link.springer.com/article/10.1007/s11042-026-21684-x)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11042--026--21684--x-blue?style=flat-square)](https://doi.org/10.1007/s11042-026-21684-x)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)

Editors can alter a video’s timeline by inserting, deleting, or duplicating frames — often with further post-processing so the cut is hard to see. This codebase implements a five-stage pipeline that flags those temporal forgeries in both static and dynamic clips, and classifies each video as **original**, **frame-insertion**, **frame-deletion**, or **frame-duplication**.

<br/>

[LinkedIn](https://www.linkedin.com/in/baheesafatima/)
·
[ResearchGate](https://www.researchgate.net/profile/Baheesa-Fatima-2)
·
[ORCID](https://orcid.org/0009-0003-2757-5672)
·
[Email](mailto:baheesafatima@gmail.com)

</div>

**Keywords:** inter-frame video forgery detection · video forgery detection · digital video forensics · frame insertion · frame deletion · frame duplication · temporal forgery · stacking ensemble · temporal convolutional network · TCN · optical flow · HOG · edge difference · multimedia forensics · passive forensics · video tampering detection · TensorFlow · OpenCV · VFD · TDTVD · VIFFD

---

## What this is

Editors can alter a video’s timeline by **inserting**, **deleting**, or **duplicating** frames — often with further post-processing so the cut is hard to see. That kind of edit is called **inter-frame video forgery** (also temporal forgery or video tampering).

This repository is the official Python implementation of our paper on **inter-frame video forgery detection** and classification:

> Fatima, B., Bakhshi, A.D. & Ghafoor, A. (2026).  
> [*Enhanced inter-frame video forgery detection using convolutional network and stacking ensemble*](https://link.springer.com/article/10.1007/s11042-026-21684-x).  
> *Multimedia Tools and Applications*, **85**, 497.  
> DOI: [10.1007/s11042-026-21684-x](https://doi.org/10.1007/s11042-026-21684-x)

The five-stage pipeline flags temporal forgeries in both static and dynamic clips, and classifies each video as **original**, **frame-insertion**, **frame-deletion**, or **frame-duplication**.

**Reported F1-scores:** VFD **0.994** · TDTVD **0.975** · VIFFD **0.940**

If you use this code or method in your own work, a citation means a lot — thank you.

---

## Method

The approach combines classical video cues with deep temporal encoding and an ensemble classifier:

<p align="center">
  <img src="assets/method_diagram.png" alt="Inter-frame video forgery detection pipeline: preprocessing, edge difference optical flow HOG features, TCN encoding, stacking ensemble classification, and forgery localisation" width="100%"/>
</p>

<p align="center"><sub>Fig. 1 from the paper — preprocessing, feature extraction, TCN encoding, stacking classification, and localisation.</sub></p>

| Stage | What happens |
|:-----:|--------------|
| **1** | **Preprocess** — convert frames to grayscale, denoise (NLM), sharpen, normalise, and optionally augment |
| **2** | **Features** — extract edge difference (**D**), optical flow (**O**), and HOG (**H**) as complementary forgery cues |
| **3** | **TCN encode** — map the concatenated features into a fixed **64-D** temporal representation (Conv1D + LSTM) |
| **4** | **Classify** — stacking ensemble of RF, GBoost, SVM, and kNN with a logistic regression meta-classifier |
| **5** | **Localise** — wavelet histogram difference (∆w) + Otsu thresholding to mark forged frame ranges |

This repository implements Stages **1–4** (plus an annotation step). Stage **5** is described in the paper (§2.2.5).

---

## Setup

```bash
git clone https://github.com/baheesa/inter_frame_forgery_detection.git
cd inter_frame_forgery_detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data layout

Organise videos by class folder (the parent folder name becomes the label):

```text
<dataset_root>/
  original/
  insert/
  delete/
  duplicate/
```

Supported formats: `.mp4`, `.avi`, `.mov`.

### Run the pipeline

Edit the absolute paths in each script’s `main()` so they point to your dataset. Keep the `split_data` flag **the same** across all four scripts.

```bash
python annotate.py              # write annotation JSON from the folder layout
python preprocess_extract.py    # preprocess frames and extract D / O / H
python tcn_feat.py              # encode features with the TCN → 64-D vectors
python ensemble_class.py        # train / evaluate the stacking ensemble
```

| Script | Input | Output |
|--------|-------|--------|
| `annotate.py` | video folders | `output/annotations/*_annotate.json` |
| `preprocess_extract.py` | annotations | `output/*_preprocess_feat.json` |
| `tcn_feat.py` | preprocess JSON | `output/*_tcn.npy` |
| `ensemble_class.py` | annotations + TCN `.npy` | `output/stacking_model.pkl`, metrics |

---

## Datasets

Evaluated on three public **video forgery detection** benchmarks:

| Dataset | Link | Notes |
|---------|------|-------|
| **VFD** | [Kaggle](https://www.kaggle.com/datasets/rajshah1/video-forgery--dataset) | Largest set · static & dynamic |
| **TDTVD** | [Paper](https://doi.org/10.1007/s11042-020-09205-w) | Temporal forgeries · single & multiple |
| **VIFFD** | [Mendeley](https://data.mendeley.com/) | Mostly static · multi-resolution |

In the paper, VIFFD and TDTVD were balanced with controlled synthesis from disjoint authentic seeds so train and test do not leak. See the article for the full protocol.

---

## Citation

If you use this code or method, please cite:

```bibtex
@article{Fatima2026InterFrame,
  title   = {Enhanced inter-frame video forgery detection using convolutional network and stacking ensemble},
  author  = {Fatima, Baheesa and Bakhshi, Asim Dilawar and Ghafoor, Abdul},
  journal = {Multimedia Tools and Applications},
  volume  = {85},
  pages   = {497},
  year    = {2026},
  doi     = {10.1007/s11042-026-21684-x}
}
```

---

## Authors

National University of Sciences and Technology (NUST), Islamabad  
**Baheesa Fatima** · **Asim Dilawar Bakhshi** (corresponding) · **Abdul Ghafoor**

<p>
  <a href="mailto:baheesafatima@gmail.com"><img src="https://img.shields.io/badge/Gmail-baheesafatima-1e293b?style=flat-square&logo=gmail&logoColor=e2e8f0" alt="Gmail"/></a>
  <a href="mailto:bfatima.phdsemcs@student.nust.edu.pk"><img src="https://img.shields.io/badge/Email-NUST-1e293b?style=flat-square&logo=gmail&logoColor=e2e8f0" alt="NUST"/></a>
  <a href="https://www.linkedin.com/in/baheesafatima/"><img src="https://img.shields.io/badge/LinkedIn-baheesafatima-1e293b?style=flat-square&logo=linkedin&logoColor=e2e8f0" alt="LinkedIn"/></a>
  <a href="https://www.researchgate.net/profile/Baheesa-Fatima-2"><img src="https://img.shields.io/badge/ResearchGate-Baheesa--Fatima--2-1e293b?style=flat-square&logo=researchgate&logoColor=e2e8f0" alt="ResearchGate"/></a>
  <a href="https://orcid.org/0009-0003-2757-5672"><img src="https://img.shields.io/badge/ORCID-0009--0003--2757--5672-1e293b?style=flat-square&logo=orcid&logoColor=e2e8f0" alt="ORCID"/></a>
</p>

Corresponding: [asim.dilawar@mcs.edu.pk](mailto:asim.dilawar@mcs.edu.pk)

---

## Licence

Released for academic research and reproducibility — please cite the paper if you use this work. Fig. 1 and the article text remain under Springer Nature rights. Datasets belong to their original providers and are not redistributed here.

© 2026 Baheesa Fatima, Asim Dilawar Bakhshi & Abdul Ghafoor
