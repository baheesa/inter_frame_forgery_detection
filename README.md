# Enhanced Inter-Frame Video Forgery Detection

<p align="center">
  <em>Convolutional network + stacking ensemble · Multimedia Tools and Applications (2026)</em>
</p>

<p align="center">
  <a href="https://link.springer.com/article/10.1007/s11042-026-21684-x"><img src="https://img.shields.io/badge/Paper-Springer-0F4C81?style=for-the-badge&logo=springer&logoColor=white" alt="Paper"/></a>
  <a href="https://doi.org/10.1007/s11042-026-21684-x"><img src="https://img.shields.io/badge/DOI-10.1007%2Fs11042--026--21684--x-blue?style=for-the-badge" alt="DOI"/></a>
</p>

<p align="center">
  <a><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a><img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/></a>
  <a><img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"/></a>
  <a><img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="sklearn"/></a>
</p>

Official code for detecting **frame-insertion**, **frame-deletion**, and **frame-duplication** forgeries using spatial/temporal features, a TCN encoder, and a stacking ensemble.

**Paper:** Fatima, B., Bakhshi, A.D. & Ghafoor, A. — [*Enhanced inter-frame video forgery detection using convolutional network and stacking ensemble*](https://link.springer.com/article/10.1007/s11042-026-21684-x). *Multimed Tools Appl* **85**, 497 (2026).

**F1-scores:** VFD **0.994** · TDTVD **0.975** · VIFFD **0.940**

---

## Method

<p align="center">
  <img src="assets/method_diagram.png" alt="Fig. 1 — Proposed method" width="100%"/>
</p>

<p align="center"><sub>Fig. 1 — Preprocess → features (D, O, H) → TCN → stacking ensemble → wavelet ∆w + Otsu localisation.</sub></p>

This repo covers annotation through classification (Stages 1–4). Localisation (Stage 5) is described in the paper.

---

## Setup

```bash
git clone https://github.com/baheesa/inter_frame_forgery_detection.git
cd inter_frame_forgery_detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Update dataset paths in each script’s `main()`. Keep `split_data` consistent. Expected layout:

```text
<dataset>/original|insert|delete|duplicate/*.mp4
```

Run in order:

```bash
python annotate.py
python preprocess_extract.py
python tcn_feat.py
python ensemble_class.py
```

| Script | Role |
|--------|------|
| `annotate.py` | Build forgery-type annotations |
| `preprocess_extract.py` | Preprocess + extract D / O / H |
| `tcn_feat.py` | Encode to 64-D TCN features |
| `ensemble_class.py` | Train stacking ensemble (RF, GBoost, SVM, kNN → LR) |

Outputs are written under `output/`.

**Datasets:** [VFD](https://www.kaggle.com/datasets/rajshah1/video-forgery--dataset) · [TDTVD](https://doi.org/10.1007/s11042-020-09205-w) · [VIFFD](https://data.mendeley.com/)

---

## Citation

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

NUST, Islamabad — **Baheesa Fatima**, **Asim Dilawar Bakhshi** (corresponding), **Abdul Ghafoor**

<p>
  <a href="mailto:baheesafatima@gmail.com"><img src="https://img.shields.io/badge/Gmail-baheesafatima-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Gmail"/></a>
  <a href="mailto:bfatima.phdsemcs@student.nust.edu.pk"><img src="https://img.shields.io/badge/Email-NUST-1B5E20?style=flat-square&logo=gmail&logoColor=white" alt="NUST"/></a>
  <a href="https://www.linkedin.com/in/baheesafatima/"><img src="https://img.shields.io/badge/LinkedIn-baheesafatima-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="https://www.researchgate.net/profile/Baheesa-Fatima-2"><img src="https://img.shields.io/badge/ResearchGate-Baheesa--Fatima--2-00CCBB?style=flat-square&logo=researchgate&logoColor=white" alt="ResearchGate"/></a>
  <a href="https://orcid.org/0009-0003-2757-5672"><img src="https://img.shields.io/badge/ORCID-0009--0003--2757--5672-A6CE39?style=flat-square&logo=orcid&logoColor=white" alt="ORCID"/></a>
</p>

Corresponding: [asim.dilawar@mcs.edu.pk](mailto:asim.dilawar@mcs.edu.pk)

---

## Licence

Code is released for academic research and reproducibility — please cite the paper if you use it. Fig. 1 and article text remain under Springer Nature rights. Datasets belong to their original providers and are not redistributed here.

© 2026 Baheesa Fatima, Asim Dilawar Bakhshi & Abdul Ghafoor
