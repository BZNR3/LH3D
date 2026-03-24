<p align="center">
  <h1 align="center">Learnability-Driven Submodular Optimization for Active Roadside 3D Detection</h1>
  <p align="center">
    <strong>Author 1</strong>
    ·
    <strong>Author 2</strong>
    ·
    <strong>Author 3</strong>
    ·
    <strong>Author 4</strong>
  </p>
  <h2 align="center">Conference / Journal 20XX</h2>
  <div align="center">
    <img src="./assets/teaser_intro.jpg" alt="Teaser" width="88%">
  </div>
  <p align="center">
    <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
    <br></br>
    <a href="#">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
  </p>
</p>

This repository contains the official implementation of **LH3D** — a learnability-driven active learning framework for vision-based roadside 3D object detection. Built on top of [BEVHeight](https://arxiv.org/abs/2303.08498), LH3D selects the most informative training samples under a fixed annotation budget using a three-stage submodular selection strategy driven by depth learnability, spatial diversity, and geometric similarity.

---

# News

- [20XX/XX] Code and paper are released!

---

# Overview

Annotating roadside 3D perception data is expensive. LH3D addresses this with a **pool-based active learning** loop that selects images maximally informative for the BEVHeight detector, measured along three complementary axes:

| Stage | Criterion | What it selects |
|---|---|---|
| **A** | Low depth entropy + depth coverage | Images where the model is **confidently wrong** in under-covered depth ranges |
| **B** | Object diversity + class balance | Images with **diverse object categories** that fix labeled-set class imbalance |
| **C** | Gaussian NLL similarity / dissimilarity | Images **similar** to the labeled scene distribution (80%) + **outliers** per class (20%) |

The annotation budget is tracked in **object counts** (GT boxes), not image counts, for realistic comparison.

---

# Getting Started

### Installation

See [docs/install.md](docs/install.md) for environment setup.

### Dataset Preparation

See [docs/prepare_dataset.md](docs/prepare_dataset.md) for DAIR-V2X-I and Rope3D setup.

### Fully-Supervised Training (BEVHeight baseline)

```bash
# Train with 8 GPUs
python exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py \
  --amp_backend native -b 8 --gpus 8

# Evaluate
python exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py \
  --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```

### Active Learning with LH3D

```bash
# DAIR-V2X-I
python exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_active.py \
  --al_enabled \
  --al_method lh3d \
  --al_init_size 100 \
  --al_query_size 100 \
  --al_rounds 5 \
  --al_epochs_per_round 10 \
  --al_max_objects 3000 \
  --devices 1

# Rope3D
python exps/rope3d/bev_height_lss_r50_864_1536_128x128_active.py \
  --al_enabled \
  --al_method lh3d \
  --al_init_size 100 \
  --al_query_size 100 \
  --al_rounds 5 \
  --al_epochs_per_round 10 \
  --al_max_objects 3000 \
  --devices 1
```

**Available `--al_method` options:**

| Method | Description |
|---|---|
| `random` | Uniform random selection (lower bound) |
| `entropy` | Class-logit entropy uncertainty |
| `lh3d` | **Our method** — learnability-driven submodular selection |

---

# Acknowledgment

This project builds on the following works:

- [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight) — roadside 3D detection backbone (CVPR 2023)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) — LSS-based depth estimation
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) — dataset and evaluation toolkit
- [pypcd](https://github.com/dimatura/pypcd) — point cloud utilities

---

# Citation

If you find this work useful, please cite our paper:

```bibtex
@article{mao2026learnability,
  title={Learnability-Driven Submodular Optimization for Active Roadside 3D Detection},
  author={Mao, Ruiyu and Zhang, Baoming and Ruozzi, Nicholas and Guo, Yunhui},
  journal={arXiv preprint arXiv:2601.01695},
  year={2026}
}
```
