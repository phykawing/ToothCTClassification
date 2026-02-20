# Development of an AI Tool to Identify Teeth from 3D Imaging

**COMP5709 Capstone Project — University of Sydney, 2023**

A deep learning pipeline that segments individual teeth from 3D CT scan volumes and classifies them into one of 10 tooth types using convolutional neural networks.

---

## Pipeline

```
Raw CT Scan (.nrrd)
       │
       ▼
Isotropic Voxel Resampling
       │
       ▼
Hounsfield Unit Thresholding  ──→  Binary mask (HU ≥ 0)
       │
       ▼
3D Connected Components + Dust Removal
       │
       ▼
Individual Tooth Volumes (.nrrd)
       │
       ▼
CNN Classification  ──→  Tooth Type (10 classes)
```

---

## Notebooks

### [`Thresholding.ipynb`](Thresholding.ipynb) — Preprocessing & Segmentation
Processes a raw CT volume into individually segmented tooth objects:
1. Reads `.nrrd` CT volumes (`pynrrd`)
2. Resamples voxels to isotropic resolution using `scipy.ndimage.zoom` (e.g. 0.43 × 0.43 × 0.50 mm → 0.43 mm isotropic)
3. Applies HU thresholding at 0 to isolate hard tissue (enamel, bone)
4. Runs 3D connected-component labelling (`cc3d`) with 6-connectivity and dust removal (< 1000 voxels)
5. Crops each component to its bounding box and saves as individual `.nrrd` files with a CSV manifest

### [`Classification.ipynb`](Classification.ipynb) — CNN-Based Tooth Classification
Benchmarks multiple architectures for 10-class tooth type classification on the segmented volumes:

- **Dataset**: ~360 labelled tooth volumes; stratified train / validation / test split
- **Class imbalance**: handled via inverse-frequency class weights in `CrossEntropyLoss` and `WeightedRandomSampler`
- **Custom transforms**: `Resize3D` (→ 128×64×64), `Pad3D` (→ 128×91×91), 3D random rotation augmentation

| Architecture | Backbone | Type |
|---|---|---|
| ResNet | ResNet-34 (ImageNet) | 2D (mid-slice) |
| VGG16 | VGG-16 (ImageNet) | 2D (mid-slice) |
| AlexNet | AlexNet (ImageNet) | 2D (mid-slice) |
| AlexNet3D | Custom `Conv3d` | 3D volumetric |
| ResNet3D | R3D-18 (Kinetics-400) | 3D volumetric |

---

## Results

All results on the held-out test set (n = 92), trained with Adam optimiser and cross-entropy loss.

| Model | Test Accuracy | Micro-F1 |
|---|---|---|
| AlexNet3D | 62.0% | 0.620 |
| AlexNet (2D) | 67.4% | 0.674 |
| ResNet (2D) | 68.5% | 0.685 |
| VGG16 (2D) | 68.5% | 0.685 |
| ResNet3D | 72.8% | 0.728 |
| Ensemble — soft voting | 77.1% | 0.771 |
| **Final ResNet3D** ⭐ | **79.3%** | **0.793** |

The best single model is **Final ResNet3D** (pretrained on Kinetics-400, fine-tuned for 23 epochs), achieving 79.3% accuracy. A 5-model ensemble with soft voting reached 77.1% on a validation subset.

---

## Setup

**Environment (Anaconda):**
```bash
conda activate PyTorch291
jupyter notebook
```

See [`requirements.txt`](requirements.txt) for the full package list.

> **Data paths are hardcoded.** Update the `path` variable in `Classification.ipynb` and the `wd` variable in `Thresholding.ipynb` to point to your local data directory before running.

---

## Repository Contents

| File | Description |
|---|---|
| [`Thresholding.ipynb`](Thresholding.ipynb) | CT scan preprocessing and tooth segmentation |
| [`Classification.ipynb`](Classification.ipynb) | Multi-architecture CNN tooth classification |
| [`Final Report.pdf`](Final Report.pdf) | Full project report |
| [`Presentation Slides.pptx`](Presentation Slides.pptx) | Project presentation slides |

---

*University of Sydney · COMP5709 Capstone Project · 2023*
