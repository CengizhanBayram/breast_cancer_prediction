# CBIS-DDSM Breast Lesion Classification (PyTorch + torchvision)

Binary classification of breast lesions (benign vs malignant) on **CBIS-DDSM** ROI images.  
Built with **PyTorch** and **torchvision** (EfficientNetV2-S), designed to run **end-to-end on Google Colab**.

> **Validation results:** AUC ≈ **0.954**, ACC ≈ **0.842**, F1 ≈ **0.795** (with TTA AUC ≈ **0.955**)

---
![Uploading image.png…]()

## ✨ Features
- ✅ **Colab-first**: single notebook, minimal dependencies
- ✅ **Robust labeling**: matches dataset JPGs to CSV labels (fallbacks included)
- ✅ **Optional DICOM → PNG** conversion
- ✅ **Torchvision** augments (no Albumentations/OpenCV/timm)
- ✅ **Metrics**: AUC/ACC/F1, PR-AUC (AP), threshold sweep, confusion matrix
- ✅ **TTA** (flip; optional advanced TTA incl. FiveCrop)
- ✅ **Deterministic-ish** seeding & group-aware split

---

## 📦 Dataset
- **CBIS-DDSM (Kaggle)** – download via Kaggle API in the notebook.
- The code unpacks nested archives and builds train/val splits with labels from CSVs.

> You need a valid `kaggle.json` for downloading.

---

## 🚀 Quick Start (Colab)
1. Open the notebook in Colab.
2. Upload `kaggle.json` when prompted.
3. Run all cells.  
   Outputs will be saved under:  
   `/content/drive/MyDrive/breast_cancer/`
4. (Optional) Enable TTA & threshold optimization cells at the end.

---

## 🧠 Model
- **Backbone:** `torchvision.models.efficientnet_v2_s` (IMAGENET1K_V1 weights)
- **Head:** single-logit linear layer (BCEWithLogitsLoss with `pos_weight`)
- **Augmentations:** Resize → CenterCrop → Flip/Rotation → Normalize
- **Train:** AdamW + CosineLR, AMP (autocast via `torch.amp.autocast`)
- **Early stopping:** by AUC (option to switch to val_loss)

---

## 📊 Results (Validation)
- **AUC:** ~0.954  
- **ACC:** ~0.842  
- **F1:** ~0.795  
- **TTA AUC:** ~0.955  
- See `/content/drive/MyDrive/breast_cancer/` for:
  - `metrics.json`, `metrics_bar.png`
  - `roc_curve.png`, `pr_curve.png`
  - `confusion_matrix*.png`
  - `val_predictions*.csv`
  - `errors_false_negatives.csv`, `errors_false_positives.csv`
  - `best_cbis_ddsm.pt`

> Threshold is tunable (we sweep 0.05–0.95 and store best-F1 threshold).

---

## 🛠 Requirements
- Python ≥ 3.10 (Colab default)
- `torch`, `torchvision`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`
- (Optional) `pydicom` + `pylibjpeg` (for DICOM → PNG)

All installs are handled inside the notebook.

---

## 🗂 Repo Layout (suggested)
