# EEG-based Parkinson's Disease Detection using Empirical Wavelet Transform and Convolutional Neural Networks

This repository contains the official implementation for the manuscript:

> **"EEG-Based Computer-Aided Diagnosis of Parkinson's Disease Using Time-Frequency Representations and Deep Learning"**
> Azadnouran et al., *Biomedical Signal Processing and Control* (Elsevier), under review.

---

## Overview

This work presents a computer-aided diagnosis (CAD) framework for Parkinson's disease (PD) detection from resting-state EEG signals. EEG recordings are transformed into 2D time-frequency representation (TFR) images using four methods — **EWT, CWT, STFT, and DWT** — and classified using both a custom residual CNN trained from scratch and three pretrained CNN architectures as comparative baselines.

**Proposed method:** Empirical Wavelet Transform (EWT)

**Baselines:** Continuous Wavelet Transform (CWT), Short-Time Fourier Transform (STFT), Discrete Wavelet Transform (DWT)

**Classification task:** Healthy Control (HC) vs. Parkinson's Disease (PD)

**Dataset:** Iowa EEG dataset (28 subjects: 14 HC, 14 PD)

---

## Repository Structure

```
eeg-pd-detection/
├── training/
│   ├── custom_residual_cnn.py        # Custom residual CNN trained from scratch
│   ├── densenet121.py                # DenseNet-121 (pretrained, frozen backbone)
│   ├── efficientnet_b0.py            # EfficientNet-B0 (pretrained, frozen backbone)
│   └── resnet50.py                   # ResNet-50 (pretrained, frozen backbone)
├── explainability/
│   └── gradcam_visualization.py      # Grad-CAM saliency maps for trained models
├── clinical_app/
│   └── app.py                        # Streamlit clinical decision-support app (GPT-4o integration)
├── requirements.txt
└── README.md
```

---

## Models

### Custom Residual CNN (`custom_residual_cnn.py`)
- Trained from scratch on grayscale TFR images (single-channel input)
- Architecture: 3-stage residual blocks with spatial Dropout2d, Global Average Pooling head
- Optimizer: Adam, lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-6)
- Early stopping: patience=3, min_epochs=7

### Pretrained Baselines (`densenet121.py`, `efficientnet_b0.py`, `resnet50.py`)
- ImageNet-pretrained backbones with all layers frozen
- Only the final classification head is replaced and trained: `Linear(in_features → 1)`
- RGB input with ImageNet normalization
- Optimizer: Adam, lr=1e-4, weight_decay=1e-4
- No LR scheduler (pretrained models start near a good solution)

### Shared training settings across all models
| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 32 |
| Loss | BCEWithLogitsLoss |
| Early stopping patience | 3 |
| Min epochs before ES | 7 |
| Input size | 224 × 224 |
| Evaluation | 5-fold subject-aware GroupKFold |

---

## Evaluation Protocol

**Protocol 1 — Manual Balanced Subject-Aware GroupKFold (primary, reported in paper)**
- Subjects are assigned to folds independently per class (HC/PD), guaranteeing balanced validation sets:
  - Folds 1–4: 3 HC + 3 PD validation subjects
  - Fold 5: 2 HC + 2 PD validation subjects
- No subject appears in both training and validation splits (no data leakage)
- Metrics computed at the image level: TP, TN, FP, FN → Accuracy, Sensitivity, Specificity, Precision, MCC, AUC

**Protocol 2 — Standard StratifiedKFold (5-fold, supplementary)**
- Image-level stratified split
- Included for completeness

---

## Explainability (`gradcam_visualization.py`)

Grad-CAM saliency maps are generated for the best-performing trained model to visualize which regions of the TFR images are most discriminative for PD classification. Output is saved as overlay figures for qualitative analysis.

---

## Clinical Decision-Support App (`app.py`)

A Streamlit-based prototype application that:
- Accepts a TFR image as input
- Runs inference using the trained EWT model
- Displays the classification result (HC vs PD) with confidence score
- Generates a GPT-4o-powered clinical summary and treatment plan suggestion based on the prediction

To run the app locally:
```bash
streamlit run clinical_app/app.py
```

An OpenAI API key is required. Set it as an environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

> **Note:** This app is a research prototype intended for demonstration purposes only. It is not a certified medical device and should not be used for clinical decision-making.

---

## Requirements

See `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

---

## Dataset

This study uses the publicly available **Iowa EEG dataset**. TFR images were generated from preprocessed EEG recordings using MATLAB (MSPCA artifact removal, ASR, bandpass/notch filtering, CAR referencing).

> Update the dataset path constants at the top of each training script to match your local directory structure before running.

---

## HPC / Cluster Usage

Experiments were run on the **ACES cluster** at Texas A&M University using SLURM job scheduling. If running locally or on a different HPC system:
- Adjust `NUM_WORKERS` in each script based on available CPU cores
- Adjust `BATCH_SIZE` based on GPU memory
- Single GPU training is assumed (`cuda:0`)

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{azadnouran2026eeg,
  title   = {EEG-Based Computer-Aided Diagnosis of Parkinson's Disease Using
             Time-Frequency Representations and Deep Learning},
  author  = {Azadnouran, Amir and others},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

> Update this entry with the final DOI and volume/page numbers upon acceptance.

---

## Related Publication

Study 1 of this project is published:

> Azadnouran et al. (2026). *BioMedInformatics*, MDPI.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
