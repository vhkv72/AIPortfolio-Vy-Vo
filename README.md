# 👁️ Macular Degeneration Disease Classifier

**Course:** *ITAI 1378*  
**Student:** Vy Vo | Houston City College

---

## Problem Statement

Age-related Macular Degeneration (AMD) is a leading cause of vision loss affecting **196 million people worldwide**. Early detection from retinal fundus images can significantly improve patient outcomes. This project builds an automated deep learning classifier that identifies four retinal conditions — AMD, Cataract, Diabetic Retinopathy, and Normal — from fundus images using transfer learning.

---

## Approach

| Component | Choice |
|-----------|--------|
| CV Technique | Image Classification |
| Model | ResNet-18 (pretrained on ImageNet) |
| Framework | PyTorch + TorchVision |
| Compute | Local (Mac M-series, MPS GPU) |
| Regularization | Dropout (p=0.4) + Data Augmentation |
| Training Strategy | Transfer Learning + Early Stopping |

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/orvile/macular-degeneration-disease-dataset
- **Name:** AMDNet23 Fundus Image Dataset for Age-Related Macular Degeneration Disease Detection
- **Total Images:** 1,994 retinal fundus photographs
- **Labels:** AMD, Cataract, Diabetes (Diabetic Retinopathy), Normal
- **Split:** 1,594 train / 400 validation (80/20)
- **Class Distribution:**

| Class | Train | Validation |
|-------|-------|------------|
| AMD | 394 | 100 |
| Cataract | 400 | 100 |
| Diabetes | 400 | 100 |
| Normal | 400 | 100 |

> **Note:** Dataset not uploaded to GitHub. Download from the Kaggle link above.

---

## How to Run

1. Clone this repository
2. Download the dataset from the Kaggle link above and unzip it
3. Install dependencies:
   ```bash
   pip3 install torch torchvision
   ```
4. Train the model:
   ```bash
   python3 train.py
   ```
5. Run predictions on validation images:
   ```bash
   python3 predict.py
   ```
6. Generate confusion matrix:
   ```bash
   pip3 install matplotlib
   python3 confusion.py
   ```

---

## Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **98.2%** |
| Train Accuracy (at best epoch) | 97.4% |
| Train/Val Gap (overfitting) | 0.8% |
| Best Epoch | 8 / 20 (early stopping) |

### Per-Class Performance

| Class | Correct | Errors | Accuracy |
|-------|---------|--------|----------|
| AMD | 99/100 | 1 | **99.0%** |
| Cataract | 99/100 | 1 | **99.0%** |
| Diabetes | 97/100 | 3 | **97.0%** |
| Normal | 98/100 | 2 | **98.0%** |

> The small confusion between Diabetes and Normal is clinically expected — early diabetic retinopathy can present subtly and resemble a healthy retina.

---

## Model Architecture

```
ResNet-18 (pretrained on ImageNet)
    └── All convolutional layers (frozen feature extractor)
    └── FC layer replaced with:
            Dropout(p=0.4)
            Linear(512 → 4 classes)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Max Epochs | 20 (early stopping patience=3) |
| Input Size | 224 x 224 |

### Data Augmentation (Training Only)

- Random horizontal & vertical flips
- Random rotation +/-15 degrees
- Color jitter (brightness & contrast +/-0.2)
- ImageNet normalization ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.14 | Programming language |
| PyTorch 2.11 | Deep learning framework |
| TorchVision 0.26 | Pretrained models & transforms |
| Matplotlib | Confusion matrix visualization |
| VS Code | Development environment |
| Mac MPS (Metal) | GPU acceleration |

---

## Project Files

| File | Description |
|------|-------------|
| `train.py` | Model training with augmentation + early stopping |
| `predict.py` | Run predictions on single images with confidence scores |
| `confusion.py` | Generate confusion matrix on validation set |
| `eye_disease_model.pth` | Saved best model weights (98.2% val accuracy) |
| `dataset.csv` | Dataset class distribution summary |

---

## Key Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Deeply nested dataset folder structure | Used os.walk() to discover actual image paths |
| Mac SSL certificate error blocking model download | Ran /Applications/Python 3.14/Install Certificates.command |
| Overfitting (3% train/val gap in first model) | Added Dropout + data augmentation + early stopping, reduced to 0.8% |
| Kaggle notebook showing 0 images | Switched to local VS Code workflow with correct path resolution |

---

## AI Usage Log

This project was developed with the assistance of Claude (Anthropic) for:
- Step-by-step guidance on dataset setup and environment configuration
- Debugging SSL certificate and folder path issues on macOS
- Writing and iterating on `train.py`, `predict.py`, and `confusion.py`
- Explaining transfer learning, dropout, and early stopping concepts
- Generating the confusion matrix visualization

All code was reviewed, run, and validated locally by the student.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385

2. Orvile. (2023). *AMDNet23 Macular Degeneration Disease Dataset*. Kaggle. https://www.kaggle.com/datasets/orvile/macular-degeneration-disease-dataset

3. Wong, W. L., et al. (2014). Global prevalence of age-related macular degeneration. *The Lancet Global Health*, 2(2), e106-e116.

4. Ting, D. S. W., et al. (2017). Development and Validation of a Deep Learning System for Diabetic Retinopathy. *JAMA*, 318(22), 2211-2223.

5. PyTorch Team. (2024). *TorchVision Models Documentation*. https://pytorch.org/vision/stable/models.html
