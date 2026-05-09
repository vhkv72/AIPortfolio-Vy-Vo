# 💵 Fake Currency Detector

**Course:** ITAI 1378 – Computer Vision  
**Student:** Vy Vo | Houston City College  
**Project Tier:** Tier 1

---

## Problem Statement

Counterfeit money causes significant financial losses for businesses and individuals. Small businesses often lack advanced detection tools. This project builds a practical image classification system using deep learning to help detect fake vs. real currency from a banknote image.

## Approach

| Component | Choice |
|-----------|--------|
| CV Technique | Image Classification |
| Model | ResNet50 (Transfer Learning) |
| Framework | PyTorch |
| Compute | Google Colab |

**Why ResNet50?** ResNet50's residual connections make it excellent for image classification tasks. Transfer learning from ImageNet allows strong performance even with smaller datasets by leveraging pre-learned visual features.

## Dataset

- **Source:** [Fake Currency Data](https://www.kaggle.com/datasets/mdladla/fake-currency-data)
- **Labels:** `real`, `fake`
- **Note:** Download the dataset from Kaggle and place it as:
  ```
  data/
  ├── real/   ← real banknote images
  └── fake/   ← fake banknote images
  ```
- Do **not** upload the dataset to GitHub (Kaggle terms of use).

## How to Run

1. Open `Fake_Currency_Detector.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Download the dataset from the Kaggle link above and upload to Colab
3. Run all cells from top to bottom
4. Results are saved to the `results/` folder

## Results

| Metric | Value | Target |
|--------|-------|--------|
| Test Accuracy | ≥ 90% | ≥ 90% |
| Precision | — | — |
| Recall | — | — |
| Inference Speed | < 50 ms | < 1000 ms |

> _Results populate after running the notebook._

## Key Techniques

- **Transfer Learning:** Pretrained ResNet50 (ImageNet weights), fine-tuned on currency images
- **Data Augmentation:** Random flips, rotations, color jitter to reduce overfitting
- **Regularization:** Dropout layers + label smoothing + AdamW weight decay
- **Scheduler:** Cosine annealing LR for smoother convergence
- **Early Stopping:** Prevents overfitting on small datasets

## Technologies Used

`Python` · `PyTorch` · `Torchvision` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Pillow` · `Google Colab`

## Results Files (generated after running notebook)

```
results/
├── sample_images.png          # Grid of training samples
├── class_distribution.png     # Class balance bar chart
├── accuracy_loss_plot.png     # Training/validation curves
├── confusion_matrix.png       # Confusion matrix (counts + normalized)
├── metrics.json               # Final test metrics
└── best_model.pth             # Best model weights
```

## AI Usage Log

- Used **ChatGPT** to help organize the proposal slides and GitHub README.
- Used **Claude** to structure the notebook and assist with documentation.
- All model code, training decisions, and analysis are the student's own work.

## References

- Dataset: [Kaggle – Fake Currency Data](https://www.kaggle.com/datasets/mdladla/fake-currency-data)
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- [PyTorch Documentation](https://pytorch.org/docs/)
