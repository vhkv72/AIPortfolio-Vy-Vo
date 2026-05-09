# Vy Vo – Applied AI Portfolio
**Houston City College · Applied AI & Robotics Program**

---

## About Me

I am an Applied AI student at Houston City College specializing in computer vision and machine learning. This portfolio showcases my hands-on project work across the Applied AI & Robotics program, where I built end-to-end AI systems — from raw image datasets to trained, evaluated deep learning models. I am passionate about applying AI to solve problems in healthcare and financial security.

---

## Repository Structure

```
Vy-Vo-AI-Portfolio/
├── README.md
├── ComputerVision-ITAI1378/
│   ├── README.md
│   ├── Fake-Currency-Detector/
│   │   ├── README.md
│   │   ├── Fake_Currency_Detector.ipynb
│   │   ├── requirements.txt
│   │   └── results/
│   │       ├── sample_images.png
│   │       ├── class_distribution.png
│   │       ├── accuracy_loss_plot.png
│   │       ├── confusion_matrix.png
│   │       └── metrics.json
│   └── Eye-Disease-Classifier/
│       ├── README.md
│       ├── train.py
│       ├── predict.py
│       ├── confusion.py
│       └── results/
│           └── confusion_matrix.png
└── MachineLearning-ITAI1371/
    ├── README.md
    └── Student-Performance-Predictor/
        ├── README.md
        ├── student_performance_classifier.ipynb
        ├── student_performance_train.py
        └── results/
            ├── eda_overview.png
            ├── feature_importance.png
            └── confusion_matrix.png
```

---

## Technical Skills

**Deep Learning & Computer Vision**
- Transfer learning with ResNet18 and ResNet50 (PyTorch / Torchvision)
- Image classification pipelines: data loading, augmentation, training, evaluation
- Data augmentation: random flips, rotations, color jitter
- Regularization: Dropout, AdamW weight decay, cosine annealing LR
- Early stopping and best-model checkpointing

**Machine Learning**
- scikit-learn pipelines with `ColumnTransformer` and `RandomizedSearchCV`
- Random Forest classification with balanced class weighting
- Feature importance analysis and EDA visualizations

**Tools & Workflow**
- Python 3 · PyTorch · Torchvision · OpenCV
- scikit-learn · pandas · NumPy · matplotlib · seaborn
- Jupyter Notebook · Google Colab
- GitHub version control

---

## Featured Courses & Projects

### 🔵 Computer Vision (ITAI 1378)

#### [Fake Currency Detector](./ComputerVision-ITAI1378/Fake-Currency-Detector/)
Built a binary image classifier to detect real vs. counterfeit banknotes, helping small businesses protect themselves from financial loss without expensive hardware.
- **Model:** ResNet50 fine-tuned via transfer learning (ImageNet weights)
- **Dataset:** [Fake Currency Data](https://www.kaggle.com/datasets/mdladla/fake-currency-data) · Labels: `real`, `fake`
- **Key techniques:** Data augmentation, Dropout, AdamW, cosine annealing LR, early stopping
- **Technologies:** Python · PyTorch · Torchvision · scikit-learn · matplotlib · Pillow

#### [Eye Disease Classifier](./ComputerVision-ITAI1378/Eye-Disease-Classifier/)
Built a 4-class retinal image classifier to detect AMD, Cataract, Diabetic Retinopathy, and Normal eyes from fundus photographs — supporting early diagnosis of vision-threatening conditions.
- **Model:** ResNet18 with Dropout (p=0.4) before the final linear layer
- **Dataset:** [Macular Degeneration Disease Dataset](https://www.kaggle.com/datasets/orvile/macular-degeneration-disease-dataset) · 4 classes: `amd`, `cataract`, `diabetes`, `normal`
- **Key techniques:** Transfer learning, per-class accuracy analysis, confusion matrix evaluation
- **Technologies:** Python · PyTorch · Torchvision · matplotlib · NumPy

---

### 🟢 Machine Learning (ITAI 1371)

#### [Student Performance Classifier](./MachineLearning-ITAI1371/Student-Performance-Predictor/)
Predicted student final grade categories from 14,003 records using 14 behavioral and demographic features, enabling educators to identify at-risk learners before they fall behind.
- **Model:** Random Forest Classifier with `RandomizedSearchCV` (12 candidates, 3-fold CV)
- **Dataset:** [Student Performance and Learning Behavior Dataset](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style) · 14,003 records · 16 features
- **Key techniques:** ColumnTransformer pipeline, One-Hot Encoding, feature importance analysis
- **Technologies:** Python · scikit-learn · pandas · matplotlib · seaborn

---

## AI Usage Disclosure

- **ChatGPT** was used to help organize proposal slides and structure GitHub README files.
- **Claude** was used to assist with notebook documentation and code structure.
- All model training decisions, hyperparameter choices, experimental analysis, and conclusions are my own work.

---

## Contact

- **LinkedIn:** [linkedin.com/in/vy-vo](https://www.linkedin.com/in/vy-v-25a8a026b/)
