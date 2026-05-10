# Vy Vo – Machine Learning Course (ITAI 1371)
**Houston City College · Applied AI & Robotics Program**

---

## About Me

Hi, I'm Vy Vo — an Applied AI student at Houston City College passionate about using machine learning to solve real-world problems. In this course I explored the full ML pipeline: from data cleaning and exploratory analysis through feature engineering, model training, hyperparameter optimization, and evaluation. I'm particularly interested in how behavioral and demographic data can help educators identify at-risk students early.

---

## Repository Structure

```
Vy-Vo-ML-Course/
├── README.md
├── Labs/
│   ├── Lab1/
│   │   ├── README.md
│   │   └── lab1_notebook.ipynb
│   └── Lab2/
│       ├── README.md
│       └── lab2_notebook.ipynb
├── Assignments/
│   ├── Assignment1/
│   │   ├── README.md
│   │   └── assignment1_notebook.ipynb
│   └── Assignment2/
│       ├── README.md
│       └── assignment2_notebook.ipynb
└── Projects/
    ├── MidTerm Project/
    │   ├── README.md
    │   └── midterm_notebook.ipynb
    └── Final Project/
        ├── README.md
        ├── student_performance_train.py
        ├── student_performance_classifier.ipynb
        ├── student_performance.csv
        └── results/
            ├── eda_overview.png
            ├── feature_importance.png
            └── confusion_matrix.png
```

---

## Labs Completed

| Lab | Topic | Description |
|-----|-------|-------------|
| Lab 1 | Introduction to ML & Python | Setting up the environment, working with NumPy and pandas, exploring a dataset |
| Lab 2 | Data Preprocessing & EDA | Handling missing values, visualizing distributions, encoding categorical variables |

---

## Assignments Completed

| Assignment | Topic | Description |
|------------|-------|-------------|
| Assignment 1 | Supervised Learning | Trained and compared classification models on a structured dataset |
| Assignment 2 | Model Evaluation | Applied cross-validation, confusion matrices, and classification reports |

---

## Projects

### 🎓 Final Project – Student Performance Predictor

**Goal:** Predict a student's final grade category from behavioral, demographic, and academic engagement features to help educators identify at-risk learners early.

**Dataset:** `student_performance.csv` — 14,003 student records, 16 features (StudyHours, Attendance, Motivation, StressLevel, Internet access, LearningStyle, and more).

**Approach:**
- Random Forest Classifier with balanced class weighting
- Hyperparameter tuning via `RandomizedSearchCV` (12 candidates, 3-fold stratified CV)
- `ColumnTransformer` pipeline: median imputation for numeric features, One-Hot Encoding for 9 categorical features
- Leakage prevention: `ExamScore` column excluded from training features

**Results & Outputs:**
- `results/eda_overview.png` — class distribution + correlation heatmap
- `results/feature_importance.png` — top 15 predictive features
- `results/confusion_matrix.png` — model performance across grade classes
- `student_performance_model.pkl` — serialized best model

**Technologies:** Python · scikit-learn · pandas · matplotlib · seaborn · Jupyter Notebook

📁 [View Final Project folder](./Projects/Final%20Project/)

---

## Technical Skills Learned

- **Core ML:** Supervised classification, cross-validation, hyperparameter tuning
- **Data Handling:** pandas pipelines, missing value imputation, One-Hot Encoding
- **Evaluation:** Confusion matrices, precision/recall/F1, feature importance
- **Tools:** scikit-learn, pandas, matplotlib, seaborn, Jupyter Notebook, Google Colab, GitHub

---

## Contact

- **GitHub:** [github.com/VyVo](https://github.com/vhkv72)
- **LinkedIn:** [linkedin.com/in/vy-vo](https://www.linkedin.com/in/vy-v-25a8a026b/)
