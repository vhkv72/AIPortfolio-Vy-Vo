# 🎓 Student Performance & Learning Predictor

**Course:** ITAI 1378

**Student:** Vy Vo | Houston City College

---

## Problem Statement

Student academic success depends on many overlapping factors — how much a student studies, whether they show up to class, their stress levels, and more. Early identification of at-risk students allows educators to provide targeted support before it's too late.

This project builds a machine learning classifier that predicts a student's **final academic grade** (0–3) using 14 behavioral, motivational, and demographic features — with no access to their exam score. The goal is not just accurate prediction, but understanding *which factors matter most* so that insights can be acted on.

---

## Approach

| Component | Choice |
|---|---|
| ML Technique | Multi-class Classification (4 classes) |
| Model | Random Forest Classifier |
| Framework | scikit-learn |
| Hyperparameter Tuning | RandomizedSearchCV + StratifiedKFold (3-fold) |
| Compute | Local — VS Code + Python virtual environment |

**Why Random Forest?**
Random Forest was chosen because it handles non-linear relationships between features, is robust to mixed feature types (numeric + ordinal), and provides built-in feature importance scores — directly supporting the goal of understanding what drives grades.

**Pipeline design:**
A scikit-learn `Pipeline` bundles preprocessing and the model together so the saved `.pkl` file can make predictions on raw data without any separate preprocessing step.

```
Raw CSV
  → Drop ExamScore (leakage prevention)
  → Train / Test Split (80/20, stratified)
  → ColumnTransformer (median imputation + OneHotEncoding)
  → RandomForestClassifier
  → RandomizedSearchCV (12 candidates)
  → Evaluate → Save model + plots
```

---

## Dataset

- **Source:** [Student Performance and Learning Behavior Dataset — Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style)
- **Size:** 14,003 rows × 16 columns, zero missing values
- **Features:** StudyHours, Attendance, AssignmentCompletion, Age, OnlineCourses, LearningStyle, StressLevel, Motivation, Resources, Gender, Internet, Extracurricular, Discussions, EduTech
- **Target:** `FinalGrade` — ordinal label (0 = highest, 3 = lowest)
- **Key decision:** `ExamScore` was dropped before training. It is a direct numeric encoding of `FinalGrade` (r = −0.97) and would cause data leakage, making the task trivially easy but meaningless.

> The dataset is not uploaded to GitHub due to size. Download it from the Kaggle link above and pass the path via `--data-path`.

---

## How to Run

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo and navigate to this project folder
cd Projects/MidTerm\ Project

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run full pipeline with hyperparameter tuning (recommended)
python student_performance_train.py --data-path path/to/student_performance.csv

# 4. Or skip tuning for a faster run
python student_performance_train.py --data-path path/to/student_performance.csv --no-tuning
```

Outputs are saved to `artifacts/`:
```
artifacts/
├── student_performance_model.pkl   # Trained pipeline — ready for inference
├── eda_overview.png                # Class distribution + correlation heatmap
├── confusion_matrix.png            # Predicted vs. true grades on test set
└── feature_importance.png          # Top 15 features by importance
```

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **87.0%** |
| Baseline (majority class) | 27.4% |
| Lift over baseline | +59.6 percentage points |
| Macro Precision | 87.0% |
| Macro Recall | 87.0% |
| Macro F1-Score | 87.0% |

**Per-class breakdown:**

| Grade | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (highest) | 86.9% | 87.1% | 87.0% | 766 |
| 1 | 87.3% | 85.1% | 86.2% | 662 |
| 2 | 88.3% | 89.2% | 88.7% | 724 |
| 3 (lowest) | 85.7% | 86.6% | 86.1% | 649 |

Performance is consistent across all four grade classes — the model does not favor any particular group. Remaining errors are almost entirely between **adjacent grade classes** (e.g., predicting Grade 1 when the true label is Grade 0), which is the expected failure mode for an ordinal target.

---

## Key Findings

**The top 5 features drive the majority of predictive signal:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | AssignmentCompletion | ~15% |
| 2 | Attendance | ~14.5% |
| 3 | OnlineCourses | ~13.5% |
| 4 | StudyHours | ~13% |
| 5 | Age | ~11% |

The three most actionable findings:
- **Completing assignments matters most.** It's the single strongest predictor — more than attendance or raw study hours.
- **Showing up is second.** Attendance is nearly as important as assignment completion and more controllable than demographic factors.
- **Internet access, extracurriculars, and discussions barely matter.** These features had near-zero importance, suggesting they don't meaningfully differentiate grade outcomes in this dataset.
- **Individual feature correlations are misleading.** Each feature alone has |r| < 0.05 with FinalGrade, yet the model reaches 87% accuracy by learning *combinations* — a clear case where a correlation heatmap alone would lead you to the wrong conclusion.

---

## What I Learned

- **Data leakage is the #1 trap in ML.** Including `ExamScore` would have given 95%+ accuracy but learned nothing real. Identifying and removing it was the most important decision in the project.
- **Framing the task correctly changes everything.** The project initially looked like a regression problem. Exploring the data revealed `FinalGrade` has only 4 discrete values — it's a classification problem, and accuracy/F1 are the right metrics, not RMSE.
- **Pipelines make ML reproducible.** Wrapping preprocessing + model in a scikit-learn `Pipeline` ensures the saved `.pkl` works correctly at inference time without any separate preprocessing step.
- **Feature importance reveals what matters, not just what correlates.** The Random Forest found strong signal in combinations of features that look uncorrelated individually — something linear methods would miss entirely.

---

## Technologies Used

- Python 3.14
- scikit-learn (Pipeline, RandomForestClassifier, RandomizedSearchCV, StratifiedKFold)
- pandas, NumPy
- Matplotlib, Seaborn
- VS Code + Python virtual environment

---

## AI Usage Log

- **Claude (Anthropic):** Used throughout the project for EDA interpretation, model selection guidance, pipeline code review, identifying the ExamScore leakage issue, and README writing. All code was reviewed, understood, and run locally.

---

## File Structure

```
MidTerm Project/
├── README.md                              # This file
├── student_performance_train.py           # Training script (CLI, fully reusable)
├── student_performance_classifier.ipynb   # Jupyter notebook walkthrough
├── MidTerm_Project_Report.docx            # Written report
└── artifacts/
    ├── student_performance_model.pkl
    ├── eda_overview.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## References

- Shamim, A. (2024). *Student Performance and Learning Behavior Dataset*. Kaggle. https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
