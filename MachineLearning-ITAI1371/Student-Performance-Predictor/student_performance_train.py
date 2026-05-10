from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET_COLUMN = "FinalGrade"
LEAKAGE_COLUMNS = ["ExamScore"]
CATEGORICAL_COLUMNS = [
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "LearningStyle",
    "Discussions",
    "EduTech",
    "StressLevel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a student performance classifier.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/Users/rosavo/Downloads/student_performance.csv"),
        help="Path to the student performance CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where the trained model and plots will be saved.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip hyperparameter tuning and train the default Random Forest pipeline.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the data reserved for the test split.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="Number of RandomizedSearchCV candidates to evaluate.",
    )
    return parser.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def build_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_columns = [column for column in feature_columns if column not in CATEGORICAL_COLUMNS]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def tune_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int) -> RandomizedSearchCV:
    param_distributions = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.7],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, str, pd.DataFrame, pd.Series]:
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, digits=4)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return accuracy, report, importance_df, y_pred


def save_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    eda_df = df.drop(columns=LEAKAGE_COLUMNS)
    corr = eda_df.corr(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    class_colors = sns.color_palette("viridis", n_colors=len(target_counts))
    axes[0].bar(target_counts.index.astype(str), target_counts.values, color=class_colors)
    axes[0].set_title("FinalGrade Class Distribution")
    axes[0].set_xlabel("FinalGrade")
    axes[0].set_ylabel("Count")

    sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Correlation Heatmap Without ExamScore")
    plt.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(importance_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x="importance", y="feature", color="#2a9d8f")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    y_pred = model.predict(X_test)
    class_labels = [str(label) for label in sorted(y_test.unique())]
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_labels,
        cmap="Blues",
        values_format="d",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in {args.data_path}")

    feature_columns = [column for column in df.columns if column not in [TARGET_COLUMN, *LEAKAGE_COLUMNS]]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_pipeline = build_pipeline(feature_columns)

    if args.no_tuning:
        base_pipeline.fit(X_train, y_train)
        best_model = base_pipeline
        best_params = {}
        best_cv_score = None
    else:
        search = tune_model(base_pipeline, X_train, y_train, args.n_iter)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

    accuracy, report, importance_df, y_pred = evaluate_model(best_model, X_test, y_test)

    save_eda_plots(df, output_dir)
    save_feature_importance_plot(importance_df, output_dir)
    save_confusion_matrix(best_model, X_test, y_test, output_dir)

    model_path = output_dir / "student_performance_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(best_model, handle)

    print(f"Data shape: {df.shape}")
    print(f"Saved model to: {model_path}")
    print(f"Saved plots to: {output_dir}")
    if best_cv_score is not None:
        print(f"Best cross-validated accuracy: {best_cv_score:.4f}")
        print(f"Best parameters: {best_params}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)
    print("Top 10 features:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()