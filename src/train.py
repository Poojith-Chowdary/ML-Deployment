# --- path bootstrap (makes `python -m src.train` work from any shell) ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -------------------------------------------------------------------------

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

from src.config import MODEL_PATH, MLFLOW_TRACKING_URI, MODEL_VERSION
from src.preprocess import load_raw_data, clean, build_preprocessor


def tune_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in [i / 100 for i in range(10, 70)]:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t


def train():
    print("Loading data...")
    df = load_raw_data()
    X, y = clean(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Class balance (test): {y_test.value_counts().to_dict()}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn-prediction")

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
    }

    with mlflow.start_run(run_name=f"rf-v{MODEL_VERSION}"):
        mlflow.log_params(params)
        mlflow.log_param("model_version", MODEL_VERSION)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        preprocessor = build_preprocessor()
        clf = RandomForestClassifier(**params)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        print("Training model...")
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]

        threshold = tune_threshold(y_test, y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        print(f"Optimal decision threshold: {threshold}")

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "decision_threshold": threshold,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_param("decision_threshold", threshold)

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["Retain", "Churn"], zero_division=0))

        mlflow.sklearn.log_model(pipeline, "model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "threshold": threshold, "version": MODEL_VERSION}, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return pipeline, threshold


if __name__ == "__main__":
    train()