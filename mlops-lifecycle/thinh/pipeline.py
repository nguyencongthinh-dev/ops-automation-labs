"""
pipeline.py — Train TWO anomaly detectors trên baseline data, log vào MLflow, register model.

Detectors:
  1. IsolationForest — tree-based, fast, handles high-dimensional data well.
     Anomaly score = average path length in random trees (shorter = more anomalous).
  2. LOF (Local Outlier Factor) — density-based, detects local outliers in clusters.
     Anomaly score = ratio of local density to k-nearest-neighbor densities.

Both detectors train on baseline.csv. The PRIMARY model registered in MLflow Registry
is IsolationForest (as required). LOF metrics are logged for comparison and research.

Supports AutoML hyperparameter tuning via Optuna (--tune flag) for IsolationForest.

Usage:
    # Standard training (both detectors):
    uv run python pipeline.py --data data/baseline.csv

    # AutoML tuning (tìm hyperparameters tốt nhất tự động):
    uv run python pipeline.py --data data/baseline.csv --tune --n-trials 50

    # Manual hyperparameters:
    uv run python pipeline.py --data data/baseline.csv --contamination 0.03 --n-estimators 100
"""

import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

EXPERIMENT_NAME = "anomaly-detection"
MODEL_NAME = "anomaly-detector"
FEATURES = ["latency_p99", "error_rate", "rps"]

logging.basicConfig(level=logging.INFO, format="[pipeline] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────── Data Loading ───────────────────────────────────

def load_features(csv_path: str) -> pd.DataFrame:
    """Load and validate feature columns from a CSV file."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return df[FEATURES].dropna()


# ─────────────────────────── AutoML Tuning ──────────────────────────────────

def _isolation_forest_objective(trial, X_scaled: np.ndarray) -> float:
    """Optuna objective: maximize the mean anomaly score magnitude (higher = more separable).

    We use the average absolute score_samples as a proxy for model quality:
    a model that assigns clearly negative scores to anomalies and scores close
    to 0 for normals is considered better. This is an unsupervised proxy metric.
    """
    contamination = trial.suggest_float("contamination", 0.01, 0.15, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)
    max_samples = trial.suggest_categorical("max_samples", ["auto", 128, 256])

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    scores = model.score_samples(X_scaled)
    # Maximise the spread (std) between normal and anomaly score distributions
    labels = model.predict(X_scaled)
    normal_scores = scores[labels == 1]
    anomaly_scores = scores[labels == -1]
    if len(anomaly_scores) == 0 or len(normal_scores) == 0:
        return 0.0
    # Higher separation = better model
    separation = float(normal_scores.mean() - anomaly_scores.mean())
    return separation


def tune_hyperparameters(X_scaled: np.ndarray, n_trials: int = 30) -> dict:
    """Run Optuna study to find best IsolationForest hyperparameters.

    Returns the best_params dict from the study.
    """
    import optuna
    # Silence Optuna's verbose trial logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="isolation-forest-tuning",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(
            lambda trial: _isolation_forest_objective(trial, X_scaled),
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=False,
        )

    best = study.best_params
    log.info("Optuna best params: %s", best)
    log.info("Optuna best score : %.4f", study.best_value)
    return best, study.best_value


# ─────────────────────────── LOF Detector ───────────────────────────

def train_lof(
    X_scaled: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.03,
) -> tuple["LocalOutlierFactor", float, float]:
    """Train LOF detector and return (model, anomaly_rate, negative_outlier_factor_mean).

    LOF uses k-nearest neighbor density estimation:
    - LOF score close to 1   = density similar to neighbors = NORMAL
    - LOF score >> 1         = much lower density than neighbors = OUTLIER (anomaly)

    Returns (lof_model, anomaly_rate, mean_lof_score)
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,   # fit_predict mode: trained on same dataset
        n_jobs=-1,
    )
    labels = lof.fit_predict(X_scaled)  # -1=anomaly, 1=normal
    anomaly_rate = float((labels == -1).mean())
    # negative_outlier_factor_: more negative = more outlier
    mean_lof = float(lof.negative_outlier_factor_.mean())
    return lof, anomaly_rate, mean_lof


# ─────────────────────────── Training ───────────────────────────

def train(
    data_path: str,
    contamination: float = 0.03,
    n_estimators: int = 100,
    random_state: int = 42,
    use_optuna: bool = False,
    n_trials: int = 30,
) -> None:
    """Train (optionally with Optuna AutoML), log, and register the model."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X = load_features(data_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── AutoML Tuning ────────────────────────────────────────────────────────
    best_score = None
    if use_optuna:
        log.info("Starting Optuna hyperparameter search (%d trials)...", n_trials)
        best_params, best_score = tune_hyperparameters(X_scaled, n_trials=n_trials)
        contamination = best_params["contamination"]
        n_estimators = best_params["n_estimators"]
        max_features = best_params.get("max_features", 1.0)
        max_samples = best_params.get("max_samples", "auto")
        log.info("Using tuned params: contamination=%.4f n_estimators=%d", contamination, n_estimators)
    else:
        max_features = 1.0
        max_samples = "auto"

    # ── Model Training ────────────────────────────────────────────────────────
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_features=max_features,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    labels = model.predict(X_scaled)
    anomaly_rate = float((labels == -1).mean())
    scores = model.score_samples(X_scaled)
    mean_score = float(scores.mean())

    # ── LOF Detector (Detector #2) ─────────────────────────────────────
    lof_model, lof_anomaly_rate, lof_mean_score = train_lof(
        X_scaled, n_neighbors=20, contamination=contamination
    )
    # Agreement rate: fraction of rows where both detectors agree
    if_labels = model.predict(X_scaled)  # -1/1
    lof_labels = lof_model.fit_predict(X_scaled)  # -1/1
    agreement_rate = float((if_labels == lof_labels).mean())

    # ── Print comparison table ─────────────────────────────────────────
    print()
    print("  Detector Comparison on Training Data")
    print("  " + "-" * 52)
    print(f"  {'Detector':<25} {'Anomaly Rate':>12} {'Mean Score':>12}")
    print("  " + "-" * 52)
    print(f"  {'IsolationForest':<25} {anomaly_rate:>12.4f} {mean_score:>12.4f}")
    print(f"  {'LOF (n_neighbors=20)':<25} {lof_anomaly_rate:>12.4f} {lof_mean_score:>12.4f}")
    print("  " + "-" * 52)
    print(f"  Detector agreement rate : {agreement_rate:.4f} ({agreement_rate*100:.1f}% of rows agree)")
    print()
    log.info("IsoForest anomaly_rate=%.4f, LOF anomaly_rate=%.4f, agreement=%.4f",
             anomaly_rate, lof_anomaly_rate, agreement_rate)

    # ── MLflow Logging ────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_samples", str(max_samples))
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("training_rows", len(X))
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("use_optuna", use_optuna)
        mlflow.log_param("detectors", "IsolationForest,LOF")
        if use_optuna:
            mlflow.log_param("n_trials", n_trials)

        # IsolationForest metrics (PRIMARY)
        mlflow.log_metric("train_anomaly_rate", anomaly_rate)
        mlflow.log_metric("feature_count", len(FEATURES))
        mlflow.log_metric("mean_score_sample", mean_score)
        if best_score is not None:
            mlflow.log_metric("optuna_best_separation", best_score)

        # LOF comparison metrics
        mlflow.log_metric("lof_anomaly_rate", lof_anomaly_rate)
        mlflow.log_metric("lof_mean_score", lof_mean_score)
        mlflow.log_metric("detector_agreement_rate", agreement_rate)

        # Log scaler as artifact
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(scaler, f)
            scaler_path = f.name
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        os.unlink(scaler_path)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X.head(3),
        )

        run_id = run.info.run_id
        log.info("Run ID      : %s", run_id)
        log.info("Anomaly rate: %.4f", anomaly_rate)
        if use_optuna:
            log.info("Optuna separation score: %.4f", best_score)

    # ── Register alias 'production' ───────────────────────────────────────────
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "production", latest.version)
    log.info("Registered  : %s v%s -> alias 'production'", MODEL_NAME, latest.version)
    log.info("MLflow UI   : %s/#/models/%s", tracking_uri, MODEL_NAME)


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train anomaly detection model (IsolationForest + optional Optuna AutoML)"
    )
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--contamination", type=float, default=0.03,
                        help="Fraction of expected anomalies (ignored if --tune is set)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees (ignored if --tune is set)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tune", action="store_true", default=False,
        help="Enable Optuna AutoML hyperparameter search (overrides --contamination / --n-estimators)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="Number of Optuna trials (only used when --tune is set, default: 30)",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        use_optuna=args.tune,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
