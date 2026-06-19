"""
drift_detector.py — Evidently DataDriftPreset wrapper cho anomaly detection pipeline.

Computes per-feature drift score giữa reference (baseline) và current (production window).
Flags drift khi dataset-level drift score vượt threshold.

Outputs:
  - HTML Report (outputs/drift_reports/drift-report-<label>-<ts>.html)
  - Summary JSON (outputs/drift_reports/drift-summary-<label>-<ts>.json)
  - Logs artifact + metrics vào MLflow (nếu --log-mlflow)

Usage:
    uv run python drift_detector.py \\
        --reference data/baseline.csv \\
        --current   data/drifted.csv \\
        --threshold 0.15

    # Performance check (concept drift detection):
    uv run python drift_detector.py \\
        --reference data/baseline.csv \\
        --current   data/drifted.csv \\
        --check-mode performance \\
        --labeled-current data/drifted.csv \\
        --model-uri models:/anomaly-detector@production

    # Combined (default): runs both data + performance checks
    uv run python drift_detector.py \\
        --reference data/baseline.csv \\
        --current   data/drifted.csv \\
        --check-mode combined \\
        --labeled-current data/drifted.csv \\
        --model-uri models:/anomaly-detector@production
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

FEATURES = ["latency_p99", "error_rate", "rps"]
DEFAULT_THRESHOLD = 0.15
DEFAULT_PERF_THRESHOLD = 0.70   # minimum acceptable precision on labeled holdout
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "drift_reports"
)

logging.basicConfig(level=logging.INFO, format="[drift_detector] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────── Data Classes ────────────────────────────────────

@dataclass
class FeatureDriftDetail:
    """Per-feature drift breakdown."""
    feature: str
    drift_detected: bool
    stat_test: str = ""
    p_value: Optional[float] = None
    drift_score: Optional[float] = None


@dataclass
class DriftResult:
    score: float            # fraction of features drifted (0.0-1.0)
    is_drift: bool
    threshold: float
    drifted_features: list
    report_path: str
    summary_path: str
    timestamp: str
    feature_details: list = field(default_factory=list)  # list[FeatureDriftDetail]
    # Performance check fields
    perf_precision: Optional[float] = None
    perf_recall: Optional[float] = None
    perf_is_degraded: bool = False
    perf_threshold: float = DEFAULT_PERF_THRESHOLD


# ─────────────────────────── Core Detection ──────────────────────────────────

def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    report_label: str = "",
) -> DriftResult:
    """
    Chay Evidently DataDriftPreset + DataQualityPreset, tra ve DriftResult.

    reference_df: training distribution (baseline)
    current_df:   production window data
    threshold:    drift score nguong (0.0-1.0). Score = fraction of features drifted.
    """
    ref = reference_df[FEATURES].copy()
    cur = current_df[FEATURES].copy()

    # ── Run Evidently Report (DataDrift + DataQuality) ───────────────────────
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)

    result_dict = report.as_dict()

    # ── Parse DataDriftPreset results ────────────────────────────────────────
    drift_metrics = result_dict["metrics"][0]["result"]

    # Find the metric that contains 'drift_by_columns' (handles Evidently version differences)
    table_metrics: dict = {}
    for metric in result_dict.get("metrics", []):
        if "drift_by_columns" in metric.get("result", {}):
            table_metrics = metric["result"]
            break

    share_drifted = drift_metrics.get("share_of_drifted_columns", 0.0)
    per_feature = table_metrics.get("drift_by_columns", {})
    drifted_features = [
        feat for feat, info in per_feature.items()
        if info.get("drift_detected", False)
    ]

    # ── Rich per-feature details ──────────────────────────────────────────────
    feature_details: list[FeatureDriftDetail] = []
    for feat in FEATURES:
        info = per_feature.get(feat, {})
        feature_details.append(FeatureDriftDetail(
            feature=feat,
            drift_detected=info.get("drift_detected", False),
            stat_test=info.get("stattest_name", ""),
            p_value=info.get("p_value"),
            drift_score=info.get("drift_score"),
        ))

    # ── Save HTML Report ──────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    label = f"-{report_label}" if report_label else ""
    report_filename = f"drift-report{label}-{ts}.html"
    report_path = os.path.join(REPORT_DIR, report_filename)
    report.save_html(report_path)
    log.info("HTML report saved: %s", report_path)

    # ── Save Summary JSON ─────────────────────────────────────────────────────
    summary = {
        "timestamp": ts,
        "label": report_label or "unlabeled",
        "drift_score": float(share_drifted),
        "is_drift": float(share_drifted) > threshold,
        "threshold": threshold,
        "drifted_features": drifted_features,
        "feature_details": [
            {
                "feature": d.feature,
                "drift_detected": d.drift_detected,
                "stat_test": d.stat_test,
                "p_value": d.p_value,
                "drift_score": d.drift_score,
            }
            for d in feature_details
        ],
        "reference_rows": len(ref),
        "current_rows": len(cur),
        "report_path": report_path,
    }
    summary_filename = f"drift-summary{label}-{ts}.json"
    summary_path = os.path.join(REPORT_DIR, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    log.info("Summary JSON saved: %s", summary_path)

    return DriftResult(
        score=float(share_drifted),
        is_drift=float(share_drifted) > threshold,
        threshold=threshold,
        drifted_features=drifted_features,
        report_path=report_path,
        summary_path=summary_path,
        timestamp=ts,
        feature_details=feature_details,
    )


# ─────────────────────────── Performance Drift ───────────────────────────────

def check_performance_drift(
    labeled_df: pd.DataFrame,
    model_uri: str,
    perf_threshold: float = DEFAULT_PERF_THRESHOLD,
) -> tuple[float, float, bool]:
    """Evaluate model precision/recall tren labeled holdout de phat hien concept drift.

    labeled_df phai co cot `anomaly_label` (0=normal, 1=anomaly).
    model_uri: MLflow model URI, e.g. 'models:/anomaly-detector@production'.

    Returns (precision, recall, is_degraded).
    is_degraded = True neu precision < perf_threshold.
    """
    import mlflow.pyfunc

    if "anomaly_label" not in labeled_df.columns:
        raise ValueError("labeled_df must contain 'anomaly_label' column (0=normal, 1=anomaly)")

    model = mlflow.pyfunc.load_model(model_uri)
    X = labeled_df[FEATURES].dropna()
    y_true = labeled_df.loc[X.index, "anomaly_label"].values

    # IsolationForest predict: -1=anomaly, 1=normal -> remap to 1/0
    raw_preds = model.predict(pd.DataFrame(X, columns=FEATURES))
    if hasattr(raw_preds, "values"):
        raw_preds = raw_preds.values
    # Handle both sklearn (-1/1) and already-remapped (0/1) outputs
    if set(raw_preds).issubset({-1, 1}):
        y_pred = (raw_preds == -1).astype(int)
    else:
        y_pred = raw_preds.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    is_degraded = precision < perf_threshold

    return precision, recall, is_degraded


# ─────────────────────────── MLflow Logging ──────────────────────────────────

def log_to_mlflow(result: DriftResult, experiment_name: str = "anomaly-detection-drift") -> None:
    """Log drift score, HTML report, and summary JSON vao MLflow de visualize trend."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"drift-check-{result.timestamp}"):
        mlflow.log_metric("drift_score", result.score)
        mlflow.log_metric("is_drift", float(result.is_drift))
        mlflow.log_metric("n_drifted_features", len(result.drifted_features))
        mlflow.log_param("threshold", result.threshold)
        mlflow.log_param("drifted_features", ",".join(result.drifted_features) or "none")

        # Log per-feature drift details as metrics
        for detail in result.feature_details:
            safe_name = detail.feature.replace(" ", "_")
            mlflow.log_metric(f"drift_{safe_name}", float(detail.drift_detected))
            if detail.p_value is not None:
                mlflow.log_metric(f"pval_{safe_name}", float(detail.p_value))

        # Log HTML report + summary JSON as artifacts
        if result.report_path and os.path.exists(result.report_path):
            mlflow.log_artifact(result.report_path, artifact_path="drift_reports")
        if result.summary_path and os.path.exists(result.summary_path):
            mlflow.log_artifact(result.summary_path, artifact_path="drift_reports")

        if result.perf_precision is not None:
            mlflow.log_metric("perf_precision", result.perf_precision)
            mlflow.log_metric("perf_recall", result.perf_recall)
            mlflow.log_metric("perf_is_degraded", float(result.perf_is_degraded))

    log.info("Drift results logged to MLflow experiment '%s'", experiment_name)


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect data drift between two CSVs")
    parser.add_argument("--reference", required=True, help="Path to reference (baseline) CSV")
    parser.add_argument("--current", required=True, help="Path to current (production window) CSV")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Drift score threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument(
        "--check-mode", choices=["data", "performance", "combined"], default="combined",
        help="data: Evidently DataDriftPreset only; performance: precision/recall on labeled data; "
             "combined: both (default)",
    )
    parser.add_argument("--labeled-current", default=None,
                        help="CSV with anomaly_label column -- required for performance/combined mode")
    parser.add_argument("--model-uri", default="models:/anomaly-detector@production",
                        help="MLflow model URI for performance evaluation")
    parser.add_argument("--perf-threshold", type=float, default=DEFAULT_PERF_THRESHOLD,
                        help=f"Minimum acceptable precision (default: {DEFAULT_PERF_THRESHOLD})")
    parser.add_argument("--log-mlflow", action="store_true", default=False,
                        help="Log drift score and reports to MLflow tracking server")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference)
    cur_df = pd.read_csv(args.current)

    # ── Data drift check ──────────────────────────────────────────────────────
    if args.check_mode in ("data", "combined"):
        result = detect_drift(ref_df, cur_df, threshold=args.threshold)
        print(f"[drift_detector] check_mode       : {args.check_mode}")
        print(f"[drift_detector] Drift score      : {result.score:.4f}")
        print(f"[drift_detector] Threshold        : {result.threshold}")
        print(f"[drift_detector] Drift detected   : {result.is_drift}")
        print(f"[drift_detector] Drifted features : {result.drifted_features}")
        print(f"[drift_detector] HTML Report      : {result.report_path}")
        print(f"[drift_detector] Summary JSON     : {result.summary_path}")
        print()
        print("[drift_detector] Per-feature breakdown:")
        for detail in result.feature_details:
            status = "DRIFT" if detail.drift_detected else "OK   "
            p_str = f"p={detail.p_value:.4f}" if detail.p_value is not None else "p=N/A"
            print(f"  [{status}] {detail.feature:<20} {p_str}  test={detail.stat_test}")
    else:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        result = DriftResult(
            score=0.0, is_drift=False, threshold=args.threshold,
            drifted_features=[], report_path="", summary_path="", timestamp=ts,
        )

    # ── Performance (concept drift) check ────────────────────────────────────
    if args.check_mode in ("performance", "combined"):
        if not args.labeled_current:
            parser.error("--labeled-current is required for performance/combined mode")
        labeled_df = pd.read_csv(args.labeled_current)
        precision, recall, is_degraded = check_performance_drift(
            labeled_df, args.model_uri, perf_threshold=args.perf_threshold,
        )
        result.perf_precision = precision
        result.perf_recall = recall
        result.perf_is_degraded = is_degraded
        result.perf_threshold = args.perf_threshold
        print(f"[drift_detector] Perf precision   : {precision:.4f}  (threshold {args.perf_threshold})")
        print(f"[drift_detector] Perf recall      : {recall:.4f}")
        print(f"[drift_detector] Perf degraded    : {is_degraded}")

    any_drift = result.is_drift or result.perf_is_degraded

    if args.log_mlflow:
        log_to_mlflow(result)

    # Push metrics to Prometheus Pushgateway
    try:
        from metrics_util import push_drift_score, push_model_eval
        push_drift_score(result.score, result.threshold)
        if result.perf_precision is not None:
            f1 = 0.0
            if (result.perf_precision + result.perf_recall) > 0:
                f1 = (2 * result.perf_precision * result.perf_recall
                      / (result.perf_precision + result.perf_recall))
            push_model_eval("current", result.perf_precision, result.perf_recall, f1)
    except ImportError:
        pass

    raise SystemExit(1 if any_drift else 0)


if __name__ == "__main__":
    main()
