"""
test_pipeline.py — Unit test suite cho MLOps pipeline (TDD).

Tests cover:
  - Feature loading (valid + invalid CSV)
  - Drift detection (no drift + significant drift)
  - Model training (train_model_on_df)
  - Pydantic data validation in serve.py (strict business rules)
  - Optuna hyperparameter tuning integration
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the 'thinh' directory to sys.path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drift_detector import DriftResult, FeatureDriftDetail, detect_drift
from pipeline import FEATURES, load_features, train_lof, tune_hyperparameters
from retrain import train_model_on_df

# ─────────────────────────── Fixtures ────────────────────────────────────────

@pytest.fixture
def normal_df():
    """A small normally-distributed DataFrame (no anomalies)."""
    np.random.seed(42)
    return pd.DataFrame({
        "latency_p99": np.random.normal(120, 1, 100),
        "error_rate": np.random.normal(0.8, 0.05, 100),
        "rps": np.random.normal(450, 5, 100)
    })


@pytest.fixture
def drifted_df():
    """A DataFrame with heavily shifted distributions (simulates drift)."""
    np.random.seed(0)
    return pd.DataFrame({
        "latency_p99": np.random.normal(300, 10, 100),   # shifted from 120 -> 300
        "error_rate": np.random.normal(3.0, 0.2, 100),   # shifted from 0.8 -> 3.0
        "rps": np.random.normal(900, 15, 100)             # shifted from 450 -> 900
    })


# ─────────────────────────── Feature Loading Tests ───────────────────────────

def test_load_features_valid(tmp_path):
    """load_features must return correct columns and row count."""
    df_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-18", periods=5, freq="10min"),
        "latency_p99": [120.0, 122.0, 118.0, 125.0, 121.0],
        "error_rate": [0.8, 0.9, 0.7, 1.0, 0.8],
        "rps": [450, 460, 440, 470, 455]
    })
    csv_file = tmp_path / "valid_data.csv"
    df_data.to_csv(csv_file, index=False)

    loaded_df = load_features(str(csv_file))
    assert list(loaded_df.columns) == FEATURES
    assert len(loaded_df) == 5


def test_load_features_missing_cols(tmp_path):
    """load_features must raise ValueError when required columns are absent."""
    df_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-18", periods=5, freq="10min"),
        "latency_p99": [120.0, 122.0, 118.0, 125.0, 121.0],
        "error_rate": [0.8, 0.9, 0.7, 1.0, 0.8]
        # 'rps' intentionally omitted
    })
    csv_file = tmp_path / "invalid_data.csv"
    df_data.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_features(str(csv_file))


# ─────────────────────────── Drift Detection Tests ───────────────────────────

def test_detect_drift_no_drift(normal_df):
    """Identical distributions should not trigger drift detection."""
    np.random.seed(42)
    cur_df = pd.DataFrame({
        "latency_p99": np.random.normal(120, 1, 100),
        "error_rate": np.random.normal(0.8, 0.05, 100),
        "rps": np.random.normal(450, 5, 100)
    })

    result = detect_drift(normal_df, cur_df, threshold=0.15, report_label="test-no-drift")
    assert isinstance(result, DriftResult)
    assert result.is_drift is False
    assert os.path.exists(result.report_path)
    assert os.path.exists(result.summary_path)

    # Verify feature_details are populated correctly
    assert len(result.feature_details) == len(FEATURES)
    for detail in result.feature_details:
        assert isinstance(detail, FeatureDriftDetail)
        assert detail.feature in FEATURES

    # Cleanup
    for path in [result.report_path, result.summary_path]:
        if os.path.exists(path):
            os.remove(path)


def test_detect_drift_with_drift(normal_df, drifted_df):
    """Heavily shifted distributions must trigger drift detection."""
    result = detect_drift(normal_df, drifted_df, threshold=0.15, report_label="test-drift")
    assert isinstance(result, DriftResult)
    assert result.is_drift is True
    assert len(result.drifted_features) > 0

    # Summary JSON must exist and be non-empty
    assert os.path.exists(result.summary_path)
    import json
    with open(result.summary_path) as fp:
        summary = json.load(fp)
    assert summary["is_drift"] is True
    assert len(summary["drifted_features"]) > 0
    assert len(summary["feature_details"]) == len(FEATURES)

    # Cleanup
    for path in [result.report_path, result.summary_path]:
        if os.path.exists(path):
            os.remove(path)


def test_drift_result_has_feature_details(normal_df, drifted_df):
    """detect_drift must populate per-feature breakdown."""
    result = detect_drift(normal_df, drifted_df, threshold=0.15, report_label="test-details")
    assert len(result.feature_details) == 3
    feature_names = {d.feature for d in result.feature_details}
    assert feature_names == set(FEATURES)

    # Cleanup
    for path in [result.report_path, result.summary_path]:
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────── Model Training Tests ────────────────────────────

def test_train_model_on_df(normal_df):
    """train_model_on_df must return a valid fitted model and scaler."""
    model, scaler, anomaly_rate, n_rows = train_model_on_df(
        normal_df, contamination=0.03, n_estimators=10
    )
    assert model is not None
    assert scaler is not None
    assert 0.0 <= anomaly_rate <= 1.0
    assert n_rows == 100


def test_train_model_predictions_shape(normal_df):
    """Model predictions must have correct shape and valid label set."""
    model, scaler, _, _ = train_model_on_df(normal_df, contamination=0.03, n_estimators=10)
    X_scaled = scaler.transform(normal_df[FEATURES])
    preds = model.predict(X_scaled)
    assert preds.shape == (100,)
    assert set(preds).issubset({-1, 1}), f"Unexpected label values: {set(preds)}"


# ─────────────────────────── Pydantic Validation Tests ───────────────────────

class TestPredictRequestValidation:
    """Tests for serve.py FeatureRow strict business-rule validation."""

    def _get_schemas(self):
        from serve import FeatureRow, PredictRequest
        return PredictRequest, FeatureRow

    def test_valid_structured_request(self):
        """Valid structured dict input must be accepted."""
        PredictRequest, FeatureRow = self._get_schemas()
        req = PredictRequest(features=[{"latency_p99": 120.0, "error_rate": 0.01, "rps": 450.0}])
        assert len(req.features) == 1
        assert req.features[0].latency_p99 == 120.0

    def test_valid_raw_matrix_request(self):
        """Valid raw [[float, float, float]] input must be normalised and accepted."""
        PredictRequest, _ = self._get_schemas()
        req = PredictRequest(features=[[120.0, 0.01, 450.0]])
        assert len(req.features) == 1
        assert req.features[0].error_rate == 0.01

    def test_rejects_negative_latency(self):
        """latency_p99 <= 0 must be rejected."""
        from pydantic import ValidationError
        _, FeatureRow = self._get_schemas()
        with pytest.raises(ValidationError, match="greater than 0"):
            FeatureRow(latency_p99=-1.0, error_rate=0.01, rps=450.0)

    def test_rejects_zero_latency(self):
        """latency_p99 == 0 must be rejected (must be strictly > 0)."""
        from pydantic import ValidationError
        _, FeatureRow = self._get_schemas()
        with pytest.raises(ValidationError, match="greater than 0"):
            FeatureRow(latency_p99=0.0, error_rate=0.01, rps=450.0)

    def test_rejects_error_rate_above_one(self):
        """error_rate > 1.0 must be rejected."""
        from pydantic import ValidationError
        _, FeatureRow = self._get_schemas()
        with pytest.raises(ValidationError):
            FeatureRow(latency_p99=120.0, error_rate=1.5, rps=450.0)

    def test_rejects_negative_error_rate(self):
        """error_rate < 0 must be rejected."""
        from pydantic import ValidationError
        _, FeatureRow = self._get_schemas()
        with pytest.raises(ValidationError):
            FeatureRow(latency_p99=120.0, error_rate=-0.1, rps=450.0)

    def test_rejects_negative_rps(self):
        """rps < 0 must be rejected."""
        from pydantic import ValidationError
        _, FeatureRow = self._get_schemas()
        with pytest.raises(ValidationError):
            FeatureRow(latency_p99=120.0, error_rate=0.01, rps=-5.0)

    def test_accepts_zero_rps(self):
        """rps == 0 must be accepted (server could have zero traffic)."""
        _, FeatureRow = self._get_schemas()
        row = FeatureRow(latency_p99=120.0, error_rate=0.01, rps=0.0)
        assert row.rps == 0.0

    def test_accepts_boundary_error_rate(self):
        """error_rate at exact boundaries 0.0 and 1.0 must be accepted."""
        _, FeatureRow = self._get_schemas()
        assert FeatureRow(latency_p99=120.0, error_rate=0.0, rps=450.0).error_rate == 0.0
        assert FeatureRow(latency_p99=120.0, error_rate=1.0, rps=450.0).error_rate == 1.0

    def test_rejects_empty_features(self):
        """Empty features list must be rejected."""
        from pydantic import ValidationError
        PredictRequest, _ = self._get_schemas()
        with pytest.raises(ValidationError):
            PredictRequest(features=[])


# ─────────────────────────── Optuna Tuning Tests ─────────────────────────────

class TestOptunaTuning:
    """Tests for Optuna AutoML hyperparameter search."""

    def test_tune_returns_valid_params(self, normal_df):
        """tune_hyperparameters must return a dict with expected keys."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(normal_df[FEATURES])

        best_params, best_score = tune_hyperparameters(X_scaled, n_trials=5)
        assert isinstance(best_params, dict)
        assert "contamination" in best_params
        assert "n_estimators" in best_params
        assert 0.01 <= best_params["contamination"] <= 0.15
        assert 50 <= best_params["n_estimators"] <= 300
        assert isinstance(best_score, float)

    def test_tune_contamination_in_range(self, normal_df):
        """Tuned contamination must stay within the search space bounds."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(normal_df[FEATURES])

        best_params, _ = tune_hyperparameters(X_scaled, n_trials=5)
        assert 0.01 <= best_params["contamination"] <= 0.15


# ─────────────────────────── LOF Detector Tests ──────────────────────────────

class TestLOFDetector:
    """Tests for the second anomaly detector: Local Outlier Factor."""

    def test_lof_returns_valid_outputs(self, normal_df):
        """train_lof must return a model, valid anomaly_rate and mean_lof_score."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(normal_df[FEATURES])

        lof_model, anomaly_rate, mean_lof = train_lof(X_scaled, n_neighbors=20, contamination=0.03)
        assert lof_model is not None
        assert 0.0 <= anomaly_rate <= 1.0
        assert isinstance(mean_lof, float)
        assert mean_lof < 0  # negative_outlier_factor_ is always <= 0

    def test_lof_anomaly_rate_matches_contamination(self, normal_df):
        """LOF anomaly_rate should be close to contamination parameter."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(normal_df[FEATURES])

        contamination = 0.05
        _, anomaly_rate, _ = train_lof(X_scaled, n_neighbors=20, contamination=contamination)
        # LOF anomaly_rate should be approximately equal to contamination
        assert abs(anomaly_rate - contamination) < 0.02, (
            f"LOF anomaly_rate {anomaly_rate:.4f} too far from contamination {contamination}"
        )

    def test_lof_and_isoforest_agreement_on_normal_data(self, normal_df):
        """IsolationForest and LOF should have high agreement rate on unimodal normal data."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(normal_df[FEATURES])

        contamination = 0.03
        isoforest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        if_labels = isoforest.fit(X_scaled).predict(X_scaled)

        _, _, _ = train_lof(X_scaled, n_neighbors=20, contamination=contamination)
        # Re-fit LOF to get labels (train_lof uses fit_predict internally)
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
        lof_labels = lof.fit_predict(X_scaled)

        agreement = float((if_labels == lof_labels).mean())
        # On unimodal normal data both detectors should largely agree
        assert agreement >= 0.85, (
            f"Detector agreement rate {agreement:.4f} is too low on normal data"
        )

