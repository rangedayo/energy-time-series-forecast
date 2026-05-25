"""Regression tests for the FastAPI prediction service."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from app.config import API_KEY, FEATURE_ORDER, MODEL_PATH, REGION_LIST_SORTED
from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "national_test_features.csv"

VALID_PAYLOAD = {
    "timestamp": "2023-07-15T13:00:00",
    "region": "전라남도",
    "기온": 28.5,
    "강수량": 0.0,
    "습도": 65.0,
    "일조": 0.9,
    "irradiance": 3.0,
    "전운량": 2.0,
    "lag_1h": 120.0,
    "lag_2h": 110.0,
    "lag_3h": 95.0,
    "lag_24h": 130.0,
    "power_diff_1h": 10.0,
    "power_diff_2h": 25.0,
    "rolling_mean_3h": 108.3,
    "rolling_mean_6h": 95.5,
    "rolling_std_3h": 12.7,
}

AUTH_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["feature_count"] == 24
    assert body["region_count"] == 17


def test_predict_requires_api_key(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid or missing API Key"


def test_predict_valid_input(client):
    r = client.post("/predict", json=VALID_PAYLOAD, headers=AUTH_HEADERS)
    assert r.status_code == 200
    body = r.json()
    value = body["predicted_power_mwh"]
    assert value >= 0
    assert np.isfinite(value)
    assert body["region"] == "전라남도"


def test_predict_invalid_region(client):
    bad = {**VALID_PAYLOAD, "region": "화성시"}
    r = client.post("/predict", json=bad, headers=AUTH_HEADERS)
    assert r.status_code == 422
    text = str(r.json())
    for region in REGION_LIST_SORTED:
        assert region in text, f"region {region} missing from error message"


def test_predict_negative_irradiance(client):
    bad = {**VALID_PAYLOAD, "irradiance": -1.0}
    r = client.post("/predict", json=bad, headers=AUTH_HEADERS)
    assert r.status_code == 422


def test_predict_batch_under_limit(client):
    payload = {"items": [VALID_PAYLOAD] * 100}
    r = client.post("/predict_batch", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 100
    assert len(body["predictions"]) == 100


def test_predict_batch_over_limit(client):
    payload = {"items": [VALID_PAYLOAD] * 1001}
    r = client.post("/predict_batch", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 422


def _row_to_payload(row: pd.Series) -> dict:
    ts = row["timestamp"]
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    return {
        "timestamp": ts,
        "region": row["region"],
        "기온": float(row["기온"]),
        "강수량": float(row["강수량"]),
        "습도": float(row["습도"]),
        "일조": float(row["일조"]),
        "irradiance": float(row["일사량"]),
        "전운량": float(row["전운량"]),
        "lag_1h": float(row["lag_1h"]),
        "lag_2h": float(row["lag_2h"]),
        "lag_3h": float(row["lag_3h"]),
        "lag_24h": float(row["lag_24h"]),
        "power_diff_1h": float(row["power_diff_1h"]),
        "power_diff_2h": float(row["power_diff_2h"]),
        "rolling_mean_3h": float(row["rolling_mean_3h"]),
        "rolling_mean_6h": float(row["rolling_mean_6h"]),
        "rolling_std_3h": float(row["rolling_std_3h"]),
    }


def test_predict_snapshot(client):
    assert TEST_CSV.exists(), f"missing fixture data: {TEST_CSV}"
    df = pd.read_csv(TEST_CSV, encoding="utf-8-sig", parse_dates=["timestamp"])
    sample = df.sample(n=5, random_state=42).reset_index(drop=True)

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    raw_matrix = sample[FEATURE_ORDER].astype(np.float32).to_numpy()
    dmatrix = xgb.DMatrix(raw_matrix, feature_names=FEATURE_ORDER)
    raw_preds = np.maximum(
        np.asarray(booster.predict(dmatrix), dtype=np.float64), 0.0
    )

    api_preds = []
    for _, row in sample.iterrows():
        payload = _row_to_payload(row)
        r = client.post("/predict", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200, r.text
        api_preds.append(r.json()["predicted_power_mwh"])
    api_preds = np.asarray(api_preds, dtype=np.float64)

    diff = np.abs(api_preds - raw_preds)
    assert np.all(diff < 1e-4), (
        f"API/raw mismatch beyond 1e-4: diffs={diff.tolist()}, "
        f"api={api_preds.tolist()}, raw={raw_preds.tolist()}"
    )


def _make_valid_horizon_payload(horizon: int = 24) -> dict:
    """검증 통과하는 정상 페이로드 생성 헬퍼."""
    start = datetime(2023, 7, 15, 13, 0, 0)
    history = [
        {
            "timestamp": (start - timedelta(hours=24 - i)).isoformat(),
            "power_mwh": 100.0 + i,
        }
        for i in range(24)
    ]
    forecast = [
        {
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "irradiance": 2.5,
            "기온": 28.0,
            "강수량": 0.0,
            "습도": 60.0,
            "일조": 0.8,
            "전운량": 3.0,
        }
        for i in range(horizon)
    ]
    return {
        "region": "전라남도",
        "start_time": start.isoformat(),
        "horizon": horizon,
        "history": history,
        "forecast": forecast,
    }


def test_predict_horizon_valid_24(client):
    """정상 24시간 요청 → 200, predictions 길이 24, 모두 ≥ 0, step 1~24."""
    payload = _make_valid_horizon_payload(horizon=24)
    r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 24
    assert all(p["predicted_power_mwh"] >= 0 for p in body["predictions"])
    assert [p["step"] for p in body["predictions"]] == list(range(1, 25))
    assert body["method"] == "recursive_multistep"


def test_predict_horizon_valid_48(client):
    """horizon=48 정상 동작."""
    payload = _make_valid_horizon_payload(horizon=48)
    r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 200, r.text
    assert len(r.json()["predictions"]) == 48


def test_predict_horizon_horizon_out_of_range(client):
    """horizon=0, 49, 100 → 422."""
    for bad_horizon in [0, 49, 100]:
        payload = _make_valid_horizon_payload(horizon=24)
        payload["horizon"] = bad_horizon
        r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 422


def test_predict_horizon_history_length_mismatch(client):
    """history가 23개 또는 25개 → 422."""
    for bad_len in [23, 25]:
        payload = _make_valid_horizon_payload(horizon=24)
        if bad_len < 24:
            payload["history"] = payload["history"][:bad_len]
        else:
            payload["history"].append(payload["history"][-1])
        r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 422


def test_predict_horizon_forecast_length_mismatch(client):
    """horizon=24인데 forecast 23개 → 422."""
    payload = _make_valid_horizon_payload(horizon=24)
    payload["forecast"] = payload["forecast"][:23]
    r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 422


def test_predict_horizon_history_timestamp_gap(client):
    """history에 1시간 갭이 있으면 → 422."""
    payload = _make_valid_horizon_payload(horizon=24)
    h = payload["history"]
    h[10]["timestamp"] = (
        datetime.fromisoformat(h[10]["timestamp"]) + timedelta(hours=2)
    ).isoformat()
    r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 422


def test_predict_horizon_invalid_region(client):
    """가짜 region → 422 + 에러 메시지에 유효 region 목록 포함."""
    payload = _make_valid_horizon_payload()
    payload["region"] = "화성시"
    r = client.post("/predict_horizon", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 422
    assert "전라남도" in r.text


def test_predict_horizon_requires_api_key(client):
    """API Key 없으면 → 401."""
    payload = _make_valid_horizon_payload()
    r = client.post("/predict_horizon", json=payload)
    assert r.status_code == 401
