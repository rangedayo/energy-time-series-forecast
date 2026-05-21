"""XGBoost prediction logic — matches feature_engineering_national.py exactly.

The model was trained on 24 features in the order defined by
``app.config.FEATURE_ORDER``: 6 raw weather columns + region_code + 17 engineered.
"""
from __future__ import annotations

import math

import numpy as np
import xgboost as xgb

from app.schemas import PredictionRequest

_SEASON_MAP = {
    12: 1, 1: 1, 2: 1,
    3: 2, 4: 2, 5: 2,
    6: 3, 7: 3, 8: 3,
    9: 4, 10: 4, 11: 4,
}

_EXPECTED_FEATURE_COUNT = 24


def _compute_features(request: PredictionRequest, region_encoder) -> dict[str, float]:
    hour = request.timestamp.hour
    month = request.timestamp.month
    day_of_week = request.timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    season = _SEASON_MAP[month]
    solar_altitude_proxy = max(0.0, math.sin(math.pi * (hour - 6) / 12))
    irrad_x_solar = request.irradiance * solar_altitude_proxy
    is_daytime = 1 if 6 <= hour <= 18 else 0
    region_code = int(region_encoder.transform([request.region])[0])

    return {
        "기온": request.temperature,
        "강수량": request.precipitation,
        "습도": request.humidity,
        "일조": request.sunshine,
        "일사량": request.irradiance,
        "전운량": request.cloud_cover,
        "region_code": region_code,
        "hour": hour,
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "season": season,
        "solar_altitude_proxy": solar_altitude_proxy,
        "lag_1h": request.lag_1h,
        "lag_2h": request.lag_2h,
        "lag_3h": request.lag_3h,
        "lag_24h": request.lag_24h,
        "power_diff_1h": request.power_diff_1h,
        "power_diff_2h": request.power_diff_2h,
        "rolling_mean_3h": request.rolling_mean_3h,
        "rolling_mean_6h": request.rolling_mean_6h,
        "rolling_std_3h": request.rolling_std_3h,
        "irrad_x_solar": irrad_x_solar,
        "is_daytime": is_daytime,
    }


def build_feature_vector(
    request: PredictionRequest,
    region_encoder,
    feature_order: list[str],
) -> np.ndarray:
    if len(feature_order) != _EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"feature_order length must be {_EXPECTED_FEATURE_COUNT}, "
            f"got {len(feature_order)}"
        )
    features = _compute_features(request, region_encoder)
    row = [features[name] for name in feature_order]
    return np.asarray([row], dtype=np.float32)


def predict_single(
    model: xgb.Booster,
    feature_vector: np.ndarray,
    feature_order: list[str],
) -> float:
    dmatrix = xgb.DMatrix(feature_vector, feature_names=feature_order)
    raw = float(model.predict(dmatrix)[0])
    if not np.isfinite(raw):
        raise ValueError(f"Model produced non-finite prediction: {raw}")
    return max(0.0, raw)


def predict_batch(
    model: xgb.Booster,
    feature_matrix: np.ndarray,
    feature_order: list[str],
) -> np.ndarray:
    dmatrix = xgb.DMatrix(feature_matrix, feature_names=feature_order)
    raw = np.asarray(model.predict(dmatrix), dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Model produced non-finite predictions in batch")
    return np.maximum(raw, 0.0)
