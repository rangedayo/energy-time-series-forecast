"""XGBoost prediction logic — matches feature_engineering_national.py exactly.

The model was trained on 24 features in the order defined by
``app.config.FEATURE_ORDER``: 6 raw weather columns + region_code + 17 engineered.
"""
from __future__ import annotations

import math

import numpy as np
import xgboost as xgb

from app.schemas import HorizonRequest, PredictionRequest

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


def predict_horizon(
    model: xgb.Booster,
    region_encoder,
    feature_order: list[str],
    request: HorizonRequest,
) -> list[float]:
    """재귀적 multi-step 예측.

    매 step마다 이전 step의 예측값을 lag로 사용하여 t+1, t+2, ..., t+horizon을
    순차 계산한다. lag/rolling 공식은 feature_engineering_national.py와 일치
    (rolling_std는 pandas 기본 ddof=1).
    """
    history_powers: list[float] = [h.power_mwh for h in request.history]
    predictions: list[float] = []

    for step in range(request.horizon):
        forecast_point = request.forecast[step]
        target_ts = forecast_point.timestamp

        lag_1h = history_powers[-1]
        lag_2h = history_powers[-2]
        lag_3h = history_powers[-3]
        lag_24h = history_powers[-24]
        power_diff_1h = lag_1h - lag_2h
        power_diff_2h = lag_1h - lag_3h
        last_3 = np.asarray(history_powers[-3:], dtype=np.float64)
        last_6 = np.asarray(history_powers[-6:], dtype=np.float64)
        rolling_mean_3h = float(last_3.mean())
        rolling_mean_6h = float(last_6.mean())
        rolling_std_3h = float(last_3.std(ddof=1))

        pseudo_req = PredictionRequest(
            timestamp=target_ts,
            region=request.region,
            기온=forecast_point.temperature,
            강수량=forecast_point.precipitation,
            습도=forecast_point.humidity,
            일조=forecast_point.sunshine,
            irradiance=forecast_point.irradiance,
            전운량=forecast_point.cloud_cover,
            lag_1h=lag_1h,
            lag_2h=lag_2h,
            lag_3h=lag_3h,
            lag_24h=lag_24h,
            power_diff_1h=power_diff_1h,
            power_diff_2h=power_diff_2h,
            rolling_mean_3h=rolling_mean_3h,
            rolling_mean_6h=rolling_mean_6h,
            rolling_std_3h=rolling_std_3h,
        )

        feature_vector = build_feature_vector(pseudo_req, region_encoder, feature_order)
        pred = predict_single(model, feature_vector, feature_order)

        predictions.append(pred)
        history_powers.append(pred)

    return predictions
