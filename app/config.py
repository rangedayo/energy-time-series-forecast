"""Application configuration: model paths, API key, regions, and feature order.

# 이 순서는 학습 시점의 train.columns - NON_FEAT 결과를 그대로 반영한다.
# feature_list_national.json의 engineered_features는 "엔지니어링된 피처 18개"의 목록일 뿐
# 학습 모델 입력(24개)이 아니다. 원시 기상 6개가 학습에 포함되어 있음을 주의.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = str(PROJECT_ROOT / "models" / "national_xgboost_model.json")
ENCODER_PATH = str(PROJECT_ROOT / "models" / "national_region_encoder.pkl")
FEATURE_LIST_PATH = str(PROJECT_ROOT / "src" / "features" / "feature_list_national.json")

_DEFAULT_API_KEY = "dev-key-change-me"
_env_key = os.environ.get("SOLAR_API_KEY")
if _env_key:
    API_KEY = _env_key
else:
    logger.warning(
        "SOLAR_API_KEY not set - falling back to development default '%s'. "
        "Do NOT use this in production.",
        _DEFAULT_API_KEY,
    )
    API_KEY = _DEFAULT_API_KEY


# ── 운영 시뮬레이션(/simulate, /sim/meta) 설정 ──────────────────────────────
# React 프론트엔드(app_frontend)가 호출하는 BFF 엔드포인트용.
SIM_CSV_PATH = str(PROJECT_ROOT / "data" / "processed" / "national_train_features.csv")

# run_mpc_simulation 이 내부에서 /predict_horizon 을 다시 HTTP 호출할 때 쓰는 자기 주소.
SELF_BASE_URL = os.environ.get("SOLAR_SELF_URL", "http://localhost:8000")

# CORS: Vite 개발 서버(기본 5173) 허용. 콤마 구분 env 로 덮어쓸 수 있다.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("SOLAR_CORS_ORIGINS", _default_origins).split(",")
    if o.strip()
]


def _load_valid_regions() -> set[str]:
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return {str(name) for name in encoder.classes_}


VALID_REGIONS: set[str] = _load_valid_regions()
REGION_LIST_SORTED: list[str] = sorted(VALID_REGIONS)

FEATURE_ORDER: list[str] = [
    "기온",
    "강수량",
    "습도",
    "일조",
    "일사량",
    "전운량",
    "region_code",
    "hour",
    "month",
    "day_of_week",
    "is_weekend",
    "season",
    "solar_altitude_proxy",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "power_diff_1h",
    "power_diff_2h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_std_3h",
    "irrad_x_solar",
    "is_daytime",
]
