"""운영 시뮬레이션 BFF 헬퍼 (React 프론트엔드용).

/sim/meta 가 쓰는 메타 계산을 담는다. 실제 시뮬은 app_streamlit.orchestrator
의 run_mpc_simulation 을 그대로 재사용한다 (중복 구현 금지).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app_streamlit.data_loader import _load_csv_cached


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def to_jsonable(obj: Any) -> Any:
    """numpy/datetime 가 섞인 시뮬 결과를 순수 JSON 호환 구조로 변환한다."""
    return json.loads(json.dumps(obj, default=_json_default))

# load_history_and_forecast 가 요구하는 윈도우: history 24h + forecast 48h
_HISTORY_HOURS: int = 24
_FORECAST_HOURS: int = 48


def compute_sim_meta(csv_path: str) -> dict[str, Any]:
    """CSV 에서 지역 목록 + 시작 시각 허용 범위를 계산한다.

    app_streamlit.ui.inputs._load_csv_metadata 와 동일 로직이지만,
    Streamlit 캐시 데코레이터 없이 순수 함수로 둔다 (API 컨텍스트에서 호출).
    """
    df = _load_csv_cached(csv_path)
    regions: list[str] = sorted(df["region"].unique().tolist())
    ts_min = pd.Timestamp(df["timestamp"].min()).to_pydatetime()
    ts_max = pd.Timestamp(df["timestamp"].max()).to_pydatetime()
    start_min = ts_min + timedelta(hours=_HISTORY_HOURS)
    start_max = ts_max - timedelta(hours=_FORECAST_HOURS - 1)
    return {
        "regions": regions,
        "start_min": start_min.isoformat(),
        "start_max": start_max.isoformat(),
    }
