"""학습 CSV에서 (region, start_time) 기준 history 24h + forecast 48h 슬라이스.

Round 2-2 MPC 오케스트레이터 전용. orchestrator.run_mpc_simulation() 만 호출한다.

CSV 컬럼: timestamp, region, power_mwh, 기온, 강수량, 습도, 일조, 일사량, 전운량, ...
시간대: naive datetime (CLAUDE.md 가정 — 한국 표준시).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

# /predict_horizon 스키마 (app/schemas.py)와 정확히 일치해야 함
_WEATHER_FIELDS: tuple[str, ...] = (
    "기온", "강수량", "습도", "일조", "일사량", "전운량",
)
_POWER_COL = "power_mwh"


@lru_cache(maxsize=4)
def _load_csv_cached(csv_path: str) -> pd.DataFrame:
    """프로세스 수명 동안 CSV 메모리에 캐시. 테스트에서 7번 부르므로 효과적."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["timestamp"])
    missing = [c for c in ("timestamp", "region", _POWER_COL, *_WEATHER_FIELDS)
               if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.sort_values(["region", "timestamp"]).reset_index(drop=True)


def load_history_and_forecast(
    csv_path: str,
    region: str,
    start_time: datetime,
) -> tuple[list[dict], list[dict], list[float]]:
    """
    Returns:
        history: 24개. [{"timestamp", "power_mwh"}]
        forecast: 48개. [{"timestamp", "irradiance", "기온", "강수량", "습도", "일조", "전운량"}]
        actuals: 48개 float. (시뮬 정답 + mpc_oracle 비교용)

    Raises:
        ValueError: region/시간 검증 실패 시. orchestrator가 그대로 노출.
    """
    df = _load_csv_cached(csv_path)

    if region not in set(df["region"].unique()):
        raise ValueError(f"region not found: {region}")

    region_df = df[df["region"] == region].set_index("timestamp").sort_index()
    region_min = region_df.index.min().to_pydatetime()
    region_max = region_df.index.max().to_pydatetime()

    if not isinstance(start_time, datetime):
        raise TypeError(
            f"start_time must be datetime, got {type(start_time).__name__}"
        )

    if start_time < region_min or start_time > region_max:
        raise ValueError(
            f"start_time out of range: {start_time.isoformat()} "
            f"(region={region} range=[{region_min.isoformat()}, "
            f"{region_max.isoformat()}])"
        )

    # 72시간 윈도우 = history 24h + forecast 48h (start_time-24h ~ start_time+47h)
    history_times = [start_time - timedelta(hours=24 - i) for i in range(24)]
    forecast_times = [start_time + timedelta(hours=i) for i in range(48)]
    all_times = history_times + forecast_times

    region_ts_set = set(region_df.index.to_pydatetime())
    missing_times = [t for t in all_times if t not in region_ts_set]
    if missing_times:
        raise ValueError(
            f"insufficient continuous window for {region} @ "
            f"{start_time.isoformat()}: {len(missing_times)} of 72 timestamps "
            f"missing (first missing: {missing_times[0].isoformat()})"
        )

    history: list[dict] = [
        {
            "timestamp": t,
            "power_mwh": float(region_df.at[t, _POWER_COL]),
        }
        for t in history_times
    ]

    forecast: list[dict] = []
    for t in forecast_times:
        row = region_df.loc[t]
        forecast.append({
            "timestamp": t,
            "irradiance": float(row["일사량"]),
            "기온": float(row["기온"]),
            "강수량": float(row["강수량"]),
            "습도": float(row["습도"]),
            "일조": float(row["일조"]),
            "전운량": float(row["전운량"]),
        })

    actuals: list[float] = [
        float(region_df.at[t, _POWER_COL]) for t in forecast_times
    ]

    return history, forecast, actuals
