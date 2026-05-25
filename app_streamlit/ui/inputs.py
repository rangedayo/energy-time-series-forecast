"""운영 모드 좌측 입력 패널 (round 2-3).

render_input_panel() — region, 시작 시각, initial_soc 위젯 + 실행 버튼.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from app_streamlit.data_loader import _load_csv_cached

# load_history_and_forecast 가 요구하는 윈도우: history 24h + forecast 48h
_HISTORY_HOURS: int = 24
_FORECAST_HOURS: int = 48
_DEFAULT_DATE = date(2022, 6, 15)
_DEFAULT_TIME = time(9, 0)


@st.cache_data(show_spinner=False)
def _load_csv_metadata(csv_path: str) -> dict[str, Any]:
    """CSV 에서 regions + start_time 허용 범위 계산. 캐시한다."""
    df = _load_csv_cached(csv_path)
    regions: list[str] = sorted(df["region"].unique().tolist())
    ts_min = pd.Timestamp(df["timestamp"].min()).to_pydatetime()
    ts_max = pd.Timestamp(df["timestamp"].max()).to_pydatetime()
    start_min = ts_min + timedelta(hours=_HISTORY_HOURS)
    start_max = ts_max - timedelta(hours=_FORECAST_HOURS - 1)
    return {
        "regions": regions,
        "start_min": start_min,
        "start_max": start_max,
    }


def _clamp_default(value: date, lo: date, hi: date) -> date:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def render_input_panel(csv_path: str) -> dict[str, Any] | None:
    """좌측 입력 패널 렌더링.

    Returns:
        실행 버튼이 눌렸을 때만 dict 반환. 아니면 None.
        {"region": str, "start_time": datetime, "initial_soc": float}
    """
    meta = _load_csv_metadata(csv_path)
    regions: list[str] = meta["regions"]
    start_min: datetime = meta["start_min"]
    start_max: datetime = meta["start_max"]

    st.subheader("입력")

    default_region = "전라남도" if "전라남도" in regions else regions[0]
    region = st.selectbox(
        "지역 (region)",
        options=regions,
        index=regions.index(default_region),
        help="학습 CSV 에 포함된 17개 시도 중 선택",
        key="input_region",
    )

    date_min = start_min.date()
    date_max = start_max.date()
    default_date = _clamp_default(_DEFAULT_DATE, date_min, date_max)
    sim_date = st.date_input(
        "시뮬 시작 날짜",
        value=default_date,
        min_value=date_min,
        max_value=date_max,
        key="input_start_date",
    )

    sim_time = st.time_input(
        "시뮬 시작 시각",
        value=_DEFAULT_TIME,
        step=timedelta(hours=1),
        help="정시 단위로 선택",
        key="input_start_time",
    )

    initial_soc = st.slider(
        "초기 SOC",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="배터리 시작 충전 상태 (0=공, 1=만충)",
        key="input_initial_soc",
    )

    st.caption(
        f"사용 가능 범위: {date_min.isoformat()} ~ {date_max.isoformat()}"
    )

    run_clicked = st.button("실행", type="primary", use_container_width=True)
    if not run_clicked:
        return None

    if isinstance(sim_date, tuple):
        sim_date = sim_date[0]
    start_time = datetime.combine(sim_date, time(sim_time.hour, 0))

    if start_time < start_min or start_time > start_max:
        st.error(
            f"시작 시각이 허용 범위 밖입니다: "
            f"{start_min.isoformat()} ~ {start_max.isoformat()}"
        )
        return None

    return {
        "region": region,
        "start_time": start_time,
        "initial_soc": float(initial_soc),
    }
