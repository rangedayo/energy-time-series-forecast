"""3개 정책 × 5개 핵심 지표 비교 표 (round 2-3)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app_streamlit.ui.naming import display_name

_HIGHLIGHT_BG = "background-color: #D5F2DE;"  # light green
_POLICY_ORDER: tuple[str, ...] = ("naive", "xgb_lookahead", "mpc_xgb")

_METRIC_ROWS: tuple[tuple[str, str, str, str | None], ...] = (
    # (row_label, summary_key, format_str, best_direction)  best_direction in {"max","min",None}
    ("순수익 (원)", "net_revenue_krw", "{:,.0f}", "max"),
    ("자급률 (%)", "self_sufficiency_rate_pct", "{:.2f}", "max"),
    ("총 매입 (MWh)", "total_import_mwh", "{:.2f}", "min"),
    ("총 판매 (MWh)", "total_export_mwh", "{:.2f}", None),
    ("배터리 사이클", "battery_cycles", "{:.3f}", None),
)


def _extract_summary(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {p: results[p]["summary"] for p in _POLICY_ORDER if p in results}


def _best_policy(
    summaries: dict[str, dict[str, float]], key: str, direction: str | None
) -> str | None:
    if direction not in {"max", "min"}:
        return None
    values = {p: float(s.get(key, 0.0)) for p, s in summaries.items()}
    if not values:
        return None
    return (
        max(values, key=values.get) if direction == "max"
        else min(values, key=values.get)
    )


def render_metrics_table(results: dict[str, Any]) -> None:
    """3개 정책 비교 표. best 값에 하이라이트."""
    summaries = _extract_summary(results)
    if not summaries:
        st.warning("표시할 결과가 없습니다.")
        return

    columns = [display_name(p) for p in _POLICY_ORDER if p in summaries]
    index = [row[0] for row in _METRIC_ROWS]

    best_per_row: dict[str, str | None] = {}
    formatted_rows: list[list[str]] = []

    for row_label, key, fmt, direction in _METRIC_ROWS:
        best = _best_policy(summaries, key, direction)
        best_per_row[row_label] = best
        formatted_row: list[str] = []
        for p in _POLICY_ORDER:
            if p not in summaries:
                continue
            v = float(summaries[p].get(key, 0.0))
            formatted_row.append(fmt.format(v))
        formatted_rows.append(formatted_row)

    df = pd.DataFrame(formatted_rows, index=index, columns=columns)

    def _style(_: pd.DataFrame) -> pd.DataFrame:
        styled = pd.DataFrame("", index=df.index, columns=df.columns)
        for row_label in df.index:
            best = best_per_row.get(row_label)
            if best is None or best not in summaries:
                continue
            col_label = display_name(best)
            if col_label in styled.columns:
                styled.at[row_label, col_label] = _HIGHLIGHT_BG
        return styled

    styled_df = df.style.apply(_style, axis=None)
    st.dataframe(styled_df, use_container_width=True)
    st.caption("초록 배경 = 해당 지표 기준 best (순수익·자급률 max / 총 매입 min)")
