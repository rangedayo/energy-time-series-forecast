"""운영 모드 차트 4종 (round 2-3).

(d-1) render_prediction_vs_actual    : 48h 예측 vs 실측
(d-2) render_soc_curves              : 3개 정책 SOC 24h
(d-3) render_revenue_sufficiency_bars: 정책 × (순수익, 자급률) 그룹 막대
(d-4) render_hourly_trading          : 선택 정책 시간대별 매매 + SOC overlay
"""
from __future__ import annotations

import sys as _sys
from datetime import datetime
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font  # noqa: E402

_apply_font()
matplotlib.rcParams["axes.unicode_minus"] = False

from app_streamlit.ui.naming import POLICY_COLORS, display_name  # noqa: E402

# 시각 기준선 (운영자 가독성용 — 실제 SOC 제약은 ess_config_v2.SOC_MIN/MAX)
_SOC_VIS_MIN: float = 0.10
_SOC_VIS_MAX: float = 0.90

_PERIOD_BG_COLOR = {
    "off_peak": "#EAF6FF",
    "mid_peak": "#FFF8E0",
    "max_peak": "#FCE4E4",
}
_PERIOD_LABEL = {
    "off_peak": "경부하",
    "mid_peak": "중간부하",
    "max_peak": "최대부하",
}


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


# ── (d-1) ────────────────────────────────────────────────────────────────────
def render_prediction_vs_actual(
    predictions: list[dict[str, Any]],
    actuals: list[dict[str, Any]],
    sim_length: int = 24,
) -> None:
    """48시간 예측 vs 실측 라인 차트. 시뮬 구간(0~sim_length) 음영."""
    pred_times = [_parse_iso(p["timestamp"]) for p in predictions]
    pred_vals = [float(p["predicted_power_mwh"]) for p in predictions]
    actual_vals = [float(a["actual_power_mwh"]) for a in actuals]

    fig, ax = plt.subplots(figsize=(8, 4))

    if pred_times:
        ax.axvspan(
            pred_times[0],
            pred_times[min(sim_length, len(pred_times)) - 1],
            color="#E8EAF6",
            alpha=0.4,
            label="시뮬 구간",
        )

    ax.plot(
        pred_times[:sim_length],
        pred_vals[:sim_length],
        color="#1F6FEB",
        linewidth=2.0,
        label="예측 (시뮬 구간)",
    )
    if len(pred_times) > sim_length:
        ax.plot(
            pred_times[sim_length - 1:],
            pred_vals[sim_length - 1:],
            color="#1F6FEB",
            linewidth=1.4,
            linestyle="--",
            alpha=0.7,
            label="예측 (이후 24h)",
        )

    ax.plot(
        pred_times,
        actual_vals,
        color="#D63D3D",
        linewidth=1.4,
        linestyle=":",
        label="실측",
    )

    ax.set_title("발전량 예측 vs 실측 (48시간)")
    ax.set_xlabel("시각")
    ax.set_ylabel("발전량 (MWh)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── (d-2) ────────────────────────────────────────────────────────────────────
def render_soc_curves(results: dict[str, Any]) -> None:
    """3개 정책 SOC 24h 라인 차트."""
    fig, ax = plt.subplots(figsize=(8, 4))

    for policy, payload in results.items():
        hourly = payload.get("hourly", [])
        if not hourly:
            continue
        steps = [int(h["step"]) for h in hourly]
        soc = [float(h["soc"]) for h in hourly]
        ax.plot(
            steps, soc,
            color=POLICY_COLORS.get(policy, "#666666"),
            linewidth=2.0,
            label=display_name(policy),
        )

    ax.axhline(_SOC_VIS_MIN, linestyle="--", color="#888888", alpha=0.6,
               label=f"SOC_MIN ({_SOC_VIS_MIN:.2f})")
    ax.axhline(_SOC_VIS_MAX, linestyle="--", color="#888888", alpha=0.6,
               label=f"SOC_MAX ({_SOC_VIS_MAX:.2f})")

    ax.set_ylim(0.0, 1.0)
    ax.set_title("정책별 SOC 추이 (24시간)")
    ax.set_xlabel("시뮬 step (h)")
    ax.set_ylabel("SOC")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── (d-3) ────────────────────────────────────────────────────────────────────
def render_revenue_sufficiency_bars(results: dict[str, Any]) -> None:
    """정책 × (순수익, 자급률) 그룹 막대. twin axis."""
    policies = list(results.keys())
    if not policies:
        return
    revenues_m = [
        float(results[p]["summary"].get("net_revenue_krw", 0.0)) / 1_000_000.0
        for p in policies
    ]
    sufficiencies = [
        float(results[p]["summary"].get("self_sufficiency_rate_pct", 0.0))
        for p in policies
    ]

    fig, ax_left = plt.subplots(figsize=(8, 4))
    ax_right = ax_left.twinx()

    x = np.arange(len(policies), dtype=float)
    width = 0.36

    bars_rev = ax_left.bar(
        x - width / 2, revenues_m, width=width,
        color=[POLICY_COLORS.get(p, "#666666") for p in policies],
        edgecolor="#222222", linewidth=0.6,
        label="순수익",
    )
    bars_suf = ax_right.bar(
        x + width / 2, sufficiencies, width=width,
        color=[POLICY_COLORS.get(p, "#666666") for p in policies],
        alpha=0.55, edgecolor="#222222", linewidth=0.6,
        hatch="//",
        label="자급률",
    )

    for bar, val in zip(bars_rev, revenues_m):
        ax_left.annotate(
            f"{val:.1f}M",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8,
        )
    for bar, val in zip(bars_suf, sufficiencies):
        ax_right.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8,
        )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([display_name(p) for p in policies])
    ax_left.set_ylabel("순수익 (백만원)")
    ax_right.set_ylabel("자급률 (%)")
    ax_left.set_title("정책별 순수익 vs 자급률")
    ax_left.grid(True, axis="y", alpha=0.25)

    ax_left.legend(
        [bars_rev, bars_suf], ["순수익 (좌)", "자급률 (우)"],
        loc="upper left", fontsize=8,
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── (d-4) ────────────────────────────────────────────────────────────────────
def render_hourly_trading(
    results: dict[str, Any],
    selected_policy: str,
) -> None:
    """선택 정책의 시간대별 매매 스택드 바 + SOC overlay."""
    payload = results.get(selected_policy)
    if not payload or not payload.get("hourly"):
        st.info(f"{display_name(selected_policy)} 정책의 시간별 데이터가 없습니다.")
        return

    hourly = payload["hourly"]
    steps = np.asarray([int(h["step"]) for h in hourly], dtype=float)
    sells = np.asarray([float(h.get("grid_sell_mwh", 0.0)) for h in hourly])
    buys = np.asarray([float(h.get("grid_buy_mwh", 0.0)) for h in hourly])
    socs = [float(h["soc"]) for h in hourly]
    periods = [str(h.get("period") or "") for h in hourly]

    fig, ax_left = plt.subplots(figsize=(8, 4))

    drawn_period_labels: set[str] = set()
    for i, period in enumerate(periods):
        color = _PERIOD_BG_COLOR.get(period)
        if not color:
            continue
        label = _PERIOD_LABEL[period]
        ax_left.axvspan(
            steps[i] - 0.5, steps[i] + 0.5,
            color=color, alpha=0.6,
            label=label if label not in drawn_period_labels else None,
        )
        drawn_period_labels.add(label)

    ax_left.bar(
        steps, sells, width=0.7,
        color="#2E8B57", edgecolor="#1A5230", linewidth=0.4,
        label="판매 (grid_sell)",
    )
    ax_left.bar(
        steps, -buys, width=0.7,
        color="#C8443B", edgecolor="#7A2620", linewidth=0.4,
        label="매입 (grid_buy)",
    )
    ax_left.axhline(0.0, color="#222222", linewidth=0.8)

    ax_right = ax_left.twinx()
    ax_right.plot(
        steps, socs,
        color="#3A3A3A", linewidth=1.6, marker="o", markersize=3,
        label="SOC",
    )
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_ylabel("SOC")

    ax_left.set_xticks(np.arange(0, len(steps), max(1, len(steps) // 12)))
    ax_left.set_xlabel("시뮬 step (h)")
    ax_left.set_ylabel("매매 (MWh, 양수=판매 / 음수=매입)")
    ax_left.set_title(
        f"시간대별 매매 — {display_name(selected_policy)}"
    )
    ax_left.grid(True, axis="y", alpha=0.25)

    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_l + handles_r,
        labels_l + labels_r,
        loc="upper left", fontsize=7, ncol=2,
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
