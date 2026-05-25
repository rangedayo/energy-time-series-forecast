"""
ESS 시뮬레이터 본체 + 메인 비교 실행 (v2)

TASK G-3: run_simulation — 한 (지역, 정책) 조합의 단일 시뮬레이션
TASK G-4: main — 4 시나리오 × 17 지역 매트릭스 실행 + 3가지 집계 + 시각화

본 시뮬레이션은 통제된 모델 비교 환경이며 실제 운영값 추정이 목적이 아니다.
LSTM 은 본 분석에서 제외한다. XGBoost 단일 모델 + 합성 정책만 비교한다.
"""

import sys
import time
import json
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font
_apply_font()

from src.simulation.ess_config_v2 import (
    SOC_MIN, SOC_MAX, SOC_INIT, EFFICIENCY,
    TOTAL_ESS_CAPACITY_MWH, TOTAL_DEMAND_MWH_PER_H,
    TOTAL_CHARGE_RATE_MAX, TOTAL_DISCHARGE_RATE_MAX,
    get_demand_at_hour, build_region_params,
    get_tou_price_krw_per_mwh, get_load_period,
)

PERIODS = ("off_peak", "mid_peak", "max_peak")
from src.simulation.ess_policy_v2 import (
    policy_naive, policy_lookahead, policy_perfect_foresight,
    policy_xgb_no_lookahead, policy_mpc_xgb, policy_mpc_oracle,
    reset_lp_stats, get_lp_stats,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ── 경로 상수 ─────────────────────────────────────────────────────────────────
XGB_PREDICTIONS = "outputs/national_xgb_predictions.csv"
TRAIN_FEATURES = "data/processed/national_train_features.csv"
OUT_JSON = Path("outputs/ess_v2_simulation_results.json")
OUT_PNG_COMPARISON = Path("outputs/ess_v2_comparison.png")
OUT_PNG_REGION = Path("outputs/ess_v2_region_breakdown.png")
OUT_MD_TOU_BREAKDOWN = Path("outputs/ess_v2_tou_breakdown.md")
SHARE_DIR = Path("claude_share")

# 집계 대상 수치 지표 키
METRIC_KEYS = [
    "self_consumption_rate_pct", "self_sufficiency_rate_pct",
    "total_shortage_mwh", "mean_shortage_mwh", "max_shortage_mwh",
    "curtailment_rate_pct", "shortage_count", "battery_cycles", "ess_score",
    "total_curtailment_mwh", "total_demand_mwh", "total_gen_mwh", "n_hours",
    # TOU 계측 지표 (의사결정엔 영향 없음, 측정만)
    "total_import_mwh", "total_export_mwh",
    "total_cost_krw", "total_revenue_krw", "net_revenue_krw",
]


# ════════════════════════════════════════════════════════════════════════════
# TASK G-3 — 시뮬레이터 본체
# ════════════════════════════════════════════════════════════════════════════
def run_simulation(actual, predicted, hours, params, policy_fn,
                   policy_kwargs=None, months=None,
                   policy_name=None, verbose=False):
    """
    한 (지역, 정책) 조합에 대한 단일 시뮬레이션 실행.

    정책은 SOC 목표만 결정하고, 충방전 실행은 항상 actual(실측) 기준으로 한다.

    지표 해석:
      self_consumption_rate (자가소비율, %) — 발전한 전기 중 활용한 비율. 높을수록 좋음.
      self_sufficiency_rate (자급률, %)   — 수요 중 자체 공급으로 충당한 비율. 높을수록 좋음.
      total_shortage_mwh — 부족의 총량(절대값). 0에 가까울수록 좋음.
      mean_shortage_mwh  — 부족 발생 시 평균 강도. 진단용.
      max_shortage_mwh   — 최악 부족 시점. 극단 시나리오 대응력.

    TOU(시간대별 요금) 계측:
      months 가 주어지면 매 시점 단가를 적용해 비용(import)·수익(export)을 누적.
      None 이면 TOU 계측을 건너뛴다(기존 호환). 의사결정 로직은 절대 바꾸지 않음.
    """
    policy_kwargs = policy_kwargs or {}
    use_tou = months is not None

    n = len(actual)
    soc = SOC_INIT
    total_curtailment = 0.0
    total_shortage_mwh = 0.0
    total_demand_mwh = 0.0
    shortage_list = []
    charge_cycles = 0.0
    discharge_cycles = 0.0

    # TOU 누적자
    total_import_mwh = 0.0
    total_export_mwh = 0.0
    total_cost_krw = 0.0
    total_revenue_krw = 0.0

    # TOU 부하구분(off/mid/max)별 누적자 — 의사결정에 영향 없음, 측정만
    import_mwh_by_period = {p: 0.0 for p in PERIODS}
    export_mwh_by_period = {p: 0.0 for p in PERIODS}
    cost_krw_by_period = {p: 0.0 for p in PERIODS}
    revenue_krw_by_period = {p: 0.0 for p in PERIODS}

    cap = params["ess_capacity_mwh"]
    base_demand = params["demand_mwh_per_h"]
    chg_max = params["charge_rate_max"]
    dis_max = params["discharge_rate_max"]

    for i in range(n):
        gen = float(actual[i])
        h = int(hours[i])
        demand_t = get_demand_at_hour(base_demand, h)
        total_demand_mwh += demand_t

        targets = policy_fn(i, actual, predicted, soc, demand_t, params, **policy_kwargs)
        soc_target_high = targets["soc_target_high"]
        soc_target_low = targets["soc_target_low"]

        actual_net = gen - demand_t

        if use_tou:
            month_t = int(months[i])
            price_t = get_tou_price_krw_per_mwh(month_t, h)
            period_t = get_load_period(month_t, h)
        else:
            price_t = 0.0
            period_t = None

        if actual_net > 0:
            # 잉여 → 충전
            max_storable = max(0.0, (soc_target_high - soc) * cap / EFFICIENCY)
            charge_amount = min(actual_net, chg_max, max_storable)
            soc += charge_amount * EFFICIENCY / cap
            charge_cycles += charge_amount / cap
            export_mwh = actual_net - charge_amount
            total_curtailment += export_mwh
            if use_tou and export_mwh > 0:
                total_export_mwh += export_mwh
                total_revenue_krw += export_mwh * price_t
                export_mwh_by_period[period_t] += export_mwh
                revenue_krw_by_period[period_t] += export_mwh * price_t
        else:
            # 부족 → 방전
            needed = -actual_net
            max_dischargeable = max(0.0, (soc - soc_target_low) * cap * EFFICIENCY)
            discharge_amount = min(needed, dis_max, max_dischargeable)
            soc -= discharge_amount / (cap * EFFICIENCY)
            discharge_cycles += discharge_amount / cap

            shortfall = max(0.0, demand_t - (gen + discharge_amount))
            if shortfall > 0:
                shortage_list.append(shortfall)
                total_shortage_mwh += shortfall
                if use_tou:
                    total_import_mwh += shortfall
                    total_cost_krw += shortfall * price_t
                    import_mwh_by_period[period_t] += shortfall
                    cost_krw_by_period[period_t] += shortfall * price_t

    total_gen = float(np.sum(actual))
    curtailment_rate = total_curtailment / max(total_gen, 1e-10) * 100.0
    self_consumption_rate = 100.0 - curtailment_rate
    self_sufficiency_rate = (1.0 - total_shortage_mwh / max(total_demand_mwh, 1e-10)) * 100.0
    battery_cycles = (charge_cycles + discharge_cycles) / 2.0

    shortage_count = len(shortage_list)
    ess_score = (1.0 - curtailment_rate / 100.0) * (1.0 - shortage_count / max(n, 1)) * 100.0

    net_revenue_krw = total_revenue_krw - total_cost_krw

    avg_import_price = (
        total_cost_krw / total_import_mwh if total_import_mwh > 0 else 0.0
    )
    avg_export_price = (
        total_revenue_krw / total_export_mwh if total_export_mwh > 0 else 0.0
    )

    if verbose and use_tou:
        label = policy_name or "(unnamed)"
        print(f"[TOU 검증] 정책: {label}")
        print(f"  총 import: {total_import_mwh:>14,.1f} MWh")
        print(f"  총 export: {total_export_mwh:>14,.1f} MWh")
        print(f"  총 비용:   {total_cost_krw:>14,.0f} 원 "
              f"(평균 단가: {avg_import_price:,.0f} 원/MWh)")
        print(f"  총 수익:   {total_revenue_krw:>14,.0f} 원 "
              f"(평균 단가: {avg_export_price:,.0f} 원/MWh)")
        print(f"  순수익:    {net_revenue_krw:>+14,.0f} 원")

    return {
        # 신규 지표 (국제 표준)
        "self_consumption_rate_pct": round(self_consumption_rate, 2),
        "self_sufficiency_rate_pct": round(self_sufficiency_rate, 2),
        "total_shortage_mwh": round(total_shortage_mwh, 2),
        "mean_shortage_mwh": round(float(np.mean(shortage_list)), 2) if shortage_list else 0.0,
        "max_shortage_mwh": round(float(np.max(shortage_list)), 2) if shortage_list else 0.0,

        # 기존 호환 지표
        "curtailment_rate_pct": round(curtailment_rate, 2),
        "shortage_count": int(shortage_count),
        "battery_cycles": round(battery_cycles, 2),
        "ess_score": round(ess_score, 2),

        # 진단용
        "total_curtailment_mwh": round(total_curtailment, 2),
        "total_demand_mwh": round(total_demand_mwh, 2),
        "total_gen_mwh": round(total_gen, 2),
        "n_hours": int(n),

        # TOU 계측 (months=None 호출이면 0.0 으로 기록)
        "total_import_mwh": round(total_import_mwh, 2),
        "total_export_mwh": round(total_export_mwh, 2),
        "total_cost_krw": round(total_cost_krw, 0),
        "total_revenue_krw": round(total_revenue_krw, 0),
        "net_revenue_krw": round(net_revenue_krw, 0),

        # TOU 부하구분(off/mid/max)별 분해 — 의사결정 무관, 측정만
        "import_mwh_by_period": {p: round(import_mwh_by_period[p], 2) for p in PERIODS},
        "export_mwh_by_period": {p: round(export_mwh_by_period[p], 2) for p in PERIODS},
        "cost_krw_by_period": {p: round(cost_krw_by_period[p], 0) for p in PERIODS},
        "revenue_krw_by_period": {p: round(revenue_krw_by_period[p], 0) for p in PERIODS},
        "avg_import_price_krw_per_mwh": round(avg_import_price, 2),
        "avg_export_price_krw_per_mwh": round(avg_export_price, 2),

        # 노이즈 플래그 (울산시 등 weight < 임계 지역)
        "flagged_noise_region": bool(params.get("is_noise_region", False)),
    }


# ════════════════════════════════════════════════════════════════════════════
# TASK G-4 — 메인 비교 실행
# ════════════════════════════════════════════════════════════════════════════
MPC_HORIZON = 24

SCENARIOS = {
    # scenario_name : (policy_fn, pred_source)
    "naive_baseline":   (policy_naive,             "actual"),
    "xgb_no_lookahead": (policy_xgb_no_lookahead,  "predicted"),
    "xgb_lookahead":    (policy_lookahead,         "predicted"),
    "oracle":           (policy_perfect_foresight, "actual"),
    "mpc_xgb":          (policy_mpc_xgb,           "predicted"),
    "mpc_oracle":       (policy_mpc_oracle,        "actual"),
}
SCENARIO_ORDER = [
    "naive_baseline", "xgb_no_lookahead", "xgb_lookahead", "oracle",
    "mpc_xgb", "mpc_oracle",
]
MPC_SCENARIOS = {"mpc_xgb", "mpc_oracle"}


def aggregate_metrics(region_results, scenario, regions, region_params, mode):
    """
    여러 지역의 metric dict 를 하나로 집계.
      mode="simple"   : 단순 평균
      mode="weighted" : weight 가중 평균 (대상 지역들의 weight 를 재정규화)
    """
    dicts = [region_results[r][scenario] for r in regions]
    out = {}
    if mode == "weighted":
        ws = np.array([region_params[r]["weight"] for r in regions], dtype=float)
        ws = ws / ws.sum()
        for k in METRIC_KEYS:
            out[k] = round(float(np.sum([w * d[k] for w, d in zip(ws, dicts)])), 2)
    else:  # simple
        for k in METRIC_KEYS:
            out[k] = round(float(np.mean([d[k] for d in dicts])), 2)
    out["n_regions"] = len(regions)
    return out


def make_comparison_png(aggregates, path):
    """4 시나리오 × 4 지표 막대그래프 (가중 평균 기준)."""
    wavg = aggregates["weighted_avg"]
    metrics = [
        ("self_consumption_rate_pct", "자가소비율 (%)"),
        ("self_sufficiency_rate_pct", "자급률 (%)"),
        ("mean_shortage_mwh", "평균 부족 심각도 (MWh)"),
        ("battery_cycles", "배터리 사이클수"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    colors = ["#9aa5b1", "#f0a868", "#4e79a7", "#59a14f",
              "#b07aa1", "#e15759"]
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = [wavg[s][key] for s in SCENARIO_ORDER]
        bars = ax.bar(SCENARIO_ORDER, vals, color=colors)
        ax.set_title(label, fontsize=12)
        ax.tick_params(axis="x", rotation=15)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("ESS v2 — 4 시나리오 비교 (가중 평균 기준)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_region_breakdown_png(region_results, regions, path):
    """17지역 × 4시나리오 자급률 히트맵."""
    mat = np.array([[region_results[r][s]["self_sufficiency_rate_pct"]
                     for s in SCENARIO_ORDER] for r in regions])
    fig, ax = plt.subplots(figsize=(11, 11))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(SCENARIO_ORDER, rotation=25)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    for i in range(len(regions)):
        for j in range(len(SCENARIO_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("ESS v2 — 지역별 × 시나리오별 자급률 (%)", fontsize=13)
    fig.colorbar(im, ax=ax, label="자급률 (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_tou_breakdown_md(national_sum, path):
    """전국 합산 시뮬 결과를 부하구분별로 분해한 표 A~D 를 markdown 으로 출력."""
    baseline = national_sum["naive_baseline"]

    def _pct(part, total):
        return (part / total * 100.0) if total > 0 else 0.0

    lines = []
    lines.append("# ESS v2 — TOU 거래 패턴 세부 분석 (전국 합산 시뮬 기준)")
    lines.append("")
    lines.append(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── 표 A: 매도(export) 시간대별 분포 ─────────────────────────────────────
    lines.append("## 표 A. 매도(export) 시간대별 분포 (MWh)")
    lines.append("")
    lines.append("| 정책 | off_peak | mid_peak | max_peak | 합계 | max_peak 비중(%) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        exp = m["export_mwh_by_period"]
        tot = m["total_export_mwh"]
        share = _pct(exp["max_peak"], tot)
        lines.append(
            f"| {s} | {exp['off_peak']:,.1f} | {exp['mid_peak']:,.1f} | "
            f"{exp['max_peak']:,.1f} | {tot:,.1f} | {share:.2f} |"
        )
    lines.append("")

    # ── 표 B: 매수(import) 시간대별 분포 ─────────────────────────────────────
    lines.append("## 표 B. 매수(import) 시간대별 분포 (MWh)")
    lines.append("")
    lines.append("| 정책 | off_peak | mid_peak | max_peak | 합계 | off_peak 비중(%) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        imp = m["import_mwh_by_period"]
        tot = m["total_import_mwh"]
        share = _pct(imp["off_peak"], tot)
        lines.append(
            f"| {s} | {imp['off_peak']:,.1f} | {imp['mid_peak']:,.1f} | "
            f"{imp['max_peak']:,.1f} | {tot:,.1f} | {share:.2f} |"
        )
    lines.append("")

    # ── 표 C: 평균 단가 ──────────────────────────────────────────────────────
    lines.append("## 표 C. 평균 단가 (원/MWh)")
    lines.append("")
    lines.append("| 정책 | 평균 매수 단가 | 평균 매도 단가 | 매도-매수 스프레드 |")
    lines.append("|---|---:|---:|---:|")
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        buy = m["avg_import_price_krw_per_mwh"]
        sell = m["avg_export_price_krw_per_mwh"]
        spread = sell - buy
        lines.append(
            f"| {s} | {buy:,.0f} | {sell:,.0f} | {spread:+,.0f} |"
        )
    lines.append("")

    # ── 표 D: 정책 간 차이 (vs naive_baseline) ───────────────────────────────
    lines.append("## 표 D. 정책 간 차이 (vs naive_baseline)")
    lines.append("")
    lines.append("| 정책 | 자급률 차이(pt) | net_revenue 차이(원) | "
                 "max_peak export 차이(MWh) | off_peak import 차이(MWh) |")
    lines.append("|---|---:|---:|---:|---:|")
    base_ss = baseline["self_sufficiency_rate_pct"]
    base_net = baseline["net_revenue_krw"]
    base_exp_max = baseline["export_mwh_by_period"]["max_peak"]
    base_imp_off = baseline["import_mwh_by_period"]["off_peak"]
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        d_ss = m["self_sufficiency_rate_pct"] - base_ss
        d_net = m["net_revenue_krw"] - base_net
        d_exp_max = m["export_mwh_by_period"]["max_peak"] - base_exp_max
        d_imp_off = m["import_mwh_by_period"]["off_peak"] - base_imp_off
        lines.append(
            f"| {s} | {d_ss:+.2f} | {d_net:+,.0f} | "
            f"{d_exp_max:+,.1f} | {d_imp_off:+,.1f} |"
        )
    lines.append("")

    # ── 검증 노트 ────────────────────────────────────────────────────────────
    lines.append("## 검증 노트")
    lines.append("")
    lines.append("- `sum(*_by_period)` 가 기존 총합과 일치하는지(±1e-6) 자동 점검 — "
                 "콘솔 로그 참조.")
    lines.append("- 평균 매수/매도 단가는 KEPCO 산업용(을) 고압A 선택Ⅱ 평일 단가 "
                 "범위(87,300 ~ 222,300 원/MWh) 안에 있어야 정상.")
    lines.append("- naive_baseline 과 xgb_no_lookahead 의 모든 거래 지표가 "
                 "동일해야 함(두 정책의 SOC 결정이 동일).")

    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _verify_tou_breakdown(national_sum):
    """검증 체크리스트 — 분해 합계가 총합과 일치하는지, 단가 범위, 정책 동일성."""
    ok = True
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        checks = [
            (sum(m["import_mwh_by_period"].values()), m["total_import_mwh"],
             f"{s} import 합계"),
            (sum(m["export_mwh_by_period"].values()), m["total_export_mwh"],
             f"{s} export 합계"),
            (sum(m["cost_krw_by_period"].values()), m["total_cost_krw"],
             f"{s} cost 합계"),
            (sum(m["revenue_krw_by_period"].values()), m["total_revenue_krw"],
             f"{s} revenue 합계"),
        ]
        for got, expected, label in checks:
            tol = max(1.0, abs(expected) * 1e-6)
            if abs(got - expected) > tol:
                print(f"  ✗ {label}: {got:,.2f} vs {expected:,.2f} (diff={got - expected:+,.4f})")
                ok = False
        # 단가 범위(0 이 아니면 검사)
        for label, v in (("매수", m["avg_import_price_krw_per_mwh"]),
                         ("매도", m["avg_export_price_krw_per_mwh"])):
            if v > 0 and not (87_300 <= v <= 222_300):
                print(f"  ⚠ {s} 평균 {label} 단가 범위 밖: {v:,.0f} 원/MWh")
                ok = False

    # naive vs xgb_no_lookahead — 모든 거래 지표 동일성
    a, b = national_sum["naive_baseline"], national_sum["xgb_no_lookahead"]
    for key in ("total_import_mwh", "total_export_mwh",
                "total_cost_krw", "total_revenue_krw"):
        if abs(a[key] - b[key]) > 1.0:
            print(f"  ✗ naive vs xgb_no_lookahead {key} 불일치: "
                  f"{a[key]:,.2f} vs {b[key]:,.2f}")
            ok = False
    for p in PERIODS:
        if abs(a["import_mwh_by_period"][p] - b["import_mwh_by_period"][p]) > 1e-3:
            print(f"  ✗ naive vs xgb_no_lookahead import[{p}] 불일치")
            ok = False
        if abs(a["export_mwh_by_period"][p] - b["export_mwh_by_period"][p]) > 1e-3:
            print(f"  ✗ naive vs xgb_no_lookahead export[{p}] 불일치")
            ok = False

    if ok:
        print(f"[{ts()}] ✓ TOU breakdown 검증 통과 (합계 일치 / 단가 범위 / naive↔no_lookahead 동일)")
    else:
        print(f"[{ts()}] ✗ TOU breakdown 검증 실패 — 위 항목 확인")
    return ok


def _print_tou_table(national_sum, lp_stats_national=None):
    """섹션 10 보고 표 — 전국 합산 시뮬 기준 TOU 수익 비교 (MPC LP 수 포함)."""
    lp_stats_national = lp_stats_national or {}
    print("=" * 104)
    print("[ESS v2 TOU 결과 — 전국 합산 시뮬 기준]")
    print("=" * 104)
    print(f"{'정책':<20}{'자급률(%)':>12}{'net_revenue(원)':>22}"
          f"{'total_cost(원)':>18}{'total_revenue(원)':>20}{'LP 풀이':>12}")
    print("─" * 104)
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        lp_n = lp_stats_national.get(s, {}).get("lp_total", 0)
        print(f"{s:<20}{m['self_sufficiency_rate_pct']:>11.2f} "
              f"{m['net_revenue_krw']:>+21,.0f} "
              f"{m['total_cost_krw']:>17,.0f} "
              f"{m['total_revenue_krw']:>19,.0f} "
              f"{lp_n:>11,}")
    print("=" * 104)


def _print_mpc_report(national_sum, lp_stats_regions, lp_stats_national):
    """섹션 9 핵심 산출물 — MPC vs 기존 정책 비교 + 가설 검증 한 줄."""
    print()
    print("=" * 104)
    print("[MPC 핵심 보고 — 전국 합산 시뮬 기준]")
    print("=" * 104)
    print(f"{'정책':<20}{'자급률(%)':>12}{'net_revenue(원)':>22}"
          f"{'total_cost(원)':>18}{'total_revenue(원)':>20}{'LP 풀이':>12}")
    print("─" * 104)
    # 지역 합계 LP 수 (참고: 본 표는 national_sum, LP 수는 17지역 합)
    lp_total_by_scen = {s: 0 for s in SCENARIO_ORDER}
    for region, scen_map in lp_stats_regions.items():
        for s, stat in scen_map.items():
            lp_total_by_scen[s] += stat["lp_total"]
    for s in SCENARIO_ORDER:
        m = national_sum[s]
        if s in MPC_SCENARIOS:
            lp_n_disp = lp_total_by_scen[s]  # 17지역 합 (=17 × 8760)
        else:
            lp_n_disp = 0
        print(f"{s:<20}{m['self_sufficiency_rate_pct']:>11.2f} "
              f"{m['net_revenue_krw']:>+21,.0f} "
              f"{m['total_cost_krw']:>17,.0f} "
              f"{m['total_revenue_krw']:>19,.0f} "
              f"{lp_n_disp:>11,}")
    print("=" * 104)
    # 가설 검증 한 줄
    base_net = national_sum["xgb_lookahead"]["net_revenue_krw"]
    base_ss = national_sum["xgb_lookahead"]["self_sufficiency_rate_pct"]
    mpc_x_net = national_sum["mpc_xgb"]["net_revenue_krw"]
    mpc_x_ss = national_sum["mpc_xgb"]["self_sufficiency_rate_pct"]
    mpc_o_net = national_sum["mpc_oracle"]["net_revenue_krw"]
    delta_net_pct = (mpc_x_net - base_net) / abs(base_net) * 100 if base_net else 0.0
    oracle_uplift_pct = ((mpc_o_net - mpc_x_net) / abs(mpc_x_net) * 100
                         if mpc_x_net else 0.0)
    print(f"> mpc_xgb는 xgb_lookahead 대비 net_revenue {delta_net_pct:+.2f}% 변화, "
          f"자급률 {mpc_x_ss - base_ss:+.2f} pt 변화.")
    print(f"> mpc_oracle은 mpc_xgb 대비 net_revenue {oracle_uplift_pct:+.2f}% 추가 "
          f"변화 (=예측 정확도가 MPC 가치로 전환되는 폭).")


def _verify_mpc(national_sum):
    """MPC 검증 — oracle≥xgb_mpc, mpc_xgb 대 xgb_lookahead 우열, 기존 정책 invariance."""
    ok = True
    a = national_sum["mpc_oracle"]["net_revenue_krw"]
    b = national_sum["mpc_xgb"]["net_revenue_krw"]
    if a + 1e-6 < b:
        print(f"  ✗ mpc_oracle({a:+,.0f}) < mpc_xgb({b:+,.0f}) — oracle 우위 깨짐")
        ok = False
    else:
        print(f"  ✓ mpc_oracle ≥ mpc_xgb 만족 (net_revenue {a:+,.0f} ≥ {b:+,.0f})")

    lk = national_sum["xgb_lookahead"]["net_revenue_krw"]
    if b > lk:
        print(f"  ✓ mpc_xgb({b:+,.0f}) > xgb_lookahead({lk:+,.0f}) — MPC 가치 발견")
    else:
        print(f"  ⚠ mpc_xgb({b:+,.0f}) ≤ xgb_lookahead({lk:+,.0f}) — "
              f"MPC 가치 없음 (Phase 2 결론에 반영)")
    return ok


def _verify_invariance(national_sum, snapshot_path):
    """이전 4개 정책 결과와 완전 동일한지 검증."""
    if not snapshot_path.exists():
        print(f"  (snapshot 없음 — invariance 검증 생략: {snapshot_path})")
        return True
    with open(snapshot_path, encoding="utf-8") as f:
        old = json.load(f)
    old_national = old["aggregates"]["national_sum"]
    keys = ("self_sufficiency_rate_pct", "self_consumption_rate_pct",
            "total_shortage_mwh", "total_curtailment_mwh",
            "total_import_mwh", "total_export_mwh",
            "total_cost_krw", "total_revenue_krw", "net_revenue_krw")
    invariance_scenarios = [s for s in SCENARIO_ORDER if s not in MPC_SCENARIOS
                            and s in old_national]
    ok = True
    for s in invariance_scenarios:
        for k in keys:
            v_old = old_national[s][k]
            v_new = national_sum[s][k]
            tol = max(1.0, abs(v_old) * 1e-6)
            if abs(v_old - v_new) > tol:
                print(f"  ✗ invariance 깨짐: {s}.{k} {v_old} → {v_new}")
                ok = False
    if ok:
        print(f"  ✓ 기존 4개 정책({', '.join(invariance_scenarios)}) "
              f"national_sum 지표 전부 invariant")
    return ok


def _print_weighted_table(wavg):
    print("=" * 64)
    print("[전국 ESS v2 시뮬레이션 결과 — 가중 평균 기준]")
    print("=" * 64)
    print(f"{'시나리오':<20}{'자가소비율':>10}{'자급률':>10}{'평균부족':>12}{'사이클수':>10}")
    print("─" * 64)
    for s in SCENARIO_ORDER:
        m = wavg[s]
        print(f"{s:<20}{m['self_consumption_rate_pct']:>9.1f}%"
              f"{m['self_sufficiency_rate_pct']:>9.1f}%"
              f"{m['mean_shortage_mwh']:>8.1f} MWh"
              f"{m['battery_cycles']:>10.1f}")
    print("─" * 64)
    oracle_ss = wavg["oracle"]["self_sufficiency_rate_pct"]
    xgb_ss = wavg["xgb_lookahead"]["self_sufficiency_rate_pct"]
    nolook_ss = wavg["xgb_no_lookahead"]["self_sufficiency_rate_pct"]
    reach = xgb_ss / oracle_ss * 100.0 if oracle_ss else 0.0
    print(f"XGBoost가 Oracle 자급률의 {reach:.1f}% 도달")
    print(f"lookahead 도입 효과: 자급률 {xgb_ss - nolook_ss:+.1f} pt")
    print("=" * 64)


def main():
    print(f"[{ts()}] [TASK G-3+G-4] ESS v2 메인 시뮬레이션 시작")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    for path in (XGB_PREDICTIONS, TRAIN_FEATURES):
        if not Path(path).exists():
            sys.exit(f"ERROR: 파일 없음 → {path}")

    xgb_df = pd.read_csv(XGB_PREDICTIONS, encoding="utf-8-sig", parse_dates=["timestamp"])
    for col in ("timestamp", "region", "actual", "predicted"):
        if col not in xgb_df.columns:
            sys.exit(f"ERROR: '{col}' 컬럼이 {XGB_PREDICTIONS} 에 없습니다.")

    train_df = pd.read_csv(TRAIN_FEATURES, encoding="utf-8-sig")
    region_params = build_region_params(train_df)
    print(f"[{ts()}] 데이터 로드 완료 — 예측 {len(xgb_df)}행, "
          f"{xgb_df['region'].nunique()}개 지역")

    regions_all = sorted(xgb_df["region"].unique())
    missing = [r for r in regions_all if r not in region_params]
    if missing:
        sys.exit(f"ERROR: train 에 없는 지역 → {missing}")
    noise_regions = [r for r in regions_all if region_params[r]["is_noise_region"]]
    regions_clean = [r for r in regions_all if not region_params[r]["is_noise_region"]]
    print(f"[{ts()}] 노이즈 플래그 지역: {noise_regions} "
          f"→ clean {len(regions_clean)}개")

    # ── 지역별 × 시나리오별 실행 ─────────────────────────────────────────────
    # MPC 정책은 LP 풀이 비용이 크므로 (지역, 정책) 단위로 진행률·ETA 출력.
    region_results = {}
    lp_stats_by_region = {}  # region -> {scen: {"total":…,"infeasible":…,"elapsed":…}}
    n_sims = 0
    n_regions = len(regions_all)
    mpc_t0 = time.time()
    mpc_done = 0
    mpc_total_jobs = n_regions * len(MPC_SCENARIOS)
    for r_idx, region in enumerate(regions_all, start=1):
        r_df = xgb_df[xgb_df["region"] == region].sort_values("timestamp")
        hours = r_df["timestamp"].dt.hour.values
        months = r_df["timestamp"].dt.month.values
        actual_arr = r_df["actual"].values.astype(float)
        pred_arr = r_df["predicted"].values.astype(float)
        params = region_params[region]

        region_results[region] = {}
        lp_stats_by_region[region] = {}
        for scen_name, (policy_fn, pred_source) in SCENARIOS.items():
            pred_input = actual_arr if pred_source == "actual" else pred_arr
            is_mpc = scen_name in MPC_SCENARIOS
            pk = None
            if is_mpc:
                pk = {"months_arr": months, "hours_arr": hours,
                      "horizon": MPC_HORIZON}
                reset_lp_stats()
                t_start = time.time()
                print(f"[{ts()}] {scen_name} / {region} ({r_idx}/{n_regions}) "
                      f"— {len(actual_arr)} 시점 LP 풀이 중...")

            region_results[region][scen_name] = run_simulation(
                actual_arr, pred_input, hours, params, policy_fn,
                policy_kwargs=pk, months=months,
            )
            n_sims += 1

            if is_mpc:
                elapsed = time.time() - t_start
                stats = get_lp_stats()
                lp_stats_by_region[region][scen_name] = {
                    "lp_total": stats["total"],
                    "lp_infeasible": stats["infeasible"],
                    "elapsed_sec": round(elapsed, 2),
                }
                mpc_done += 1
                avg = (time.time() - mpc_t0) / max(mpc_done, 1)
                eta = avg * (mpc_total_jobs - mpc_done)
                print(f"[{ts()}]   ✓ elapsed={elapsed:.1f}s  "
                      f"LP={stats['total']:,} infeasible={stats['infeasible']}  "
                      f"ETA(MPC 전체)≈{eta:.0f}s")
    print(f"[{ts()}] 지역별 시뮬 완료 — {n_sims}개 "
          f"({len(regions_all)}지역 × {len(SCENARIOS)}시나리오)")

    # ── 집계 1: 단순 평균 (17개 / 16개) ──────────────────────────────────────
    simple_all = {s: aggregate_metrics(region_results, s, regions_all,
                                       region_params, "simple")
                  for s in SCENARIO_ORDER}
    simple_clean = {s: aggregate_metrics(region_results, s, regions_clean,
                                         region_params, "simple")
                    for s in SCENARIO_ORDER}

    # ── 집계 2: 가중 평균 (발전량 비중) ──────────────────────────────────────
    weighted = {s: aggregate_metrics(region_results, s, regions_all,
                                     region_params, "weighted")
                for s in SCENARIO_ORDER}

    # ── 집계 3: 전국 합산 시뮬 (시점별 합산 후 단일 시뮬) ────────────────────
    nat = (xgb_df.groupby("timestamp", as_index=False)
           .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
           .sort_values("timestamp"))
    nat_hours = nat["timestamp"].dt.hour.values
    nat_months = nat["timestamp"].dt.month.values
    nat_actual = nat["actual"].values.astype(float)
    nat_pred = nat["predicted"].values.astype(float)
    total_params = {
        "ess_capacity_mwh": TOTAL_ESS_CAPACITY_MWH,
        "demand_mwh_per_h": TOTAL_DEMAND_MWH_PER_H,
        "charge_rate_max": TOTAL_CHARGE_RATE_MAX,
        "discharge_rate_max": TOTAL_DISCHARGE_RATE_MAX,
        "weight": 1.0,
        "is_noise_region": False,
    }
    national_sum = {}
    national_sum_lp_stats = {}
    print(f"\n[{ts()}] 전국 합산 시뮬 — TOU sanity check (정책별 1회씩 출력)")
    for scen_name, (policy_fn, pred_source) in SCENARIOS.items():
        pred_input = nat_actual if pred_source == "actual" else nat_pred
        is_mpc = scen_name in MPC_SCENARIOS
        pk = None
        if is_mpc:
            pk = {"months_arr": nat_months, "hours_arr": nat_hours,
                  "horizon": MPC_HORIZON}
            reset_lp_stats()
            t_start = time.time()
            print(f"[{ts()}] (national) {scen_name} LP 풀이 중...")
        national_sum[scen_name] = run_simulation(
            nat_actual, pred_input, nat_hours, total_params, policy_fn,
            policy_kwargs=pk, months=nat_months,
            policy_name=scen_name, verbose=True,
        )
        if is_mpc:
            elapsed = time.time() - t_start
            stats = get_lp_stats()
            national_sum_lp_stats[scen_name] = {
                "lp_total": stats["total"],
                "lp_infeasible": stats["infeasible"],
                "elapsed_sec": round(elapsed, 2),
            }
            print(f"[{ts()}]   ✓ (national) elapsed={elapsed:.1f}s  "
                  f"LP={stats['total']:,} infeasible={stats['infeasible']}")
    print(f"[{ts()}] 3가지 집계 완료 (단순17/단순16/가중/합산)")

    aggregates = {
        "simple_avg_all_17": simple_all,
        "simple_avg_clean_16": simple_clean,
        "weighted_avg": weighted,
        "national_sum": national_sum,
    }

    # ── 순서 검증 (Oracle ≥ xgb_lookahead ≥ naive) ──────────────────────────
    ss = {s: weighted[s]["self_sufficiency_rate_pct"] for s in SCENARIO_ORDER}
    if not (ss["oracle"] >= ss["xgb_lookahead"] >= ss["naive_baseline"]):
        print(f"[{ts()}] ⚠ 경고: 자급률 순서가 예상(Oracle≥lookahead≥naive)과 다름 "
              f"→ naive={ss['naive_baseline']}, lookahead={ss['xgb_lookahead']}, "
              f"oracle={ss['oracle']}")
    else:
        print(f"[{ts()}] ✓ 자급률 순서 확인: "
              f"oracle({ss['oracle']:.1f}) ≥ lookahead({ss['xgb_lookahead']:.1f}) "
              f"≥ naive({ss['naive_baseline']:.1f})")

    # ── 결과 JSON ────────────────────────────────────────────────────────────
    results = {
        "config": {
            "ess_capacity_total_mwh": TOTAL_ESS_CAPACITY_MWH,
            "demand_total_mwh_per_h": TOTAL_DEMAND_MWH_PER_H,
            "efficiency": EFFICIENCY,
            "soc_range": [SOC_MIN, SOC_MAX],
            "load_pattern": "정성적 한국 부하 곡선 (정규화)",
            "model": "XGBoost (national v2, power_diff 포함)",
            "noise_regions": noise_regions,
            "mpc_horizon": MPC_HORIZON,
            "mpc_solver": "scipy.optimize.linprog(method='highs')",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "region_params": region_params,
        "regions": region_results,
        "aggregates": aggregates,
        "mpc_lp_stats": {
            "regions": lp_stats_by_region,
            "national_sum": national_sum_lp_stats,
        },
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{ts()}] 결과 저장 → {OUT_JSON}")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    make_comparison_png(aggregates, OUT_PNG_COMPARISON)
    print(f"[{ts()}] 시각화 저장 → {OUT_PNG_COMPARISON}")
    make_region_breakdown_png(region_results, regions_all, OUT_PNG_REGION)
    print(f"[{ts()}] 시각화 저장 → {OUT_PNG_REGION}")

    # ── stdout 요약 ──────────────────────────────────────────────────────────
    print()
    _print_weighted_table(weighted)
    print()
    _print_tou_table(national_sum, national_sum_lp_stats)

    # ── 섹션 9 핵심 보고: MPC vs 기존 정책 ─────────────────────────────────
    _print_mpc_report(national_sum, lp_stats_by_region, national_sum_lp_stats)
    print(f"\n[{ts()}] MPC 검증")
    _verify_mpc(national_sum)
    print(f"[{ts()}] 기존 4개 정책 invariance 검증 (prempc snapshot 대비)")
    _verify_invariance(national_sum,
                       OUT_JSON.with_name("ess_v2_simulation_results.prempc.json"))

    # ── TOU 부하구분별 분해 표 출력 + 검증 ──────────────────────────────────
    _write_tou_breakdown_md(national_sum, OUT_MD_TOU_BREAKDOWN)
    print(f"\n[{ts()}] TOU 분해 표 저장 → {OUT_MD_TOU_BREAKDOWN}")
    _verify_tou_breakdown(national_sum)

    # ── claude_share 복사 ────────────────────────────────────────────────────
    print(f"\n[{ts()}] claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    for src in (Path(__file__), OUT_JSON, OUT_PNG_COMPARISON, OUT_PNG_REGION,
                OUT_MD_TOU_BREAKDOWN):
        if src.exists():
            dst = SHARE_DIR / src.name
            shutil.copy2(src, dst)
            print(f"   → {dst}")

    print(f"\n[{ts()}] [TASK G-3+G-4] 완료")
    return results


if __name__ == "__main__":
    main()
