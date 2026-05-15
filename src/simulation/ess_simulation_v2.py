"""
ESS 시뮬레이터 본체 + 메인 비교 실행 (v2)

TASK G-3: run_simulation — 한 (지역, 정책) 조합의 단일 시뮬레이션
TASK G-4: main — 4 시나리오 × 17 지역 매트릭스 실행 + 3가지 집계 + 시각화

본 시뮬레이션은 통제된 모델 비교 환경이며 실제 운영값 추정이 목적이 아니다.
LSTM 은 본 분석에서 제외한다. XGBoost 단일 모델 + 합성 정책만 비교한다.
"""

import sys
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
)
from src.simulation.ess_policy_v2 import (
    policy_naive, policy_lookahead, policy_perfect_foresight,
    policy_xgb_no_lookahead,
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
SHARE_DIR = Path("claude_share")

# 집계 대상 수치 지표 키
METRIC_KEYS = [
    "self_consumption_rate_pct", "self_sufficiency_rate_pct",
    "total_shortage_mwh", "mean_shortage_mwh", "max_shortage_mwh",
    "curtailment_rate_pct", "shortage_count", "battery_cycles", "ess_score",
    "total_curtailment_mwh", "total_demand_mwh", "total_gen_mwh", "n_hours",
]


# ════════════════════════════════════════════════════════════════════════════
# TASK G-3 — 시뮬레이터 본체
# ════════════════════════════════════════════════════════════════════════════
def run_simulation(actual, predicted, hours, params, policy_fn, policy_kwargs=None):
    """
    한 (지역, 정책) 조합에 대한 단일 시뮬레이션 실행.

    정책은 SOC 목표만 결정하고, 충방전 실행은 항상 actual(실측) 기준으로 한다.

    지표 해석:
      self_consumption_rate (자가소비율, %) — 발전한 전기 중 활용한 비율. 높을수록 좋음.
      self_sufficiency_rate (자급률, %)   — 수요 중 자체 공급으로 충당한 비율. 높을수록 좋음.
      total_shortage_mwh — 부족의 총량(절대값). 0에 가까울수록 좋음.
      mean_shortage_mwh  — 부족 발생 시 평균 강도. 진단용.
      max_shortage_mwh   — 최악 부족 시점. 극단 시나리오 대응력.
    """
    policy_kwargs = policy_kwargs or {}

    n = len(actual)
    soc = SOC_INIT
    total_curtailment = 0.0
    total_shortage_mwh = 0.0
    total_demand_mwh = 0.0
    shortage_list = []
    charge_cycles = 0.0
    discharge_cycles = 0.0

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

        if actual_net > 0:
            # 잉여 → 충전
            max_storable = max(0.0, (soc_target_high - soc) * cap / EFFICIENCY)
            charge_amount = min(actual_net, chg_max, max_storable)
            soc += charge_amount * EFFICIENCY / cap
            charge_cycles += charge_amount / cap
            total_curtailment += actual_net - charge_amount
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

    total_gen = float(np.sum(actual))
    curtailment_rate = total_curtailment / max(total_gen, 1e-10) * 100.0
    self_consumption_rate = 100.0 - curtailment_rate
    self_sufficiency_rate = (1.0 - total_shortage_mwh / max(total_demand_mwh, 1e-10)) * 100.0
    battery_cycles = (charge_cycles + discharge_cycles) / 2.0

    shortage_count = len(shortage_list)
    ess_score = (1.0 - curtailment_rate / 100.0) * (1.0 - shortage_count / max(n, 1)) * 100.0

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

        # 노이즈 플래그 (울산시 등 weight < 임계 지역)
        "flagged_noise_region": bool(params.get("is_noise_region", False)),
    }


# ════════════════════════════════════════════════════════════════════════════
# TASK G-4 — 메인 비교 실행
# ════════════════════════════════════════════════════════════════════════════
SCENARIOS = {
    # scenario_name : (policy_fn, pred_source)
    "naive_baseline":   (policy_naive,             "actual"),
    "xgb_no_lookahead": (policy_xgb_no_lookahead,  "predicted"),
    "xgb_lookahead":    (policy_lookahead,         "predicted"),
    "oracle":           (policy_perfect_foresight, "actual"),
}
SCENARIO_ORDER = ["naive_baseline", "xgb_no_lookahead", "xgb_lookahead", "oracle"]


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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#9aa5b1", "#f0a868", "#4e79a7", "#59a14f"]
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
    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(SCENARIO_ORDER, rotation=20)
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
    region_results = {}
    n_sims = 0
    for region in regions_all:
        r_df = xgb_df[xgb_df["region"] == region].sort_values("timestamp")
        hours = r_df["timestamp"].dt.hour.values
        actual_arr = r_df["actual"].values.astype(float)
        pred_arr = r_df["predicted"].values.astype(float)
        params = region_params[region]

        region_results[region] = {}
        for scen_name, (policy_fn, pred_source) in SCENARIOS.items():
            pred_input = actual_arr if pred_source == "actual" else pred_arr
            region_results[region][scen_name] = run_simulation(
                actual_arr, pred_input, hours, params, policy_fn,
            )
            n_sims += 1
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
    for scen_name, (policy_fn, pred_source) in SCENARIOS.items():
        pred_input = nat_actual if pred_source == "actual" else nat_pred
        national_sum[scen_name] = run_simulation(
            nat_actual, pred_input, nat_hours, total_params, policy_fn,
        )
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
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "region_params": region_params,
        "regions": region_results,
        "aggregates": aggregates,
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

    # ── claude_share 복사 ────────────────────────────────────────────────────
    print(f"\n[{ts()}] claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    for src in (Path(__file__), OUT_JSON, OUT_PNG_COMPARISON, OUT_PNG_REGION):
        if src.exists():
            dst = SHARE_DIR / src.name
            shutil.copy2(src, dst)
            print(f"   → {dst}")

    print(f"\n[{ts()}] [TASK G-3+G-4] 완료")
    return results


if __name__ == "__main__":
    main()
