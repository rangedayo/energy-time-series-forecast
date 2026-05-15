import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font
_apply_font()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── ESS 파라미터 (기존과 동일) ─────────────────────────────────────────────────
ESS_CAPACITY_MWH    = 500.0
SOC_MIN             = 0.20
SOC_MAX             = 0.80
CHARGE_RATE_MAX     = 100.0
DISCHARGE_RATE_MAX  = 100.0
EFFICIENCY          = 0.95
DEMAND_MWH_PER_HOUR = 50.0

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
XGB_PRED    = "outputs/hybrid_xgb_predictions.csv"
OUTPUT_JSON = "outputs/hybrid_ess_simulation_results.json"
OUTPUT_PNG  = "outputs/hybrid_ess_simulation_comparison.png"

# 비교 기준값 (national_ess_simulation_results.json)
INTEGRATED_SCORES = {
    "naive_strategy": {"curtailment_rate_pct": 95.2,  "shortage_count": 509, "battery_cycles": 152.14, "ess_score": 4.38},
    "xgb_strategy":  {"curtailment_rate_pct": 86.09, "shortage_count": 508, "battery_cycles": 152.14, "ess_score": 12.69},
}
JEONNAM_OLD_ESS_SCORE = 24.6  # 통합 모델 기준 전남 ESS 점수

print(f"[{ts()}] [하이브리드 ESS 시뮬레이션 시작]")

if not os.path.exists(XGB_PRED):
    sys.exit(f"ERROR: 파일 없음 → {XGB_PRED}\n먼저 build_hybrid_predictions.py를 실행하세요.")

xgb_df = pd.read_csv(XGB_PRED, encoding="utf-8-sig", parse_dates=["timestamp"])
print(f"[{ts()}]   하이브리드 예측 로드: {len(xgb_df):,}행  {xgb_df['region'].nunique()}개 지역")


def run_simulation(actual_mwh: np.ndarray, predicted_mwh: np.ndarray, strategy_name: str) -> dict:
    n                 = len(actual_mwh)
    soc               = 0.5
    total_curtailment = 0.0
    shortage_count    = 0
    charge_cycles     = 0.0

    for i in range(n):
        gen  = float(actual_mwh[i])
        pred = float(predicted_mwh[i])

        actual_net = gen  - DEMAND_MWH_PER_HOUR
        pred_net   = pred - DEMAND_MWH_PER_HOUR
        decision   = actual_net if strategy_name == "naive_strategy" else pred_net

        if decision > 0:
            max_storable  = (SOC_MAX - soc) * ESS_CAPACITY_MWH / EFFICIENCY
            charge_amount = min(decision, CHARGE_RATE_MAX, max(0, max_storable))
            soc          += charge_amount * EFFICIENCY / ESS_CAPACITY_MWH
            charge_cycles += charge_amount / ESS_CAPACITY_MWH
            total_curtailment += max(0.0, max(0.0, actual_net) - charge_amount)
        else:
            needed            = max(0.0, -actual_net)
            max_dischargeable = (soc - SOC_MIN) * ESS_CAPACITY_MWH * EFFICIENCY
            discharge_amount  = min(needed, DISCHARGE_RATE_MAX, max(0, max_dischargeable))
            soc              -= discharge_amount / (ESS_CAPACITY_MWH * EFFICIENCY)
            if gen + discharge_amount < DEMAND_MWH_PER_HOUR:
                shortage_count += 1

    total_gen        = float(actual_mwh.sum())
    curtailment_rate = total_curtailment / max(total_gen, 1e-10) * 100.0
    ess_score        = (1.0 - curtailment_rate / 100.0) * (1.0 - shortage_count / n) * 100.0
    return {
        "curtailment_rate_pct": round(curtailment_rate, 2),
        "shortage_count":       int(shortage_count),
        "battery_cycles":       round(charge_cycles, 2),
        "ess_score":            round(ess_score, 2),
    }


# 전국 집계 (시간별 합산)
ts_col  = "timestamp"
agg_xgb = xgb_df.groupby(ts_col).agg({"actual": "sum", "predicted": "sum"}).reset_index()
actual   = agg_xgb["actual"].values
xgb_pred = agg_xgb["predicted"].values

results = {}

print(f"\n[{ts()}] [1] Naive 전략")
results["naive_strategy"] = run_simulation(actual, actual, "naive_strategy")
print(f"   → {results['naive_strategy']}")

print(f"[{ts()}] [2] XGBoost 하이브리드 전략")
results["xgb_hybrid_strategy"] = run_simulation(actual, xgb_pred, "xgb_hybrid_strategy")
print(f"   → {results['xgb_hybrid_strategy']}")

# 지역별 운영효율점수
print(f"\n[{ts()}] [지역별 운영효율점수 (하이브리드)]")
region_scores = {}
for region in sorted(xgb_df["region"].unique()):
    r_df = xgb_df[xgb_df["region"] == region]
    r_result = run_simulation(r_df["actual"].values, r_df["predicted"].values, "xgb_hybrid_strategy")
    region_scores[region] = r_result["ess_score"]
    print(f"  {region}: {r_result['ess_score']:.1f}")
results["region_ess_scores_xgb_hybrid"] = region_scores

# ── 비교 테이블 출력 ──────────────────────────────────────────────────────────
naive_ref = INTEGRATED_SCORES["naive_strategy"]
xgb_ref   = INTEGRATED_SCORES["xgb_strategy"]
hybrid_r  = results["xgb_hybrid_strategy"]
score_delta = hybrid_r["ess_score"] - xgb_ref["ess_score"]
jeonnam_new_score = region_scores.get("전라남도", 0)

print(f"\n[{ts()}] [ESS 시뮬레이션 비교]")
print(f"  {'전략':<22} {'전력낭비율':>10} {'부족횟수':>8} {'사이클수':>8} {'운영효율점수':>12}")
print(f"  {'-'*64}")
print(f"  {'Naive':<22} {naive_ref['curtailment_rate_pct']:>9.1f}%"
      f" {naive_ref['shortage_count']:>8} {naive_ref['battery_cycles']:>8.1f}"
      f" {naive_ref['ess_score']:>12.1f}")
print(f"  {'XGBoost 통합':<22} {xgb_ref['curtailment_rate_pct']:>9.1f}%"
      f" {xgb_ref['shortage_count']:>8} {xgb_ref['battery_cycles']:>8.1f}"
      f" {xgb_ref['ess_score']:>12.1f}")
print(f"  {'XGBoost 하이브리드':<22} {hybrid_r['curtailment_rate_pct']:>9.1f}%"
      f" {hybrid_r['shortage_count']:>8} {hybrid_r['battery_cycles']:>8.1f}"
      f" {hybrid_r['ess_score']:>12.1f}  ← 신규")
print(f"\n  전남 피크 MAE 25% 개선 → 운영효율점수: {xgb_ref['ess_score']:.2f} → {hybrid_r['ess_score']:.2f} ({score_delta:+.2f})")
print(f"  전남 지역 ESS 점수: {JEONNAM_OLD_ESS_SCORE} → {jeonnam_new_score:.1f}")

# ── 저장 ──────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
results["comparison_reference"] = INTEGRATED_SCORES
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[{ts()}]   결과 저장: {OUTPUT_JSON}")

# 시각화
labels       = ["Naive", "XGBoost 통합", "XGBoost 하이브리드"]
scores_plot  = [naive_ref["ess_score"], xgb_ref["ess_score"], hybrid_r["ess_score"]]
curtail_plot = [naive_ref["curtailment_rate_pct"], xgb_ref["curtailment_rate_pct"], hybrid_r["curtailment_rate_pct"]]
colors       = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(labels, scores_plot, color=colors)
axes[0].set_title("[전국] ESS 운영 효율 점수 (하이브리드 비교)")
axes[0].set_ylabel("점수")
axes[0].set_ylim(0, 20)
for i, v in enumerate(scores_plot):
    axes[0].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)

axes[1].bar(labels, curtail_plot, color=colors)
axes[1].set_title("[전국] 전력 낭비율 (%) (하이브리드 비교)")
axes[1].set_ylabel("%")
for i, v in enumerate(curtail_plot):
    axes[1].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.close()
print(f"[{ts()}]   시각화 저장: {OUTPUT_PNG}")

print(f"\n[{ts()}] [하이브리드 ESS 시뮬레이션 완료]")
