"""
ESS Sensitivity 분석 (v2) — TASK G-5

"예측 정확도가 얼마나 ESS 가치(자급률)로 전환되는가?"의 곡선.

실측값에 합성 노이즈를 단계적으로 주입한 예측으로 lookahead 정책을 돌려
noise_level → 자급률 곡선을 그리고, 그 위에 XGBoost 실제 위치를 별표로 마킹한다.

전국 합산(national_sum) 시계열을 기반으로 실행한다 (단일 시계열 → 단위 일관성).
LSTM 은 본 분석에서 제외한다.
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
    TOTAL_ESS_CAPACITY_MWH, TOTAL_DEMAND_MWH_PER_H,
    TOTAL_CHARGE_RATE_MAX, TOTAL_DISCHARGE_RATE_MAX,
)
from src.simulation.ess_policy_v2 import policy_lookahead, policy_lookahead_noisy
from src.simulation.ess_simulation_v2 import run_simulation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ── 경로 / 파라미터 상수 ──────────────────────────────────────────────────────
XGB_PREDICTIONS = "outputs/national_xgb_predictions.csv"
OUT_JSON = Path("outputs/ess_v2_sensitivity_results.json")
OUT_PNG = Path("outputs/ess_v2_sensitivity_curve.png")
SHARE_DIR = Path("claude_share")

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5]
RNG_SEEDS = [42, 123, 456]  # 견고성 확보

TOTAL_PARAMS = {
    "ess_capacity_mwh": TOTAL_ESS_CAPACITY_MWH,
    "demand_mwh_per_h": TOTAL_DEMAND_MWH_PER_H,
    "charge_rate_max": TOTAL_CHARGE_RATE_MAX,
    "discharge_rate_max": TOTAL_DISCHARGE_RATE_MAX,
    "weight": 1.0,
    "is_noise_region": False,
}


def run_with_synthetic_noise(actual, hours, noise, rng):
    """실측값에 noise_level 만큼 합성 노이즈를 주입한 예측으로 lookahead 시뮬."""
    return run_simulation(
        actual, actual, hours, TOTAL_PARAMS, policy_lookahead_noisy,
        policy_kwargs={"noise_level": noise, "rng": rng},
    )


def main():
    print(f"[{ts()}] [TASK G-5] ESS v2 Sensitivity 분석 시작")

    if not Path(XGB_PREDICTIONS).exists():
        sys.exit(f"ERROR: 파일 없음 → {XGB_PREDICTIONS}")

    xgb_df = pd.read_csv(XGB_PREDICTIONS, encoding="utf-8-sig", parse_dates=["timestamp"])
    for col in ("timestamp", "region", "actual", "predicted"):
        if col not in xgb_df.columns:
            sys.exit(f"ERROR: '{col}' 컬럼이 {XGB_PREDICTIONS} 에 없습니다.")

    # ── 전국 합산 시계열 ─────────────────────────────────────────────────────
    nat = (xgb_df.groupby("timestamp", as_index=False)
           .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
           .sort_values("timestamp"))
    nat_hours = nat["timestamp"].dt.hour.values
    nat_actual = nat["actual"].values.astype(float)
    nat_pred = nat["predicted"].values.astype(float)
    actual_std = float(np.std(nat_actual))
    print(f"[{ts()}] 전국 합산 시계열 {len(nat_actual)}시점, "
          f"actual std={actual_std:.2f} MWh")

    # ── noise level × seed 매트릭스 ──────────────────────────────────────────
    points = []
    for noise in NOISE_LEVELS:
        for seed in RNG_SEEDS:
            rng = np.random.default_rng(seed)
            sim = run_with_synthetic_noise(nat_actual, nat_hours, noise, rng)
            points.append({
                "noise_level": noise,
                "seed": seed,
                "self_sufficiency_rate": sim["self_sufficiency_rate_pct"],
                "self_consumption_rate": sim["self_consumption_rate_pct"],
                "ess_score": sim["ess_score"],
            })
    print(f"[{ts()}] sensitivity 매트릭스 완료 — "
          f"{len(NOISE_LEVELS)}개 noise × {len(RNG_SEEDS)}개 seed = {len(points)}점")

    # ── seed 평균 곡선 ───────────────────────────────────────────────────────
    curve = []
    for noise in NOISE_LEVELS:
        ss = [p["self_sufficiency_rate"] for p in points if p["noise_level"] == noise]
        sc = [p["self_consumption_rate"] for p in points if p["noise_level"] == noise]
        curve.append({
            "noise_level": noise,
            "ss_mean": float(np.mean(ss)),
            "ss_std": float(np.std(ss)),
            "sc_mean": float(np.mean(sc)),
        })
    noise_x = np.array([c["noise_level"] for c in curve])
    ss_y = np.array([c["ss_mean"] for c in curve])
    ss_err = np.array([c["ss_std"] for c in curve])

    # ── XGBoost 실제 위치 ────────────────────────────────────────────────────
    # 노이즈 모델: 주입 sigma = actual_std × noise_level → noise_level = sigma/std.
    # XGBoost 의 노이즈 등가 = 예측 오차의 std / actual std (동일 단위 기준).
    xgb_error = nat_actual - nat_pred
    xgb_error_std = float(np.std(xgb_error))
    xgb_mae = float(np.mean(np.abs(xgb_error)))
    xgb_noise_equiv = xgb_error_std / actual_std       # 곡선 x축 등가 위치
    xgb_nmae = xgb_mae / actual_std                    # 참고용 nMAE
    # XGBoost 실제 예측으로 lookahead 돌린 자급률 (합성 노이즈 아님)
    xgb_sim = run_simulation(nat_actual, nat_pred, nat_hours, TOTAL_PARAMS, policy_lookahead)
    xgb_ss = xgb_sim["self_sufficiency_rate_pct"]
    print(f"[{ts()}] XGBoost 위치 — noise 등가={xgb_noise_equiv:.3f} "
          f"(nMAE={xgb_nmae:.3f}), 자급률={xgb_ss:.2f}%")

    # ── 결론 수치 ────────────────────────────────────────────────────────────
    oracle_ss = curve[0]["ss_mean"]                    # noise=0.0
    interp_at = lambda x: float(np.interp(x, noise_x, ss_y))
    xgb_curve_ss = interp_at(xgb_noise_equiv)
    half_noise_ss = interp_at(xgb_noise_equiv * 0.5)   # 정확도 50% 개선 등가
    reach_pct = xgb_ss / oracle_ss * 100.0 if oracle_ss else 0.0
    half_gain = half_noise_ss - xgb_curve_ss
    slope = float(np.polyfit(noise_x, ss_y, 1)[0])  # pt / 노이즈 단위 (선형 적합)

    # ── JSON 저장 ────────────────────────────────────────────────────────────
    results = {
        "config": {
            "basis": "national_sum (전국 합산 시계열)",
            "policy": "policy_lookahead_noisy (합성 노이즈 주입 lookahead)",
            "noise_levels": NOISE_LEVELS,
            "rng_seeds": RNG_SEEDS,
            "actual_std_mwh": round(actual_std, 2),
            "model": "XGBoost (national v2) — LSTM 제외",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "points": points,
        "curve": curve,
        "xgboost": {
            "noise_equiv": round(xgb_noise_equiv, 4),
            "nmae": round(xgb_nmae, 4),
            "mae_mwh": round(xgb_mae, 4),
            "error_std_mwh": round(xgb_error_std, 4),
            "self_sufficiency_rate": xgb_ss,
            "self_sufficiency_on_curve": round(xgb_curve_ss, 2),
        },
        "summary": {
            "oracle_ss": round(oracle_ss, 2),
            "xgb_ss": round(xgb_ss, 2),
            "reach_pct": round(reach_pct, 2),
            "ss_at_half_noise": round(half_noise_ss, 2),
            "gain_from_50pct_accuracy": round(half_gain, 2),
            "curve_slope_pt_per_noise": round(slope, 3),
        },
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{ts()}] 결과 저장 → {OUT_JSON}")

    # ── 곡선 시각화 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7.6))
    # 곡선: 강한 빨강으로 눈에 띄게
    ax.errorbar(noise_x, ss_y, yerr=ss_err, marker="o", markersize=7,
                linewidth=2.6, capsize=4, color="#d62728",
                label="합성 노이즈 lookahead (seed 평균±std)")
    # Oracle 점: 큰 다이아몬드
    ax.scatter([0.0], [oracle_ss], s=320, marker="D", color="#2ca02c",
               edgecolors="black", linewidths=1.2, zorder=5,
               label=f"Oracle (noise=0): {oracle_ss:.1f}%")
    # XGBoost 점: 큰 별표
    ax.scatter([xgb_noise_equiv], [xgb_ss], s=620, marker="*", color="#ff7f0e",
               edgecolors="black", linewidths=1.2, zorder=6,
               label=f"XGBoost (noise~{xgb_noise_equiv:.2f}): {xgb_ss:.1f}%")
    ax.axvline(xgb_noise_equiv, ls="--", color="#ff7f0e", alpha=0.6)
    # 기울기 주석
    ax.annotate(f"기울기 ~ {slope:+.1f}pt / 노이즈 단위\n(양의 기울기 = 역설)",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=11, color="#d62728",
                bbox=dict(boxstyle="round", fc="#fff3f3", ec="#d62728"))
    ax.set_xlabel("노이즈 수준 (주입 sigma / actual std)")
    ax.set_ylabel("자급률 (%)")
    ax.set_title("ESS v2 Sensitivity — 예측 정확도 → ESS 자급률 전환 곡선")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    # 하단 캡션
    fig.text(0.5, -0.02,
             "예측 정확도가 높을수록 자급률이 오히려 미세 하락하는 역설. "
             "정책이 손해 구조일 때 노이즈가 신호를 망가뜨려 결과를 개선한다.",
             ha="center", va="top", fontsize=10, color="#444444", wrap=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{ts()}] 시각화 저장 → {OUT_PNG}")

    # ── 자동 결론 출력 ───────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("[Sensitivity 분석 결론]")
    print("=" * 64)
    print(f"Oracle (noise=0.0):           자급률 {oracle_ss:.1f}%")
    print(f"XGBoost 위치 (noise≈{xgb_noise_equiv:.2f}):    자급률 {xgb_ss:.1f}%")
    print(f"→ XGBoost는 Oracle 대비 {reach_pct:.1f}% 도달")
    print(f"→ 예측 정확도를 50% 더 높여도(noise {xgb_noise_equiv:.2f}→{xgb_noise_equiv*0.5:.2f}) "
          f"자급률은 {half_gain:+.1f} pt 만 추가")
    print(f"→ 한계는 시뮬레이터 설계(그리디 단일스텝) 또는 ESS 용량 자체에 있음을 시사")
    print("=" * 64)

    # ── claude_share 복사 ────────────────────────────────────────────────────
    print(f"\n[{ts()}] claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    for src in (Path(__file__), OUT_JSON, OUT_PNG):
        if src.exists():
            dst = SHARE_DIR / src.name
            shutil.copy2(src, dst)
            print(f"   → {dst}")

    print(f"\n[{ts()}] [TASK G-5] 완료")
    return results


if __name__ == "__main__":
    main()
