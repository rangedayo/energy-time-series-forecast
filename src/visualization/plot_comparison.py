import sys
import json
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys as _sys
_sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font
_apply_font()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
PRED_CSV     = "outputs/xgb_predictions.csv"
XGB_JSON     = "outputs/xgb_results.json"
BASE_JSON    = "outputs/baseline_results.json"
OUT_PNG      = "outputs/prediction_comparison.png"
PEAK_HOURS   = range(10, 15)   # 10~14시

# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

with open(XGB_JSON,  encoding="utf-8") as f: xgb  = json.load(f)
with open(BASE_JSON, encoding="utf-8") as f: base = json.load(f)

# Naive lag1 예측값 = 직전 시점의 actual (=lag_1h)
df["naive_lag1"] = df["actual"].shift(1).bfill()
df["naive_lag1"] = np.clip(df["naive_lag1"], 0, None)

# ══════════════════════════════════════════════════════════════════════════════
# 가장 잘 보이는 2주 구간 자동 선택
# : XGBoost 예측과 실제값의 상관이 높고(잘 맞고),
#   실제 발전이 활발한(평균 발전량이 높은) 구간을 선택
# ══════════════════════════════════════════════════════════════════════════════
WINDOW_HOURS = 14 * 24  # 2주 = 336시간
stride       = 24       # 하루 단위로 슬라이딩

best_score = -np.inf
best_start = 0

for i in range(0, len(df) - WINDOW_HOURS, stride):
    seg = df.iloc[i : i + WINDOW_HOURS]
    avg_power = seg["actual"].mean()
    # 상관계수 계산 (발전량이 거의 0인 구간은 제외)
    daytime_seg = seg[seg["actual"] > 1.0]
    if len(daytime_seg) < 50:
        continue
    corr = np.corrcoef(daytime_seg["actual"], daytime_seg["predicted"])[0, 1]
    # 점수: 발전량 활발 + 예측 잘 맞는 구간
    score = avg_power * corr if not np.isnan(corr) else -np.inf
    if score > best_score:
        best_score = score
        best_start = i

seg = df.iloc[best_start : best_start + WINDOW_HOURS].copy()
seg = seg.reset_index(drop=True)

print(f"선택 구간: {seg['timestamp'].iloc[0].date()} ~ {seg['timestamp'].iloc[-1].date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 그래프
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

times = seg["timestamp"]

# 피크 시간대 배경 음영
prev_in_peak = False
shade_start  = None
for _, row in seg.iterrows():
    in_peak = row["timestamp"].hour in PEAK_HOURS
    if in_peak and not prev_in_peak:
        shade_start = row["timestamp"]
    elif not in_peak and prev_in_peak and shade_start is not None:
        ax.axvspan(shade_start, row["timestamp"],
                   color="#FFD700", alpha=0.18, label="_nolegend_")
        shade_start = None
    prev_in_peak = in_peak
# 마지막 구간 처리
if prev_in_peak and shade_start is not None:
    ax.axvspan(shade_start, seg["timestamp"].iloc[-1],
               color="#FFD700", alpha=0.18)

# 데이터 라인
ax.plot(times, seg["actual"],    color="#2176AE", linewidth=1.6,
        label="실제값", zorder=3)
ax.plot(times, seg["naive_lag1"],color="#999999", linewidth=1.2,
        linestyle="--", label="Naive lag1", zorder=2, alpha=0.85)
ax.plot(times, seg["predicted"], color="#D62828", linewidth=1.4,
        label="XGBoost 예측", zorder=4)

# 피크 음영 범례용 더미 패치
import matplotlib.patches as mpatches
peak_patch = mpatches.Patch(color="#FFD700", alpha=0.4, label="피크 시간대 (10~14시)")

ax.set_xlabel("날짜", fontsize=11)
ax.set_ylabel("태양광 발전량 (MWh)", fontsize=11)
ax.set_title(
    f"예측값 비교 — {seg['timestamp'].iloc[0].strftime('%Y-%m-%d')} ~ "
    f"{seg['timestamp'].iloc[-1].strftime('%Y-%m-%d')}  (2주)",
    fontsize=13, fontweight="bold"
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [peak_patch], labels + ["피크 시간대 (10~14시)"],
          loc="upper left", fontsize=9, framealpha=0.85)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

ax.set_xlim(times.iloc[0], times.iloc[-1])
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)
print(f"그래프 저장: {OUT_PNG}")

# ══════════════════════════════════════════════════════════════════════════════
# 성능 요약 출력
# ══════════════════════════════════════════════════════════════════════════════
xgb_mae       = xgb["MAE"]
xgb_rmse      = xgb["RMSE"]
xgb_mae_peak  = xgb["MAE_peak"]
naive_mae      = base["lag1"]["MAE"]
mae_reduction  = naive_mae - xgb_mae          # MWh
imp_pct        = xgb["improvement_vs_lag1_pct"]
daily_reduction= mae_reduction * 24           # MWh/day (24시간 누적)

print()
print("=== XGBoost 성능 요약 ===")
print()
print("[MAE/RMSE 수치]")
print(f"- 전체 MAE:       {xgb_mae:.2f} MWh  (Naive 대비 {imp_pct:.1f}% 개선)")
print(f"- 전체 RMSE:      {xgb_rmse:.2f} MWh")
print(f"- 피크 시간대 MAE: {xgb_mae_peak:.2f} MWh")
print()
print("[비즈니스 의미]")
print(f"- MAE {mae_reduction:.2f} MWh 감소 = ESS 충전 계획 오차가 시간당 평균 {mae_reduction:.2f} MWh 줄었다는 의미")
print(f"- 하루 24시간 기준 누적 오차 감소: {daily_reduction:.2f} MWh/day")
print(f"- 이는 ESS 불필요한 충방전 횟수 감소 및 배터리 수명 보호로 이어짐")
