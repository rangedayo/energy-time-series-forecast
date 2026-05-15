import sys
import platform
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from statsmodels.tsa.seasonal import seasonal_decompose

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
def set_korean_font():
    system = platform.system()
    if system == "Windows":
        candidates = ["Malgun Gothic", "맑은 고딕", "NanumGothic"]
    elif system == "Darwin":
        candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]
    else:
        candidates = ["NanumGothic", "UnDotum", "DejaVu Sans"]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

DATA_PATH  = "data/raw/한국동서발전(주)_제주 기상관측 및 태양광 발전 현황_20240531.csv"
OUTPUT_DIR = "outputs"

COL_TIME  = "일시"
COL_POWER = "태양광 발전량(MWh)"
COL_IRRAD = "일사량"
WEATHER_COLS = ["기온", "강수량(mm)", "습도", "적설(cm)", "전운량(10분위)", "일조(hr)", "일사량"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 기본 전처리
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("[1] 데이터 로드 및 기본 전처리")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding="cp949")
df[COL_TIME] = pd.to_datetime(df[COL_TIME])
df = df.set_index(COL_TIME).sort_index()

print(f"로드 완료: {len(df):,}행  |  {df.index[0]} ~ {df.index[-1]}")

# ── 시간 간격 끊김 구간 탐지 ────────────────────────────────────────────────
print("\n[시간 간격 끊김 구간]")
time_diff = df.index.to_series().diff()
expected  = pd.Timedelta("1h")
gaps      = time_diff[time_diff > expected].dropna()

if gaps.empty:
    print("  끊김 없음")
else:
    for ts, delta in gaps.items():
        prev_ts = ts - delta
        print(f"  {prev_ts}  →  {ts}  (점프: {delta})")
print(f"  총 {len(gaps)}개 끊김 구간")

# ── 컬럼별 결측치 ───────────────────────────────────────────────────────────
print("\n[컬럼별 결측치]")
total = len(df)
for col in df.columns:
    n = df[col].isna().sum()
    pct = n / total * 100
    marker = " ★" if n > 0 else ""
    print(f"  {col:<22} {n:>6,}건  ({pct:.2f}%){marker}")

# ── 발전량 결측치 보간 ──────────────────────────────────────────────────────
before = df[COL_POWER].isna().sum()
df[COL_POWER] = df[COL_POWER].interpolate(method="time")
after  = df[COL_POWER].isna().sum()
print(f"\n발전량 time 보간 완료: {before:,}건 결측 → 잔여 {after:,}건")

# ── 낮 시간대(06~18시) 발전량 0 통계 ────────────────────────────────────────
daytime  = df[(df.index.hour >= 6) & (df.index.hour <= 18)]
zero_day = (daytime[COL_POWER] == 0).sum()
print(f"\n낮 시간대(06~18시) 발전량 0: {zero_day:,}건 "
      f"({zero_day / len(daytime) * 100:.2f}%)  ※ 제거하지 않음")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 전체 패턴 시각화 (4종 서브플롯)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[2] 전체 패턴 시각화 → outputs/eda_overview.png")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("태양광 발전량 EDA 개요 (2018~2024)", fontsize=15, fontweight="bold")

# ① 전체 발전량 시계열
ax = axes[0, 0]
df[COL_POWER].plot(ax=ax, lw=0.4, color="steelblue")
ax.set_title("전체 발전량 시계열")
ax.set_xlabel("")
ax.set_ylabel("발전량 (MWh)")

# ② 시간대별 평균 발전량
ax = axes[0, 1]
hourly_mean = df.groupby(df.index.hour)[COL_POWER].mean()
ax.bar(hourly_mean.index, hourly_mean.values, color="coral")
ax.set_title("시간대별 평균 발전량")
ax.set_xlabel("시 (hour)")
ax.set_ylabel("평균 발전량 (MWh)")
ax.set_xticks(range(0, 24))

# ③ 월별 평균 발전량
ax = axes[1, 0]
monthly_mean = df.groupby(df.index.month)[COL_POWER].mean()
ax.bar(monthly_mean.index, monthly_mean.values, color="mediumseagreen")
ax.set_title("월별 평균 발전량")
ax.set_xlabel("월")
ax.set_ylabel("평균 발전량 (MWh)")
ax.set_xticks(range(1, 13))

# ④ 일사량 vs 발전량 산점도 (낮 시간대만)
ax = axes[1, 1]
daytime_valid = daytime[[COL_IRRAD, COL_POWER]].dropna()
ax.scatter(
    daytime_valid[COL_IRRAD],
    daytime_valid[COL_POWER],
    s=2, alpha=0.3, color="darkorange",
)
ax.set_title("일사량 vs 발전량 (낮 06~18시)")
ax.set_xlabel("일사량 (MJ/㎡)")
ax.set_ylabel("발전량 (MWh)")

plt.tight_layout()
out1 = f"{OUTPUT_DIR}/eda_overview.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장 완료: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 시계열 분해 (statsmodels seasonal_decompose)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[3] 시계열 분해 (multiplicative, period=13)")
print("=" * 60)

# 낮 시간대 데이터만 추출
day_df = df[(df.index.hour >= 6) & (df.index.hour <= 18)].copy()

# multiplicative 분해 직전: 0값 → NaN → 보간 (0이면 분해 깨짐)
zero_mask = day_df[COL_POWER] == 0
day_df.loc[zero_mask, COL_POWER] = np.nan
day_df[COL_POWER] = day_df[COL_POWER].interpolate(method="time")
day_df[COL_POWER] = day_df[COL_POWER].ffill().bfill()

print(f"분해 대상: {len(day_df):,}행  (낮 시간대, 0→NaN→보간 처리)")

result = seasonal_decompose(
    day_df[COL_POWER],
    model="multiplicative",
    period=13,
    extrapolate_trend="freq",
)

fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
fig.suptitle(
    "태양광 발전량 시계열 분해 (Multiplicative, period=13)",
    fontsize=14, fontweight="bold",
)

components = [
    (day_df[COL_POWER], "Observed (발전량 MWh)", "steelblue"),
    (result.trend,      "Trend (추세)",           "darkorange"),
    (result.seasonal,   "Seasonal (계절성)",       "mediumseagreen"),
    (result.resid,      "Residual (잔차)",         "crimson"),
]
for ax, (series, title, color) in zip(axes, components):
    series.plot(ax=ax, lw=0.5, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("")

axes[-1].set_xlabel("날짜")
plt.tight_layout()
out2 = f"{OUTPUT_DIR}/decompose_result.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장 완료: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 상관관계 분석
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[4] 상관관계 분석 (낮 시간대 06~18시 기준)")
print("=" * 60)

corr_df      = daytime[[COL_POWER] + WEATHER_COLS].dropna()
correlations = corr_df.corr()[COL_POWER].drop(COL_POWER).sort_values(ascending=False)

print(f"\n{'변수':<22} {'상관계수':>10}")
print("-" * 34)
for col, val in correlations.items():
    print(f"  {col:<20} {val:>10.4f}")

irrad_corr = correlations.get(COL_IRRAD)
if irrad_corr is not None:
    flag = "[O] 0.8 이상" if irrad_corr >= 0.8 else "[X] 0.8 미만"
    print(f"\n일사량-발전량 상관계수: {irrad_corr:.4f}  [{flag}]")

print("\n모든 단계 완료.")
