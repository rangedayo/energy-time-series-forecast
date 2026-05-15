import sys
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
TRAIN_IN   = "data/processed/national_train_ready.csv"
TEST_IN    = "data/processed/national_test_ready.csv"
TRAIN_OUT  = "data/processed/national_train_features.csv"
TEST_OUT   = "data/processed/national_test_features.csv"
FEAT_LIST  = "src/features/feature_list_national.json"

COL_POWER = "power_mwh"
COL_IRRAD = "일사량"

print(f"[{ts()}] [TASK B] 전국 피처 엔지니어링 시작")

for path in [TRAIN_IN, TEST_IN]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: 파일 없음 → {path}")

train = pd.read_csv(TRAIN_IN, encoding="utf-8-sig", parse_dates=["timestamp"])
test  = pd.read_csv(TEST_IN,  encoding="utf-8-sig", parse_dates=["timestamp"])

print(f"[{ts()}]   Train: {len(train):,}행  {train['timestamp'].min().date()} ~ {train['timestamp'].max().date()}")
print(f"[{ts()}]   Test:  {len(test):,}행  {test['timestamp'].min().date()} ~ {test['timestamp'].max().date()}")

# ── 래그/롤링 누수 방지: train+test concat 후 region별 계산 ────────────────────
train["_split"] = "train"
test["_split"]  = "test"
df = pd.concat([train, test], ignore_index=True)
df = df.sort_values(["region", "timestamp"]).reset_index(drop=True)
print(f"[{ts()}]   concat 후: {len(df):,}행")

# ── 1. 시간 피처 ──────────────────────────────────────────────────────────────
df["hour"]        = df["timestamp"].dt.hour
df["month"]       = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["season"]      = df["month"].map({
    12: 1, 1: 1, 2: 1,
    3: 2,  4: 2, 5: 2,
    6: 3,  7: 3, 8: 3,
    9: 4, 10: 4, 11: 4,
})

# ── 2. 태양 위치 피처 ─────────────────────────────────────────────────────────
df["solar_altitude_proxy"] = np.sin(np.pi * (df["hour"] - 6) / 12).clip(0)

# ── 3. 래그 피처 (region별) ───────────────────────────────────────────────────
for col_name, shift_n in [("lag_1h", 1), ("lag_2h", 2), ("lag_3h", 3), ("lag_24h", 24)]:
    df[col_name] = df.groupby("region")[COL_POWER].shift(shift_n)

# ── 3-1. 변화량 피처 (region별) ───────────────────────────────────────────────
df["power_diff_1h"] = df.groupby("region")[COL_POWER].diff(1)
df["power_diff_2h"] = df.groupby("region")[COL_POWER].diff(2)

# ── 4. 롤링 통계 (region별) ───────────────────────────────────────────────────
df["rolling_mean_3h"] = df.groupby("region")[COL_POWER].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df["rolling_mean_6h"] = df.groupby("region")[COL_POWER].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)
df["rolling_std_3h"] = df.groupby("region")[COL_POWER].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0)
)

# ── 5. 기상 교호작용 ──────────────────────────────────────────────────────────
df["irrad_x_solar"] = df[COL_IRRAD] * df["solar_altitude_proxy"]

# ── 6. 야간 마스크 ────────────────────────────────────────────────────────────
df["is_daytime"] = df["hour"].between(6, 18).astype(int)

# ── 분리 및 NaN 제거 ──────────────────────────────────────────────────────────
train_out = df[df["_split"] == "train"].drop(columns=["_split"])
test_out  = df[df["_split"] == "test" ].drop(columns=["_split"])

before_tr = len(train_out)
before_te = len(test_out)
train_out = train_out.dropna().reset_index(drop=True)
test_out  = test_out.dropna().reset_index(drop=True)

print(f"[{ts()}]   dropna Train: {before_tr:,} → {len(train_out):,}행  (제거: {before_tr - len(train_out)})")
print(f"[{ts()}]   dropna Test : {before_te:,} → {len(test_out):,}행  (제거: {before_te - len(test_out)})")

# ── 데이터 기댓값 테스트 ──────────────────────────────────────────────────────
def validate_features_national(df, name):
    required = [COL_POWER, COL_IRRAD, "lag_1h", "is_daytime", "hour", "region_code"]
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"
    assert df[COL_POWER].min() >= 0, f"[{name}] 발전량 음수 존재"
    assert df[COL_IRRAD].min() >= 0, f"[{name}] 일사량 음수 존재"
    nan_total = df.isnull().sum().sum()
    assert nan_total == 0, f"[{name}] NaN {nan_total}건 존재"
    assert len(df) > 10000, f"[{name}] 데이터 부족: {len(df)}행"
    print(f"  [{name}] 피처 기댓값 테스트 통과 ✓  ({len(df):,}행)")

print(f"\n[{ts()}] [데이터 기댓값 테스트]")
validate_features_national(train_out, "national_train_features")
validate_features_national(test_out,  "national_test_features")

# ── 저장 ─────────────────────────────────────────────────────────────────────
train_out.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
test_out.to_csv(TEST_OUT,   index=False, encoding="utf-8-sig")
print(f"\n[{ts()}]   저장: {TRAIN_OUT}")
print(f"[{ts()}]   저장: {TEST_OUT}")

# 피처 목록 저장
engineered = [
    "hour", "month", "day_of_week", "is_weekend", "season",
    "solar_altitude_proxy", "irrad_x_solar", "is_daytime",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h",
    "power_diff_1h", "power_diff_2h",
    "rolling_mean_3h", "rolling_mean_6h", "rolling_std_3h",
    "region_code",
]
feature_list = {
    "target": COL_POWER,
    "engineered_features": engineered,
    "all_columns": train_out.columns.tolist(),
}
with open(FEAT_LIST, "w", encoding="utf-8") as f:
    json.dump(feature_list, f, ensure_ascii=False, indent=2)
print(f"[{ts()}]   피처 목록 저장: {FEAT_LIST}")

print(f"\n[{ts()}] [검증]")
print(f"  Train shape: {train_out.shape}")
print(f"  Test  shape: {test_out.shape}")
print(f"  피처 목록  : {engineered}")
print(f"\n[{ts()}] [TASK B] 전국 피처 엔지니어링 완료")
