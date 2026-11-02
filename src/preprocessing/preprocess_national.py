import sys
import os
import glob
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
POWER_FILES = [
    "data/raw/170101_230228_지역별_시간별_태양광_발전량.csv",
    "data/raw/230601_230831_지역별_시간별 _태양광_발전량.csv",
    "data/raw/230901_231130_지역별_시간대별_태양광_발전량.csv",
]
ASOS_PATTERN    = "data/raw/*_OBS_ASOS_TIM.csv"
ASOS_EXTRA      = ["data/raw/OBS_ASOS_TIM_서산_170101_231231.csv"]
TRAIN_OUT       = "data/processed/national_train_ready.csv"
TEST_OUT        = "data/processed/national_test_ready.csv"
ENCODER_OUT     = "models/national_region_encoder.pkl"

STATION_TO_REGION = {
    "춘천": "강원도",  "수원": "경기도",  "창원": "경상남도", "포항": "경상북도",
    "광주": "광주시",  "대구": "대구시",  "대전": "대전시",   "부산": "부산시",
    "서울": "서울시",  "세종": "세종시",  "울산": "울산시",   "인천": "인천시",
    "목포": "전라남도","전주": "전라북도","제주": "제주도",   "서산": "충청남도",
    "청주": "충청북도",
}

print(f"[{ts()}] [TASK A] 전국 데이터 전처리 시작")

# ══════════════════════════════════════════════════════════════════════════════
# A-1. 발전량 데이터 전처리
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] [A-1] 발전량 데이터 로드 및 전처리")

power_frames = []
for path in POWER_FILES:
    if not os.path.exists(path):
        sys.exit(f"ERROR: 파일 없음 → {path}")
    df = pd.read_csv(path, encoding="cp949")
    df.columns = df.columns.str.strip()

    # 발전량 컬럼 통일
    power_col = next(
        (c for c in df.columns if "태양광" in c and ("발전량" in c or "Mwh" in c or "MWh" in c)),
        None,
    )
    if power_col is None:
        sys.exit(f"ERROR: 발전량 컬럼 없음 → {path}\n컬럼: {df.columns.tolist()}")
    df = df.rename(columns={power_col: "power_mwh"})

    # 지역 컬럼 통일
    region_col = next((c for c in df.columns if c in ("지역", "지역명")), None)
    if region_col is None:
        sys.exit(f"ERROR: 지역 컬럼 없음 → {path}")
    df = df.rename(columns={region_col: "region"})

    # 날짜/시간 컬럼 통일
    df = df.rename(columns={"거래일자": "date", "거래시간": "hour"})

    # 발전량 숫자 변환 (빈 문자열 → NaN)
    df["power_mwh"] = pd.to_numeric(df["power_mwh"], errors="coerce")

    power_frames.append(df[["date", "hour", "region", "power_mwh"]])
    print(f"[{ts()}]   로드: {os.path.basename(path)} → {len(df):,}행")

power_df = pd.concat(power_frames, ignore_index=True)
print(f"[{ts()}]   발전량 합계: {len(power_df):,}행")

# 거래시간 1~24 → 0~23 변환 (24시 → 다음날 0시)
mask_24 = power_df["hour"] == 24
power_df.loc[mask_24, "date"] = (
    pd.to_datetime(power_df.loc[mask_24, "date"]) + pd.Timedelta(days=1)
).dt.strftime("%Y-%m-%d")
power_df.loc[mask_24, "hour"] = 0
print(f"[{ts()}]   24시 → 익일 0시 변환: {mask_24.sum()}건")

# timestamp 생성
power_df["timestamp"] = (
    pd.to_datetime(power_df["date"]) + pd.to_timedelta(power_df["hour"], unit="h")
).dt.floor("h")

# 발전량 음수 제거
before = len(power_df)
power_df = power_df[power_df["power_mwh"].fillna(0) >= 0]
print(f"[{ts()}]   음수 제거: {before - len(power_df)}건")

# NaN 발전량 → 0 처리
power_df["power_mwh"] = power_df["power_mwh"].fillna(0)

# 중복 제거
power_df = power_df.drop_duplicates(subset=["timestamp", "region"])
power_df = power_df.sort_values(["region", "timestamp"]).reset_index(drop=True)
power_df = power_df[["timestamp", "region", "power_mwh"]]
print(f"[{ts()}]   최종 발전량: {len(power_df):,}행  {power_df['region'].nunique()}개 지역")

# ══════════════════════════════════════════════════════════════════════════════
# A-2. 기상 데이터 전처리
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] [A-2] 기상 데이터 로드 및 전처리")

asos_files = sorted(glob.glob(ASOS_PATTERN))
if not asos_files:
    sys.exit(f"ERROR: ASOS 파일 없음 → {ASOS_PATTERN}")

weather_frames = []
for path in asos_files + ASOS_EXTRA:
    df = pd.read_csv(path, encoding="cp949")
    weather_frames.append(df)
    print(f"[{ts()}]   로드: {os.path.basename(path)} → {len(df):,}행")

weather = pd.concat(weather_frames, ignore_index=True)
print(f"[{ts()}]   기상 합계: {len(weather):,}행  지점: {weather['지점명'].nunique()}개")

# timestamp 파싱
weather["timestamp"] = pd.to_datetime(weather["일시"]).dt.floor("h")

# 관측소명 → 시도명 변환
weather["region"] = weather["지점명"].map(STATION_TO_REGION)
before_region = len(weather)
weather = weather.dropna(subset=["region"])
print(f"[{ts()}]   매핑 제외: {before_region - len(weather)}행")

# 컬럼명 통일
weather = weather.rename(columns={
    "기온(°C)":    "기온",
    "강수량(mm)":  "강수량",
    "습도(%)":     "습도",
    "일조(hr)":    "일조",
    "일사(MJ/m2)": "일사량",
    "전운량(10분위)": "전운량",
})

weather = weather[["timestamp", "region", "기온", "강수량", "습도", "일조", "일사량", "전운량"]]

# 세종 2022 일사량 → 대전 값으로 대체
daejeon_2022 = weather[
    (weather["region"] == "대전시") & (weather["timestamp"].dt.year == 2022)
][["timestamp", "일사량"]].rename(columns={"일사량": "_daejeon_irrad"})

sejong_2022_mask = (
    (weather["region"] == "세종시") & (weather["timestamp"].dt.year == 2022)
)
if sejong_2022_mask.sum() > 0:
    weather = weather.merge(daejeon_2022, on="timestamp", how="left")
    sejong_2022_mask = (weather["region"] == "세종시") & (weather["timestamp"].dt.year == 2022)
    weather.loc[sejong_2022_mask, "일사량"] = weather.loc[sejong_2022_mask, "_daejeon_irrad"]
    weather = weather.drop(columns=["_daejeon_irrad"])
    print(f"[{ts()}]   세종 2022년 일사량 → 대전 대체: {sejong_2022_mask.sum()}건")
else:
    print(f"[{ts()}]   세종 2022년 데이터 없음 (ASOS 미설치) — 스킵")

# 야간 일조/일사 결측치 → 0
night_mask = (weather["timestamp"].dt.hour <= 5) | (weather["timestamp"].dt.hour >= 19)
weather.loc[night_mask, "일조"]   = weather.loc[night_mask, "일조"].fillna(0)
weather.loc[night_mask, "일사량"] = weather.loc[night_mask, "일사량"].fillna(0)

# 겨울철 강수량 선형 보간
winter_mask = weather["timestamp"].dt.month.isin([11, 12, 1, 2, 3])
weather.loc[winter_mask, "강수량"] = (
    weather.loc[winter_mask]
    .groupby("region")["강수량"]
    .transform(lambda x: x.interpolate(method="linear"))
)
weather["강수량"] = weather.groupby("region")["강수량"].transform(
    lambda x: x.fillna(0)
)

# 나머지 기상 변수 선형 보간
for col in ["기온", "습도", "전운량", "일조", "일사량"]:
    weather[col] = weather.groupby("region")[col].transform(
        lambda x: x.interpolate(method="linear").fillna(0)
    )

weather = weather.sort_values(["region", "timestamp"]).reset_index(drop=True)
print(f"[{ts()}]   최종 기상 데이터: {len(weather):,}행  {weather['region'].nunique()}개 지역")

# ══════════════════════════════════════════════════════════════════════════════
# A-3. 발전량 + 기상 데이터 결합
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] [A-3] 데이터 결합")

merged = pd.merge(power_df, weather, on=["timestamp", "region"], how="inner")
print(f"[{ts()}]   inner merge 후: {len(merged):,}행  {merged['region'].nunique()}개 지역")

# 이상치 제거 (낮 시간 일사량=0 & 발전량>0)
daytime_mask = merged["timestamp"].dt.hour.between(6, 18)
anomaly_mask = daytime_mask & (merged["일사량"] == 0) & (merged["power_mwh"] > 0)
merged = merged[~anomaly_mask].reset_index(drop=True)
print(f"[{ts()}]   이상치 제거: {anomaly_mask.sum()}건")

# 결측치 보간 (지역별)
for col in ["power_mwh", "기온", "강수량", "습도", "일조", "일사량", "전운량"]:
    merged[col] = merged.groupby("region")[col].transform(
        lambda x: x.interpolate(method="linear").fillna(0)
    )

# 시도명 → 숫자 코드 변환 (train 기준 fit)
le = LabelEncoder()
merged["region_code"] = le.fit_transform(merged["region"])
os.makedirs("models", exist_ok=True)
with open(ENCODER_OUT, "wb") as f:
    pickle.dump(le, f)
print(f"[{ts()}]   지역 코드 매핑: {dict(zip(le.classes_, le.transform(le.classes_).tolist()))}")

# train/test 분리 (시간 순)
merged = merged.sort_values(["region", "timestamp"]).reset_index(drop=True)
train = merged[merged["timestamp"] < "2023-01-01"].copy().reset_index(drop=True)
test  = merged[merged["timestamp"] >= "2023-01-01"].copy().reset_index(drop=True)

print(f"[{ts()}]   Train: {len(train):,}행  {train['timestamp'].min()} ~ {train['timestamp'].max()}")
print(f"[{ts()}]   Test:  {len(test):,}행  {test['timestamp'].min()} ~ {test['timestamp'].max()}")

# ══════════════════════════════════════════════════════════════════════════════
# A-4. 데이터 기댓값 테스트
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] [A-4] 데이터 기댓값 테스트")

def validate_national(df, name):
    required = ["power_mwh", "일사량", "기온", "습도", "전운량", "region", "region_code"]
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"
    assert df["power_mwh"].min() >= 0, f"[{name}] 발전량 음수 존재"
    assert df["일사량"].min() >= 0,    f"[{name}] 일사량 음수 존재"
    nan_total = df.isnull().sum().sum()
    assert nan_total == 0,             f"[{name}] NaN {nan_total}건 존재"
    assert len(df) > 10000,            f"[{name}] 데이터 부족: {len(df)}행"
    assert df["region"].nunique() >= 15, \
        f"[{name}] 지역 수 이상: {df['region'].nunique()}개"
    dup = df.duplicated(subset=["timestamp", "region"]).sum()
    assert dup == 0, f"[{name}] 중복 행 {dup}건 존재"
    print(f"  [{name}] 기댓값 테스트 통과 ✓  ({len(df):,}행, {df['region'].nunique()}개 지역)")

validate_national(train, "national_train")
validate_national(test,  "national_test")

# 저장
os.makedirs("data/processed", exist_ok=True)
train.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
test.to_csv(TEST_OUT,   index=False, encoding="utf-8-sig")
print(f"\n[{ts()}]   저장: {TRAIN_OUT}")
print(f"[{ts()}]   저장: {TEST_OUT}")
print(f"[{ts()}]   저장: {ENCODER_OUT}")
print(f"\n[{ts()}] [TASK A] 전국 데이터 전처리 완료")
