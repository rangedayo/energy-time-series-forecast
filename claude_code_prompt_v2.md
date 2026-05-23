# Claude Code 실행 프롬프트 v2
# 태양광 발전량 예측 ML 프로젝트 — 전국 데이터 확장

---

## 🔰 프로젝트 컨텍스트

이 프로젝트는 **전국 17개 시도의 태양광 발전량 + 기상 데이터로 미래 발전량을 예측**하는 ML 시스템이다.
기존에 제주 단일 지역으로 완료된 파이프라인(TASK 0~7)을 전국 데이터로 확장한다.

---

## ✅ 기존에 완료된 것 (건드리지 말 것)

- 제주 단일 지역 기반 전처리/학습 파이프라인 전체 완료
- `data/processed/train_ready.csv`, `test_ready.csv` — 제주 데이터
- `data/processed/train_features.csv`, `test_features.csv` — 제주 피처
- `models/xgboost_model.json`, `lstm_model_state.pt` 등 — 제주 모델
- `outputs/` 하위 결과물 전체 — 제주 기준 결과

---

## 🚧 신규 작업 (전국 데이터 확장)

### 데이터 현황

**발전량 데이터 (data/raw/)**
```
170101_230228_지역별_시간별_태양광_발전량.csv   # 2017.01~2023.02, 전국 17개 시도
230601_230831_지역별_시간별_태양광_발전량.csv    # 2023.06~2023.08
230901_231130_지역별_시간대별_태양광_발전량.csv  # 2023.09~2023.11
```

**기상 데이터 (data/raw/)**
```
170101_171231_OBS_ASOS_TIM.csv   # 2017년, 16개 관측소
180101_181231_OBS_ASOS_TIM.csv   # 2018년
190101_191231_OBS_ASOS_TIM.csv   # 2019년
200101_201231_OBS_ASOS_TIM.csv   # 2020년
210101_211231_OBS_ASOS_TIM.csv   # 2021년
220101_221231_OBS_ASOS_TIM.csv   # 2022년
230101_231231_OBS_ASOS_TIM.csv   # 2023년
```

**ASOS 컬럼**: 지점, 지점명, 일시, 기온(°C), 강수량(mm), 습도(%), 일조(hr), 일사(MJ/m2), 전운량(10분위)

### 매핑 테이블 (고정값으로 사용)

**① 관측소명 → 시도명 매핑** (기상 데이터에만 적용 — 지점명을 발전량 데이터의 지역명과 일치시키는 작업)
```python
STATION_TO_REGION = {
    "춘천": "강원도",
    "수원": "경기도",
    "창원": "경상남도",
    "포항": "경상북도",
    "광주": "광주시",
    "대구": "대구시",
    "대전": "대전시",
    "부산": "부산시",
    "서울": "서울시",
    "세종": "세종시",   # 세종 관측소 (239) 존재 확인
    "울산": "울산시",
    "인천": "인천시",
    "목포": "전라남도",
    "전주": "전라북도",
    "제주": "제주도",
    "보령": "충청남도",
    "청주": "충청북도",
}
```

**② 시도명 → 숫자 코드 변환** (merge 완료 후 모델 입력용으로 통합 데이터에 한 번만 적용)
```python
# LabelEncoder로 시도명을 숫자로 변환
# 예: 강원도→0, 경기도→1, 경상남도→2, ...
# encoder를 models/national_region_encoder.pkl로 저장
# test에는 transform만 적용 (fit 금지)
```

---

## 📋 신규 TASK 목록

---

### TASK A — 전국 데이터 전처리 (`preprocess_national.py`)

**출력**: `data/processed/national_train_ready.csv`, `data/processed/national_test_ready.csv`

#### A-1. 발전량 데이터 전처리

```python
# 1. 3개 파일 로드 및 합치기
# 컬럼명이 파일마다 다르므로 통일 필요:
# ' 태양광 발전량(MWh) ', '태양광발전량(MWh)', '태양광발전량(Mwh)' → 'power_mwh'
# '거래일자' → 'date', '거래시간' → 'hour', '지역'/'지역명' → 'region'

# 2. 거래시간 변환: 1~24 → 0~23
# 24시 → 다음날 0시로 변환
df.loc[df['hour'] == 24, 'date'] = (
    pd.to_datetime(df.loc[df['hour'] == 24, 'date']) + pd.Timedelta(days=1)
)
df.loc[df['hour'] == 24, 'hour'] = 0

# 3. timestamp 컬럼 생성 후 정시로 통일 (초 단위 제거)
df['timestamp'] = (
    pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
).dt.floor('h')

# 4. 발전량 음수 제거
df = df[df['power_mwh'] >= 0]

# 5. 중복 행 제거 (timestamp + region 기준)
df = df.drop_duplicates(subset=['timestamp', 'region'])

# 6. 시간 순 정렬
df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
```

#### A-2. 기상 데이터 전처리

```python
# 1. 연도별 7개 파일 합치기
import glob
asos_files = sorted(glob.glob("data/raw/*_OBS_ASOS_TIM.csv"))
weather = pd.concat([pd.read_csv(f, encoding='cp949') for f in asos_files])

# 2. timestamp 파싱 후 정시로 통일
# ASOS 형식: "2023-01-01 1:00" (1자리 시간) → pandas가 자동 파싱
# dt.floor('h')로 초 단위 제거하여 발전량 데이터와 형식 통일
weather['timestamp'] = pd.to_datetime(weather['일시']).dt.floor('h')

# 3. 관측소명 → 시도명 변환 (STATION_TO_REGION 사용)
weather['region'] = weather['지점명'].map(STATION_TO_REGION)
weather = weather.dropna(subset=['region'])  # 매핑 안 된 행 제거

# 4. 컬럼명 통일
weather = weather.rename(columns={
    '기온(°C)': '기온',
    '강수량(mm)': '강수량',
    '습도(%)': '습도',
    '일조(hr)': '일조',
    '일사(MJ/m2)': '일사량',
    '전운량(10분위)': '전운량',
})

# 5. 세종시 특수 처리
# - 2017~2018년: 세종 관측소 미설치 → 해당 연도 세종 데이터 없음 (자연스럽게 제외)
# - 2019~2021, 2023년: 야간 일사량 결측 → 5단계에서 0으로 채움 (정상 처리)
# - 2022년: 일사량 센서 장애로 연간 전체 결측 → 인접한 대전 일사량으로 대체
daejeon_solar = weather[
    (weather['region'] == '대전시') &
    (weather['timestamp'].dt.year == 2022)
][['timestamp', '일사량']].rename(columns={'일사량': '일사량_대전'})

sejong_2022_mask = (
    (weather['region'] == '세종시') &
    (weather['timestamp'].dt.year == 2022)
)
weather = weather.merge(daejeon_solar, on='timestamp', how='left')
weather.loc[sejong_2022_mask, '일사량'] = weather.loc[sejong_2022_mask, '일사량_대전']
weather = weather.drop(columns=['일사량_대전'])
print(f"세종 2022년 일사량 → 대전 값으로 대체 완료: {sejong_2022_mask.sum()}건")

# 6. 야간(00~05시, 19~23시) 일조/일사 결측치 → 0으로 채우기
# 야간에는 물리적으로 일조/일사가 0이어야 함
night_mask = (weather['timestamp'].dt.hour <= 5) | (weather['timestamp'].dt.hour >= 19)
weather.loc[night_mask, '일조']   = weather.loc[night_mask, '일조'].fillna(0)
weather.loc[night_mask, '일사량'] = weather.loc[night_mask, '일사량'].fillna(0)

# 7. 겨울철(11~3월) 강수량 처리
# 문제: 3시간마다 한 번씩만 기록 → 나머지 시간은 NaN
# 예: 1시=NaN, 2시=NaN, 3시=3.0mm(합산값) → 시간당 1.0mm로 균등 분배
# 처리: 선형 보간으로 시간당 값으로 분배
winter_mask = weather['timestamp'].dt.month.isin([11, 12, 1, 2, 3])
weather.loc[winter_mask, '강수량'] = (
    weather.loc[winter_mask]
    .groupby('region')['강수량']
    .transform(lambda x: x.interpolate(method='linear'))
)
# 보간 후에도 남은 NaN(데이터 맨 앞/뒤) → 0으로 채우기
weather['강수량'] = weather.groupby('region')['강수량'].transform(
    lambda x: x.fillna(0)
)

# 8. 나머지 기상 변수 결측치 → 선형 보간 후 0 채우기
for col in ['기온', '습도', '전운량', '일조', '일사량']:
    weather[col] = weather.groupby('region')[col].transform(
        lambda x: x.interpolate(method='linear').fillna(0)
    )
```

#### A-3. 발전량 + 기상 데이터 결합

```python
# timestamp + region 기준으로 inner merge
# (두 데이터 모두 dt.floor('h') 적용했으므로 형식 일치 보장)
merged = pd.merge(power_df, weather_df, on=['timestamp', 'region'], how='inner')

# 이상치 제거 (기존 제주 전처리와 동일)
# 낮 시간(06~18시) 일사량=0 & 발전량>0 → 센서 오류로 판단하여 제거
daytime_mask = merged['timestamp'].dt.hour.between(6, 18)
anomaly_mask = daytime_mask & (merged['일사량'] == 0) & (merged['power_mwh'] > 0)
merged = merged[~anomaly_mask]
print(f"이상치 제거: {anomaly_mask.sum()}건")

# 결측치 보간 (지역별로 수행)
for col in ['power_mwh', '기온', '강수량', '습도', '일조', '일사량', '전운량']:
    merged[col] = merged.groupby('region')[col].transform(
        lambda x: x.interpolate(method='time').fillna(0)
    )

# ※ 발전효율 컬럼은 추가하지 않음
# 이유 1: 새 데이터에 설비용량 컬럼이 없음
# 이유 2: 발전효율 = 발전량/설비용량으로 타겟과 거의 동일한 정보 → 데이터 누수 유발

# 시도명 → 숫자 코드 변환 (모델 입력용)
from sklearn.preprocessing import LabelEncoder
import pickle
le = LabelEncoder()
merged['region_code'] = le.fit_transform(merged['region'])
with open("models/national_region_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"지역 코드 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# train/test 시간 순 분리
# train: 2017.01 ~ 2022.12 / test: 2023.01 ~ 2023.11
# (2024년 전국 데이터 없으므로 2023년을 test로 사용)
train = merged[merged['timestamp'] < '2023-01-01'].copy()
test  = merged[merged['timestamp'] >= '2023-01-01'].copy()
print(f"Train: {len(train):,}행  ({train['timestamp'].min()} ~ {train['timestamp'].max()})")
print(f"Test:  {len(test):,}행  ({test['timestamp'].min()} ~ {test['timestamp'].max()})")
```

#### A-4. 데이터 기댓값 테스트

```python
def validate_national(df, name):
    required = ['power_mwh', '일사량', '기온', '습도', '전운량', 'region', 'region_code']
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"
    assert df['power_mwh'].min() >= 0, f"[{name}] 발전량 음수 존재"
    assert df['일사량'].min() >= 0,    f"[{name}] 일사량 음수 존재"
    assert df.isnull().sum().sum() == 0, f"[{name}] NaN {df.isnull().sum().sum()}건 존재"
    assert len(df) > 10000, f"[{name}] 데이터 부족: {len(df)}행"
    assert df['region'].nunique() == 17, \
        f"[{name}] 지역 수 이상: {df['region'].nunique()}개"
    dup = df.duplicated(subset=['timestamp', 'region']).sum()
    assert dup == 0, f"[{name}] 중복 행 {dup}건 존재"
    print(f"  [{name}] 기댓값 테스트 통과 ✓  ({len(df):,}행, {df['region'].nunique()}개 지역)")

validate_national(train, "national_train")
validate_national(test,  "national_test")
```

저장:
- `data/processed/national_train_ready.csv`
- `data/processed/national_test_ready.csv`
- `models/national_region_encoder.pkl`

---

### TASK B — 전국 피처 엔지니어링 (`src/features/feature_engineering_national.py`)

**입력**: `data/processed/national_train_ready.csv`, `data/processed/national_test_ready.csv`
**출력**: `data/processed/national_train_features.csv`, `data/processed/national_test_features.csv`

기존 제주 피처와 동일하게 생성하되 두 가지가 다르다.
**① region_code 피처 추가** — 지역별 발전 패턴 학습에 필수
**② 래그/롤링 피처는 반드시 region별 groupby로 계산** — 지역 경계 누수 방지

```python
# 래그 피처: region별로 따로 계산
for col_name, shift_n in [('lag_1h',1),('lag_2h',2),('lag_3h',3),('lag_24h',24)]:
    df[col_name] = df.groupby('region')['power_mwh'].shift(shift_n)

# 롤링 통계: region별로 따로 계산
df['rolling_mean_3h'] = df.groupby('region')['power_mwh'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['rolling_mean_6h'] = df.groupby('region')['power_mwh'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)
df['rolling_std_3h'] = df.groupby('region')['power_mwh'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0)
)

# 나머지 피처 (region 무관)
df['hour']                 = df['timestamp'].dt.hour
df['month']                = df['timestamp'].dt.month
df['day_of_week']          = df['timestamp'].dt.dayofweek
df['is_weekend']           = (df['day_of_week'] >= 5).astype(int)
df['season']               = df['month'].map({
    12:1,1:1,2:1, 3:2,4:2,5:2, 6:3,7:3,8:3, 9:4,10:4,11:4
})
df['solar_altitude_proxy'] = np.sin(np.pi * (df['hour'] - 6) / 12).clip(0)
df['irrad_x_solar']        = df['일사량'] * df['solar_altitude_proxy']
df['is_daytime']           = df['hour'].between(6, 18).astype(int)
# region_code는 TASK A에서 이미 추가됨

# NaN 제거 (래그 피처로 인한 초반 행)
df = df.dropna()
```

**타겟 컬럼**: `power_mwh`
피처 목록을 `src/features/feature_list_national.json`에 저장.

데이터 기댓값 테스트:
```python
def validate_features_national(df, name):
    required = ['power_mwh', '일사량', 'lag_1h', 'is_daytime', 'hour', 'region_code']
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"
    assert df['power_mwh'].min() >= 0, f"[{name}] 발전량 음수 존재"
    assert df['일사량'].min() >= 0,    f"[{name}] 일사량 음수 존재"
    assert df.isnull().sum().sum() == 0, f"[{name}] NaN 존재"
    assert len(df) > 10000, f"[{name}] 데이터 부족: {len(df)}행"
    print(f"  [{name}] 피처 기댓값 테스트 통과 ✓  ({len(df):,}행)")

validate_features_national(train_out, "national_train_features")
validate_features_national(test_out,  "national_test_features")
```

---

### TASK C — Naive Baseline (전국) (`src/models/baseline_naive_national.py`)

기존 제주 baseline과 동일한 방식. `lag1`, `rolling24` 두 전략.
평가: 전체 + 피크 시간대(10~14시) + **지역별 MAE/RMSE**
결과: `outputs/national_baseline_results.json`

---

### TASK D — XGBoost (전국) (`src/models/train_xgboost_national.py`)

기존 제주 XGBoost와 동일한 구조. `region_code`가 피처로 추가된 것이 차이점.

#### D-1. 암기 테스트
각 지역에서 30행씩 추출(총 17×30=510행)하여 과적합 확인.
MAE가 national_baseline lag1 MAE의 30% 이하면 통과.

#### D-2. 본 학습
```python
params = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "random_state": 42, "n_jobs": -1,
}
```
early_stopping_rounds=30, validation = train 마지막 20% (시간 순)

#### D-3. 평가 및 저장
- 전체 MAE/RMSE + 피크 MAE/RMSE + 지역별 MAE (오차 큰 상위 5개 표시)
- naive 대비 개선율
- 모델: `models/national_xgboost_model.json`
- feature importance: `outputs/national_xgb_feature_importance.png`
- 예측값: `outputs/national_xgb_predictions.csv` (timestamp, region, actual, predicted)
- 결과: `outputs/national_xgb_results.json`
- W&B: WANDB_MODE=offline, project="solar-power-forecast", name="xgboost-national"

---

### TASK E — 행동 테스트 (전국) (`src/tests/behavioral_tests_national.py`)

#### 테스트 1 — NaN/Inf 검증
전국 test set 예측값에 NaN/Inf 없는지 확인.

#### 테스트 2 — 방향성 테스트
각 지역에서 샘플 10개씩 추출(총 170개).
일사량 +0.5 MJ/m² 증가 시 예측값 증가 비율 90% 이상이면 통과.

#### 테스트 3 — 불변성 테스트
동일 입력 5회 반복 호출 시 표준편차 0 확인.

#### 테스트 4 — 정확성 테스트
피크 시간대 MAE < national_baseline lag1 MAE_peak이면 통과.

#### 테스트 5 — 지역 불변성 테스트 (신규)
동일한 기상 피처에서 region_code만 다르게 바꿨을 때 예측값이 달라지는지 확인.
모든 지역 코드에 대해 예측값이 다르면 통과.

결과: `outputs/national_behavioral_test_results.json`

---

### TASK F — LSTM (전국) (`src/models/train_lstm_national.py`)

**코랩 엔터프라이즈(GPU)에서 실행할 파일이다.**
경로 상수를 스크립트 상단에 집중 정의하여 코랩 환경에서 쉽게 수정 가능하도록 작성하라.

```python
# ── 경로 상수 (코랩 실행 시 이 부분만 수정) ──────────────
TRAIN_FEAT   = "data/processed/national_train_features.csv"
TEST_FEAT    = "data/processed/national_test_features.csv"
SCALER_OUT   = "models/national_lstm_scaler.pkl"
MODEL_STATE  = "models/national_lstm_model_state.pt"
MODEL_SCRIPT = "models/national_lstm_model_scripted.pt"
MODEL_ONNX   = "models/national_lstm_model.onnx"
RESULT_OUT   = "outputs/national_lstm_results.json"
PRED_OUT     = "outputs/national_lstm_predictions.csv"
LOSS_PNG     = "outputs/national_lstm_loss_curve.png"
VERIFY_OUT   = "outputs/national_model_save_verify_results.json"
# ─────────────────────────────────────────────────────────
```

#### F-1. 데이터셋 구성
- seq_len = 24
- **region별로 시퀀스 생성** — 지역 경계에서 시퀀스 끊기
```python
# 올바른 방법: region별 분리 후 각각 시퀀스 생성
sequences = []
for region in df['region'].unique():
    region_df = df[df['region'] == region].sort_values('timestamp')
    for i in range(len(region_df) - seq_len):
        sequences.append(region_df.iloc[i:i+seq_len+1])
```
- StandardScaler: train fit → train/test transform
- scaler 저장: `models/national_lstm_scaler.pkl`

#### F-2. 모델 아키텍처
```python
class NationalSolarLSTM(nn.Module):
    # hidden_size=128, num_layers=2, dropout=0.2
    # 출력층: Linear(128 → 1)
```

#### F-3. 학습 설정
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{ts()}] 사용 디바이스: {device}")
# Adam lr=0.001 / MSELoss / epochs=50 / patience=10 / batch_size=128
# validation = train 마지막 20% (시간 순)
```

#### F-4. 암기 테스트
각 지역에서 2개 배치(256행) 추출, 100 epoch 학습 (dropout=0).
초기 loss의 1% 이하 달성 시 통과.

#### F-5. 평가 및 모델 출력 검증
```python
pred_tensor = torch.tensor(pred)
assert torch.isfinite(pred_tensor).all(), "LSTM 출력 NaN/Inf 감지"

# 야간 클리핑
pred = np.where((hours <= 5) | (hours >= 19), 0, np.clip(pred, 0, None))
```
- 전체 + 피크 MAE/RMSE + 지역별 MAE
- XGBoost 대비 비교
- loss 곡선, 예측값, 결과 저장

#### F-6. 모델 저장 및 검증
```python
model = model.cpu().eval()
sample_input = X_test_tensor[:10].cpu()
with torch.no_grad():
    pred_before = model(sample_input).numpy()

# state_dict / TorchScript / ONNX(opset_version=14) 저장
# 저장 후 각각 불러와서 np.allclose(pred_before, pred_x, atol=1e-5) 검증
# 배치 추론 테스트: batch_size in [1, 8, 64]
```
검증 결과: `outputs/national_model_save_verify_results.json`

---

### TASK G — ESS 시뮬레이션 (전국) (`src/simulation/ess_simulation_national.py`)

```python
ESS_CAPACITY_MWH    = 500.0
SOC_MIN, SOC_MAX    = 0.20, 0.80
CHARGE_RATE_MAX     = 100.0
DISCHARGE_RATE_MAX  = 100.0
EFFICIENCY          = 0.95
DEMAND_MWH_PER_HOUR = 50.0
```

전략 3가지 (naive / xgb / lstm), 지역별 운영효율점수도 출력.
결과: `outputs/national_ess_simulation_results.json`, `outputs/national_ess_simulation_comparison.png`

---

### TASK H — 최종 리포트 (전국) (`src/reporting/final_report_national.py`)

모든 `outputs/national_*.json`을 읽어 아래 형식으로 출력 후 `outputs/national_final_report.md` 저장.

```
=== [전국] 모델 성능 비교 (Test Set 2023) ===
| 모델      | MAE | RMSE | 피크 MAE | Naive 대비 개선율 |

=== [전국] 지역별 MAE (XGBoost 기준) ===
| 지역 | MAE | Naive 대비 개선율 |

=== [전국] ESS 시뮬레이션 비교 ===
| 전략 | 전력낭비율 | 부족횟수 | 사이클수 | 운영효율점수 |

=== [전국] 테스트 결과 ===
[데이터 기댓값 테스트] national_train / national_test: PASS/FAIL
[XGBoost 행동 테스트] NaN/Inf / 방향성 / 불변성 / 정확성 / 지역불변성: PASS/FAIL
[LSTM 테스트] 암기 / NaN/Inf / state_dict / TorchScript / ONNX / 배치(1/8/64): PASS/FAIL
```

---

## ⚠️ 전체 공통 규칙

1. **시간 순 분리**: train ~2022년 / test 2023년. random split 절대 금지.
2. **지역별 래그/롤링**: region 기준 groupby 후 계산. 지역 경계 누수 방지.
3. **데이터 누수 금지**: scaler, LabelEncoder는 train 기준으로만 fit.
4. **발전효율 컬럼 추가 금지**: 설비용량 데이터 없고 데이터 누수 유발.
5. **야간 클리핑**: 00~05시, 19~23시 예측값 → 0으로 클리핑.
6. **timestamp 형식 통일**: 발전량/기상 데이터 모두 `dt.floor('h')` 적용 후 merge.
7. **재현성**: random_state=42 고정.
8. **경로 상수**: 스크립트 상단에 집중 정의.
9. **로그**: 각 단계 타임스탬프 출력.
10. **에러 처리**: 파일/컬럼 없으면 명확한 메시지와 함께 종료.
11. **W&B**: WANDB_MODE=offline.
12. **LSTM 저장**: .cpu().eval() 상태에서 저장.

---

## 🚀 실행 순서

```bash
.venv/bin/python preprocess_national.py
.venv/bin/python src/features/feature_engineering_national.py
.venv/bin/python src/models/baseline_naive_national.py
.venv/bin/python src/models/train_xgboost_national.py
.venv/bin/python src/tests/behavioral_tests_national.py
# TASK F: 코랩에서 실행 (경로 상수 수정 후)
# 코랩 설치: !pip install wandb onnx onnxruntime onnxscript
.venv/bin/python src/simulation/ess_simulation_national.py
.venv/bin/python src/reporting/final_report_national.py
```

---

## 📁 완성 후 추가될 파일 구조

```
data/processed/
├── national_train_ready.csv      ← TASK A
├── national_test_ready.csv       ← TASK A
├── national_train_features.csv   ← TASK B
└── national_test_features.csv    ← TASK B

models/
├── national_region_encoder.pkl   ← TASK A
├── national_lstm_scaler.pkl      ← TASK F
├── national_xgboost_model.json   ← TASK D
├── national_lstm_model_state.pt  ← TASK F
├── national_lstm_model_scripted.pt ← TASK F
└── national_lstm_model.onnx      ← TASK F

outputs/
├── national_baseline_results.json
├── national_xgb_results.json
├── national_xgb_predictions.csv
├── national_xgb_feature_importance.png
├── national_behavioral_test_results.json
├── national_lstm_results.json
├── national_lstm_predictions.csv
├── national_lstm_loss_curve.png
├── national_model_save_verify_results.json
├── national_ess_simulation_results.json
├── national_ess_simulation_comparison.png
└── national_final_report.md

preprocess_national.py            ← TASK A
src/
├── features/feature_engineering_national.py  ← TASK B
├── models/
│   ├── baseline_naive_national.py            ← TASK C
│   ├── train_xgboost_national.py             ← TASK D
│   └── train_lstm_national.py                ← TASK F
├── tests/behavioral_tests_national.py        ← TASK E
├── simulation/ess_simulation_national.py     ← TASK G
└── reporting/final_report_national.py        ← TASK H
```
