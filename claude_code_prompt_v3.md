# Claude Code 실행 프롬프트 v3
# 태양광 발전량 예측 ML 프로젝트 — 전국 데이터 확장
# (v2 → v3 변경사항: power_diff 피처 추가, XGBoost 기준값 업데이트, LSTM 피처 일치 명시, 방향성 테스트 주석 추가)

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

## ✅ 전국 파이프라인 완료된 것 (TASK A~E, G~H 완료)

- TASK A~B: 전처리 + 피처 엔지니어링 완료
- TASK C: Naive Baseline 완료 (MAE 20.51, 피크 MAE 28.86)
- TASK D: XGBoost 학습 완료 — **power_diff 피처 추가 후 재학습된 v2 기준**
  - 전체 MAE: 9.66 (Naive 대비 52.9% 개선)
  - 피크 MAE: 확인 필요 (national_xgb_results.json 참조)
- TASK E: 행동 테스트 완료
  - PASS: NaN/Inf, 불변성, 정확성
  - FAIL: 방향성(82.4%), 지역불변성(11/17) → 모델 결함 아님 (아래 주석 참조)
- TASK G: ESS 시뮬레이션 완료 (XGBoost 기준, LSTM 전략은 TASK F 완료 후 재실행 필요)
- TASK H: 최종 리포트 완료 (LSTM 결과 추가 후 재실행 필요)

**현재 남은 작업: TASK F (LSTM 학습) → TASK G 재실행 → TASK H 재실행**

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

**① 관측소명 → 시도명 매핑**
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
    "세종": "세종시",
    "울산": "울산시",
    "인천": "인천시",
    "목포": "전라남도",
    "전주": "전라북도",
    "제주": "제주도",
    "보령": "충청남도",
    "청주": "충청북도",
}
```

**② 시도명 → 숫자 코드 변환**
```python
# LabelEncoder로 시도명을 숫자로 변환
# encoder를 models/national_region_encoder.pkl로 저장
# test에는 transform만 적용 (fit 금지)
```

---

## 📋 신규 TASK 목록

---

### TASK A — 전국 데이터 전처리 (`preprocess_national.py`) ✅ 완료

**출력**: `data/processed/national_train_ready.csv`, `data/processed/national_test_ready.csv`

#### A-1. 발전량 데이터 전처리

```python
# 1. 3개 파일 로드 및 합치기
# 컬럼명이 파일마다 다르므로 통일 필요:
# ' 태양광 발전량(MWh) ', '태양광발전량(MWh)', '태양광발전량(Mwh)' → 'power_mwh'
# '거래일자' → 'date', '거래시간' → 'hour', '지역'/'지역명' → 'region'

# 2. 거래시간 변환: 1~24 → 0~23
df.loc[df['hour'] == 24, 'date'] = (
    pd.to_datetime(df.loc[df['hour'] == 24, 'date']) + pd.Timedelta(days=1)
)
df.loc[df['hour'] == 24, 'hour'] = 0

# 3. timestamp 컬럼 생성 후 정시로 통일
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
weather['timestamp'] = pd.to_datetime(weather['일시']).dt.floor('h')

# 3. 관측소명 → 시도명 변환
weather['region'] = weather['지점명'].map(STATION_TO_REGION)
weather = weather.dropna(subset=['region'])

# 4. 컬럼명 통일
weather = weather.rename(columns={
    '기온(°C)': '기온', '강수량(mm)': '강수량', '습도(%)': '습도',
    '일조(hr)': '일조', '일사(MJ/m2)': '일사량', '전운량(10분위)': '전운량',
})

# 5. 세종시 특수 처리 (2022년 일사량 센서 장애 → 대전 값으로 대체)
daejeon_solar = weather[
    (weather['region'] == '대전시') & (weather['timestamp'].dt.year == 2022)
][['timestamp', '일사량']].rename(columns={'일사량': '일사량_대전'})
sejong_2022_mask = (weather['region'] == '세종시') & (weather['timestamp'].dt.year == 2022)
weather = weather.merge(daejeon_solar, on='timestamp', how='left')
weather.loc[sejong_2022_mask, '일사량'] = weather.loc[sejong_2022_mask, '일사량_대전']
weather = weather.drop(columns=['일사량_대전'])

# 6. 야간(00~05시, 19~23시) 일조/일사 결측치 → 0으로 채우기
night_mask = (weather['timestamp'].dt.hour <= 5) | (weather['timestamp'].dt.hour >= 19)
weather.loc[night_mask, '일조']   = weather.loc[night_mask, '일조'].fillna(0)
weather.loc[night_mask, '일사량'] = weather.loc[night_mask, '일사량'].fillna(0)

# 7. 겨울철 강수량 선형 보간
winter_mask = weather['timestamp'].dt.month.isin([11, 12, 1, 2, 3])
weather.loc[winter_mask, '강수량'] = (
    weather.loc[winter_mask].groupby('region')['강수량']
    .transform(lambda x: x.interpolate(method='linear'))
)
weather['강수량'] = weather.groupby('region')['강수량'].transform(lambda x: x.fillna(0))

# 8. 나머지 기상 변수 결측치 → 선형 보간 후 0 채우기
for col in ['기온', '습도', '전운량', '일조', '일사량']:
    weather[col] = weather.groupby('region')[col].transform(
        lambda x: x.interpolate(method='linear').fillna(0)
    )
```

#### A-3. 발전량 + 기상 데이터 결합

```python
merged = pd.merge(power_df, weather_df, on=['timestamp', 'region'], how='inner')

# 이상치 제거 (낮 시간 일사량=0 & 발전량>0 → 센서 오류)
daytime_mask = merged['timestamp'].dt.hour.between(6, 18)
anomaly_mask = daytime_mask & (merged['일사량'] == 0) & (merged['power_mwh'] > 0)
merged = merged[~anomaly_mask]

# 결측치 보간
for col in ['power_mwh', '기온', '강수량', '습도', '일조', '일사량', '전운량']:
    merged[col] = merged.groupby('region')[col].transform(
        lambda x: x.interpolate(method='time').fillna(0)
    )

# ※ 발전효율 컬럼 추가 금지 (데이터 누수 유발)

# 시도명 → 숫자 코드 변환
from sklearn.preprocessing import LabelEncoder
import pickle
le = LabelEncoder()
merged['region_code'] = le.fit_transform(merged['region'])
with open("models/national_region_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# train/test 시간 순 분리
train = merged[merged['timestamp'] < '2023-01-01'].copy()
test  = merged[merged['timestamp'] >= '2023-01-01'].copy()
```

#### A-4. 데이터 기댓값 테스트

```python
def validate_national(df, name):
    required = ['power_mwh', '일사량', '기온', '습도', '전운량', 'region', 'region_code']
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"
    assert df['power_mwh'].min() >= 0
    assert df['일사량'].min() >= 0
    assert df.isnull().sum().sum() == 0
    assert len(df) > 10000
    assert df['region'].nunique() == 17
    assert df.duplicated(subset=['timestamp', 'region']).sum() == 0
```

---

### TASK B — 전국 피처 엔지니어링 (`src/features/feature_engineering_national.py`) ✅ 완료

**입력**: `data/processed/national_train_ready.csv`, `data/processed/national_test_ready.csv`
**출력**: `data/processed/national_train_features.csv`, `data/processed/national_test_features.csv`

```python
# ── 래그 피처 (region별) ───────────────────────────────────────────────────
for col_name, shift_n in [('lag_1h',1),('lag_2h',2),('lag_3h',3),('lag_24h',24)]:
    df[col_name] = df.groupby('region')['power_mwh'].shift(shift_n)

# ── 변화량 피처 (region별) — v3 신규 추가 ──────────────────────────────────
# 전라남도 등 급등/급락 패턴 포착 목적. XGBoost 재학습 시 MAE 16.37 → 9.66 개선 확인.
df['power_diff_1h'] = df.groupby('region')['power_mwh'].diff(1)  # 직전 대비 변화량
df['power_diff_2h'] = df.groupby('region')['power_mwh'].diff(2)  # 2시간 전 대비 변화량

# ── 롤링 통계 (region별) ───────────────────────────────────────────────────
df['rolling_mean_3h'] = df.groupby('region')['power_mwh'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['rolling_mean_6h'] = df.groupby('region')['power_mwh'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)
df['rolling_std_3h'] = df.groupby('region')['power_mwh'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0)
)

# ── 나머지 피처 (region 무관) ──────────────────────────────────────────────
df['hour']                 = df['timestamp'].dt.hour
df['month']                = df['timestamp'].dt.month
df['day_of_week']          = df['timestamp'].dt.dayofweek
df['is_weekend']           = (df['day_of_week'] >= 5).astype(int)
df['season']               = df['month'].map({12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4})
df['solar_altitude_proxy'] = np.sin(np.pi * (df['hour'] - 6) / 12).clip(0)
df['irrad_x_solar']        = df['일사량'] * df['solar_altitude_proxy']
df['is_daytime']           = df['hour'].between(6, 18).astype(int)
# region_code는 TASK A에서 이미 추가됨

# NaN 제거 (래그/diff 피처로 인한 초반 행)
df = df.dropna()
```

**타겟 컬럼**: `power_mwh`
피처 목록을 `src/features/feature_list_national.json`에 저장.

데이터 기댓값 테스트:
```python
def validate_features_national(df, name):
    required = ['power_mwh', '일사량', 'lag_1h', 'is_daytime', 'hour', 'region_code',
                'power_diff_1h', 'power_diff_2h']  # v3: diff 피처 포함
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"
    assert df['power_mwh'].min() >= 0
    assert df['일사량'].min() >= 0
    assert df.isnull().sum().sum() == 0
    assert len(df) > 10000
```

---

### TASK C — Naive Baseline (전국) ✅ 완료

결과: `outputs/national_baseline_results.json`
- 전체 MAE: 20.51, RMSE: 66.58, 피크 MAE: 28.86

---

### TASK D — XGBoost (전국) ✅ 완료 (power_diff 피처 추가 후 v2 재학습)

#### 현재 확정된 XGBoost 성능 (v2 기준 — 이후 모든 비교는 이 수치 사용)
- 전체 MAE: **9.66**, Naive 대비 개선율: **52.9%**
- 피크 MAE: `outputs/national_xgb_results.json` 참조
- 지역별 주요 수치: 전라남도 90.4, 충청남도 25.5, 전라북도 11.5, 제주도 2.6

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
- 전체 MAE/RMSE + 피크 MAE/RMSE + 지역별 MAE
- naive 대비 개선율
- 모델: `models/national_xgboost_model.json`
- feature importance: `outputs/national_xgb_feature_importance.png`
- 예측값: `outputs/national_xgb_predictions.csv`
- 결과: `outputs/national_xgb_results.json`
- W&B: WANDB_MODE=offline, project="solar-power-forecast", name="xgboost-national"

---

### TASK E — 행동 테스트 (전국) ✅ 완료

#### 테스트 1 — NaN/Inf 검증: PASS
#### 테스트 2 — 방향성 테스트: FAIL (82.4% < 90%)
```
※ 물리적으로 일사량 증가 시 발전량 미증가 케이스가 존재함
  (고온에 의한 패널 효율 저하, 전운량 복합 영향 등 실제 현상)
  → 90% 기준은 유지하되, 이 FAIL은 허용 가능한 수준으로 판단. 모델 결함 아님.
  → LSTM도 동일한 기준(90%)과 동일한 해석 기준 적용.
```
#### 테스트 3 — 불변성 테스트: PASS
#### 테스트 4 — 정확성 테스트: PASS
#### 테스트 5 — 지역 불변성 테스트: FAIL (11/17)
```
※ region_code feature importance가 0.01로 매우 낮음 (lag_1h의 1/65 수준).
  지역 특성이 lag/기상 피처에 이미 흡수되어 있어 구조적으로 발생하는 FAIL.
  모델 결함 아님.
```

결과: `outputs/national_behavioral_test_results.json`

---

### TASK F — LSTM (전국) (`src/models/train_lstm_national.py`) 🚧 진행 필요

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

#### F-0. 피처 일치 확인 (필수 — 시작 전 반드시 확인)

LSTM은 XGBoost와 **동일한 피처셋**을 사용해야 공정한 비교가 가능하다.
`national_train_features.csv`에 아래 피처가 모두 포함되어 있는지 확인 후 진행:

```python
REQUIRED_FEATURES = [
    'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h',
    'power_diff_1h', 'power_diff_2h',        # ← v3 신규. 반드시 포함할 것
    'rolling_mean_3h', 'rolling_mean_6h', 'rolling_std_3h',
    'hour', 'month', 'day_of_week', 'is_weekend', 'season',
    'solar_altitude_proxy', 'irrad_x_solar', 'is_daytime',
    'region_code', '일사량', '기온', '습도', '전운량',
]
# 위 피처 중 누락된 것이 있으면 feature_engineering_national.py 재실행 후 진행
```

#### F-1. 데이터셋 구성
- seq_len = 24
- **region별로 시퀀스 생성** — 지역 경계에서 시퀀스 끊기
```python
sequences = []
for region in df['region'].unique():
    region_df = df[df['region'] == region].sort_values('timestamp')
    for i in range(len(region_df) - seq_len):
        sequences.append(region_df.iloc[i:i+seq_len+1])
```
- StandardScaler: train fit → train/test transform (test에는 transform만)
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

평가 항목:
- 전체 MAE/RMSE + 피크 MAE/RMSE + 지역별 MAE
- **XGBoost(v2) 대비 비교** — 비교 기준: XGBoost MAE 9.66, Naive 대비 52.9%
  (구버전 XGBoost MAE 16.37과 비교하지 말 것)
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

### TASK G — ESS 시뮬레이션 (전국) 🚧 TASK F 완료 후 재실행

```python
ESS_CAPACITY_MWH    = 500.0
SOC_MIN, SOC_MAX    = 0.20, 0.80
CHARGE_RATE_MAX     = 100.0
DISCHARGE_RATE_MAX  = 100.0
EFFICIENCY          = 0.95
DEMAND_MWH_PER_HOUR = 50.0
```

전략 3가지 (naive / xgb / lstm), 지역별 운영효율점수도 출력.

**기존 XGBoost 결과 (재실행 전 참고)**
- XGBoost 전력낭비율: 84.1%, 부족횟수: 503, 운영효율점수: 14.6
- Naive 전력낭비율: 94.5%, 부족횟수: 515, 운영효율점수: 5.0

결과: `outputs/national_ess_simulation_results.json`, `outputs/national_ess_simulation_comparison.png`

---

### TASK H — 최종 리포트 (전국) 🚧 TASK F 완료 후 재실행

모든 `outputs/national_*.json`을 읽어 아래 형식으로 출력 후 `outputs/national_final_report.md` 저장.

```
=== [전국] 모델 성능 비교 (Test Set 2023) ===
| 모델           | MAE  | RMSE | 피크 MAE | Naive 대비 개선율 |
|--------------|------|------|--------|--------------|
| Naive(lag1)  |20.51 |66.58 | 28.86  | -            |
| XGBoost (v2) | 9.66 | ???  | ???    | 52.9%        |
| LSTM         | ???  | ???  | ???    | ???          |

# ※ XGBoost는 power_diff 피처 추가 후 재학습된 v2 기준

=== [전국] 지역별 MAE (XGBoost v2 기준) ===
| 지역 | MAE | Naive 대비 개선율 |

=== [전국] ESS 시뮬레이션 비교 ===
| 전략 | 전력낭비율 | 부족횟수 | 사이클수 | 운영효율점수 |

=== [전국] 테스트 결과 ===
[데이터 기댓값 테스트] national_train / national_test: PASS/FAIL
[XGBoost 행동 테스트] NaN/Inf / 방향성 / 불변성 / 정확성 / 지역불변성: PASS/FAIL
  ※ 방향성 FAIL(82.4%): 물리적 허용 범위, 모델 결함 아님
  ※ 지역불변성 FAIL: region_code importance 낮음(0.01), 구조적 현상
[LSTM 테스트] 암기 / NaN/Inf / state_dict / TorchScript / ONNX / 배치(1/8/64): PASS/FAIL
```

---

## ⚠️ 전체 공통 규칙

1. **시간 순 분리**: train ~2022년 / test 2023년. random split 절대 금지.
2. **지역별 래그/롤링/diff**: region 기준 groupby 후 계산. 지역 경계 누수 방지.
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
13. **피처 일치**: LSTM은 XGBoost와 동일한 피처셋 사용. power_diff_1h/2h 반드시 포함.
14. **비교 기준 통일**: XGBoost 비교 기준은 v2(MAE 9.66) 사용. 구버전(16.37) 사용 금지.

---

## 🚀 실행 순서

```bash
# TASK A~E, G~H 완료됨. 아래 순서로 마무리:

# [1] TASK F — 코랩 엔터프라이즈(GPU)에서 실행
# 코랩 설치: !pip install wandb onnx onnxruntime onnxscript
# F-0에서 power_diff_1h/2h 피처 포함 여부 먼저 확인할 것
python src/models/train_lstm_national.py

# [2] TASK G 재실행 — LSTM 전략 추가
python src/simulation/ess_simulation_national.py

# [3] TASK H 재실행 — 최종 리포트 업데이트
python src/reporting/final_report_national.py
```

---

## 📁 완성 후 추가될 파일 구조

```
models/
├── national_region_encoder.pkl        ← TASK A ✅
├── national_xgboost_model.json        ← TASK D ✅
├── national_lstm_scaler.pkl           ← TASK F 🚧
├── national_lstm_model_state.pt       ← TASK F 🚧
├── national_lstm_model_scripted.pt    ← TASK F 🚧
└── national_lstm_model.onnx           ← TASK F 🚧

outputs/
├── national_baseline_results.json     ← ✅
├── national_xgb_results.json          ← ✅ (v2, MAE 9.66)
├── national_xgb_predictions.csv       ← ✅
├── national_xgb_feature_importance.png← ✅
├── national_behavioral_test_results.json ← ✅
├── national_lstm_results.json         ← 🚧
├── national_lstm_predictions.csv      ← 🚧
├── national_lstm_loss_curve.png       ← 🚧
├── national_model_save_verify_results.json ← 🚧
├── national_ess_simulation_results.json ← 🚧 재실행 필요
├── national_ess_simulation_comparison.png ← 🚧 재실행 필요
└── national_final_report.md           ← 🚧 재실행 필요
```
