# Claude Code 실행 프롬프트
# 태양광 발전량 예측 ML 프로젝트

---

## 🔰 프로젝트 컨텍스트 (먼저 읽어라)

이 프로젝트는 **태양광 발전소의 과거 발전량 + 기상 데이터로 미래 발전량을 예측**하는 ML 시스템이다.
최종 목표는 단순 예측이 아니라, **ESS(에너지 저장 시스템) 운영 효율을 baseline 대비 수치로 개선**하는 것이다.

## ✅ 완료된 TASK (실행하지 말 것)
- TASK 0: requirements.in/txt 이미 존재
- TASK 1: feature_engineering.py 완료 — 단, validate_features() 함수 추가 필요
- TASK 2: baseline_naive.py 완료
- TASK 3: train_xgboost.py 완료

## 🔧 수정이 필요한 TASK
- TASK 1: feature_engineering.py에 1-2 데이터 기댓값 테스트 코드 추가
- TASK 0: src/check_env.py 신규 생성

## 🚧 신규 작업 TASK
- TASK 4, 5, 6, 7: 파일 없음, 처음부터 작성

### 현재 디렉토리 구조 (이미 존재함)
```
ENERGY-TIME-SERIES-FORECAST/
├── .venv/                                          # 가상환경
├── .python-version
├── data/
│   ├── raw/
│   │   ├── 한국동서발전(주)_제주 기상관측 및 태양광 발전 현황_20240531.csv   # 메인 데이터
│   │   ├── 170101_171231_OBS_ASOS_TIM.csv
│   │   ├── 170101_230228_지역별_시간별_태양광_발전량.csv
│   │   ├── 180101_181231_OBS_ASOS_TIM.csv
│   │   ├── 190101_191231_OBS_ASOS_TIM.csv
│   │   ├── 200101_201231_OBS_ASOS_TIM.csv
│   │   ├── 210101_211231_OBS_ASOS_TIM.csv
│   │   ├── 220101_221231_OBS_ASOS_TIM.csv
│   │   ├── 230101_231231_OBS_ASOS_TIM.csv
│   │   ├── 230601_230831_지역별_시간별 _태양광_발전량.csv
│   │   ├── 230901_231130_지역별_시간대별_태양광_발전량.csv
│   │   └── ess_simulation/
│   │       ├── 180101-201231_태양광_지역별_시간별_전력거래량.csv
│   │       ├── 230301_230531_한국전력거래소_지역별_시간별_태양광_발전량.csv
│   │       ├── 231201_231231_한국전력거래소_지역별 _시간별_태양광 _발전량.csv
│   │       ├── 240101_241231_한국전력거래소_지역별_시간대별_태양광_발전량.csv
│   │       └── 251231_251231_한국전력거래소_지역별_시간별_태양광_발전량.csv
│   └── processed/
│       ├── train_ready.csv                         # 완료 (2018~2023, 이상치 제거)
│       ├── test_ready.csv                          # 완료 (2024년)
│       ├── train_features.csv                      # 완료 (피처 엔지니어링)
│       └── test_features.csv                       # 완료 (피처 엔지니어링)
├── eda/
│   └── decompose.py                                # 완료 (EDA + 시계열 분해)
├── models/
│   ├── xgboost_model.json                          # 완료 (TASK 3)
│   ├── lstm_model_state.pt                         # 완료 (TASK 5 - state_dict)
│   ├── lstm_model_scripted.pt                      # 완료 (TASK 5 - TorchScript)
│   ├── lstm_model.onnx                             # 완료 (TASK 5 - ONNX)
│   └── lstm_scaler.pkl                             # 완료 (TASK 5 - scaler)
├── outputs/
│   ├── baseline_results.json                       # 완료 (TASK 2)
│   ├── xgb_results.json                            # 완료 (TASK 3)
│   ├── xgb_predictions.csv                         # 완료 (TASK 3)
│   ├── xgb_feature_importance.png                  # 완료 (TASK 3)
│   ├── lstm_results.json                           # 완료 (TASK 5)
│   ├── lstm_predictions.csv                        # 완료 (TASK 5)
│   ├── lstm_loss_curve.png                         # 완료 (TASK 5)
│   ├── model_save_verify_results.json              # 완료 (TASK 5)
│   ├── ess_simulation_results.json                 # 완료 (TASK 6)
│   ├── ess_simulation_comparison.png               # 완료 (TASK 6)
│   ├── final_report.md                             # 완료 (TASK 7)
│   ├── correlation_heatmap.png
│   ├── decompose_result.png
│   ├── eda_overview.png
│   └── prediction_comparison.png
├── src/
│   ├── check_env.py                                # 완료 (TASK 0)
│   ├── features/
│   │   ├── feature_engineering.py                  # 완료 + validate_features() 추가 (TASK 1)
│   │   └── feature_list.json
│   ├── models/
│   │   ├── baseline_naive.py                       # 완료 (TASK 2)
│   │   ├── train_xgboost.py                        # 완료 (TASK 3)
│   │   └── train_lstm.py                           # 완료 (TASK 5)
│   ├── simulation/
│   │   └── ess_simulation.py                       # 완료 (TASK 6)
│   ├── tests/
│   │   └── behavioral_tests.py                     # 완료 (TASK 4)
│   ├── reporting/
│   │   └── final_report.py                         # 완료 (TASK 7)
│   ├── utils/
│   │   └── font_setting.py
│   └── visualization/
│       └── plot_comparison.py
├── wandb/                                          # W&B 오프라인 로그 (자동 생성됨)
├── preprocess.py                                   # 완료
├── prepare_splits.py                               # 완료
├── requirements.in
├── requirements.txt
├── ML Test Suite.md
├── check-env script.md
├── claude_code_prompt.md
├── model trouble shooting.md
├── tea_debug.log
├── 실행계획서.md
└── 사전조사내용정리.md
```

### 핵심 데이터 정보
- **타겟**: `태양광 발전량(MWh)` — 1시간 단위 시계열
- **주요 피처**: 일사량(상관 0.71), 발전효율(파생, 상관 0.89), 일조(hr)(0.57), 전운량(-0.43), 습도(-0.36), 기온(0.16)
- **train**: 2018~2023년 / **test**: 2024년
- **이미 완료된 전처리**: 낮 시간(06~18시) 일사량=0 & 발전량>0 이상치 제거, time 보간, 발전효율 컬럼 추가

---

## 📋 TASK 목록 (순서대로 실행하라)

---

### TASK 0 — 환경 설정 및 검증

`.venv`가 이미 존재하므로 `.venv/bin/python` 기준으로 작업한다.

#### 0-1. requirements.in / requirements.txt 작성

**requirements.in** 파일을 프로젝트 루트에 생성하라:
```
pandas
numpy
scikit-learn
xgboost
torch
onnx
onnxruntime
matplotlib
seaborn
statsmodels
wandb
pip-tools
```

그 다음 pip-compile로 requirements.txt를 생성하라:
```bash
.venv/bin/pip install pip-tools
.venv/bin/pip-compile requirements.in --output-file requirements.txt
.venv/bin/pip install -r requirements.txt
```

#### 0-2. 환경 검증 스크립트 실행

패키지 설치 완료 후 아래 내용을 실행하여 핵심 라이브러리가 정상 설치되었는지 확인하라.
모든 항목이 출력되면 "✅ 환경 검증 통과"를 출력하고, 하나라도 실패하면 에러 메시지와 함께 종료하라.

```python
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

import xgboost
print(f"XGBoost: {xgboost.__version__}")

import pandas
print(f"pandas: {pandas.__version__}")

import numpy
print(f"numpy: {numpy.__version__}")

import statsmodels
print(f"statsmodels: {statsmodels.__version__}")

import onnx, onnxruntime
print(f"ONNX: {onnx.__version__}  /  ONNXRuntime: {onnxruntime.__version__}")

print("\n✅ 환경 검증 통과")
```

---

### TASK 1 — 피처 엔지니어링 (`src/features/feature_engineering.py`)

`src/features/` 디렉토리를 생성하고, `feature_engineering.py`를 작성하라.

**입력**: `data/processed/train_ready.csv`, `data/processed/test_ready.csv`
**출력**: `data/processed/train_features.csv`, `data/processed/test_features.csv`

#### 1-1. 피처 생성

구현할 피처 목록:
1. **시간 피처**: hour, month, day_of_week, is_weekend, season(1~4)
2. **태양 위치 피처**: solar_altitude_proxy = sin(π × (hour-6)/12) — 낮 시간 기준 정규화
3. **래그 피처**: 발전량 lag_1h, lag_2h, lag_3h, lag_24h (전날 같은 시각)
4. **롤링 통계**: 발전량 rolling_mean_3h, rolling_mean_6h, rolling_std_3h
5. **기상 교호작용**: irrad_x_solar = 일사량 × solar_altitude_proxy
6. **야간 마스크**: is_daytime = 1 if 6 ≤ hour ≤ 18 else 0

중요 규칙:
- **래그 피처는 train과 test를 concat 후 생성하고 다시 분리할 것** (데이터 누수 방지)
- NaN이 생기는 초반 행은 dropna()로 제거
- 최종 피처 목록을 `src/features/feature_list.json`에 저장

#### 1-2. 데이터 기댓값 테스트 (피처 엔지니어링 완료 직후 실행)

`train_features.csv`와 `test_features.csv` 저장 직후, 아래 검증을 수행하라.
하나라도 실패하면 에러 메시지와 함께 종료하라.

```python
def validate_features(df, name):
    # 1. 필수 컬럼 존재 여부
    required = ["태양광 발전량(MWh)", "일사량", "lag_1h", "is_daytime", "hour"]
    for col in required:
        assert col in df.columns, f"[{name}] 필수 컬럼 누락: {col}"

    # 2. 물리적으로 불가능한 값 (느슨한 경계 — 오탐 방지)
    assert df["태양광 발전량(MWh)"].min() >= 0, f"[{name}] 발전량 음수 존재"
    assert df["일사량"].min() >= 0,             f"[{name}] 일사량 음수 존재"

    # 3. 결측치 없음
    nan_count = df.isnull().sum().sum()
    assert nan_count == 0, f"[{name}] NaN {nan_count}건 존재"

    # 4. 시간 순서 정렬
    assert df.index.is_monotonic_increasing, f"[{name}] timestamp 정렬 깨짐"

    # 5. 데이터 크기
    assert len(df) > 1000, f"[{name}] 데이터가 너무 적음: {len(df)}행"

    print(f"  [{name}] 데이터 기댓값 테스트 통과 ✓  ({len(df):,}행)")

validate_features(train_out, "train_features")
validate_features(test_out,  "test_features")
```

스크립트 마지막에 검증 출력:
- train/test shape
- NaN 개수
- 피처 목록

---

### TASK 2 — Naive Baseline (`src/models/baseline_naive.py`)

**전략 2가지**를 모두 구현하고 MAE/RMSE를 비교하라:
1. `lag1`: 직전 시점 발전량을 그대로 예측값으로 사용
2. `rolling24`: 직전 24시간 이동 평균을 예측값으로 사용

평가 대상: **test set 전체** + **피크 시간대(10~14시)만 따로**

결과를 `outputs/baseline_results.json`에 저장:
```json
{
  "lag1":      {"MAE": ..., "RMSE": ..., "MAE_peak": ..., "RMSE_peak": ...},
  "rolling24": {"MAE": ..., "RMSE": ..., "MAE_peak": ..., "RMSE_peak": ...}
}
```

---

### TASK 3 — XGBoost 모델 (`src/models/train_xgboost.py`)

#### 3-1. 암기 테스트 (소규모 과적합 확인)
train set에서 **처음 500행만** 추출하여 XGBoost를 학습시킨 뒤,
같은 500행에 대한 MAE가 naive baseline의 30% 이하면 "암기 테스트 통과"로 출력하고 진행하라.
통과하지 못하면 오류를 출력하고 중단하라.

#### 3-2. 본 학습
- **피처**: TASK 1에서 만든 `train_features.csv` 사용
- **하이퍼파라미터** (초기값, 나중에 조정 가능):
  ```python
  params = {
      "n_estimators": 500,
      "max_depth": 6,
      "learning_rate": 0.05,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "random_state": 42,
      "n_jobs": -1,
  }
  ```
- **early stopping**: validation set을 train의 마지막 20%로 분리 (시간 순 분리 필수), early_stopping_rounds=30

#### 3-3. 평가 및 저장
- test set MAE, RMSE, 피크 시간대 MAE/RMSE 계산
- naive baseline 대비 개선율(%) 출력
- 모델을 `models/xgboost_model.json`으로 저장
- feature importance 상위 15개를 `outputs/xgb_feature_importance.png`로 저장
- 예측값을 `outputs/xgb_predictions.csv`로 저장 (timestamp, actual, predicted 컬럼)
- 결과를 `outputs/xgb_results.json`에 저장

#### 3-4. W&B 로깅
wandb.init(project="solar-power-forecast", name="xgboost-v1")으로 초기화하고
하이퍼파라미터, MAE, RMSE, 개선율을 로깅하라.
**단, `WANDB_MODE=offline`으로 설정하여 오프라인 모드로 실행하라.**

---

### TASK 4 — 행동 테스트 (`src/tests/behavioral_tests.py`)

학습된 XGBoost 모델(`models/xgboost_model.json`)을 로드한 뒤 아래 4가지 테스트를 수행하라.

#### 테스트 1 — 모델 출력 NaN/Inf 검증
test set 전체에 대해 XGBoost 예측값을 생성한 뒤,
NaN 또는 Inf가 하나라도 존재하면 FAIL로 처리하고 에러 메시지를 출력하라.

```python
import numpy as np
pred = model.predict(X_test)
assert np.isfinite(pred).all(), \
    f"NaN/Inf 감지: NaN={np.isnan(pred).sum()}건, Inf={np.isinf(pred).sum()}건"
print("  [테스트 1] 모델 출력 NaN/Inf 검증 통과 ✓")
```

#### 테스트 2 — 방향성 테스트 (Directional Test)
test set에서 낮 시간대 샘플 100개를 임의 추출한다.
각 샘플의 일사량을 +0.5 MJ/㎡ 증가시킨 뒤 예측값이 **증가**하는지 확인.
증가 비율이 90% 이상이면 통과.

#### 테스트 3 — 불변성 테스트 (Invariance Test)
동일한 피처 행을 입력했을 때, 여러 번 호출해도 예측값이 동일한지 확인 (결정론적 모델).
5회 반복 호출 후 표준편차가 0이면 통과.

#### 테스트 4 — 정확성 테스트 (Accuracy Test)
피크 시간대(10~14시) MAE가 naive baseline의 lag1 MAE_peak보다 낮으면 통과.

4개 테스트 결과를 콘솔에 출력하고 `outputs/behavioral_test_results.json`으로 저장하라.

---

### TASK 5 — LSTM 모델 (`src/models/train_lstm.py`)

#### 5-1. 데이터셋 구성
- **시퀀스 길이(seq_len)**: 24 (24시간 look-back)
- **타겟**: 다음 1시간 후 발전량
- `torch.utils.data.Dataset`을 상속한 `SolarDataset` 클래스 구현
- 피처는 TASK 1의 피처 그대로 사용
- **스케일링**: StandardScaler를 train에 fit하고 train/test 모두 transform
  - scaler를 `models/lstm_scaler.pkl`로 저장

#### 5-2. 모델 아키텍처
```python
class SolarLSTM(nn.Module):
    # input_size: 피처 수
    # hidden_size: 128
    # num_layers: 2
    # dropout: 0.2
    # 출력층: Linear(128 → 1)
```

#### 5-3. 학습 설정
- optimizer: Adam, lr=0.001
- loss: MSELoss
- epochs: 50 (조기 종료: val_loss가 10 epoch 동안 개선 없으면 중단)
- batch_size: 64
- GPU 우선 사용:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[{ts()}] 사용 디바이스: {device}")
  ```
- train의 마지막 20%를 validation으로 분리 (시간 순 분리 필수)

#### 5-4. 암기 테스트 (학습 시작 전)
train set에서 **처음 2개 배치(batch_size=64 기준, 128행)만** 추출하여 100 epoch 학습한 뒤,
같은 128행에 대한 MSELoss가 초기 loss의 1% 이하로 떨어지면 "LSTM 암기 테스트 통과"를 출력하고 본 학습으로 진행하라.
통과하지 못하면 오류를 출력하고 중단하라.

단, 암기 테스트 시에는 dropout을 0으로 설정하여 모델이 암기에 집중하도록 하라.

#### 5-5. 평가 및 모델 출력 검증
- test set 예측값 생성 후 **NaN/Inf 검증**을 먼저 수행하라:
  ```python
  import torch
  pred_tensor = torch.tensor(pred)
  assert torch.isfinite(pred_tensor).all(), \
      f"LSTM 출력 NaN/Inf 감지 — 학습 불안정 의심"
  ```
- test set MAE, RMSE, 피크 시간대 MAE/RMSE 계산
- XGBoost 대비 비교 출력
- epoch별 train/val loss 곡선을 `outputs/lstm_loss_curve.png`로 저장
- 예측값을 `outputs/lstm_predictions.csv`로 저장
- 결과를 `outputs/lstm_results.json`에 저장
- W&B 로깅 (WANDB_MODE=offline)

#### 5-6. 모델 저장 및 검증

**저장 전 준비**
```python
# GPU로 학습했더라도 저장은 반드시 CPU에서 수행 (환경 독립성 확보)
model = model.cpu().eval()

# 저장 전 원본 추론 결과 기록 (검증 기준값)
sample_input = X_test_tensor[:10].cpu()   # 테스트 샘플 10개
with torch.no_grad():
    pred_before = model(sample_input).numpy()
print(f"[{ts()}]   저장 전 예측값 기록 완료 (샘플 10개)")
```

**3가지 방식으로 저장**
```python
# 1. state_dict
torch.save(model.state_dict(), "models/lstm_model_state.pt")

# 2. TorchScript
scripted = torch.jit.script(model)
scripted.save("models/lstm_model_scripted.pt")

# 3. ONNX
dummy_input = torch.randn(1, seq_len, input_size)  # (batch, seq, feature)
torch.onnx.export(
    model, dummy_input,
    "models/lstm_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
```

**저장 후 검증 — 3가지 방식 각각 불러와서 저장 전 결과와 1:1 대조**
```python
TOL = 1e-5  # 허용 오차

# 1. state_dict 검증
model_loaded = SolarLSTM(...)
model_loaded.load_state_dict(torch.load("models/lstm_model_state.pt"))
model_loaded.eval()
with torch.no_grad():
    pred_state = model_loaded(sample_input).numpy()
assert np.allclose(pred_before, pred_state, atol=TOL), \
    f"state_dict 불일치 — 최대 오차: {np.abs(pred_before - pred_state).max():.2e}"
print(f"  [state_dict] 검증 통과 ✓")

# 2. TorchScript 검증
scripted_loaded = torch.jit.load("models/lstm_model_scripted.pt")
scripted_loaded.eval()
with torch.no_grad():
    pred_script = scripted_loaded(sample_input).numpy()
assert np.allclose(pred_before, pred_script, atol=TOL), \
    f"TorchScript 불일치 — 최대 오차: {np.abs(pred_before - pred_script).max():.2e}"
print(f"  [TorchScript] 검증 통과 ✓")

# 3. ONNX 검증
import onnxruntime as ort
sess = ort.InferenceSession("models/lstm_model.onnx")
pred_onnx = sess.run(None, {"input": sample_input.numpy()})[0]
assert np.allclose(pred_before, pred_onnx, atol=TOL), \
    f"ONNX 불일치 — 최대 오차: {np.abs(pred_before - pred_onnx).max():.2e}"
print(f"  [ONNX] 검증 통과 ✓")
```

**배치 추론 테스트 — 여러 크기의 입력이 들어올 때 shape과 값이 올바른지 확인**
```python
for batch_size in [1, 8, 64]:
    x_batch = X_test_tensor[:batch_size].cpu()
    with torch.no_grad():
        out = model_loaded(x_batch)
    assert out.shape == (batch_size, 1), \
        f"배치 크기 {batch_size}: 출력 shape 오류 {out.shape}"
    assert torch.isfinite(out).all(), \
        f"배치 크기 {batch_size}: NaN/Inf 감지"
    print(f"  [배치 추론] batch={batch_size} → shape {tuple(out.shape)} ✓")
```

검증 결과를 `outputs/model_save_verify_results.json`으로 저장하라.

---

### TASK 6 — ESS 시뮬레이션 (`src/simulation/ess_simulation.py`)

예측 모델이 있을 때와 없을 때 ESS 운영 효율을 비교하라.

#### ESS 파라미터 (고정값)
```python
ESS_CAPACITY_MWH = 1.0    # 배터리 용량 (kWh)
SOC_MIN = 0.20               # 최소 충전율 20%
SOC_MAX = 0.80               # 최대 충전율 80%
CHARGE_RATE_MAX = 0.2      # 최대 충전 속도 (kW/h)
DISCHARGE_RATE_MAX = 0.2   # 최대 방전 속도 (kW/h)
EFFICIENCY = 0.95            # 충방전 효율 95%
DEMAND_MWH_PER_HOUR = 0.15  # 시간당 수요 (kWh, 고정)
```

#### 시뮬레이션 전략 3가지
1. **naive_strategy**: 예측 없이, 현재 발전량이 수요보다 많으면 무조건 충전, 적으면 방전
2. **xgb_strategy**: XGBoost 예측값을 기반으로 충전/방전 결정
3. **lstm_strategy**: LSTM 예측값을 기반으로 충전/방전 결정

#### 평가 지표 계산
- **전력 낭비율**: curtailment(버려진 전력) / 총 발전량 × 100 (%)
- **전력 부족 횟수**: 방전해도 수요를 못 채운 시간 수
- **배터리 사이클 수**: 충방전 횟수 합산 / 2
- **ESS 운영 효율 점수**: (1 - 낭비율/100) × (1 - 부족횟수/총시간) × 100

결과를 `outputs/ess_simulation_results.json`과 `outputs/ess_simulation_comparison.png`로 저장하라.

---

### TASK 7 — 최종 비교 리포트 (`src/reporting/final_report.py`)

모든 `outputs/*.json` 파일을 읽어 다음 표를 콘솔에 출력하고 `outputs/final_report.md`로 저장하라.

```
=== 모델 성능 비교 (Test Set) ===
| 모델          | MAE    | RMSE   | 피크 MAE | Naive 대비 개선율 |
|--------------|--------|--------|----------|-----------------|
| Naive(lag1)  | XX.XX  | XX.XX  | XX.XX    | -               |
| XGBoost      | XX.XX  | XX.XX  | XX.XX    | XX.X%           |
| LSTM         | XX.XX  | XX.XX  | XX.XX    | XX.X%           |

=== ESS 시뮬레이션 비교 ===
| 전략          | 전력낭비율 | 부족횟수 | 사이클수 | 운영효율점수 |
|--------------|----------|---------|---------|-----------|
| Naive        | XX.X%    | XXX     | XX.X    | XX.X      |
| XGBoost 기반 | XX.X%    | XXX     | XX.X    | XX.X      |
| LSTM 기반    | XX.X%    | XXX     | XX.X    | XX.X      |

=== 테스트 결과 ===
[데이터 기댓값 테스트]
- train_features 검증: PASS/FAIL
- test_features 검증:  PASS/FAIL

[XGBoost 행동 테스트]
- NaN/Inf 출력 검증:  PASS/FAIL
- 방향성 테스트:       PASS/FAIL (증가 비율: XX%)
- 불변성 테스트:       PASS/FAIL
- 정확성 테스트:       PASS/FAIL

[LSTM 테스트]
- 암기 테스트:         PASS/FAIL
- NaN/Inf 출력 검증:  PASS/FAIL
- state_dict 검증:    PASS/FAIL
- TorchScript 검증:   PASS/FAIL
- ONNX 검증:          PASS/FAIL
- 배치 추론 (1):      PASS/FAIL
- 배치 추론 (8):      PASS/FAIL
- 배치 추론 (64):     PASS/FAIL
```

---

## ⚠️ 전체 공통 규칙

1. **시간 순 분리 원칙**: train/validation/test 분리 시 항상 날짜 기준으로 앞뒤를 나눠라. 절대 random split 하지 말 것.
2. **데이터 누수 금지**: scaler, lag 피처 등은 반드시 train 기준으로만 fit하고, test에는 transform만 적용하라.
3. **야간 시간대 처리**: 발전량 예측에서 00~05시 / 19~23시는 발전이 없으므로 0으로 클리핑하라 (예측값이 음수 또는 소수점이 나올 경우).
4. **재현성**: 모든 random_state=42로 고정.
5. **경로**: 모든 파일 경로는 스크립트 상단에 상수로 정의하라.
6. **로그**: 각 단계 시작/완료 시 타임스탬프와 함께 출력하라.
7. **에러 처리**: 파일이 없거나 컬럼이 없으면 명확한 에러 메시지와 함께 종료하라.
8. **W&B**: 모든 모델 학습에 `WANDB_MODE=offline`으로 로깅하라.
9. **모델 저장**: LSTM은 반드시 `.cpu().eval()` 상태에서 저장하라 (GPU 환경 독립성 확보).

---

## 🚀 실행 순서 요약

```bash
# TASK 0: 환경 설정 및 검증
.venv/bin/pip install pip-tools
.venv/bin/pip-compile requirements.in --output-file requirements.txt
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/check_env.py          # 환경 검증

# TASK 1: 피처 엔지니어링 + 데이터 기댓값 테스트
.venv/bin/python src/features/feature_engineering.py

# TASK 2: Naive Baseline
.venv/bin/python src/models/baseline_naive.py

# TASK 3: XGBoost (암기 테스트 포함)
.venv/bin/python src/models/train_xgboost.py

# TASK 4: 행동 테스트 (NaN/Inf + 방향성 + 불변성 + 정확성)
.venv/bin/python src/tests/behavioral_tests.py

# TASK 5: LSTM (암기 테스트 + NaN/Inf 검증 + 모델 저장 검증 포함)
.venv/bin/python src/models/train_lstm.py

# TASK 6: ESS 시뮬레이션
.venv/bin/python src/simulation/ess_simulation.py

# TASK 7: 최종 리포트
.venv/bin/python src/reporting/final_report.py
```

---

## 📁 완성 후 예상 디렉토리 구조

```
ENERGY-TIME-SERIES-FORECAST/
├── .venv/
├── data/
│   ├── raw/
│   └── processed/
│       ├── train_ready.csv
│       ├── test_ready.csv
│       ├── train_features.csv              ← TASK 1
│       └── test_features.csv               ← TASK 1
├── eda/
├── src/
│   ├── check_env.py                        ← TASK 0
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_list.json
│   ├── models/
│   │   ├── baseline_naive.py
│   │   ├── train_xgboost.py
│   │   └── train_lstm.py
│   ├── simulation/
│   │   └── ess_simulation.py
│   ├── tests/
│   │   └── behavioral_tests.py
│   └── reporting/
│       └── final_report.py
├── models/
│   ├── xgboost_model.json
│   ├── lstm_model_state.pt                 ← TASK 5 (state_dict)
│   ├── lstm_model_scripted.pt              ← TASK 5 (TorchScript)
│   ├── lstm_model.onnx                     ← TASK 5 (ONNX)
│   └── lstm_scaler.pkl
├── outputs/
│   ├── baseline_results.json
│   ├── xgb_results.json
│   ├── xgb_feature_importance.png
│   ├── xgb_predictions.csv
│   ├── lstm_results.json
│   ├── lstm_loss_curve.png
│   ├── lstm_predictions.csv
│   ├── behavioral_test_results.json
│   ├── model_save_verify_results.json      ← TASK 5
│   ├── ess_simulation_results.json
│   ├── ess_simulation_comparison.png
│   └── final_report.md
├── requirements.in                         ← TASK 0
├── requirements.txt                        ← TASK 0
├── preprocess.py
├── prepare_splits.py
├── 실행계획서.md
└── 사전조사내용정리.md
```