# 전국 태양광 발전량 예측 (Energy Time-Series Forecast)

전국 17개 시도의 시간별 태양광 발전량을 기상 데이터로 예측하는 머신러닝 시스템입니다.
2017~2022년 데이터로 학습하고 2023년 데이터로 평가하며, ESS(에너지 저장 시스템) 운영
시뮬레이션까지 포함합니다.

## 주요 성과

2023년 테스트셋 기준 모델 성능 비교:

| 모델 | MAE | RMSE | 피크 MAE | Naive 대비 개선율 |
|------|-----|------|----------|-------------------|
| Naive (lag1) | 21.74 | 67.03 | 30.85 | — |
| **XGBoost (통합)** | **9.61** | **46.90** | **7.87** | **55.8%** |
| LSTM (통합) | 17.82 | 67.50 | 30.71 | 18.0% |

XGBoost 통합 모델이 모든 지표에서 우수하며, 17개 시도 중 16개 지역에서
Naive 대비 60% 이상 개선했습니다. 전라남도(MAE 90.04)는 발전 규모가 커
절대 오차가 크게 나타나며 별도 분석 대상입니다.

## 데이터

- **출처**: 기상청 ASOS 시간별 관측 자료 + 한국전력거래소 지역별 시간별 태양광 발전량
- **기간**: 2017~2023년
- **분리 방식**: 시간 순 분리 (train ≤ 2022년 / test = 2023년) — random split 미사용

> ⚠️ 원본 데이터(`data/`)와 학습된 모델 가중치(`models/`)는 용량 문제로
> 저장소에 포함되지 않습니다. 전처리 스크립트로 재생성하거나 별도로 받아야 합니다.

## 프로젝트 구조

```
.
├── preprocess_national.py        # 원본 → 학습용 데이터 전처리
├── src/
│   ├── features/                 # 피처 엔지니어링
│   ├── models/                   # Baseline / XGBoost / LSTM 학습
│   ├── simulation/               # ESS 운영 시뮬레이션
│   ├── diagnostics/              # 데이터 진단 / 분포 변화 점검
│   ├── reporting/                # 지표 산출 / 최종 보고서 생성
│   ├── tests/                    # 행동 테스트 (behavioral test)
│   ├── visualization/            # 비교 그래프
│   └── utils/                    # 한글 폰트 설정 등 공용 유틸
├── outputs/                      # 평가 결과 (json / csv / png / 보고서)
├── archive/                      # 종료된 실험 기록 (분리 학습, 하이브리드 등)
├── eda/                          # 탐색적 데이터 분석
└── requirements.txt              # 의존성 (pip-compile 생성)
```

## 설치 및 실행

Python 3.11 기준입니다.

```bash
# 가상환경 생성 및 의존성 설치
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

데이터를 `data/raw/`에 배치한 뒤 파이프라인을 순서대로 실행합니다:

```bash
# 1. 전처리
python preprocess_national.py

# 2. 피처 엔지니어링
python src/features/feature_engineering_national.py

# 3. 모델 학습
python src/models/baseline_naive_national.py
python src/models/train_xgboost_national.py
python src/models/train_lstm_national.py

# 4. ESS 시뮬레이션 및 최종 보고서
python src/simulation/ess_simulation_national.py
python src/reporting/final_report_national.py
```

## 모델링 규칙

재현성과 데이터 누수 방지를 위해 다음 규칙을 따릅니다:

- **시간 순 분리**: train ≤ 2022년, test = 2023년. random split 금지.
- **데이터 누수 금지**: scaler·LabelEncoder는 train 기준으로만 fit.
- **재현성**: `random_state=42` 고정.
- **야간 클리핑**: 00~05시, 19~23시 예측값은 0으로 클리핑.
- **인코딩**: CSV 입출력은 `utf-8-sig` / `utf-8`.

## 기술 스택

pandas · numpy · scikit-learn · XGBoost · PyTorch · matplotlib · seaborn · statsmodels · wandb
