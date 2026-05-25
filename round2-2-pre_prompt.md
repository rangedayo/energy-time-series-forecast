# 작업: 48h 재귀 multi-step 예측 정확도 검증 (round 2-2-pre)

## 프로젝트 컨텍스트
- round 2-1에서 `/predict_horizon` 엔드포인트(horizon=1~48 지원, 재귀 multi-step)가 완성됨
- 다음 단계(round 2-2)에서 Streamlit 운영자 대시보드용 MPC 오케스트레이터를 만들 예정
- MPC는 매 시점 24h lookahead 윈도우를 보고 LP를 풀이 (Rolling Horizon)
- 운영자 화면이 24h 시뮬을 보여주려면 시뮬 끝 시점도 24h lookahead가 필요 → **미래 데이터 48h 필요**
- 그러나 재귀 multi-step의 본질상 t+24~t+47 구간 예측은 t+1~t+23보다 부정확할 가능성
- Phase 2 결과(mpc_xgb ≈ mpc_oracle, 차이 0.08%)는 MPC가 예측 부정확성에 robust함을 시사하지만, 이는 24h 윈도우 기준 결과
- 48h까지 확장해도 그 robustness가 유지되는지 데이터로 직접 확인 필요

## round 2-2-pre 목표
우리 학습된 XGBoost 모델로 horizon=48 예측을 수행하고, **단기(t+1~t+23) vs 장기(t+24~t+47) 구간의 예측 오차를 정량 비교**한다. 그 결과로 round 2-2에서 (b-1) 48h 슬라이스 방식의 정당성을 확정한다.

**스코프 제한**:
- 새 코드/엔드포인트 추가 없음 — 기존 `/predict_horizon` 호출만
- 모델 재학습/튜닝 없음
- 단순 진단 스크립트 1개 작성 + 결과 보고서 1개 생성

## 작업 내역

### (a) 진단 스크립트 작성

`src/diagnostics/diagnose_horizon_accuracy.py` 신규 파일:

```python
"""
48h 재귀 multi-step 예측 정확도 진단.

목적: round 2-2의 (b-1) 48h 슬라이스 방식이 정당한지 확인.
즉, 우리 XGBoost 모델이 t+24~t+47 구간에서도 사용 가능한 정확도를 내는가?

방법:
1. 학습 CSV에서 (region, start_time) 페어 N개 샘플링
2. 각 페어에 대해 /predict_horizon (horizon=48) 호출
3. 응답의 48개 예측을 실측과 비교
4. 단기(step 1~23) vs 장기(step 24~48) 오차 통계 비교
5. JSON + Markdown 보고서 출력
"""

import json
import random
import requests
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.font_setting import apply as _apply_font
_apply_font()

# 설정
N_SAMPLES = 200                    # 샘플링할 (region, start_time) 페어 수
SHORT_STEPS = range(1, 24)         # step 1~23 (단기)
LONG_STEPS = range(24, 49)         # step 24~48 (장기)
API_URL = "http://localhost:8000/predict_horizon"
API_KEY = "dev-key-change-me"
TRAIN_CSV = "data/processed/national_train_features.csv"
OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
```

**주요 로직**:

1. **CSV 로딩 및 샘플링**
   - `national_train_features.csv` 로드 (region, timestamp, power_mwh + 기상 6개 컬럼 있음)
   - 각 region별로 가능한 start_time 후보 추출:
     - 조건: start_time-24h부터 start_time+47h까지의 시간이 그 region 데이터에 모두 존재 (history 24h + 미래 48h 보장)
     - 즉 region마다 timestamp 정렬 후 (24번째 이후, 끝-47번째 이전) 범위에서 후보
   - 17개 region에서 균등하게 샘플링하되 총 N_SAMPLES개 (region당 약 12개)
   - `random.seed(RANDOM_SEED)`로 재현성 보장

2. **각 샘플마다 API 호출**
   - history 24h, forecast 48h 페이로드 구성
   - history의 timestamp들은 `start_time - timedelta(hours=24..1)` 정확히 일치하게 만들기 (round 2-1 검증 통과 조건)
   - forecast의 기상 6개 값은 학습 CSV의 해당 시점 실측 기상 사용 (현실에선 기상예보, 진단에선 ex-post 정확값으로 모델 자체 정확도만 분리 측정)
   - POST 요청, response의 predictions 48개 수집
   - 같은 시점의 실측값 48개도 CSV에서 추출

3. **오차 계산**
   - 각 샘플마다 step별 절대오차 `|predicted - actual|` 계산
   - step별로 누적 → step 1~48 각각의 평균/중앙값/표준편차
   - 단기 집계: SHORT_STEPS의 모든 (샘플 × step) 오차들에 대한 RMSE, MAE, MAPE
   - 장기 집계: LONG_STEPS의 모든 (샘플 × step) 오차들에 대한 RMSE, MAE, MAPE
   - 비교 비율: `장기_RMSE / 단기_RMSE`, `장기_MAE / 단기_MAE`

4. **이상 케이스 처리**
   - API 호출 실패 (5xx, timeout) → 해당 샘플 스킵, 카운트만 기록
   - 실측값이 0인 시점 → MAPE 계산 시 제외 (0 나눗셈 방지)
   - 야간 시간대 (실측 0, 예측 0) → 정상으로 처리 (오차 0)

5. **시각화 3장**
   - `step_wise_rmse.png`: x축 step 1~48, y축 RMSE 선 그래프. 단기/장기 경계(step 23~24)에 vertical line.
   - `error_distribution_compare.png`: 단기/장기 절대오차 분포를 boxplot 또는 violin plot으로 나란히 비교.
   - `sample_trajectories.png`: 무작위 샘플 6개 골라서 각각 "실측 vs 예측" 48시간 라인 플롯 (2×3 grid). 예측이 멀어질수록 어떻게 어긋나는지 직관적으로 보기.

6. **JSON 저장**: `outputs/diagnostics/horizon_accuracy.json`
   ```json
   {
     "config": {"n_samples": 200, "short_steps": [1,23], "long_steps": [24,48], "seed": 42},
     "n_samples_succeeded": 198,
     "n_samples_failed": 2,
     "short_term": {"rmse": ..., "mae": ..., "mape": ..., "n_points": ...},
     "long_term": {"rmse": ..., "mae": ..., "mape": ..., "n_points": ...},
     "ratio_long_to_short": {"rmse": ..., "mae": ..., "mape": ...},
     "step_wise_rmse": [step1_rmse, step2_rmse, ..., step48_rmse],
     "verdict": "PASS|MARGINAL|FAIL",
     "verdict_criteria": "ratio_rmse < 2.0 = PASS, 2.0~3.0 = MARGINAL, ≥3.0 = FAIL"
   }
   ```

7. **Markdown 보고서**: `outputs/diagnostics/horizon_accuracy_report.md`
   - 제목, 목적, 방법, 결과 표, 시각화 3장 임베드, 결론 한 줄
   - 결론 문장 예시:
     - PASS: "장기/단기 RMSE 비율 X.XX → (b-1) 48h 슬라이스 방식이 정당함. round 2-2 진행 가능."
     - MARGINAL: "비율 X.XX → 주의가 필요하나 (b-1) 가능. round 2-2 결과에서 끝쪽 시점의 액션 합리성을 추가 점검 권고."
     - FAIL: "비율 X.XX → (b-2) 24h 슬라이스로 후퇴 권고. round 2-2 설계 재논의 필요."

### (b) 실행 및 보고

```bash
# API 서버 먼저 띄움
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 진단 실행
python -m src.diagnostics.diagnose_horizon_accuracy
```

실행 후 stdout에 다음을 보여줘:
- 실행 시간
- N_SAMPLES 중 성공/실패 카운트
- 핵심 표 (단기/장기 RMSE, MAE, MAPE, 비율)
- 최종 평결 (PASS/MARGINAL/FAIL)

## 검증 방법

1. **재현성**: 같은 RANDOM_SEED로 2번 돌려서 결과 완전 동일한지 확인
2. **데이터 정상성**: n_samples_succeeded ≥ 0.9 * N_SAMPLES (실패율 10% 이내)
3. **시각화 sanity check**: step_wise_rmse.png가 우상향(또는 적어도 비감소) 곡선이어야 정상. 만약 평탄하거나 우하향이면 코드 버그 의심
4. **수동 sanity check**: sample_trajectories.png에서 무작위 6개 샘플 중 적어도 절반은 예측이 실측의 일반적 모양을 따라가야 함

## 주의사항

- **API 서버 별도 띄워야 함**: 진단 스크립트가 HTTP 요청 보내니까. README에 두 줄 명령으로 안내.
- **API rate limit 없음**: 우리 서버는 dev라 200개 요청 빠르게 보내도 됨. 단 요청 사이 sleep 안 넣어도 되는지 확인하고, 필요하면 0.05s 정도 추가.
- **history 24개의 timestamp 엄격 일치**: round 2-1 검증이 매우 엄격하니까 (`start_time - timedelta(hours=24-i) for i in range(24)`), 페이로드 만들 때 정확히 맞춰야 422 안 남.
- **forecast의 timestamp도 동일**: `start_time + timedelta(hours=i) for i in range(48)`.
- **실측값 매칭**: 예측은 step 1~48 (timestamps start_time + 0h ~ +47h), 실측도 같은 timestamp에서 추출. region 필터 잊지 말 것.
- **MAPE 계산 안전성**: `|actual| < threshold (예: 1.0 MWh)`이면 MAPE 계산에서 제외. 야간 시간대 0 근처 값들에서 폭주 방지.
- **N_SAMPLES=200 시간**: 샘플당 ~50ms (round 2-1 측정값) → 10초. 단 행이 많으면 CSV 로딩만 10초 걸릴 수 있음. 전체 1분 안쪽이면 OK.
- **claude_share 복사**: 진단 스크립트도 모듈이라 자기 자신 복사.

## 작업 끝나면 알려줘야 할 것

1. 추가된 파일 목록
2. 실행 stdout 전체
3. 핵심 표 (단기/장기 RMSE, MAE, MAPE, 비율, 평결)
4. 3장의 시각화 (이미지 첨부 또는 경로)
5. JSON, MD 보고서 경로
6. **너의 추천**: 결과를 보고 (b-1) 진행이 안전한지, (b-2)로 후퇴해야 하는지 한 줄 의견
