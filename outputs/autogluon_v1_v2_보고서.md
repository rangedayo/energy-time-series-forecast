# AutoGluon v1/v2 실험 보고서 — XGBoost와의 비교

> **결론 요약**: AutoGluon은 XGBoost보다 약 2배 더 큰 오차를 보였다. v2의 트랜스포머 추가 실험은 성능 개선에 기여하지 못했다(앙상블 가중치 0%). 본 데이터(태양광 발전량 예측)는 외생변수(일사량·전운량 등)에 의한 즉시적 함수 관계가 지배적이라 트리 모델이 구조적으로 유리하며, 직접 튜닝한 XGBoost가 AutoML 자동화보다 우수했다.

**작성일**: 2026-05-20
**데이터**: 전국 17개 지역 시간단위 태양광 발전량 (train: 2018-2022, test: 2023-01 ~ 2023-03)
**평가 지표**: MAE (MWh), 단순평균 / 가중평균 / 전남 단일

---

## 1. 실험 개요

본 실험의 동기는 **"AutoML 도구가 직접 튜닝한 XGBoost를 능가할 수 있는가?"** 였다. AutoGluon v1으로 베이스라인을 확립한 후, v1의 약점(RecursiveTabular 누적오차, 트랜스포머 미시도)을 개선한 v2를 별도 진행했다.

| 항목 | v1 | v2 |
|---|---|---|
| 모델 선택 | `preset="best_quality"` (자동) | `hyperparameters=` 수동 명시 |
| 시도된 모델 | 10개 (Chronos 계열 다수) | 7개 (트랜스포머 집중) |
| 트랜스포머 | TFT 1개 (9분 학습) | TFT × 2 + PatchTST × 2 (각 50 epoch) |
| RecursiveTabular | 포함 | **제외** (누적오차 우려) |
| num_val_windows | 3 | 5 (분포 shift 완화) |
| 학습 시간 | 18분 (8h 한도) | 57분 (3h 한도) |

---

## 2. 핵심 결과

### 2.1 종합 성능 (테스트 MAE, MWh)

| 지표 | XGBoost | AutoGluon v1 | AutoGluon v2 | v2 vs XGBoost |
|---|---|---|---|---|
| **Simple Avg MAE** | **9.61** | 18.21 | 18.10 | +88% (악화) |
| **Weighted MAE** | ~29.55 | 57.53 | 56.89 | +93% (악화) |
| **전남 MAE** | **90.04** | 143.45 | 141.56 | +57% (악화) |

> **핵심**: AutoGluon은 모든 지표에서 XGBoost 대비 약 **2배 큰 오차**를 보였다.
> v1 → v2 개선폭은 1.1%로 미미함.

### 2.2 AutoGluon Leaderboard (Validation -MASE, 낮을수록 좋음)

| 모델 | v1 | v2 |
|---|---|---|
| WeightedEnsemble | **-0.496** (최고) | **-0.599** (최고) |
| RecursiveTabular | -0.546 | (제외) |
| DirectTabular | -0.595 | -0.633 |
| SeasonalNaive | -0.721 | -0.840 |
| AutoETS | -1.709 | -1.746 |
| TemporalFusionTransformer | -1.938 | -1.838 ~ -1.899 |
| PatchTST | — | **-3.057 ~ -3.121** (최악) |

### 2.3 앙상블 가중치 — 트랜스포머는 결국 0%

```
v1 앙상블: RecursiveTabular(0.49) + DirectTabular(0.45) + SeasonalNaive(0.06)
v2 앙상블: DirectTabular(0.73) + SeasonalNaive(0.27)
         → TFT, PatchTST, AutoETS 모두 0% (제외됨)
```

v2에서 **트랜스포머 4개를 추가 학습했으나 앙상블에 단 하나도 포함되지 않았다.** AutoGluon이 "도움 안 됨"으로 판단한 것이며, 트랜스포머 추가 가설은 기각.

---

## 3. 왜 이런 결과가 나왔는가

### 3.1 태양광 발전은 "즉시적 함수 매핑 문제"

```
발전량(t) ≈ f(일사량(t), 전운량(t), 기온(t)) + 작은 잡음
```

본 데이터에선 **현재 시점의 기상 변수가 거의 모든 것을 결정**한다. 과거 자기 패턴(autoregression)의 정보 가치가 낮다.

- **트리 모델(XGBoost, DirectTabular)**: 입력 → 출력의 직접 매핑에 최적화 → 구조적 유리
- **트랜스포머(TFT)**: covariates를 잘 다루지만 시간 맥락 통합이 강점 → 본 문제에선 강점 발휘 못 함
- **PatchTST**: target 자기 패턴 위주 학습 → 외생변수 활용도 낮음 → 최악 성능 (-3.05~)

### 3.2 XGBoost가 AutoGluon DirectTabular보다 나은 이유

같은 트리 계열인데도 격차가 있다:
- **Feature engineering**: XGBoost는 도메인 지식 기반 24개 feature(시간·기상 파생)를 직접 설계
- **하이퍼파라미터**: `n_estimators=500, max_depth=6, lr=0.05` 등 데이터에 맞춰 튜닝
- **AutoGluon DirectTabular**: 시계열 자동 처리 + 일반화된 lag/window feature → 본 문제 특성 반영 부족

**자동화의 일반화 vs 수동 튜닝의 특화** 구도이며, 본 데이터에선 후자가 명확히 이김.

### 3.3 전남이 모든 모델의 공통 약점

| 모델 | 전남 MAE |
|---|---|
| XGBoost | 90.04 |
| AutoGluon v1 | 143.45 |
| AutoGluon v2 | 141.56 |

전남은 데이터에서 가중치 30%를 차지하며 평균 발전량(167 MWh)이 가장 큰 지역. **모델 종류와 무관하게 가장 어려운 지역**으로, 이는 모델 문제가 아닌 **데이터 분포 shift(2022 하반기 → 2023) 또는 발전소 설비 변화** 같은 외부 요인 가능성이 높다.

---

## 4. v1 → v2 개선 가설 검증

| 가설 | 검증 결과 |
|---|---|
| **RecursiveTabular 제외 → 누적오차 감소** | ✅ 부분 검증. Validation은 v1(-0.496)이 더 좋았으나 테스트 MAE는 v2가 1% 개선. 누적오차 영향이 테스트 시점에서 실제로 존재함을 시사 |
| **트랜스포머 추가(TFT, PatchTST)** | ❌ 기각. 앙상블 가중치 0%. 본 데이터에 부적합한 모델 구조 |
| **num_val_windows 3 → 5 (분포 shift 완화)** | △ 효과 불명확. 테스트 MAE 1% 개선의 일부 기여 추정 |
| **학습 시간 8h → 3h, max_epochs 50** | ✅ 효율적. 57분으로 충분한 학습 |

---

## 5. 최종 결론

1. **본 문제(태양광 발전 예측)에서는 직접 튜닝한 XGBoost가 AutoGluon보다 약 2배 우수**하다. 자동화 도구가 만능이 아님을 확인.

2. **트랜스포머 계열은 본 데이터에 구조적으로 부적합**하다. 외생변수 의존이 큰 즉시적 매핑 문제에서는 트리 모델이 유리하다.

3. **AutoGluon의 가치는 "다양한 모델을 빠르게 비교"하는 데 있다.** 절대 성능은 도메인 특화 튜닝보다 떨어질 수 있으나, 모델 선택의 합리적 근거를 제공한다는 점에서 의미가 있다.

4. **다음 개선 방향은 AutoGluon이 아닌 XGBoost 자체에 있다.**
   - 전남 단일 모델 분리 (지역별 분포 차이 반영)
   - 데이터 분포 shift 처리 (2022 하반기 → 2023)
   - Feature engineering 추가 (시간대별 평균 발전량, 기상-발전 lag 관계)

---

## 부록 A. 환경 트러블슈팅

본 실험에서 환경 문제로 인한 모델 실패가 다수 발생했다. 재현성을 위해 기록.

### A.1 v1: Chronos 계열 4개 모델 일괄 실패

```
ImportError: cannot import name 'size_hint' from 'torch.fx.experimental.symbolic_shapes'
```

- **실패 모델**: Chronos2, Chronos2SmallFineTuned, ChronosWithRegressor[bolt_small], DeepAR
- **원인**: Colab Enterprise 기본 torch 2.10.0 + cu128과 chronos/transformers 라이브러리의 ABI 불일치
- **조치**: v1에선 미해결 (그대로 학습 진행). 살아남은 모델 6개로 결과 도출.

### A.2 v2: TFT/PatchTST 환경 충돌

```
AttributeError: partially initialized module 'torchvision' has no attribute 'extension'
(most likely due to a circular import)
```

- **실패 모델 (초기)**: TemporalFusionTransformer × 2, PatchTST × 2
- **원인**: torchvision 0.25.0+cu128과 torch 2.9.1 (AutoGluon 다운그레이드 결과)의 빌드 ABI 불일치
- **조치**:
  1. Colab Enterprise 런타임 완전 삭제 → 신규 생성
  2. `autogluon.timeseries==1.5.0` 설치 (`-U` 옵션 제거)
  3. `torchvision==0.24.1`, `torchaudio==2.9.1` 명시적 다운그레이드
  4. 세션 재시작 후 환경 검증
- **결과**: TFT/PatchTST 4개 모델 모두 정상 학습 성공

### A.3 핵심 교훈

- AutoGluon의 deep learning 모델은 PyTorch 생태계 버전 일관성에 매우 민감
- Colab Enterprise 기본 환경(torch 2.10) ≠ AutoGluon 요구사항(torch <2.10) → 항상 충돌 위험
- **학습 시작 전 환경 검증 셀(import 체인 테스트)을 분리해 사전 차단** 필요

### A.4 적용된 hyperparameters 키 오류

v2 초안에서 사용한 `DirectTabular`의 `tabular_hyperparameters` 키는 AutoGluon 1.5.0에서 미지원:

```python
# 무효 (경고 발생, 3개 모두 동일 모델로 학습됨)
"DirectTabular": [
    {"tabular_hyperparameters": {"GBM": {}}},
    {"tabular_hyperparameters": {"XGB": {}}},
    {"tabular_hyperparameters": {"CAT": {}}},
]

# 수정 (단일 인스턴스로 변경)
"DirectTabular": {}
```

---

## 부록 B. 실험 환경

- **플랫폼**: Google Colab Enterprise
- **하드웨어**: T4 GPU (22GB), 4 CPU, 15.6GB RAM
- **AutoGluon**: 1.5.0
- **PyTorch**: 2.9.1 + CUDA 12.8 (v2), 2.10.0 (v1)
- **학습 데이터**: 872,382행 × 17개 시계열
- **테스트 평가 기간**: 2023-01-01 ~ 2023-03-01 (59 steps)

---

## 부록 C. 참고 파일

- `national_autogluon_results.json` — v1 결과 (4.4 KB)
- `national_autogluon_v2_results.json` — v2 결과 (5.7 KB)
- `national_xgb_results.json` — XGBoost 베이스라인
- `leaderboard.csv` (v1, v2 각각) — 모델별 validation 점수
- `national_autogluon_v2_region_comparison.csv` — 지역별 MAE 비교
- `solar_autogluon_v1.ipynb`, `solar_autogluon_v2.ipynb` — 실행 노트북
