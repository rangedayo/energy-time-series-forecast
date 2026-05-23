# Claude Code 실행 프롬프트 — ESS 시뮬레이터 재설계 v2
# 학술적 모델 비교용 ESS 시뮬레이션 환경 구축
# (변경사항: LSTM 시나리오 제거, claude_share 자동 복사 강제)

---

## 🔰 프로젝트 컨텍스트

전국 17개 시도 태양광 발전량 예측 ML 시스템의 ESS 시뮬레이션 단계를 재설계한다.
모델 학습(TASK A~F)과 행동 테스트(TASK E)는 완료된 상태이며,
**ESS 시뮬레이션(TASK G)과 최종 리포트(TASK H)만 재설계 대상**이다.

LSTM은 학습 완료했으나 **후속 비교에 사용하지 않는다**. 모든 ESS 시뮬 시나리오에서
LSTM 관련 코드, 데이터 로드, 비교 항목을 제외한다. 비교 대상은 **XGBoost 단일 모델**과
**합성 정책(naive / lookahead / oracle / 노이즈 주입)**으로 충분하다.

### 재설계 목적: (A) 학술적 모델 비교용 환경

본 시뮬레이션은 **실제 운영값 추정이 아닌 모델 간 상대 비교**가 목적이다.
세 가지 원칙을 따른다.

1. **동일한 환경에 모든 정책 적용** — 환경 자체의 비현실성은 모든 정책에 동일 작용, 정책 간 차이는 보존.
2. **외부 데이터 도입 최소화** — KPX 실데이터, 통계청 데이터 등은 도입하지 않음. 너희가 이미 가진 데이터에서 비례 추정.
3. **시뮬레이터의 한계를 검증 장치로 보완** — perfect foresight, sensitivity 분석 등으로 결과 해석 가능성 확보.

---

## 📦 산출물 공유 규칙 (claude_share)

이 프로젝트는 `extended_metrics.py`, `distribution_shift_check.py` 등에서
관행적으로 사용해 온 **claude_share 자동 복사 패턴**을 따른다.

### 규칙

**각 스크립트 종료 직전**, 그 스크립트가 **새로 만들거나 수정한 모든 파일**을
`claude_share/` 폴더로 복사한다. 복사 대상은 다음 4종을 포함한다:

1. 스크립트 자기 자신 (`__file__`)
2. 생성된 JSON/CSV 결과 파일
3. 생성된 PNG/HTML 시각화 파일
4. 생성된 MD 리포트 파일

### 구현 패턴 (모든 스크립트에 적용)

```python
import shutil
from pathlib import Path

SHARE_DIR = Path("claude_share")

# ... 메인 로직 ...

# 스크립트 종료 직전
print(f"{ts()} claude_share 복사 중...")
SHARE_DIR.mkdir(exist_ok=True)
for src, dst in [
    (Path(__file__),      SHARE_DIR / Path(__file__).name),
    (OUT_JSON,            SHARE_DIR / OUT_JSON.name),
    (OUT_PNG,             SHARE_DIR / OUT_PNG.name),
    # ... 그 스크립트가 만든 모든 파일 ...
]:
    if Path(src).exists():
        shutil.copy2(src, dst)
        print(f"   → {dst}")
```

### 적용 범위

본 프롬프트의 **모든 TASK 스크립트(G-1~G-5, H)에 동일 적용**한다.
한 스크립트가 여러 파일을 만들면 그 파일 전부를 복사한다.
복사 후에는 stdout에 복사된 파일 목록을 출력한다.

이 규칙은 누락 시 작업 완료로 인정하지 않는다.

---

## ✅ 기존 시뮬레이터의 6가지 문제 (모두 해결 대상)

### 문제 1: 단위 버그 잔재
`claude_code_prompt.md` 시점에 `ESS_CAPACITY_MWH=1.0` (kWh 단위 주석)으로 시작했고,
이후 500.0 (MWh)으로 수정됐으나 두 버전이 문서에 공존. 단위 버그 수정 흔적을
정리하고 현재 버전만 남긴다.

### 문제 2: 결정 시점 = 정보 가용 시점
```python
# 기존 코드 (ess_simulation_national.py 73~75줄)
actual_net = gen - DEMAND_MWH_PER_HOUR
pred_net   = pred - DEMAND_MWH_PER_HOUR
decision   = actual_net if naive else pred_net
```
t 시점에 t 시점 정보로 결정한다. 예측 모델의 시간적 우위(lookahead)를 활용 못 한다.
→ **예측의 가치가 시뮬에 반영되지 않는 구조적 결함**.

### 문제 3: shortage_count 비대칭 카운팅
방전 분기 안에서만 부족이 카운트된다. 예측이 양수(충전 분기)인데 실제로는 음수면
부족이 누락된다. 또한 부족의 심각도(1 MWh vs 49 MWh)를 binary로만 셈.

### 문제 4: 모든 지역에 동일 ESS 파라미터
500 MWh, 50 MWh/h 수요가 전국 17개 지역에 똑같이 적용된다. 발전량이 100배 차이 나는
지역에 같은 ESS는 비현실적. 결과 해석에 노이즈 다발.

### 문제 5: 24시간 고정 수요
새벽 2시도 50 MWh, 한낮 12시도 50 MWh. 한낮 발전 피크는 항상 낭비로 잡힘.

### 문제 6: 단일 지표 의존
`ess_score` 하나로 정책 비교. 부족의 심각도, 자급률 등 다각도 지표 부재.

---

## 🚧 신규 작업 (TASK G 재설계)

### 작업 단위
- **TASK G-1**: ESS 파라미터 모듈화 + 지역별 차등 + 시간대별 수요 패턴
- **TASK G-2**: 결정 정책 함수화 (naive / lookahead / perfect_foresight / noisy)
- **TASK G-3**: 시뮬레이터 본체 재작성 (지표 다양화 포함)
- **TASK G-4**: 메인 비교 실행 (정책 × XGBoost 시나리오 매트릭스)
- **TASK G-5**: Sensitivity 분석 (예측 정확도 → ESS 가치 곡선)
- **TASK H 재실행**: 최종 리포트 업데이트 (LSTM 제외)

### 산출 파일 (모두 신규, `*_v2_*` 네이밍)
```
src/simulation/ess_config_v2.py             # 파라미터 모듈
src/simulation/ess_policy_v2.py             # 결정 정책 함수들
src/simulation/ess_simulation_v2.py         # 시뮬레이터 본체
src/simulation/ess_sensitivity_v2.py        # sensitivity 분석
src/reporting/final_report_v2.py            # 최종 리포트 (LSTM 제외)
outputs/ess_v2_simulation_results.json      # 메인 결과
outputs/ess_v2_sensitivity_results.json     # sensitivity 결과
outputs/ess_v2_comparison.png               # 정책별 4지표 비교
outputs/ess_v2_region_breakdown.png         # 지역별 결과 히트맵
outputs/ess_v2_sensitivity_curve.png        # sensitivity 곡선
outputs/national_final_report_v2.md         # 새 리포트
```

기존 `ess_simulation_national.py`, `national_ess_simulation_results.json`,
`national_final_report.md` 등은 **보존**한다(`CLAUDE.md` 규칙).
새 결과는 모두 `*_v2_*` 네이밍으로 분리.

**위 모든 파일은 생성·수정 시 claude_share/로 자동 복사.**

---

## TASK G-1 — ESS 파라미터 모듈 (`src/simulation/ess_config_v2.py`)

### 1-1. 기본 파라미터 (상수)

```python
"""
ESS 시뮬레이션 파라미터 (v2)

본 시뮬레이션은 통제된 모델 비교 환경이며 실제 운영값 추정이 목적이 아니다.
파라미터는 산업 통상 범위 내에서 선정했으며, 절대값 해석이 아닌
정책(naive/lookahead/oracle) 간 상대 비교에만 유효하다.
"""

# 전국 합산 기준 ESS 파라미터 (17개 광역 단위 시뮬용)
# 근거: 한국에너지공단 신재생 REC 가이드라인상 ESS 방전출력은 태양광 설비용량의
# 70% 이내. 산업부 2025년 호남·제주 ESS 입찰 평균 70 MW/곳, 4시간 저장 기준.
TOTAL_ESS_CAPACITY_MWH = 500.0 * 17     # 17개 광역 합산 ESS 용량
TOTAL_DEMAND_MWH_PER_H = 50.0 * 17      # 17개 광역 합산 평균 시간당 수요
TOTAL_CHARGE_RATE_MAX  = 100.0 * 17     # 합산 충전 속도 상한
TOTAL_DISCHARGE_RATE_MAX = 100.0 * 17   # 합산 방전 속도 상한

SOC_MIN     = 0.20  # DOD 60% 보호. LFP 배터리 수명-가용량 균형의 산업 통상값.
SOC_MAX     = 0.80
SOC_INIT    = 0.50
EFFICIENCY  = 0.90  # 시스템 RTE (Round-Trip Efficiency) 90%, 산업 통상값
                    # (셀 단 95%가 아닌 시스템 단 90%로 보수 조정)
```

### 1-2. 시간대별 수요 패턴

```python
import numpy as np

# 한국 일반 전력 부하의 정성적 패턴 (KPX 통계 정성적 참조)
# 절대값 정확도 아닌 시간 변동성 도입이 목적.
# 평균 = 1.0 으로 정규화, 시간당 평균 수요에 곱해서 사용.
HOURLY_LOAD_FACTOR = np.array([
    0.70, 0.65, 0.60, 0.60, 0.65, 0.75,  # 0~5시: 새벽 저점
    0.85, 0.95, 1.05, 1.15, 1.20, 1.25,  # 6~11시: 오전 상승
    1.25, 1.20, 1.15, 1.15, 1.10, 1.15,  # 12~17시: 한낮 피크 후 둔덕
    1.25, 1.20, 1.10, 1.00, 0.90, 0.80,  # 18~23시: 저녁 피크 후 하강
])
assert abs(HOURLY_LOAD_FACTOR.mean() - 1.0) < 0.01, "부하 패턴 평균 정규화 깨짐"
```

### 1-3. 지역별 파라미터 빌더

```python
import pandas as pd

def build_region_params(train_df: pd.DataFrame) -> dict:
    """
    train 데이터의 지역별 평균 발전량 비중으로 ESS 파라미터를 차등 분배.

    근거: 발전 인프라가 큰 지역은 ESS 용량도 크고 수요도 크다는 자연스러운 가정.
    너희 데이터에서 직접 도출되므로 외부 자료 의존성 없음.

    Returns:
        dict[region_name -> dict[param_name -> float]]
    """
    region_mean_gen = train_df.groupby("region")["power_mwh"].mean()
    total_mean_gen = region_mean_gen.sum()

    params = {}
    for region, mean_gen in region_mean_gen.items():
        weight = float(mean_gen / total_mean_gen)
        params[region] = {
            "ess_capacity_mwh":    TOTAL_ESS_CAPACITY_MWH * weight,
            "demand_mwh_per_h":    TOTAL_DEMAND_MWH_PER_H * weight,
            "charge_rate_max":     TOTAL_CHARGE_RATE_MAX * weight,
            "discharge_rate_max":  TOTAL_DISCHARGE_RATE_MAX * weight,
            "weight":              weight,
        }
    return params


def get_demand_at_hour(base_demand: float, hour: int) -> float:
    """시간대별 수요 = 평균 수요 × 부하 패턴."""
    return base_demand * HOURLY_LOAD_FACTOR[hour]
```

### 1-4. 검증 출력 (__main__ 블록)
스크립트 직접 실행 시 다음을 출력 후 종료:
- 17개 지역 파라미터 표 (region | weight | ess_capacity | demand)
- weight 합산 = 1.0 검증 (assert)
- 부하 패턴 평균 = 1.0 검증 (assert)

### 1-5. claude_share 복사
이 스크립트는 **순수 모듈**이라 단독 산출물이 없다. 따라서 **자기 자신만 복사**:
```python
import shutil
from pathlib import Path

SHARE_DIR = Path("claude_share")
SHARE_DIR.mkdir(exist_ok=True)
shutil.copy2(__file__, SHARE_DIR / Path(__file__).name)
print(f"   → {SHARE_DIR / Path(__file__).name}")
```

---

## TASK G-2 — 결정 정책 함수 (`src/simulation/ess_policy_v2.py`)

### 2-1. 정책 인터페이스

각 정책은 동일한 시그니처를 따른다:

```python
def policy_xxx(
    t: int,               # 현재 시점 인덱스
    actual: np.ndarray,   # 실측 발전량 (전 구간)
    predicted: np.ndarray,# 예측 발전량 (전 구간)
    soc: float,           # 현재 SOC
    demand_t: float,      # t 시점 수요 (시간대별)
    params: dict,         # ESS 파라미터 (지역별)
) -> dict:
    """
    Returns:
        {
            "soc_target_high": float,  # 충전 분기에서 도달 목표 SOC (SOC_MAX 이하)
            "soc_target_low":  float,  # 방전 분기에서 유지 하한 SOC (SOC_MIN 이상)
        }
    """
```

정책은 **SOC 목표만 결정**하고, 실제 충방전 실행은 시뮬레이터 본체가 `actual` 기준으로 수행한다.
이게 "충방전은 항상 실측 기준, 예측은 강도 조절에만 사용"의 구현이다.

### 2-2. 정책 4가지

```python
import numpy as np

def policy_naive(t, actual, predicted, soc, demand_t, params):
    """
    가장 단순한 베이스라인. "다음 시점도 지금과 같다"는 persistence 가정.
    예측을 전혀 사용하지 않음. 사실상 lag1 가정.
    """
    return {
        "soc_target_high": 0.80,
        "soc_target_low":  0.20,
    }


def policy_lookahead(t, actual, predicted, soc, demand_t, params, horizon=1):
    """
    예측을 기반으로 다음 N시점을 보고 SOC 목표를 동적 조정.

    - 다음 시점이 잉여 예상 → 지금 덜 채워서 공간 남김
    - 다음 시점이 부족 예상 → 지금 덜 빼서 비축

    horizon=1이 기본. 멀티스텝은 후속 과제.
    """
    n = len(predicted)
    next_t = min(t + horizon, n - 1)
    forecast_next = float(predicted[next_t])
    forecast_net = forecast_next - demand_t

    if forecast_net > 0:
        return {"soc_target_high": min(0.80, soc + 0.10), "soc_target_low": 0.20}
    else:
        return {"soc_target_high": 0.80, "soc_target_low": min(0.80, 0.20 + 0.10)}


def policy_perfect_foresight(t, actual, predicted, soc, demand_t, params, horizon=1):
    """
    Oracle. 예측 대신 실측값을 lookahead로 사용 → 예측 오차 0인 가상 케이스.
    ESS 운영 효율의 이론 상한을 정의한다.
    """
    return policy_lookahead(t, actual, actual, soc, demand_t, params, horizon)


def policy_lookahead_noisy(t, actual, predicted, soc, demand_t, params,
                            noise_level=0.0, horizon=1, rng=None):
    """
    Sensitivity 분석용. 실측값에 합성 노이즈를 주입한 예측으로 lookahead.
    """
    rng = rng or np.random.default_rng(42)
    n = len(actual)
    next_t = min(t + horizon, n - 1)
    sigma = float(np.std(actual)) * noise_level
    noisy = actual[next_t] + rng.normal(0, sigma)
    fake_pred = predicted.copy()
    fake_pred[next_t] = noisy
    return policy_lookahead(t, actual, fake_pred, soc, demand_t, params, horizon)
```

### 2-3. 검증 (__main__ 블록)
각 정책에 임의 입력 넣고 반환값이 `SOC_MIN ≤ low ≤ high ≤ SOC_MAX` 범위인지 assert.

### 2-4. claude_share 복사
이 스크립트도 모듈이라 자기 자신만 복사.
```python
shutil.copy2(__file__, SHARE_DIR / Path(__file__).name)
```

---

## TASK G-3 — 시뮬레이터 본체 (`src/simulation/ess_simulation_v2.py`)

### 3-1. run_simulation 함수

```python
import numpy as np
import pandas as pd

def run_simulation(
    actual: np.ndarray,
    predicted: np.ndarray,
    hours: np.ndarray,            # 각 시점의 시간(0~23)
    params: dict,                 # 단일 지역 또는 전국 ESS 파라미터
    policy_fn,                    # 정책 함수
    policy_kwargs: dict = None,
) -> dict:
    """
    한 (지역, 정책) 조합에 대한 단일 시뮬레이션 실행.
    """
    policy_kwargs = policy_kwargs or {}
    from src.simulation.ess_config_v2 import (
        SOC_MIN, SOC_MAX, SOC_INIT, EFFICIENCY,
        get_demand_at_hour,
    )

    n = len(actual)
    soc = SOC_INIT
    total_curtailment = 0.0
    total_shortage_mwh = 0.0
    total_demand_mwh = 0.0
    shortage_list = []
    charge_cycles = 0.0
    discharge_cycles = 0.0

    cap = params["ess_capacity_mwh"]
    base_demand = params["demand_mwh_per_h"]
    chg_max = params["charge_rate_max"]
    dis_max = params["discharge_rate_max"]

    for i in range(n):
        gen = float(actual[i])
        h = int(hours[i])
        demand_t = get_demand_at_hour(base_demand, h)
        total_demand_mwh += demand_t

        targets = policy_fn(i, actual, predicted, soc, demand_t, params, **policy_kwargs)
        soc_target_high = targets["soc_target_high"]
        soc_target_low  = targets["soc_target_low"]

        actual_net = gen - demand_t

        if actual_net > 0:
            max_storable = max(0.0, (soc_target_high - soc) * cap / EFFICIENCY)
            charge_amount = min(actual_net, chg_max, max_storable)
            soc += charge_amount * EFFICIENCY / cap
            charge_cycles += charge_amount / cap
            total_curtailment += actual_net - charge_amount
        else:
            needed = -actual_net
            max_dischargeable = max(0.0, (soc - soc_target_low) * cap * EFFICIENCY)
            discharge_amount = min(needed, dis_max, max_dischargeable)
            soc -= discharge_amount / (cap * EFFICIENCY)
            discharge_cycles += discharge_amount / cap

            shortfall = max(0.0, demand_t - (gen + discharge_amount))
            if shortfall > 0:
                shortage_list.append(shortfall)
                total_shortage_mwh += shortfall

    total_gen = float(np.sum(actual))
    curtailment_rate = total_curtailment / max(total_gen, 1e-10) * 100.0
    self_consumption_rate = 100.0 - curtailment_rate
    self_sufficiency_rate = (1.0 - total_shortage_mwh / max(total_demand_mwh, 1e-10)) * 100.0
    battery_cycles = (charge_cycles + discharge_cycles) / 2.0

    shortage_count = len(shortage_list)
    ess_score = (1.0 - curtailment_rate / 100.0) * (1.0 - shortage_count / n) * 100.0

    return {
        # 신규 지표 (국제 표준)
        "self_consumption_rate_pct": round(self_consumption_rate, 2),
        "self_sufficiency_rate_pct": round(self_sufficiency_rate, 2),
        "total_shortage_mwh": round(total_shortage_mwh, 2),
        "mean_shortage_mwh": round(float(np.mean(shortage_list)), 2) if shortage_list else 0.0,
        "max_shortage_mwh":  round(float(np.max(shortage_list)), 2)  if shortage_list else 0.0,

        # 기존 호환 지표
        "curtailment_rate_pct": round(curtailment_rate, 2),
        "shortage_count": int(shortage_count),
        "battery_cycles": round(battery_cycles, 2),
        "ess_score": round(ess_score, 2),

        # 진단용
        "total_curtailment_mwh": round(total_curtailment, 2),
        "total_demand_mwh": round(total_demand_mwh, 2),
        "total_gen_mwh": round(total_gen, 2),
        "n_hours": int(n),
    }
```

### 3-2. 지표 해석 (코드 주석에 박을 것)
```
self_consumption_rate (자가소비율, %) — 발전한 전기 중 활용한 비율. 높을수록 좋음.
self_sufficiency_rate (자급률, %) — 수요 중 자체 공급으로 충당한 비율. 높을수록 좋음.
total_shortage_mwh — 부족의 총량(절대값). 0에 가까울수록 좋음.
mean_shortage_mwh — 부족 발생 시 평균 강도. 진단용.
max_shortage_mwh — 최악 부족 시점. 극단 시나리오 대응력.
```

---

## TASK G-4 — 메인 비교 실행 (TASK G-3 스크립트의 main 함수)

### 4-1. 시나리오 매트릭스

LSTM 제외. **XGBoost 단일 모델 × 정책 3개 + Oracle = 4개 시나리오**:

```
시나리오                | 정책                       | 사용 데이터
───────────────────────┼───────────────────────────┼─────────────────
1. naive_baseline      | policy_naive              | actual만 사용
2. xgb_lookahead       | policy_lookahead          | XGBoost 예측 사용
3. oracle              | policy_perfect_foresight  | actual을 예측 자리에
4. xgb_no_lookahead    | (기존 시뮬 로직 재현)      | XGBoost 예측 + t시점 결정
```

**4번 `xgb_no_lookahead`를 추가한 이유**: 기존 시뮬과 새 시뮬의 차이를 직접 측정.
"lookahead 도입이 얼마나 개선했나"를 분리해서 보여주는 효과.

```python
def policy_xgb_no_lookahead(t, actual, predicted, soc, demand_t, params):
    """
    기존 시뮬 재현용. t 시점 예측값으로 결정하되 lookahead 없음.
    pred_net > 0이면 적극 충전, <0이면 적극 방전 (기존과 동일).
    """
    pred_net = float(predicted[t]) - demand_t
    if pred_net > 0:
        return {"soc_target_high": 0.80, "soc_target_low": 0.20}
    else:
        return {"soc_target_high": 0.80, "soc_target_low": 0.20}
    # → 사실상 naive와 같은 SOC 범위. 결정 분기 자체에는 영향 없음.
    # 차이는 시뮬레이터 본체의 actual_net 사용으로 나타남.
```

> **주의**: `policy_xgb_no_lookahead`는 새 시뮬 구조 안에서 기존 동작을 근사한
> 재현이며 100% 동일하지 않다. 이 한계를 리포트에 명시.

### 4-2. 지역별 × 시나리오별 실행

```python
# 의사코드
xgb_df = pd.read_csv(XGB_PREDICTIONS)  # national_xgb_predictions.csv
train_df = pd.read_csv(NATIONAL_TRAIN_FEATURES)
region_params_map = build_region_params(train_df)

scenarios = {
    "naive_baseline":     (policy_naive,             "actual"),
    "xgb_lookahead":      (policy_lookahead,         "predicted"),
    "oracle":             (policy_perfect_foresight, "actual"),
    "xgb_no_lookahead":   (policy_xgb_no_lookahead,  "predicted"),
}

results = {"scenarios": {}, "regions": {}}

for region in sorted(xgb_df["region"].unique()):
    r_df = xgb_df[xgb_df["region"] == region].sort_values("timestamp")
    hours = pd.to_datetime(r_df["timestamp"]).dt.hour.values
    actual_arr = r_df["actual"].values
    pred_arr   = r_df["predicted"].values
    params     = region_params_map[region]

    results["regions"][region] = {}
    for scen_name, (policy_fn, pred_source) in scenarios.items():
        pred_input = actual_arr if pred_source == "actual" else pred_arr
        results["regions"][region][scen_name] = run_simulation(
            actual_arr, pred_input, hours, params, policy_fn,
        )
```

### 4-3. 집계 3가지

```python
# 1. 단순 평균 (17개 평등)
# 2. 가중 평균 (발전량 비중)
# 3. 전국 합산 시뮬 (시점별 합산 후 단일 시뮬)
```

세 집계를 모두 `results["aggregates"]` 아래에 저장.

### 4-4. 시각화

1. **`outputs/ess_v2_comparison.png`** — 4 시나리오 × 4 지표 (자가소비율, 자급률, 평균 부족 심각도, 사이클수)
2. **`outputs/ess_v2_region_breakdown.png`** — 17지역 × 4시나리오 히트맵 (자급률 기준)

### 4-5. 결과 JSON 구조

```json
{
  "config": {
    "ess_capacity_total_mwh": 8500.0,
    "demand_total_mwh_per_h": 850.0,
    "efficiency": 0.90,
    "soc_range": [0.20, 0.80],
    "load_pattern": "정성적 한국 부하 곡선 (정규화)",
    "model": "XGBoost (national v2, power_diff 포함)"
  },
  "region_params": {region: {weight, ess_capacity_mwh, demand_mwh_per_h, ...}},
  "regions": {region: {scenario: metrics}},
  "aggregates": {
    "simple_avg":   {scenario: metrics},
    "weighted_avg": {scenario: metrics},
    "national_sum": {scenario: metrics}
  }
}
```

### 4-6. stdout 요약 출력

```
============================================================
[전국 ESS v2 시뮬레이션 결과 — 가중 평균 기준]
============================================================
시나리오                 자가소비율   자급률   평균부족   사이클수
─────────────────────────────────────────────────────────────
naive_baseline             XX.X%    XX.X%   XX.X MWh   XX.X
xgb_no_lookahead           XX.X%    XX.X%   XX.X MWh   XX.X
xgb_lookahead              XX.X%    XX.X%   XX.X MWh   XX.X
oracle                     XX.X%    XX.X%   XX.X MWh   XX.X
─────────────────────────────────────────────────────────────
XGBoost가 Oracle 자급률의 XX.X% 도달
lookahead 도입 효과: 자급률 +X.X pt
============================================================
```

### 4-7. claude_share 복사

```python
print(f"{ts()} claude_share 복사 중...")
SHARE_DIR.mkdir(exist_ok=True)
for src, dst in [
    (Path(__file__),                          SHARE_DIR / "ess_simulation_v2.py"),
    (Path("outputs/ess_v2_simulation_results.json"), SHARE_DIR / "ess_v2_simulation_results.json"),
    (Path("outputs/ess_v2_comparison.png"),          SHARE_DIR / "ess_v2_comparison.png"),
    (Path("outputs/ess_v2_region_breakdown.png"),    SHARE_DIR / "ess_v2_region_breakdown.png"),
]:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"   → {dst}")
```

---

## TASK G-5 — Sensitivity 분석 (`src/simulation/ess_sensitivity_v2.py`)

### 5-1. 목적
"예측 정확도가 얼마나 ESS 가치로 전환되는가?"의 곡선.

### 5-2. 실행 절차

```python
noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5]
rng_seeds = [42, 123, 456]  # 견고성 확보

results = []
for noise in noise_levels:
    for seed in rng_seeds:
        rng = np.random.default_rng(seed)
        # 가중 평균 또는 전국 합산 시뮬로 실행
        sim_result = run_with_synthetic_noise(actual, noise, rng, region_params)
        results.append({
            "noise_level": noise,
            "seed": seed,
            "self_sufficiency_rate": sim_result["self_sufficiency_rate_pct"],
            "self_consumption_rate": sim_result["self_consumption_rate_pct"],
            "ess_score": sim_result["ess_score"],
        })
```

### 5-3. 곡선 위에 XGBoost 점 찍기

```python
# XGBoost의 실제 nMAE를 노이즈 등가로 환산
# (actual.std() 기준 정규화)
nmae_xgb = compute_nmae_from_predictions(xgb_df)
xgb_noise_equiv = nmae_xgb / actual.std()
# 곡선 위 (xgb_noise_equiv, xgb의 자급률) 점을 별표로 마킹
```

### 5-4. 자동 결론 출력

스크립트 종료 시 stdout에:
```
============================================================
[Sensitivity 분석 결론]
============================================================
Oracle (noise=0.0):       자급률 XX.X%
XGBoost 위치 (noise≈X.XX): 자급률 XX.X%
→ XGBoost는 Oracle 대비 XX% 도달
→ 예측 정확도를 50% 더 높여도 자급률은 +X.X pt만 추가
→ 한계는 시뮬레이터 설계 또는 ESS 용량 자체에 있음을 시사
============================================================
```

### 5-5. claude_share 복사

```python
for src, dst in [
    (Path(__file__),                                 SHARE_DIR / "ess_sensitivity_v2.py"),
    (Path("outputs/ess_v2_sensitivity_results.json"), SHARE_DIR / "ess_v2_sensitivity_results.json"),
    (Path("outputs/ess_v2_sensitivity_curve.png"),    SHARE_DIR / "ess_v2_sensitivity_curve.png"),
]:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"   → {dst}")
```

---

## TASK H 재실행 — 최종 리포트 (`src/reporting/final_report_v2.py`)

`outputs/ess_v2_simulation_results.json`과 `outputs/ess_v2_sensitivity_results.json`을 읽어
**기존 `national_final_report.md`를 보존**하고, **새 파일 `outputs/national_final_report_v2.md`** 생성.

LSTM 관련 항목은 **모두 제외**. XGBoost 단일 모델과 정책 비교에만 집중.

### 리포트 구조

```markdown
# 최종 비교 리포트 v2 (ESS 시뮬레이터 재설계)

생성일시: YYYY-MM-DD HH:MM:SS

## 1. 시뮬레이터 재설계 배경
[기존 v1 시뮬의 6가지 문제 → 본 보고서의 해결 방식 요약]

## 2. 환경 설정 (통제된 비교 환경)
- 지역별 차등 ESS 파라미터 (17개 지역, 발전량 비례)
- 시간대별 수요 패턴 (정성적 정규화 곡선)
- 시스템 RTE 90%, DOD 60%
- 4개 정책 (naive / xgb_no_lookahead / xgb_lookahead / oracle)
- 사용 모델: XGBoost 통합 v2 (power_diff 포함) — LSTM은 본 분석 제외

## 3. XGBoost 모델 성능 (Test Set 2023)
[기존 `national_xgb_results.json` 요약, MAE/RMSE/피크/지역별]

## 4. ESS 시뮬레이션 비교 — 새 지표 기반
### 4-1. 단순 평균 (17개 지역 평등)
| 시나리오 | 자가소비율 | 자급률 | 평균 부족 심각도 | 사이클수 |

### 4-2. 가중 평균 (발전량 비중)
[동일 형식]

### 4-3. Oracle 대비 도달률
| 시나리오 | 자급률 / Oracle 자급률 (%) |

### 4-4. lookahead 도입 효과
xgb_no_lookahead → xgb_lookahead 변화량 표

### 4-5. 지역별 ESS 영향 (히트맵 첨부)

## 5. Sensitivity 분석
[곡선 PNG + 자동 출력된 결론]

## 6. 시뮬레이터 한계 (정직한 명시)
본 시뮬은 (A) 통제된 모델 비교 환경이며 다음은 후속 과제로 남긴다:
- 실제 KPX 시간대별 수요 데이터 결합
- REC 가중치 / SMP 가격 반영
- 출력제어 시나리오 모델링
- 멀티스텝 MPC 최적화
- 지역별 ESS 용량 실측 매칭
- xgb_no_lookahead의 기존 시뮬 완전 재현 한계

## 7. 핵심 발견 (포트폴리오 시그니처)
- **MAE ≠ ESS 점수** — XGBoost의 절댓값 정확도와 운영 결정 품질이 분리됨
- **예측 정확도-운영 가치 전환 곡선** — 정확도 개선의 한계 효용 측정
- **지역별 모델 성능 → ESS 영향 차등 전파** — 17개 지역 검증
```

### claude_share 복사

```python
for src, dst in [
    (Path(__file__),                                  SHARE_DIR / "final_report_v2.py"),
    (Path("outputs/national_final_report_v2.md"),     SHARE_DIR / "national_final_report_v2.md"),
]:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"   → {dst}")
```

---

## ⚠️ 전체 공통 규칙

1. **기존 파일 보호**: `CLAUDE.md`에 명시된 파일은 절대 덮어쓰지 말 것. 새 결과는 모두 `*_v2_*` 네이밍.
2. **LSTM 관련 코드 금지**: import, 데이터 로드, 시나리오 추가 모두 금지. XGBoost 단일.
3. **재현성**: random_state=42 고정. sensitivity 다중 seed는 [42, 123, 456] 명시.
4. **인코딩**: `utf-8-sig` 읽기, `sys.stdout.reconfigure(encoding="utf-8")` 쓰기.
5. **한글 폰트**: matplotlib 사용 시 `src.utils.font_setting.apply()` 호출.
6. **야간 클리핑**: XGBoost 예측값은 이미 클리핑된 상태일 가능성 — 중복 적용 방지.
7. **로그**: 각 단계 `[HH:MM:SS]` 타임스탬프.
8. **에러 처리**: 파일/컬럼 부재 시 `sys.exit()` + 명확한 메시지.
9. **claude_share 자동 복사**: 모든 스크립트에서 강제. 누락 시 작업 미완료.
10. **W&B 사용 안 함** (학습이 아닌 시뮬레이션 단계).

---

## 🚀 실행 순서

```bash
# TASK G-1: 파라미터 모듈 (검증 + claude_share 복사)
.venv/bin/python -m src.simulation.ess_config_v2

# TASK G-2: 정책 함수 (단위 테스트 + claude_share 복사)
.venv/bin/python -m src.simulation.ess_policy_v2

# TASK G-3 + G-4: 메인 시뮬 (4 시나리오 × 17 지역)
.venv/bin/python -m src.simulation.ess_simulation_v2

# TASK G-5: sensitivity 분석
.venv/bin/python -m src.simulation.ess_sensitivity_v2

# TASK H: 최종 리포트
.venv/bin/python -m src.reporting.final_report_v2
```

---

## ✅ 완료 조건

각 TASK는 다음을 모두 만족해야 완료:

### TASK G-1
- [ ] 17개 지역 파라미터 표 stdout 출력
- [ ] weight 합 = 1.0 ± 0.001 검증 통과
- [ ] 부하 패턴 평균 1.0 검증 통과
- [ ] `claude_share/ess_config_v2.py` 생성됨

### TASK G-2
- [ ] 4개 정책 모두 임의 입력에 대해 SOC 범위 내 반환 검증
- [ ] `policy_naive`는 항상 (0.80, 0.20) 반환
- [ ] `claude_share/ess_policy_v2.py` 생성됨

### TASK G-3 + G-4
- [ ] 4 시나리오 × 17 지역 = 68개 시뮬 실행 완료
- [ ] 3가지 집계(단순/가중/합산) 모두 산출
- [ ] `outputs/ess_v2_simulation_results.json` 생성
- [ ] `outputs/ess_v2_comparison.png`, `ess_v2_region_breakdown.png` 생성
- [ ] Oracle 자급률 ≥ xgb_lookahead ≥ naive 순서 확인 (assert 또는 stdout 경고)
- [ ] `claude_share/` 에 4개 파일 복사 완료

### TASK G-5
- [ ] 9개 noise level × 3 seed = 27개 점 산출
- [ ] sensitivity 곡선 PNG 생성, XGBoost 점 마킹됨
- [ ] Oracle 대비 도달률 stdout 출력
- [ ] `claude_share/` 에 3개 파일 복사 완료

### TASK H
- [ ] `outputs/national_final_report_v2.md` 생성 (v1 덮어쓰기 X)
- [ ] LSTM 언급 0건 (검증: grep -i lstm 결과 0)
- [ ] 7개 섹션 모두 포함
- [ ] `claude_share/` 에 2개 파일 복사 완료

---

## 💬 보고 시점

다음 시점마다 사용자에게 진행 보고하고 응답 대기:

1. **TASK G-1 완료 후** → 지역별 파라미터 표 보여주고 진행 승인 요청
2. **TASK G-4 완료 후** → 4 시나리오 핵심 지표 표 보여주고 sensitivity 진행 승인
3. **TASK G-5 완료 후** → Oracle 대비 도달률 보여주고 리포트 진행 승인
4. **TASK H 완료** → 최종 결과 요약 + claude_share 파일 목록

임의로 다음 단계 진행 금지. 의도 불명확 시 즉시 질문.
