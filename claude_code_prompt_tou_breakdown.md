# Claude Code 작업 지시: TOU 거래 패턴 세부 분석 출력 추가

## 1. 배경

이전 TOU 도입 작업에서 다음 결과가 나왔다:

| 정책 | 자급률(%) | net_revenue(원) |
|---|---|---|
| naive_baseline | 80.39 | +162,045,876,733 |
| xgb_no_lookahead | 80.39 | +162,045,876,733 |
| xgb_lookahead | 79.06 | +168,921,944,499 |
| oracle | 79.05 | +169,073,145,447 |

자급률 ↑ ↔ net_revenue ↓ 의 **음의 상관관계**가 관측됐다. 가설은 다음과 같다:

> lookahead 계열이 "충전을 미루는" 행동을 통해, *우연히* TOU 차익거래(낮 비싼 시간 매도, 밤 싼 시간 매수)와 정렬됐다.

이 가설이 맞다면 정책별 거래의 *시간대별 분포*에 차이가 보여야 한다. 본 작업은 이를 데이터로 검증할 수 있게 세부 출력을 추가한다.

## 2. 작업 범위 — 무엇을 하고 무엇을 안 하나

### 하는 것

1. `ess_simulation_v2.py`의 `run_simulation()`에서 시점별 import/export를 **부하구분(off_peak/mid_peak/max_peak)별로 분해 집계**
2. 결과 dict에 분해 지표 추가 (`import_mwh_by_period`, `export_mwh_by_period`, `cost_krw_by_period`, `revenue_krw_by_period`)
3. 정책별 **평균 매수 단가** 및 **평균 매도 단가** 계산해서 결과 dict에 추가
4. 메인 실행 결과를 기존 JSON 파일에 덮어쓰기 (이전 지표 유지 + 새 지표 추가)
5. 분석용 markdown 표 별도 출력 (다음 섹션 형식)

### 안 하는 것

- 정책 함수 수정 ❌
- 의사결정 로직 변경 ❌ (측정만 추가)
- 기존 지표 변경 ❌
- 새 시뮬 시나리오 추가 ❌
- 그래프/시각화 생성 ❌ (표만)

## 3. 구현 세부 사항

### 3-1. `run_simulation()` 안에 누적 변수 추가

매 시점 루프 진입 전 초기화:

```python
import_mwh_by_period = {"off_peak": 0.0, "mid_peak": 0.0, "max_peak": 0.0}
export_mwh_by_period = {"off_peak": 0.0, "mid_peak": 0.0, "max_peak": 0.0}
cost_krw_by_period = {"off_peak": 0.0, "mid_peak": 0.0, "max_peak": 0.0}
revenue_krw_by_period = {"off_peak": 0.0, "mid_peak": 0.0, "max_peak": 0.0}
```

매 시점 루프 안 (TOU 단가 계산하는 자리 근처):

```python
period = get_load_period(month, hour)  # 이미 호출되고 있다면 재사용
# import 발생 시:
import_mwh_by_period[period] += import_mwh
cost_krw_by_period[period] += import_mwh * price_t
# export 발생 시:
export_mwh_by_period[period] += export_mwh
revenue_krw_by_period[period] += export_mwh * price_t
```

### 3-2. 평균 단가 계산

루프 종료 후:

```python
avg_import_price_krw_per_mwh = (
    total_cost_krw / total_import_mwh if total_import_mwh > 0 else 0.0
)
avg_export_price_krw_per_mwh = (
    total_revenue_krw / total_export_mwh if total_export_mwh > 0 else 0.0
)
```

### 3-3. 결과 dict에 추가

`run_simulation()` 반환 dict에 다음 키 추가:

```
"import_mwh_by_period":       {"off_peak": ..., "mid_peak": ..., "max_peak": ...},
"export_mwh_by_period":       {"off_peak": ..., "mid_peak": ..., "max_peak": ...},
"cost_krw_by_period":         {...},
"revenue_krw_by_period":      {...},
"avg_import_price_krw_per_mwh": float,
"avg_export_price_krw_per_mwh": float,
```

기존 키(`total_import_mwh`, `total_export_mwh`, `total_cost_krw`, `total_revenue_krw`, `net_revenue_krw`)는 그대로 유지.

### 3-4. JSON 저장

`outputs/ess_v2_simulation_results.json`에 위 새 키들이 정책별로 포함되도록 저장. 기존 키 보존.

### 3-5. 분석용 표 출력

시뮬 완료 후 stdout (또는 `outputs/ess_v2_tou_breakdown.md`로 저장)에 다음 4개 표 출력:

**표 A: 매도(export) 시간대별 분포 (MWh)**

| 정책 | off_peak | mid_peak | max_peak | 합계 | max_peak 비중(%) |
|---|---|---|---|---|---|
| naive_baseline | ... | ... | ... | ... | ... |
| xgb_no_lookahead | ... | ... | ... | ... | ... |
| xgb_lookahead | ... | ... | ... | ... | ... |
| oracle | ... | ... | ... | ... | ... |

**표 B: 매수(import) 시간대별 분포 (MWh)**

| 정책 | off_peak | mid_peak | max_peak | 합계 | off_peak 비중(%) |
|---|---|---|---|---|---|
| naive_baseline | ... | ... | ... | ... | ... |
| ... |

**표 C: 평균 단가 (원/MWh)**

| 정책 | 평균 매수 단가 | 평균 매도 단가 | 매도-매수 스프레드 |
|---|---|---|---|
| naive_baseline | ... | ... | ... |
| ... |

**표 D: 정책 간 차이 (vs naive_baseline)**

| 정책 | 자급률 차이(pt) | net_revenue 차이(원) | max_peak export 차이(MWh) | off_peak import 차이(MWh) |
|---|---|---|---|---|
| xgb_no_lookahead | 0.00 | 0 | 0 | 0 |
| xgb_lookahead | -1.33 | +6,876,067,766 | ... | ... |
| oracle | -1.34 | +7,027,268,714 | ... | ... |

## 4. 어떤 시뮬 결과를 대상으로 하나

기존 시뮬에서 사용한 동일 매트릭스 (17 지역 × 4 정책) 그대로. 합산 결과(전국 합산 시뮬 기준)에 대해 위 표 출력.

지역별 분해는 이번 작업에서는 안 함. 합산 수준에서 4개 정책 비교만 충분.

## 5. 산출물

```
src/simulation/ess_simulation_v2.py        # 분해 집계 로직 추가
outputs/ess_v2_simulation_results.json     # 새 지표 포함 재저장
outputs/ess_v2_tou_breakdown.md            # 표 A~D 출력 (신규 파일)
claude_share/                              # 위 3개 파일 복사
```

## 6. 검증 체크리스트

- [ ] `sum(import_mwh_by_period.values()) == total_import_mwh` (이전 작업의 총합과 일치)
- [ ] `sum(export_mwh_by_period.values()) == total_export_mwh`
- [ ] `sum(cost_krw_by_period.values()) == total_cost_krw`
- [ ] `sum(revenue_krw_by_period.values()) == total_revenue_krw`
- [ ] 평균 매수 단가가 87,300 ~ 222,300 원/MWh 범위 안
- [ ] 평균 매도 단가가 87,300 ~ 222,300 원/MWh 범위 안
- [ ] **기존 지표 invariance**: 자급률·자가소비율·부족 강도·사이클수·net_revenue 모두 이전 결과와 동일
- [ ] naive와 xgb_no_lookahead의 모든 새 지표가 서로 동일 (현재 두 정책은 동일 SOC 결정 → 거래 패턴도 동일해야 함)

## 7. 주의사항

- 의사결정 로직은 절대 손대지 말 것. 측정만 추가.
- 부동소수점 오차로 합계 검증이 정확히 0이 아닐 수 있음 → 1e-6 이하 차이는 OK.
- `get_load_period()` 함수는 이미 `ess_config_v2.py`에 있음. import해서 재사용.
- 표 D의 자급률·net_revenue 차이는 이미 알려진 값이므로 하드코딩 말고 결과 dict에서 계산해 출력할 것 (재현성).
