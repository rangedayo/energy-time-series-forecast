# Claude Code 작업 지시: ESS 시뮬레이터에 TOU(시간대별 요금) 도입

## 1. 배경 및 목적

본 작업은 Phase 2(MPC 도입)의 첫 단계다. 현재 v2 시뮬레이터는 "시점마다 전기 가치가 동등"하다는 가정 위에 돌아간다. 그래서 4개 정책(naive / xgb_no_lookahead / xgb_lookahead / oracle) 모두 **SOC 범위를 가장 넓게 쓰는 naive가 탐욕 최적**이고, 예측 정확도가 ESS 가치로 전환되지 않는다(발견 2: Oracle ≤ Naive).

TOU(Time-of-Use, 시간대별 요금)를 도입하면 **시점별로 전기 가치가 달라진다**. 시간대 차익거래가 가능해지므로 정책 간 차별화 여지가 생긴다. 이번 작업은 비용/수익 계산 인프라까지만 만들고, MPC 정책 자체는 다음 단계로 분리한다.

## 2. 작업 범위 — 무엇을 하고 무엇을 안 하나

### 하는 것

1. `ess_config_v2.py`에 TOU 단가 매트릭스 + 시간 구분 함수 추가
2. `ess_simulation_v2.py`의 `run_simulation()`에 비용/수익 계산 로직 추가
3. 새 지표 `total_cost`, `total_revenue`, `net_revenue` 추가 (기존 지표는 유지)
4. 정책 함수 시그니처는 **변경하지 않는다** (호환성 유지)
5. 메인 실행 결과 JSON에 새 지표 포함
6. 기존 정책 4개로 메인 시뮬 재실행 → 결과 비교 가능 상태로

### 안 하는 것

- 정책 함수 자체에 TOU 인식 로직 추가 ❌ (다음 단계의 MPC가 그 역할)
- 송전 한도, 사이클 비용 도입 ❌ (인수인계 문서 결정대로 TOU만)
- 기본요금, 부가세, 기후환경요금 ❌ (단순화)
- 토요일·공휴일 특례 ❌ (평일 기준만)
- Sensitivity 분석 재실행 ❌ (별도 작업)
- 11월본 요금 적용 ❌ (5월본 하나로 통일)
- 새 정책 함수 추가 ❌
- 자급률·자가소비율 등 기존 지표 변경 ❌ (추가만, 변경 X)

## 3. 단가 매트릭스 — 이 숫자 그대로 사용

KEPCO 산업용(을) 고압A 선택Ⅱ, 2023-05-16 시행본, 평일 기준 (원/kWh):

| 시간대 | 여름철 (6~8월) | 봄·가을철 (3~5월, 9~10월) | 겨울철 (11~2월) |
|---|---|---|---|
| 경부하 | 87.3 | 87.3 | 94.3 |
| 중간부하 | 140.2 | 109.8 | 140.4 |
| 최대부하 | 222.3 | 140.5 | 197.9 |

## 4. 시간 구분 매핑 — 이 규칙 그대로 사용

여름철·봄·가을철 (3~10월):
- 경부하: 22:00 ~ 익일 08:00
- 중간부하: 08:00~11:00, 12:00~13:00, 18:00~22:00
- 최대부하: 11:00~12:00, 13:00~18:00

겨울철 (11~2월):
- 경부하: 22:00 ~ 익일 08:00
- 중간부하: 08:00~09:00, 12:00~16:00, 19:00~22:00
- 최대부하: 09:00~12:00, 16:00~19:00

## 5. 비용/수익 계산 규칙

- **매수**(grid_import, 외부에서 사옴): `import_mwh × price_per_mwh`로 비용 누적
- **매도**(grid_export, 잉여를 그리드로 보냄): `export_mwh × price_per_mwh`로 수익 누적
- **매수가 = 매도가** (양방향 동일 가격, 단순화 가정)
- **단위 환산 주의**: 표는 원/kWh, 시뮬은 MWh 단위. **× 1000 필요**.
- `net_revenue = total_revenue - total_cost`

## 6. 시뮬레이터 동작 매핑 — 어디서 매수/매도가 일어나는가

현재 시뮬 흐름:
```
잉여 발생 (gen > demand)
   ↓
수요 차감 (자체 수요 충당) ← 여기는 매수/매도 모두 발생 안 함 (자체 소비)
   ↓
남은 잉여만 ESS로 충전
   ↓
ESS 못 받은 건 그리드로 매도  ← export
```

부족 시:
```
발전 부족 (gen < demand)
   ↓
ESS에서 방전 시도 (SOC 하한까지)
   ↓
ESS로 못 채운 부족분은 그리드에서 매수  ← import
```

→ 즉 `total_cost`는 시뮬 내 "부족분으로 외부에서 사온 양"의 누적, `total_revenue`는 "ESS 못 받고 그리드로 흘려보낸 잉여"의 누적이다. 현재 시뮬 변수명으로 보면 `shortage_mwh` 누적이 import, curtailment(또는 그에 해당하는 변수)가 export에 해당한다. 정확한 변수명은 `ess_simulation_v2.py`를 보고 매핑할 것.

## 7. 구현 세부 사항

### 7-1. `ess_config_v2.py`에 추가

```python
# TOU 단가 매트릭스 (원/kWh)
# 출처: KEPCO 산업용(을) 고압A 선택Ⅱ, 2023-05-16 시행
# 단순화: 평일 기준, 기본요금/부가세/기후환경요금/연료비조정요금 무시
TOU_PRICES_KRW_PER_KWH = {
    "summer": {"off_peak": 87.3, "mid_peak": 140.2, "max_peak": 222.3},
    "spring_autumn": {"off_peak": 87.3, "mid_peak": 109.8, "max_peak": 140.5},
    "winter": {"off_peak": 94.3, "mid_peak": 140.4, "max_peak": 197.9},
}

def get_season(month: int) -> str:
    """월 → 계절 구분."""
    if month in (6, 7, 8):
        return "summer"
    elif month in (3, 4, 5, 9, 10):
        return "spring_autumn"
    else:  # 11, 12, 1, 2
        return "winter"


def get_load_period(month: int, hour: int) -> str:
    """월·시간 → 부하 구분 (경부하/중간부하/최대부하).
    평일 기준. 토·공휴일 특례 미적용."""
    if month in (11, 12, 1, 2):  # 겨울철
        if 22 <= hour or hour < 8:
            return "off_peak"
        elif hour in (9, 10, 11, 16, 17, 18):
            return "max_peak"
        else:  # 8, 12, 13, 14, 15, 19, 20, 21
            return "mid_peak"
    else:  # 여름·봄·가을철
        if 22 <= hour or hour < 8:
            return "off_peak"
        elif hour in (11, 13, 14, 15, 16, 17):
            return "max_peak"
        else:  # 8, 9, 10, 12, 18, 19, 20, 21
            return "mid_peak"


def get_tou_price_krw_per_mwh(month: int, hour: int) -> float:
    """월·시간 → 단가 (원/MWh). MWh 단위 변환 완료."""
    season = get_season(month)
    period = get_load_period(month, hour)
    return TOU_PRICES_KRW_PER_KWH[season][period] * 1000.0
```

### 7-2. `ess_simulation_v2.py`의 `run_simulation()` 수정

매 시점 루프 안에:
- 시점 t의 timestamp에서 month, hour 추출
- `price_t = get_tou_price_krw_per_mwh(month, hour)` 호출
- import 발생 시 `total_cost += import_mwh × price_t`
- export 발생 시 `total_revenue += export_mwh × price_t`

루프 종료 후:
- `net_revenue = total_revenue - total_cost`
- 반환 dict에 `total_cost_krw`, `total_revenue_krw`, `net_revenue_krw` 추가

### 7-3. 자가검증 출력 (디버깅용)

`run_simulation()` 종료 시점에 stdout으로 sanity check:

```
[TOU 검증] 정책: {policy_name}
  총 import: XX,XXX MWh
  총 export: XX,XXX MWh
  총 비용: XXX,XXX,XXX 원 (평균 단가: XXX 원/MWh)
  총 수익: XXX,XXX,XXX 원 (평균 단가: XXX 원/MWh)
  순수익: ±XXX,XXX,XXX 원
```

평균 단가가 87,300 ~ 222,300 원/MWh 범위(=87.3~222.3 원/kWh × 1000) 안에 있는지 눈으로 확인 가능.

## 8. 산출물

다음 파일을 수정·생성하고 모두 `claude_share/`에 복사:

```
src/simulation/ess_config_v2.py            # TOU 함수 3개 추가
src/simulation/ess_simulation_v2.py        # run_simulation 수정
outputs/ess_v2_simulation_results.json     # 새 지표 포함 재실행
```

기존 파일들(`national_final_report_v2.md`, `ess_v2_sensitivity_results.json` 등)은 **보존**. 이번 작업에서는 손대지 않는다.

## 9. 검증 체크리스트

PR 또는 작업 완료 보고 시 다음을 모두 확인:

- [ ] `get_season(6) == "summer"`, `get_season(11) == "winter"` 등 경계값 확인
- [ ] `get_load_period(7, 12) == "mid_peak"` (여름 12시), `get_load_period(7, 14) == "max_peak"` (여름 14시), `get_load_period(12, 10) == "max_peak"` (겨울 10시)
- [ ] `get_tou_price_krw_per_mwh(7, 14) == 222300.0` (여름 최대부하)
- [ ] 4개 정책 모두 새 시뮬 정상 실행
- [ ] 기존 지표(자급률·자가소비율·부족 강도·사이클수)가 **이전 결과와 동일**한지 확인 (TOU는 측정만 하고 결정 로직은 안 바꾸므로, 기존 지표는 변하지 않아야 함)
- [ ] 4개 정책의 `net_revenue` 차이가 자급률 차이와 같은 부호인지 확인 (가설 검증용 출력)
- [ ] sanity check stdout 출력에서 평균 단가가 87,300 ~ 222,300 원/MWh 범위 안에 있는지 확인

## 10. 결과 보고 형식

작업 완료 후 다음 표를 stdout 또는 별도 markdown으로 출력:

| 정책 | 자급률(%) | net_revenue(원) | total_cost(원) | total_revenue(원) |
|---|---|---|---|---|
| naive_baseline | ... | ... | ... | ... |
| xgb_no_lookahead | ... | ... | ... | ... |
| xgb_lookahead | ... | ... | ... | ... |
| oracle | ... | ... | ... | ... |

이 표는 다음 단계(가설 검증)에서 핵심 입력이 된다.

## 11. 주의사항

- **기존 지표가 바뀌면 코드 버그**다. TOU는 측정만 추가하는 것이지 의사결정 로직을 바꾸는 게 아니다. 자급률 등 기존 숫자는 이전 결과와 동일해야 한다. (오차 부동소수점 수준 이내)
- 단위 환산(원/kWh → 원/MWh) 빠뜨리지 말 것. ×1000 한 번 안 하면 결과가 1/1000로 나옴.
- 정책 함수 시그니처 변경 금지. 비용 계산은 시뮬레이터 본체에서만.
- 새 변수명은 기존 코드 컨벤션 따를 것 (`total_cost_krw`처럼 단위 명시 권장).
