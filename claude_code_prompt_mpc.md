# Claude Code 작업 지시: MPC 정책 도입 (Phase 2 핵심)

## 1. 배경

이전 작업들로 다음이 갖춰졌다:
- TOU 단가 매트릭스 + 시간 구분 함수 (`ess_config_v2.py`)
- 비용/수익 계산 인프라 (`ess_simulation_v2.py`의 `run_simulation`)
- 시간대별 거래 분해 분석 (`ess_v2_tou_breakdown.md`)

기존 4개 정책(naive / xgb_no_lookahead / xgb_lookahead / oracle)은 **TOU 가격을 전혀 안 본다.** 발전량 패턴만 보고 SOC를 결정한다. 그럼에도 음의 상관관계(자급률↓ → net_revenue↑)가 관측됐는데, 이는 우연한 정렬일 뿐이다.

본 작업은 **TOU를 *명시적으로* 보고 최적 의사결정을 내리는 MPC 정책 2종**을 추가한다. 이것이 Phase 2의 핵심 산출물이다.

## 2. MPC 정책 사양

### 공통 사양

- **방법**: 매 시점 t에서 다음 24시점(t..t+23)을 보고 선형계획법(LP)으로 최적 충방전 시퀀스 산출 후, t 시점 행동만 실행. 다음 시점 t+1에서 다시 풀이.
- **목적함수**: net_revenue 최대화 (= revenue - cost 최대화 = cost - revenue 최소화)
- **자급률 페널티 없음**: net_revenue만 단순 최대화. 자급률은 결과로 측정만.
- **Horizon**: N=24
- **물리 모델**: 기존 시뮬레이터(`ess_simulation_v2.py`)와 **완전히 동일**한 SOC 동역학·효율·제약을 LP에 옮길 것. 시뮬과 LP의 SOC 궤적이 같은 입력에서 일치해야 함.

### 두 변형

| 정책 이름 | 미래 정보 출처 | 의미 |
|---|---|---|
| `mpc_xgb` | XGBoost 예측 N시점 (`predicted[t..t+23]`) | 현실적 MPC |
| `mpc_oracle` | 실측값 N시점 (`actual[t..t+23]`) | 이론적 상한 MPC |

이 둘의 차이가 **"예측 정확도가 MPC 가치로 얼마나 전환되나"**를 정량화한다.

## 3. LP 정식화

### 3-1. 의사결정 변수 (시점 t 기준, h=0..23)

```
charge[h]    >= 0    # 시점 t+h 충전량 (MWh)
discharge[h] >= 0    # 시점 t+h 방전량 (MWh)
import_g[h]  >= 0    # 시점 t+h 그리드 매수량 (MWh)
export_g[h]  >= 0    # 시점 t+h 그리드 매도량 (MWh)
```

SOC는 변수 아니고, 충방전에서 누적식으로 계산.

### 3-2. 효율 처리 — 기존 시뮬과 동일하게

`ess_simulation_v2.py`의 `run_simulation()`을 먼저 읽어서 **효율(EFFICIENCY=0.90)이 어디에 곱해지는지** 정확히 파악할 것. LP의 SOC 갱신식이 시뮬 본체와 *동일*해야 함. 추측 금지.

전형적 두 패턴:
- 충전 손실형: `SOC[h+1] = SOC[h] + EFFICIENCY * charge[h] / capacity - discharge[h] / capacity`
- 방전 손실형: `SOC[h+1] = SOC[h] + charge[h] / capacity - discharge[h] / EFFICIENCY / capacity`

→ 시뮬 본체 읽고 **동일한 식**을 LP에 적용. 어느 패턴인지 코드 주석으로 명시.

### 3-3. 제약

각 시점 h=0..23에 대해:

```
# 수요-공급 균형 (gen + discharge + import = demand + charge + export)
gen[h] + discharge[h] + import_g[h] = demand[h] + charge[h] + export_g[h]

# SOC 범위
SOC_MIN <= SOC[h] <= SOC_MAX     (h=0..24)

# 충방전 속도 상한
charge[h]    <= CHARGE_RATE_MAX
discharge[h] <= DISCHARGE_RATE_MAX
```

- `gen[h]`: XGBoost 예측 (mpc_xgb) 또는 실측값 (mpc_oracle)
- `demand[h]`: `get_demand_at_hour()` 로부터 받음 (기존 시뮬과 동일)
- `SOC[0]`: 현재 시점의 실제 SOC (시뮬 본체에서 받음)

### 3-4. 목적함수

```
minimize: sum over h of (import_g[h] * price[h]) - (export_g[h] * price[h])
```

여기서 `price[h] = get_tou_price_krw_per_mwh(month[t+h], hour[t+h])`. 이건 미래 시점의 *알려진 결정론적* 값이므로 LP 계수로 들어감.

## 4. 정책 함수 구현

### 4-1. 시그니처

기존 정책 시그니처와 동일하게 유지:

```python
def policy_mpc_xgb(t, actual, predicted, soc, demand_t, params, **kwargs):
    """MPC (XGBoost 예측 기반). horizon=24, LP."""
    ...
    return {"soc_target_high": ..., "soc_target_low": ...}

def policy_mpc_oracle(t, actual, predicted, soc, demand_t, params, **kwargs):
    """MPC (실측값 기반, 이론적 상한). horizon=24, LP."""
    return policy_mpc_xgb(t, actual, actual, soc, demand_t, params, **kwargs)
```

### 4-2. SOC 목표 산출

LP를 풀고 `charge[0], discharge[0]`을 얻으면 다음 시점의 SOC 목표가 결정됨:

```
soc_target = SOC[0] + (effective charge change) - (effective discharge change)
```

이 `soc_target`을 시뮬레이터에게 어떻게 전달할지 두 방식:

**방식 A (시도 1순위)**: 좁은 SOC 밴드로 반환

```python
return {
    "soc_target_high": soc_target,
    "soc_target_low":  soc_target,
}
```

기존 시뮬 본체는 이 밴드 안에서 탐욕 동작하므로, 사실상 LP가 결정한 SOC로 유도됨.
**다만 시뮬 본체의 탐욕 로직이 정확히 이 SOC에 도달하지 못할 수 있음** → 검증 시 LP가 산출한 시뮬 결과의 SOC 궤적이 LP 내부 SOC와 차이가 큰지 확인.

**방식 B (방식 A가 안 맞을 때)**: 시뮬 본체에 새 분기 추가

`run_simulation()` 안에서 `policy_fn` 호출 결과가 `{"action_charge": ..., "action_discharge": ...}` 형태면 SOC 목표 해석 단계를 건너뛰고 직접 행동 적용. 정책 함수는 LP 결과를 그대로 충방전량으로 넘김.

→ **방식 A로 먼저 구현 → 검증 후 SOC 궤적이 LP와 다르면 방식 B로 전환.** 두 방식 중 어느 게 채택됐는지 작업 보고에 명시.

## 5. 성능 최적화

매 시점 × 8760 시점 × 17 지역 × 2 정책 = 약 30만 LP. 단순 구현 시 수 시간 소요.

### 5-1. 권장 전략

LP를 매 *시점*에서 푸는 대신, 각 지역×정책에 대해 **24시간 윈도우를 슬라이딩하며 1회 LP 풀이로 1시간 진행** 구조. 즉 정책 함수 내부에서 LP 풀이 결과를 캐시하지 말고, 매 시점 새로 풀이 (=진정한 MPC). 다만 LP solver 자체를 빠른 것 사용.

### 5-2. Solver 선택

- **추천**: `scipy.optimize.linprog(method="highs")` — 빠르고 안정적.
- 대안: `cvxpy` (코드는 깔끔하나 느림). 시도해보고 1지역 1정책 시뮬에 5분 넘으면 scipy로 전환.

### 5-3. 진행률 출력

지역별, 정책별 진행률을 시간 추정과 함께 stdout에 출력:

```
[hh:mm:ss] mpc_xgb / 전라남도 (1/17) — 8760 시점 LP 풀이 중...
[hh:mm:ss]   1000/8760 (11.4%) elapsed=12.3s  ETA=1m 48s
...
```

전체 작업이 1시간 안쪽이면 OK. 넘으면 solver 옵션 튜닝 또는 LP 차원 축소 고려.

## 6. 작업 범위 — 무엇을 하고 무엇을 안 하나

### 하는 것

1. `src/simulation/ess_policy_v2.py`에 `policy_mpc_xgb`, `policy_mpc_oracle` 추가
2. (필요 시) `src/simulation/ess_simulation_v2.py`의 `run_simulation()`에 방식 B 분기 추가 — 방식 A로 충분하면 변경 없음
3. 메인 시뮬(`main()`)의 시나리오 리스트에 두 정책 추가 → 6개 정책 매트릭스 실행
4. 결과 JSON, 비교 PNG, region breakdown PNG 갱신
5. 새 TOU breakdown 표(`outputs/ess_v2_tou_breakdown.md`)도 6개 정책으로 갱신
6. 결과 보고: MPC 도입 후 가설 검증용 핵심 표

### 안 하는 것

- 송전 한도, 사이클 비용 도입 ❌
- 자급률 페널티 ❌
- 다른 horizon 변형 (N=12, N=48 등) ❌
- Sensitivity 분석 재실행 ❌ (별도 작업)
- 최종 리포트(`national_final_report_v2.md`) 업데이트 ❌ (Phase 2 완료 후 별도)
- 새 시각화 추가 ❌ (기존 PNG 자동 갱신만)

## 7. 검증 체크리스트

- [ ] LP의 SOC 동역학이 시뮬 본체와 동일한지 코드 비교 (효율 적용 위치, 충방전 부호 등)
- [ ] LP가 infeasible 나는 시점이 있는지 카운트 (있으면 SOC_MIN/MAX를 살짝 완화하는 fallback 가능)
- [ ] mpc_oracle ≥ mpc_xgb 인지 (net_revenue 기준). Oracle이 더 잘해야 정상.
- [ ] **두 MPC 정책 모두 net_revenue가 xgb_lookahead보다 큰가?** (이게 MPC의 핵심 가치 — 안 그러면 MPC가 lookahead보다 못한 거)
- [ ] 자급률 invariance: naive·xgb_no_lookahead·xgb_lookahead·oracle 4개 정책의 자급률·net_revenue는 이전 결과와 **완전히 동일**해야 함 (MPC 추가가 기존 정책에 영향 주면 안 됨)
- [ ] LP 풀이 횟수 카운트 출력 (디버그용)
- [ ] 전체 실행 시간 1시간 이내

## 8. 산출물

```
src/simulation/ess_policy_v2.py            # MPC 정책 2개 추가
src/simulation/ess_simulation_v2.py        # (필요 시) 방식 B 분기 추가
outputs/ess_v2_simulation_results.json     # 6개 정책 결과
outputs/ess_v2_comparison.png              # 6개 정책 비교 (자동 갱신)
outputs/ess_v2_region_breakdown.png        # 자동 갱신
outputs/ess_v2_tou_breakdown.md            # 6개 정책 TOU 분해 (갱신)
claude_share/                              # 위 파일들 복사
```

## 9. 결과 보고 형식

다음 표를 stdout 또는 별도 markdown으로 출력:

| 정책 | 자급률(%) | net_revenue(원) | total_cost(원) | total_revenue(원) | LP 풀이 횟수 |
|---|---|---|---|---|---|
| naive_baseline | (이전과 동일) | (이전과 동일) | ... | ... | 0 |
| xgb_no_lookahead | (이전과 동일) | (이전과 동일) | ... | ... | 0 |
| xgb_lookahead | (이전과 동일) | (이전과 동일) | ... | ... | 0 |
| oracle | (이전과 동일) | (이전과 동일) | ... | ... | 0 |
| **mpc_xgb** | ... | ... | ... | ... | 17 × 8760 |
| **mpc_oracle** | ... | ... | ... | ... | 17 × 8760 |

추가로 한 줄 결론:
> mpc_xgb는 xgb_lookahead 대비 net_revenue +X.X% 향상, 자급률 ±X.X pt 변화.
> mpc_oracle은 mpc_xgb 대비 net_revenue +X.X% 추가 향상 (=예측 정확도가 MPC 가치로 전환되는 폭).

이 표가 Phase 2 완료의 핵심 산출물이다.

## 10. 주의사항

- **시뮬 본체의 SOC 동역학을 정확히 복제할 것.** 추측·근사 금지. 코드를 직접 읽고 동일한 식을 옮길 것.
- 효율(EFFICIENCY=0.90) 적용 위치를 LP와 시뮬 사이 *반드시 일치*시킬 것. 다르면 LP 최적해가 시뮬에서 실현 안 됨.
- 정책 함수 시그니처는 가능한 한 유지. 방식 B 채택 시에만 시그니처 확장.
- 기존 4개 정책 결과가 바뀌면 코드 버그. invariance 체크 필수.
- LP infeasibility는 SOC 제약을 살짝 완화(예: SOC_MIN=0.19)하는 fallback으로 처리. 발생 횟수는 카운트해서 보고.
- LP solver는 무난한 옵션 사용. 너무 정밀하게 설정하면 느려짐.
- 시간 추정 출력으로 폭주 시 조기 발견 가능하게 할 것.
