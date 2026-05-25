#### 동작 흐름

```markdown
[운영자]
   ↓ "시뮬레이션 실행" 클릭
   
┌──────────────────────────────────────────┐
│  Streamlit 앱                              │
│  (모든 흐름 조정자)                          │
│                                            │
│  ① API에 예측 요청 ─────┐                  │
│                          ↓                 │
│              ┌────────────────────┐       │
│              │  FastAPI 서버       │       │
│              │  /predict_horizon  │       │
│              │  ② 모델 24번 호출   │       │
│              │  ③ 예측 배열 반환   │       │
│              └────────────────────┘       │
│                          │                 │
│  ④ 예측값 받음 ◄─────────┘                  │
│                                            │
│  ⑤ MPC 솔버 호출 (같은 프로세스 안)         │
│     LP 풀이 → 액션 + 수익 계산              │
│                                            │
│  ⑥ 화면에 그래프/숫자 그리기                │
└──────────────────────────────────────────┘
   ↓
[운영자가 결과 봄]
```

발전량 예측은 FastAPI 서버로 분리 + MPC 최적화(LP Solver)는 '전용 내부 로직'이므로 Streamlit 프로세스 내부에서 직접 구동(Internal Import)

**1. 역할의 성격 차이 (공용 서비스 vs 전용 로직)**

- 모델 추론 (FastAPI): 태양광 발전량 예측은 현재 개발 중인 운영 대시보드뿐만 아니라, 향후 모바일 알림 봇, 사내 모니터링 시스템 등 다양한 외부 플랫폼에서도 필요로 하는 범용 데이터다. 따라서 언제 어디서나 호출할 수 있도록 인터넷 창구(API)로 분리하는 것이 아키텍처 관점에서 올바르다고 판단했다.
- MPC 최적화 (Streamlit 내장): MPC 솔버는 운영자가 대시보드 화면(UI)에서 실시간으로 조절하는 파라미터(ESS 용량, TOU 요금제 변경, 현재 SOC 상태 등)를 입력받아 이 화면만을 위해 구동되는 독립적 계산기다. 외부에 노출할 필요가 없는 폐쇄적 로직이므로 화면을 실행하는 프로그램(Streamlit) 내부에서 직접 처리하는 것이 효율적임.

**2. 불필요한 네트워크 통신(HTTP) 비용 절감**

- MPC 최적화를 풀기 위해서는 대량의 제약조건과 시뮬레이션 파라미터가 오가야 한다. 만약 MPC까지 별도 API 서버로 만들면, 사용자가 버튼을 누를 때마다 수많은 내부 운영 매개변수를 HTTP 패킷에 담아 주고받아야 하므로 인터넷 통신 지연(Latency)과 서버 자원 낭비가 발생할 수 있다. **→ 추후 더 알아보기**
- 공용 데이터인 '발전량 예측값 배열'만 API로 깔끔하게 받아오고, 이를 활용한 헤비한 수학적 연산(LP 풀이)은 대시보드를 구동하는 파이썬 프로세스 메모리 상에서 직접 처리하여 속도를 극대화한다.

**3. 실제 서비스 환경의 안정성(장애 격리) 및 확장성 확보**

- 장애 격리: 만약 기상 데이터 파싱 오류나 모델 자체의 문제로 인해 예측 프로세스에 런타임 에러가 발생하더라도, FastAPI 서버만 일시적으로 영향을 받을 뿐 운영자가 보고 있는 UI 화면(Streamlit)은 정상적으로 유지된다. 화면이 통째로 뻗는 치명적인 장애를 방지할 수 있다. **→ 추후 더 알아보기**
- 데이터 검증: FastAPI 입구에서 데이터 규격(Pydantic)을 엄격하게 검증하여, 오염되거나 잘못된 데이터가 모델 내부 깊숙이 침투해 시스템 전체를 마비시키는 것을 원천 차단한다.

### 1단계 : Multi-step 예측 엔드포인트 (`/predict_horizon`)

- 목표를 "현장 운영자가 쓸 수 있는 도구"로 재정의 → 최종 산출물을 **Streamlit 대시보드**로 잡음
- Streamlit이 호출할 24시간 예측 엔드포인트가 필요해서 추가

**결정 사항**

| 항목 | 결정 |
| --- | --- |
| 정책 선택 (MPC/lookahead) | API에 안 받음 (Streamlit에서 처리) |
| 출력 깊이 | 예측 시퀀스만 (SOC/수익은 호출자가 계산) |
| 기상 데이터 | 클라이언트가 페이로드로 제공 |
| horizon 범위 | 1~48 (MPC 기본 24, 그 2배 여유) |
| history 길이 | 정확히 24개 (엄격) |
| start_time 검증 | 1900~2100 (명백한 비정상만 거름) |
| history power_mwh | 0 ≤ x ≤ 50000 (일괄 상한) |

**구현 핵심: 재귀 multi-step**

t+1 예측: 클라이언트 제공 실측 lag 사용
t+2 예측: t+1 예측값을 lag_1h로 (자기참조 시작)
… 
t+N 예측: lag/rolling이 모두 누적 예측값 기반

→ 오차 누적이 단점이지만 학술 표준 방식.

**작업 결과**

- `app/schemas.py`: 4개 모델 추가 (HistoryPoint, ForecastPoint, HorizonRequest, HorizonResponse)
- `app/inference.py`: `predict_horizon()` 재귀 함수 추가
- `app/main.py`: `/predict_horizon` 엔드포인트 + run_in_executor 처리
- `app/tests/test_api.py`: 회귀 테스트 8개 추가 (총 16개 통과)
- README: 엔드포인트 + 호출 예시 갱신

**성능 측정**

- horizon=24 → 48ms
- horizon=48 → 50ms

**→ XGBoost 추론보다 Pydantic/Python 오버헤드가 dominate → 향후 최적화 여지 있음**

**회귀테스트**

| 정상 경로 | 테스트 | 검증 내용 |
| --- | --- | --- |
| 1 | `valid_24` | horizon=24 정상 요청 → 200, predictions 24개, 모두 ≥0, step 1~24, method="recursive_multistep" |
| 2 | `valid_48` | horizon=48 정상 요청 → 200, predictions 48개 |

| 입력 검증 위반 | 테스트 | 검증 내용 |
| --- | --- | --- |
| 3 | `horizon_out_of_range` | horizon=0, 49, 100 → 422 (1~48 범위 강제) |
| 4 | `history_length_mismatch` | history 23개/25개 → 422 (정확히 24개 강제) |
| 5 | `forecast_length_mismatch` | horizon=24인데 forecast 23개 → 422 (길이 일치 강제) |
| 6 | `history_timestamp_gap` | history 중간에 2시간 갭 → 422 (1시간 연속성 강제) |
| 7 | `invalid_region` | region="화성시" → 422 + 에러 메시지에 유효 region 목록 노출 |

| 보안 | 테스트 | 검증 내용 |
| --- | --- | --- |
| 8 | `requires_api_key` | X-API-Key 헤더 없음 → 401 |

#### MPC 작동 방법

MPC는 매 1시간마다 "그 시점부터 24시간 미래"를 다시 보고 LP를 풀어 첫 액션만 실행한다. 그래서 24시간 시뮬을 보여주려면 미래 48시간 데이터가 필요하다.

```markdown
[입력]
  운영자 선택: "전라남도, 2023-07-15 09시 시작, 정책=mpc_xgb, SOC=50%"
  
       ↓
       
[데이터 준비]
  학습 CSV에서 슬라이스:
    history:  어제 9시 ~ 오늘 8시 실측 발전량 (24h)
    forecast: 오늘 9시 ~ 모레 8시 기상 (48h)
  
       ↓
       
[API 1번 호출]
  POST /predict_horizon (horizon=48)
  → 예측값 48개 받음
  
       ↓
       
[시뮬 시작 — 오늘 9시부터 24시간]
  매 시점마다:
  
    t=0 (09시): SOC=0.5
      MPC가 보는 미래: predicted[0:24]  ← 9시~다음날 8시
      LP 풀이 → 24개 액션 시퀀스 계산
      첫 액션만 실행 (예: 충전 80 MWh) → SOC 0.5 → 0.644
    
    t=1 (10시): SOC=0.644
      MPC가 보는 미래: predicted[1:25]  ← 10시~다음날 9시 (한 시간 밀림!)
      LP 풀이 → 새 24개 액션
      첫 액션만 실행
    
    t=2 (11시): MPC가 보는 미래: predicted[2:26]  ← 또 밀림
    ...
    t=23 (다음날 8시): MPC가 보는 미래: predicted[23:47]  ← 여기서 48h 필요
  
       ↓
       
[결과]
  24시간 SOC 궤적, 액션 시퀀스, 수익/자급률 계산
  → UI에 표시
```

**일반 시뮬과의 차이**

|  | 일반 시뮬 (xgb_lookahead) | MPC (mpc_xgb) |
| --- | --- | --- |
| 예측 보는 방식 | 한 번 받은 24개 그대로 사용 | 24h 윈도우가 매 시점 뒤로 밀림 |
| 필요 데이터 | 미래 24h | 미래 48h |
| 계산 횟수 | 시뮬 24번 (가벼움) | 시뮬 24번 + LP 풀이 24번 |
| 응답 시간 | 빠름 (~50ms) | 느림 (~200ms) |
| 정확도 | 단순 | 더 정교한 의사결정 |

### 2단계 : 48h 예측 정확도 검증

**왜 필요했나**

1단계에서 `/predict_horizon`은 horizon=1~48 지원하지만, 재귀 multi-step은 멀어질수록 부정확.
다음 단계(MPC 오케스트레이터)에서 Rolling Horizon 방식으로 시뮬 24h를 보여주려면 미래 48h 데이터가 필요
→ "48h 예측의 끝쪽 시점도 신뢰 가능한가?"를 직접 측정해야 함

- N=200 샘플 (17 region × 약 12 시점, random.seed=42)
    
    각 샘플에서 `/predict_horizon` (horizon=48) 호출 → 학습 CSV 실측값과 비교
    
- step별 그룹화: 단기(step 1~23) vs 장기(step 24~48)

**결과: PASS**

| 구간 | RMSE | MAE | MAPE | n |
| --- | --- | --- | --- | --- |
| 단기 (1~23) | 129.66 | 57.04 | 628.60% | 4,048 |
| 장기 (24~48) | 161.31 | 77.66 | 697.31% | 4,400 |
| 장기/단기 비율 | **1.24** | 1.36 | 1.11 | — |
- step-wise: step 1 RMSE=20.6 → step 14에서 161.8 (1일차 피크) → step 30에서 209.4 (2일차 피크). 2일차 피크가 1일차 대비 +30%로 누적 오차 합리적 수준.
- MAPE 절댓값(628%/697%)은 크지만 비율은 통과. 새벽/저녁 저발전 시점이 분모 작아서 발생하는 현상 — **round 2-3 UI에서 "예측 신뢰도" 지표 쓸 때는 RMSE 기반으로 갈 것.**

의사결정: (b-1) 48h 슬라이스 채택. 다음 단계로 진행.

| # | 항목 | 결정 |
| --- | --- | --- |
| (1) | 입력 범위 | **단일 지역** (운영자가 select로 1개 선택) |
| (2) | history/forecast 데이터 소스 | **학습 CSV 재활용** (`data/processed/national_train_features.csv` 슬라이스) |
| (3) | MPC 솔버 호출 방식 | **옵션 X** — 기존 `run_simulation()`을 24시간 길이로 재호출 (검증된 코드 그대로 재활용) |
| (4) | API 호출 정책 | **옵션 P** — 정책별로 필요할 때만 |
| (4-디테일) | API 호출 횟수 | **1번만** (`horizon=48`). 결과를 lookahead/mpc_xgb가 공유. naive는 안 씀 |
| (5) | 정책 처리 방식 | **3개 정책 일괄 실행** (naive, xgb_lookahead, mpc_xgb). 결과 비교 가능한 dict 반환 |
| (b) | MPC lookahead 처리 | **(b-1) 48시간 슬라이스** — 단, 사전 검증(round 2-2-pre)으로 정당성 확보 후 |
| (c) | initial_soc 입력 | **(c-1) 사용자 슬라이더 입력** (기본값 0.5) |
| - | 디렉토리 구조 | `app_streamlit/orchestrator.py`, `app_streamlit/data_loader.py`, `app_streamlit/tests/` |
| - | 반환 dict 스키마 | `{meta, results: {정책별 결과}, api_calls}` 구조 (자세한 내용은 프롬프트에서 확정) |

### 3단계: MPC 오케스트레이터

목표: Streamlit이 호출할 단일 진입점 함수 `run_mpc_simulation()` 구현. 입력 (start_time, region, initial_soc) → 출력 3개 정책(naive/xgb_lookahead/mpc_xgb) 결과 dict.

**사전 점검에서 발견된 갭** (원본 `run_simulation()`)

| # | 항목 | 문제 |
| --- | --- | --- |
| A | sim_length 인자 | 없음. 입력 배열 길이로 결정 → OK, 24로 슬라이스 |
| B | initial_soc 인자 | 없음. 항상 `SOC_INIT=0.5` 고정 → UI 슬라이더 요구사항 위반 |
| C | hourly 시계열 | 반환 안 함. aggregate만 → Streamlit 그래프 그릴 데이터 없음 |

**1차 결정 (헬퍼 복제 노선)** — 이후 재검토됨

- 사전 점검 시 옵션 4개 중 B-2 + C-1 (헬퍼 함수로 복제) 채택
- 근거: "원본 절대 수정 금지" 원칙 + 코드 중복은 명시적이고 추적 가능
- `app_streamlit/simulator.py::run_simulation_with_hourly()` 헬퍼 작성

**구현 결과**

```markdown
app_streamlit/
├── __init__.py
├── data_loader.py       — CSV 슬라이스 (history 24 + forecast 48 + actuals 48), lru_cache 사용
├── simulator.py         — 헬퍼 (원본 SOC 동역학 복제 + initial_soc + hourly)
├── orchestrator.py      — run_mpc_simulation() 본체
└── tests/
    ├── __init__.py
    └── test_orchestrator.py — 9개 테스트
```

**테스트 결과**: 9/9 통과

1. test_normal_case (3개 정책 결과 검증)
2. test_invalid_region (ValueError)
3. test_start_time_out_of_range (2023년, ValueError)
4. test_insufficient_window (CSV 끝 근처, ValueError)
5. test_initial_soc_boundary (0.0/1.0 OK, 그 밖 ValueError)
6. test_api_unreachable (잘못된 포트, RuntimeError)
7. test_predictions_shared (predictions 48개, API 1회 호출 확인)
8. test_helper_matches_original (헬퍼 ≡ 원본, naive 경로) — max diff = 0.000e+00
9. test_helper_matches_original_lookahead (lookahead 경로) — max diff = 0.000e+00

**Phase 2 패턴 재현 (단일 케이스 / 전라남도 / 2022-06-15 09:00 / initial_soc=0.5)**

| 지표 | mpc_xgb − xgb_lookahead | Phase 2 기대 | 부호 |
| --- | --- | --- | --- |
| net_revenue | +14.30% (+5337만원) | +49.53% | ✅ |
| self_sufficiency | −18.75pt | −17.48pt | ✅ |

→ self_sufficiency 차이가 Phase 2 평균과 거의 일치 → 헬퍼 SOC 동역학이 원본과 사실상 동일하다는 강한 간접 증거

**성능**

- elapsed_ms = 4934ms (목표 260ms 대비 19배 초과)
- 원인: LP 24회 (전라남도 대용량 ESS) + cold-start 모델 로드 + 48-step 재귀
- round 2-3 영향: Streamlit spinner 4~5초로 가리면 됨. 슬라이더 재실행 답답하면 캐싱 전략 필요 (후속 검토)

**HTTP 라이브러리**: requests 미설치 환경 회피 위해 urllib.request 유지 (round 2-2-pre와 일관)

### 4단계: 리팩토링 — 헬퍼 제거 + 원본 수정

- 직전 단계에서 `run_simulation()`을 복제한 헬퍼를 만들었으나, 정합성 테스트가 max diff = 0.000e+00을 보증 → 헬퍼 = 원본의 정확한 superset
- 결론: 원본 수정이 안전. 헬퍼 삭제 후 `run_simulation()`에 `initial_soc=SOC_INIT, include_hourly=False` 추가
- 디폴트값 덕에 Phase 1·2 기존 호출자 무수정, 산출물 JSON 비트 단위 동일 (prempc snapshot 1e-6 invariance 통과)
- 교훈: "원본 보존" 같은 자기검열적 제약은 검증 데이터 없이 박지 말 것

### 5단계: Streamlit 운영 모드 UI

**목표**: orchestrator 함수를 운영자가 실제로 쓸 수 있는 인터페이스로 노출. Streamlit 채택.

**핵심 설계 결정**

- 단일 진입점 `run_mpc_simulation()` 호출만으로 화면 그릴 데이터 다 나오는 구조 (round 2-2 완료 시점에 확정)
- `@st.cache_data` 캐싱 + 명시적 실행 버튼 병행 — 동일 입력 즉시 응답, 의도치 않은 재계산 방지
- API 다운 시 친절한 에러 메시지로 변환 (`RuntimeError` → 사용자 안내)
- `st.session_state`로 결과 보존 — 위젯 토글로 인한 rerun에도 결과 유지

**구현 결과**

- `app_streamlit/{app, ui/*}.py` 6개 파일 신규
- 기존 파일(orchestrator, data_loader, tests) 무수정
- 차트 4종 (예측 vs 실측, 정책별 SOC, 수익·자급률 비교, 시간대별 매매)
- 정책 표시명을 운영자 친화 명명으로 매핑 (mpc_xgb → "수익 최적화 (MPC)" 등)

**검증**

- 정상 케이스 / API 다운 / 입력 위젯 토글 / 캐시 hit 모두 통과
- SOC 초기값 0.5→0.7 변화 시 정책별 순수익 변화 패턴이 물리적으로 합리적 (기본 운영 +30%, MPC +4% — MPC는 초기 SOC에 덜 민감)

**얻은 것**

- 운영자가 단일 시점에 대해 3개 정책의 trade-off (수익 vs 자급률)를 한 화면에서 비교 가능
- Phase 1·2 발견을 인터랙티브 도구로 검증 가능한 형태로 노출