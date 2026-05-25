# Round 2-2: MPC 오케스트레이터 함수 구현

> **시작하기 전 사전 점검**: 기존 `run_simulation()` 함수 위치와 시그니처를 먼저 확인하고 보고할 것. `sim_length` 인자 유무에 따라 (b)-4 구현 방식이 달라진다.

## 배경

- Round 2-1에서 `/predict_horizon` 엔드포인트 완성 (horizon=48 지원)
- Round 2-2-pre 검증 결과: 48h 재귀 multi-step 예측의 장기/단기 RMSE 비율 1.24 → **PASS**. (b-1) 48h 슬라이스 방식으로 진행 확정.
- 이제 Streamlit 운영자 화면(round 2-3)에서 호출할 **MPC 오케스트레이터 함수**를 구현한다.
- 이 함수는 단일 지역에 대해 history 24h + forecast 48h를 학습 CSV에서 슬라이스하고, `/predict_horizon` 1번 호출로 받은 예측을 3개 정책(`naive`, `xgb_lookahead`, `mpc_xgb`)이 공유하며 24시간 시뮬을 돌린다.

## 목표

`run_mpc_simulation()` 함수 1개를 만든다. 입력은 (start_time, region, initial_soc), 출력은 3개 정책 결과를 담은 dict. Streamlit에서 이 함수 하나만 호출하면 운영자 화면이 그릴 모든 데이터가 나와야 한다.

## 디렉토리 구조

```
app_streamlit/
├── __init__.py
├── orchestrator.py      ← run_mpc_simulation() 정의
├── data_loader.py       ← CSV 슬라이스 유틸
└── tests/
    ├── __init__.py
    └── test_orchestrator.py
```

## 작업 내역

### (a) `app_streamlit/data_loader.py` 작성

- 학습 CSV(`data/processed/national_train_features.csv`) 로드 함수
- 함수 시그니처:
  ```python
  def load_history_and_forecast(
      csv_path: str,
      region: str,
      start_time: datetime,
  ) -> tuple[list[dict], list[dict], list[float]]:
      """
      반환:
        history: [{timestamp, generation_mwh, ...기상변수}] × 24
        forecast: [{timestamp, ...기상변수}] × 48  (generation_mwh 제외)
        actuals: [실측 발전량] × 48  (mpc_oracle 및 시뮬 정답 비교용)
      """
  ```
- 검증:
  - region이 CSV에 없으면 `ValueError("region not found: {region}")`
  - start_time이 CSV 범위 밖이면 `ValueError("start_time out of range")`
  - start_time 기준 [-24h, +48h] 72시간 연속 데이터가 없으면 `ValueError("insufficient continuous window")`
- 1시간 간격이 유지되는지 체크 (round 2-2-pre에서 했던 것과 동일)
- CSV의 컬럼명에 맞춰 history/forecast dict 구성. **기상변수는 `/predict_horizon` 스키마(`HistoryPoint`, `ForecastPoint`)가 요구하는 필드와 정확히 일치해야 함** — `app/schemas.py` 확인 후 매핑.

### (b) `app_streamlit/orchestrator.py` 작성

- 함수 시그니처:
  ```python
  def run_mpc_simulation(
      start_time: datetime,
      region: str,
      initial_soc: float = 0.5,
      policies: list[str] = ("naive", "xgb_lookahead", "mpc_xgb"),
      api_base_url: str = "http://localhost:8000",
      api_key: str = "dev-key-change-me",
      csv_path: str = "data/processed/national_train_features.csv",
      timeout: float = 10.0,
  ) -> dict:
  ```

- 동작 순서:
  1. `data_loader.load_history_and_forecast()`로 history 24h + forecast 48h + actuals 48h 슬라이스
  2. `/predict_horizon` 1번 호출 (horizon=48). `requests` 사용 (round 2-2-pre는 urllib였지만 본 작업은 requests 권장 — `requirements.txt`에 이미 있으면 그대로, 없으면 추가)
  3. 받은 predictions 48개를 3개 정책이 공유:
     - `naive`: 예측 안 씀 → 정책 내부에서 SOC 0.20~0.80 if-else
     - `xgb_lookahead`: predictions[0:24] 사용 (시뮬 24h 동안 next-1h만 보면 됨)
     - `mpc_xgb`: predictions[0:48] 사용 (Rolling Horizon용)
  4. 각 정책마다 기존 `run_simulation()` 함수 호출 (시뮬 길이 24h, initial_soc 전달)
  5. 결과 dict 조립 후 반환

- **반환 dict 스키마**:
  ```python
  {
      "meta": {
          "start_time": "2023-06-15T09:00:00",
          "region": "서울특별시",
          "initial_soc": 0.5,
          "policies": ["naive", "xgb_lookahead", "mpc_xgb"],
          "api_base_url": "http://localhost:8000",
          "horizon_used": 48,
          "sim_length": 24,
      },
      "predictions": [...],  # /predict_horizon 응답 48개 (운영자 화면 그래프용)
      "actuals": [...],      # 실측 발전량 48개 (정답 곡선용)
      "results": {
          "naive":         {hourly: [...], summary: {net_revenue, self_sufficiency, ...}},
          "xgb_lookahead": {hourly: [...], summary: {...}},
          "mpc_xgb":       {hourly: [...], summary: {...}},
      },
      "api_calls": {
          "predict_horizon": {
              "url": "...",
              "elapsed_ms": 50.2,
              "horizon": 48,
              "status": 200,
          },
      },
      "elapsed_ms": 261.4,  # 전체 함수 실행 시간
  }
  ```

- `hourly`: 시간별 SOC, charge, discharge, grid_buy, grid_sell, generation 등. `run_simulation()`이 이미 만들고 있는 구조 그대로 옮기면 됨.
- `summary`: net_revenue, self_sufficiency, total_charge, total_discharge 등 핵심 지표.

### (c) API 호출 에러 처리

- ConnectionError → `RuntimeError("API server unreachable at {url}")`
- Timeout → `RuntimeError("API server timeout after {timeout}s")`
- status_code != 200 → `RuntimeError("API error {status}: {body}")`
- 422 (validation) → 위 메시지에 응답 본문 포함 (디버깅용)

### (d) `app_streamlit/tests/test_orchestrator.py` 작성

테스트 항목 (최소 7개):

1. **test_normal_case**: 정상 입력 → 반환 dict의 meta/predictions/actuals/results/api_calls 키 존재, results 안에 3개 정책 모두 있음
2. **test_invalid_region**: 존재하지 않는 region → `ValueError`
3. **test_start_time_out_of_range**: CSV 범위 밖 시점 → `ValueError`
4. **test_insufficient_window**: start_time이 CSV 끝부분이라 +48h가 안 잡힘 → `ValueError`
5. **test_initial_soc_boundary**: initial_soc=0.0과 1.0 모두 정상 실행. 0.0 미만/1.0 초과는 `ValueError`
6. **test_api_unreachable**: api_base_url을 일부러 잘못된 포트(`http://localhost:9999`)로 지정 → `RuntimeError`
7. **test_predictions_shared**: 반환된 predictions가 정확히 48개이고, results의 각 정책이 동일 predictions를 기반으로 했는지 확인 (api_calls.predict_horizon이 정확히 1회)

테스트 실행 전 uvicorn 서버 띄워야 함. 테스트 파일 docstring에 "이 테스트는 `uvicorn app.main:app` 실행 중일 때만 통과한다" 명시. (API unreachable 테스트는 잘못된 포트로 직접 지정하므로 서버 죽일 필요 없음)

## 검증

1. `pytest app_streamlit/tests/ -v` → 7개 테스트 모두 통과
2. 정상 케이스 1회 실행 후 stdout에 출력:
   - `elapsed_ms` (~260ms 목표, ±100ms 허용)
   - 3개 정책의 net_revenue / self_sufficiency 비교 (Phase 2 발견 재현되는지 — mpc_xgb의 net_revenue가 xgb_lookahead보다 높고, self_sufficiency는 낮은 패턴 확인)
3. `api_calls.predict_horizon` 호출이 정확히 1번인지 확인

## 주의사항

- **기존 `run_simulation()` 함수 시그니처/내부 로직 절대 수정하지 말 것**. 24h 길이로 호출만 하면 됨. 만약 길이 인자가 없으면, `run_simulation()` 호출 후 결과를 24h로 슬라이스하는 식으로 우회.
- 학습 CSV의 시간대(Asia/Seoul vs UTC) 확인하고 일관되게 처리. `/predict_horizon`의 start_time 검증 로직과 충돌 없게.
- `naive` 정책은 API를 안 쓰지만, 반환 dict의 일관성을 위해 predictions는 전체 함수가 받아온 것 그대로 포함됨. naive 결과 내부에서 predictions를 참조하지 않으면 됨.
- 기존 시뮬레이터/정책 코드 import 경로 확인: round 1 시리즈에서 `src/` 어딘가에 있을 것. orchestrator.py 상단 import에서 명확히.
- requests 라이브러리 사용 — `requirements.txt`에 없으면 추가하고 보고.

## 끝나면 알려줄 것

1. 새로 만든 파일 목록
2. 테스트 결과 (`pytest -v` 출력)
3. 정상 케이스 stdout (elapsed_ms, 3개 정책 비교 지표)
4. `api_calls.predict_horizon` 호출 횟수 확인
5. requests 라이브러리 추가 여부
6. **Phase 2 발견 재현 여부**: mpc_xgb net_revenue > xgb_lookahead net_revenue, mpc_xgb self_sufficiency < xgb_lookahead self_sufficiency 패턴이 보이는지 (단일 케이스라 정확한 +49.53% / -17.48pt는 안 나올 수 있음 — 부호 방향만 일치하면 OK)
7. 막힌 부분/이상한 부분 있으면 같이
