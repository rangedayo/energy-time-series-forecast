- "사용자 피드백 수집", "단계 전환 의사결정" → **개념을 README나 보고서에 "만약 실서비스라면 이렇게 모니터링할 것이다"라는 회고 섹션으로 적는 정도가 적절.**
- "회귀 테스트" → 이미 `src/tests/behavioral_tests_national.py` 가 비슷한 역할을 하고 있음. API에서는 "예전 입력 X에 대해 예측이 Y±α 안에 들어오는가" 정도의 스냅샷 테스트를 추가하는 정도면 충분
- 클라우드 배포 → **로컬에서 Swagger UI로 호출 데모 영상을 찍는 것만으로도 포트폴리오 가치는 충분**
    - Swagger UI에서 Authorize → /predict 한 번 호출 → 응답 확인 30초 영상
    - 포트폴리오 README나 발표 자료에 임베드
    - 작업량: 5분, 효과 큼
- **Dockerfile 추가**
    - `docker build && docker run`으로 누구나 띄울 수 있게
    - 포트폴리오 reviewer가 "내 컴퓨터에서도 돌려볼 수 있다"는 신호 받음

- API 연결 준비
    - multipart, on_event 방식이 사라질 예정이라서 최신 방식인 lifespan로 바꿔서 코드 짜기
        
        → 서버 시작 시 한 번 로드해서 메모리에 들고 있어야 함. 이게 lifespan이 필요한 정확한 이유다.
        
    - 무거운 작업에는 run_in_executor 쓴다. → 모델 추론이니 당연
    - Global Exception Handler, 로깅 미들웨어 쓴다.
        
        → XGBoost 예측이 NaN/Inf 던질 수 있고 (행동 테스트 1번이 정확히 이걸 검증함), 입력값이 이상하면 예외가 날 수 있다.
        클라이언트에 스택트레이스 노출하지 않고 "예측 실패: 입력값 확인 요망" 같은 깔끔한 메시지로 변환해야 한다.
        
    - RequestBody에서 Field()를 사용
        - 피처들을 Field로 범위 검증을 박아둔다.
        - 타입 검사를 넘어 값의 범위, 길이, 패턴 등을 세밀하게 제어할 수 있다.
    - 허가된 사람만 쓸 수 있도록 api key 설정하기
        
        배포까지 한다면 누구나 호출하게 두면 안 되니까 헤더 기반 API Key 검증은 넣어두는 게 좋다.
        
- uvicorn app.main:app --reload
- http://localhost:8000/docs
- API키 : dev-key-change-me

### 디버깅

#### **1. 사용자 입력값을 모델 입력값에 맞춰 설정하지 않음.**

- 학습할 때 모델이 본 피처와 JSON 파일에 적힌 피처 목록이 서로 달랐다.
    
    `feature_list_national.json` 에는 18개 피처가 적혀 있음
    실제로 학습 코드(`train_xgboost_national.py`)는 24개 피처로 학습함
    
    → 차이 = 원시 기상 6개 (기온, 강수량, 습도, 일조, 일사량, 전운량)
    
    학습 코드는 그 JSON 파일을 안 보고, CSV에서 직접 컬럼을 뽑아서 학습했다. JSON은 학습 흐름에 끼지 않는 "참고용 메모" 같은 파일이었던 것. 
    XGBoost 모델은 학습 때 본 24개 피처를 그대로 받기를 기대하기 때문에, API에서 18개만 주면 → 즉시 에러난다.
    
- API 입력 스키마에 원시 기상 5개 (기온/강수량/습도/일조/전운량)를 추가. 일사량은 이미 있었음. → 총 16개 입력 필드 → 서버에서 파생 피처 계산해서 24개 만들어 모델에 전달했다.

☞ 교훈 : 

- API 만들 때 모델 입력 피처를 확인할 때는 반드시 학습 스크립트가 실제로 모델에 넘긴 컬럼을 봐야 한다. 별도 문서/JSON은 작성 시점과 학습 시점이 어긋날 수 있다.
- 이 불일치는 Swagger UI 띄우자마자 첫 호출에서 잡혔다. → API 한 겹 씌우는 작업 자체가 모델 검증의 한 형태로 작동함. 오프라인 배치만 돌릴 땐 안 보이던 문제가 API 인터페이스를 강제하면서 드러난 거.

### 결과

- 129.23 MWh가 전라남도 6~8월 13시 92건 표본 (평균 229.3, p10/p90 = 0/819) 분포 안에 자연스럽게 들어옴. 이 말인 즉슨,
    - 피처 순서 24개가 학습 모델과 정확히 맞음
    - 파생 피처 계산 공식이 학습 코드와 정확히 일치함
- fail-fast 검증
    - `ooster.feature_names == FEATURE_ORDER` 검증이 startup에 박혔다는 게 특히 잘한 부분
    - 앞으로 누군가 모델을 재학습하거나 피처를 바꾸면 → API 서버가 시작 자체를 거부함. 이전 24-피처 사건 같은 일이 또 일어나도 클로드 코드가 발견할 필요 없이 서버가 알아서 멈춘다. 이번 사고가 영구 안전장치로 변환됐다.
- pytest 8/8 통과
    - 24개 피처 순서가 학습과 정확히 일치
    - 파생 피처 공식 (solar_altitude_proxy, season, irrad_x_solar 등)이 학습 코드와 한 줄도 안 어긋남
    - 음수 클립 같은 후처리도 raw 결과와 동일하게 적용됨
    
    → 이 회귀 테스트가 앞으로 안전망임. 누가 piece 하나 잘못 건드려도 즉시 빨간불.
    
- Case 1 회귀 (snapshot 회귀 테스트)
    - 129.2322 MWh, ±0 — 2단계 결과와 완전히 동일. 3단계의 미들웨어/예외 핸들러/security 리팩토링이 예측 경로를 깨지 않았다는 보증.
- 미들웨어 로그
    - `[a20896c58733] POST /predict from 127.0.0.1` → `-> 200 (3.5ms)`
    - 같은 request_id가 들어옴/나감 두 줄에 일관되게 찍히는 거 좋다. 디버깅할 때 grep으로 한 요청의 전체 흐름 추적 가능. ms 단위 처리시간도 나중에 p95/p99 모니터링의 기반이 된다.
- OpenAPI 보안 스킴 노출
    - `/predict`, `/predict_batch`에 `security=[{'APIKeyHeader': []}]`가 OpenAPI 스펙에 정확히 박혔다는 게 중요 → Swagger UI의 "Authorize" 버튼이 진짜로 동작한다는 뜻
    - 사용자가 한 번 키 입력하면 그 세션 동안 모든 보호된 엔드포인트가 자동으로 헤더 붙어서 호출된다.

```markdown
[학습 파이프라인]                       [서빙 레이어 — 3단계로 완성]
preprocess_national.py                   app/
  → feature_engineering_national.py        ├── lifespan (모델/인코더 로드 + feature_names 검증)
  → train_xgboost_national.py              ├── /predict + /predict_batch (24-피처)
  → behavioral_tests_national.py           ├── Pydantic 입력 검증 (17 region + 범위)
                                            ├── API Key 인증
[모델 산출물]                              ├── 로깅 미들웨어
models/national_xgboost_model.json    ←   ├── Global Exception Handler
models/national_region_encoder.pkl    ←   └── tests/ (snapshot 회귀 포함)
```

### 스웨거ui 테스트

#### 예시1 : 새벽3시

```python
{
  "timestamp": "2023-07-15T03:00:00",
  "region": "전라남도",
  "기온": 22.0,
  "강수량": 0.0,
  "습도": 80.0,
  "일조": 0.0,
  "irradiance": 0.0,
  "전운량": 2.0,
  "lag_1h": 0.0,
  "lag_2h": 0.0,
  "lag_3h": 0.0,
  "lag_24h": 0.0,
  "power_diff_1h": 0.0,
  "power_diff_2h": 0.0,
  "rolling_mean_3h": 0.0,
  "rolling_mean_6h": 0.0,
  "rolling_std_3h": 0.0
}
```

**결과 :**

```python
{
  "predicted_power_mwh": 0.012701236642897129,
  "region": "전라남도",
  "timestamp": "2023-07-15T03:00:00",
  "model_version": "national_xgboost_v1"
}
```