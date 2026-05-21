# 작업: 태양광 발전량 예측 FastAPI (2/3단계) — 예측 엔드포인트 본체

## 전제
1단계 완료 + 추가 지침까지 적용된 상태야:
- `app/config.py`에 `VALID_REGIONS: set[str]`, `REGION_LIST_SORTED: list[str]` 이미 export됨
- `app/schemas.py`에 `PredictionRequest`의 `region` 필드 + `field_validator` 이미 구현됨 (17개 지역 검증, 에러 메시지에 목록 포함)
- 모델/인코더/feature_order는 lifespan에서 `app.state`에 로드됨

## 2단계 목표
`/predict`와 `/predict_batch` 엔드포인트를 완성한다. 입력 스키마의 나머지 11개 필드를 추가하고, 파생 피처 계산 + 예측 로직을 구현한다.

## 작업 내역

### (a) app/schemas.py — 나머지 필드 추가

`PredictionRequest`에 이미 있는 `region` 외에 다음 11개 필드를 추가 (모두 Pydantic v2 `Field`로 검증):

```python
class PredictionRequest(BaseModel):
    timestamp: datetime = Field(..., description="예측 대상 시점 (ISO 8601)", examples=["2023-07-15T13:00:00"])
    region: str = Field(...)  # 이미 구현됨, 그대로 유지
    irradiance: float = Field(..., ge=0, le=5.0, description="일사량 (MJ/m²). 음수 불가, 물리적 상한 5.0")
    lag_1h: float = Field(..., ge=0, description="1시간 전 발전량 (MWh)")
    lag_2h: float = Field(..., ge=0, description="2시간 전 발전량 (MWh)")
    lag_3h: float = Field(..., ge=0, description="3시간 전 발전량 (MWh)")
    lag_24h: float = Field(..., ge=0, description="24시간 전 발전량 (MWh)")
    power_diff_1h: float = Field(..., description="1시간 전 대비 변화량 (음수 가능)")
    power_diff_2h: float = Field(..., description="2시간 전 대비 변화량 (음수 가능)")
    rolling_mean_3h: float = Field(..., ge=0, description="직전 3시간 평균 발전량")
    rolling_mean_6h: float = Field(..., ge=0, description="직전 6시간 평균 발전량")
    rolling_std_3h: float = Field(..., ge=0, description="직전 3시간 발전량 표준편차")
```

또한 클래스 레벨에 `model_config = {"json_schema_extra": {"example": {...}}}`로 Swagger UI용 전체 예시를 넣어. 예시 값은 전라남도 2023-07-15 13:00, 일사량 3.0, lag/rolling을 적당한 양수로 (50~150 사이) 채워서.

`PredictionResponse`:
```python
class PredictionResponse(BaseModel):
    predicted_power_mwh: float
    region: str
    timestamp: datetime
    model_version: str = "national_xgboost_v1"
```

`BatchPredictionRequest`: `items: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)`.
`BatchPredictionResponse`: `predictions: list[PredictionResponse]`, `count: int`.

### (b) app/inference.py — 예측 로직 본구현

함수 `build_feature_vector(request: PredictionRequest, region_encoder, feature_order: list[str]) -> np.ndarray`:

입력 `PredictionRequest`의 raw 값들과 `timestamp` / `irradiance`로부터 파생 피처 계산. **`src/features/feature_engineering_national.py`에서 학습 시 사용한 공식과 완전히 동일하게** (한 줄이라도 다르면 학습/추론 불일치):

- `hour = timestamp.hour`
- `month = timestamp.month`
- `day_of_week = timestamp.weekday()`  (pandas dayofweek와 동일하게 월=0)
- `is_weekend = 1 if day_of_week >= 5 else 0`
- `season`: 12,1,2→1 / 3,4,5→2 / 6,7,8→3 / 9,10,11→4
- `solar_altitude_proxy = max(0, np.sin(np.pi * (hour - 6) / 12))`
- `irrad_x_solar = irradiance * solar_altitude_proxy`
- `is_daytime = 1 if 6 <= hour <= 18 else 0`
- `region_code = int(region_encoder.transform([request.region])[0])`

이후 `feature_order` 순서대로 numpy array (shape: `(1, 18)`, dtype=float32 권장) 생성해서 반환. **feature_order는 lifespan에서 로드한 `feature_list_national.json`의 `engineered_features`를 그대로 따라야 함**. 학습 시점 컬럼 순서와 다르면 예측이 완전히 망가지므로, 함수 내부에서 길이가 18이 아니면 ValueError 발생시켜.

함수 `predict_single(model, feature_vector) -> float`:
- `pred = float(model.predict(feature_vector)[0])`
- `np.isfinite(pred)`가 False면 ValueError 발생 (3단계 Global Exception Handler가 잡음)
- 음수면 0으로 클립 (`behavioral_tests_national.py`의 행동 테스트 4번과 같은 논리 — 태양광은 음수 불가)
- 반환

함수 `predict_batch(model, feature_matrix: np.ndarray) -> np.ndarray`:
- shape `(N, 18)`을 받아서 `(N,)` 반환
- NaN/Inf 체크 + 음수 클립 동일 적용

### (c) app/main.py에 엔드포인트 추가

```python
POST /predict          → PredictionRequest → PredictionResponse
POST /predict_batch    → BatchPredictionRequest → BatchPredictionResponse
```

`/predict` 구현:
- 단일 예측. `build_feature_vector` → `predict_single` 호출.
- 동기 함수지만 FastAPI가 알아서 스레드풀에서 돌리므로 추가 처리 불필요.
- `def`로 정의해도 되고 `async def`로 정의 후 내부 호출만 동기로 해도 됨. 깔끔한 쪽 선택.

`/predict_batch` 구현:
- **반드시 `asyncio.get_event_loop().run_in_executor(None, predict_batch, model, X)`로 감쌀 것.**
- 이유: 배치 크기가 커지면 XGBoost가 수십~수백 ms 잡아먹어 이벤트 루프를 막기 때문.
- 처리 흐름: 입력 items 리스트 → 각각 build_feature_vector로 (1,18) → np.vstack으로 (N,18) → run_in_executor로 predict_batch 호출 → 결과를 PredictionResponse 리스트로 매핑.

두 엔드포인트 모두 API Key 헤더 의존성으로 보호. 1단계의 placeholder `app/security.py`에 일단 stub 함수 하나 만들고 (3단계에서 본구현 채울 예정):

```python
# app/security.py (stub for stage 2)
from fastapi import Header, HTTPException
from app.config import API_KEY

async def verify_api_key(x_api_key: str = Header(None)) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key
```

엔드포인트에 `Depends(verify_api_key)` 적용. (3단계에서 `APIKeyHeader` 클래스 기반으로 더 깔끔하게 재작성할 예정이지만 일단 stub으로 동작 확인.)

### (d) FastAPI 메타데이터 보강
`main.py`에서:
```python
app = FastAPI(
    title="태양광 발전량 예측 API",
    description="전국 17개 시도의 시간당 태양광 발전량(MWh) 예측. XGBoost 모델 기반.",
    version="1.0.0",
    lifespan=lifespan,
)
```

각 엔드포인트에 `summary`, `description`, `tags=["prediction"]` 등 메타데이터 채우기.

## 검증 방법
구현 후 `uvicorn app.main:app --reload`로 띄우고 다음 5개 케이스를 직접 실행해서 결과 보여줘:

1. **정상 단일 요청**: 전라남도 2023-07-15 13:00, 일사량 3.0, lag/rolling 적당히 채워서 → 200 OK + `predicted_power_mwh` 값 (양수)
2. **잘못된 region**: "화성시" → 422, 에러 메시지에 17개 유효 region 목록 포함되는지 확인
3. **음수 irradiance**: -1.0 → 422 (`ge=0` 검증에 걸림)
4. **/predict_batch 정상**: 3개 시점 동시 → 200 OK + `count: 3`, `predictions` 길이 3
5. **/predict_batch 한도 초과**: 1001개 → 422 (`max_length=1000` 검증에 걸림)
6. **API Key 없음**: 헤더 없이 호출 → 401

각 케이스의 `curl` 명령어 + 응답 본문을 함께 출력해줘.

## 주의사항
- `feature_engineering_national.py`의 파생 피처 계산 공식과 정확히 일치해야 함 (특히 `solar_altitude_proxy`의 `clip(0)` 처리, `season` 매핑, `day_of_week`의 월=0 기준).
- `feature_order`는 반드시 lifespan에서 로드한 `app.state.feature_order` 사용 (하드코딩 금지).
- region encoder도 마찬가지로 `app.state.region_encoder` 사용 (config의 module-level 로드본은 검증용이고, 추론용은 lifespan본).