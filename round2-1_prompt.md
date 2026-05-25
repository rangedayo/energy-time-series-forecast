# 작업: 태양광 발전량 예측 FastAPI — Multi-step 예측 엔드포인트 (round 2-1)

## 프로젝트 컨텍스트
- round1-1 ~ round1-3에서 단일/배치 예측 API가 완성된 상태:
  - `/predict` (단일 시점), `/predict_batch` (최대 1000건)
  - lifespan으로 모델/인코더/feature_order 로드
  - API Key 검증, 로깅 미들웨어, Global Exception Handler 완비
  - 8개 회귀 테스트 통과 중
- Phase 2에서 MPC 도입 결과 확정: mpc_xgb가 xgb_lookahead 대비 net_revenue +49.53%, mpc_xgb ≈ mpc_oracle (차이 0.08%)
- MPC는 N=24 horizon에 대해 한 번에 24개 예측을 필요로 함
- 본 작업은 향후 Streamlit 운영자 대시보드(옵션 A: HTTP 분리 구조)가 호출할 예측 엔드포인트를 만드는 것

## round 2-1 목표
재귀적 multi-step 방식의 `/predict_horizon` 엔드포인트를 추가한다. 입력은 "과거 24시간 실측 발전량 + 미래 N시간 기상 예보", 출력은 "미래 N시간의 발전량 예측 시퀀스"이다.

**스코프 제한**:
- 정책(MPC/lookahead) 선택 입력 받지 않음 — 본 엔드포인트는 순수 예측 서비스
- 시뮬 결과(SOC/수익) 반환하지 않음 — 호출자(Streamlit)가 별도 계산
- 외부 기상 API 연동하지 않음 — forecast는 클라이언트가 페이로드로 제공

## 작업 내역

### (a) app/schemas.py에 4개 모델 추가

기존 `PredictionRequest`/`PredictionResponse`는 그대로 둔다. 다음을 추가:

```python
class HistoryPoint(BaseModel):
    timestamp: datetime = Field(..., description="과거 시점 (ISO 8601)")
    power_mwh: float = Field(..., ge=0, le=50000, description="해당 시점의 실측 발전량 (MWh). 음수 불가, 일괄 상한 50000.")

class ForecastPoint(BaseModel):
    timestamp: datetime = Field(..., description="예측 대상 시점 (ISO 8601)")
    irradiance: float = Field(..., ge=0, le=5.0, description="일사량 (MJ/m²)")
    기온: float = Field(..., alias="temperature", description="기온 (°C)")
    강수량: float = Field(..., ge=0, alias="precipitation", description="강수량 (mm)")
    습도: float = Field(..., ge=0, le=100, alias="humidity", description="습도 (%)")
    일조: float = Field(..., ge=0, le=1.0, alias="sunshine", description="일조시간 비율 (0~1)")
    전운량: float = Field(..., ge=0, le=10, alias="cloud_cover", description="전운량 (0~10)")

    model_config = {"populate_by_name": True}

class HorizonRequest(BaseModel):
    region: str = Field(..., description="17개 시도 또는 '전국합산'")
    start_time: datetime = Field(..., description="예측 시작 시점 (ISO 8601). predictions[0]의 timestamp가 됨.")
    horizon: int = Field(..., ge=1, le=48, description="예측 길이 (시간 단위). 1~48.")
    history: list[HistoryPoint] = Field(..., min_length=24, max_length=24, description="start_time 직전 24시간의 실측 발전량. 정확히 24개.")
    forecast: list[ForecastPoint] = Field(..., min_length=1, max_length=48, description="start_time부터 horizon시간의 기상 예보. 정확히 horizon개.")

    # region 검증: 기존 PredictionRequest와 동일한 field_validator 적용
    # (VALID_REGIONS 사용, 에러 메시지에 REGION_LIST_SORTED 노출)

    # start_time 검증: 1900-01-01 ~ 2100-12-31 (명백히 비정상 값만 거름)

    # model_validator(mode='after')로 cross-field 검증:
    #   1. len(forecast) == horizon
    #   2. history의 timestamp들이 start_time 직전 24시간과 정확히 일치
    #      (start_time - timedelta(hours=24), -23, ..., -1; 1시간 간격)
    #   3. forecast의 timestamp들이 start_time부터 horizon시간과 정확히 일치
    #      (start_time, +1, +2, ..., +horizon-1; 1시간 간격)
    #   위반 시 ValueError → 422

    model_config = {
        "json_schema_extra": {
            "example": {
                "region": "전라남도",
                "start_time": "2023-07-15T13:00:00",
                "horizon": 24,
                "history": [
                    {"timestamp": "2023-07-14T13:00:00", "power_mwh": 185.0},
                    # ... 24개 (예시에선 일부만 적고 주석으로 생략 표시)
                ],
                "forecast": [
                    {
                        "timestamp": "2023-07-15T13:00:00",
                        "irradiance": 2.95, "기온": 29.1, "강수량": 0.0,
                        "습도": 60.0, "일조": 0.9, "전운량": 2.0,
                    },
                    # ... horizon개
                ],
            }
        }
    }


class HorizonPrediction(BaseModel):
    timestamp: datetime
    predicted_power_mwh: float
    step: int = Field(..., description="1-indexed. step=1은 t+1 (start_time과 동일)")


class HorizonResponse(BaseModel):
    region: str
    start_time: datetime
    horizon: int
    predictions: list[HorizonPrediction]
    model_version: str = "national_xgboost_v1"
    method: str = "recursive_multistep"
```

**검증 구현 노트**:
- `field_validator`로 region 검증은 기존 `PredictionRequest`와 동일 패턴 (`VALID_REGIONS`, 에러 메시지에 `REGION_LIST_SORTED` 노출)
- `field_validator`로 `start_time` 1900~2100 검증
- `model_validator(mode='after')`로 horizon ↔ forecast 길이 일치, history/forecast timestamp 연속성 검증
- timestamp 연속성 위반 시 에러 메시지에 "기대: X, 실제: Y" 형태로 어긋난 첫 지점 명시

### (b) app/inference.py에 재귀 multi-step 로직 추가

기존 `build_feature_vector`, `predict_single`, `predict_batch`는 그대로 두고 추가:

```python
def predict_horizon(
    model,
    region_encoder,
    feature_order: list[str],
    request: HorizonRequest,
) -> list[float]:
    """
    재귀적 multi-step 예측.

    매 step마다 이전 step의 예측값을 lag로 사용하여 t+1, t+2, ..., t+horizon을 순차 계산.

    Returns:
        길이 horizon의 예측값 리스트 (MWh, 음수 클립 적용).
    """
    # 1. history를 deque/list로 들고 다님 (가장 최근값이 lag_1h가 됨)
    #    history는 timestamp 오름차순으로 정렬되어 있다고 가정 (validator가 보장)
    history_powers: list[float] = [h.power_mwh for h in request.history]  # 길이 24

    predictions: list[float] = []
    for step in range(request.horizon):
        forecast_point = request.forecast[step]
        target_ts = forecast_point.timestamp

        # 2. lag/rolling을 history_powers에서 계산
        #    feature_engineering_national.py와 정확히 동일한 공식이어야 함
        lag_1h = history_powers[-1]
        lag_2h = history_powers[-2]
        lag_3h = history_powers[-3]
        lag_24h = history_powers[-24]  # always 24시간 전. 매 step 의미는 다르지만 인덱스는 같음
        power_diff_1h = lag_1h - lag_2h
        power_diff_2h = lag_1h - lag_3h
        rolling_mean_3h = sum(history_powers[-3:]) / 3.0
        rolling_mean_6h = sum(history_powers[-6:]) / 6.0
        # 표준편차는 numpy로
        last_3 = np.array(history_powers[-3:], dtype=np.float64)
        rolling_std_3h = float(last_3.std(ddof=0))

        # 3. 가짜 PredictionRequest 만들어서 기존 build_feature_vector 재활용
        #    (alias 처리 신경쓰지 말고, 내부 호출이니 영문/한글 어느 쪽이든 통일)
        pseudo_req = PredictionRequest(
            timestamp=target_ts,
            region=request.region,
            irradiance=forecast_point.irradiance,
            기온=forecast_point.기온,
            강수량=forecast_point.강수량,
            습도=forecast_point.습도,
            일조=forecast_point.일조,
            전운량=forecast_point.전운량,
            lag_1h=lag_1h,
            lag_2h=lag_2h,
            lag_3h=lag_3h,
            lag_24h=lag_24h,
            power_diff_1h=power_diff_1h,
            power_diff_2h=power_diff_2h,
            rolling_mean_3h=rolling_mean_3h,
            rolling_mean_6h=rolling_mean_6h,
            rolling_std_3h=rolling_std_3h,
        )

        feature_vector = build_feature_vector(pseudo_req, region_encoder, feature_order)
        pred = predict_single(model, feature_vector)  # NaN/Inf/음수 처리는 기존 함수가 담당

        predictions.append(pred)
        # 4. history_powers에 이번 예측 추가 (다음 step의 lag로 쓰임)
        history_powers.append(pred)

    return predictions
```

**중요 주의**:
- `PredictionRequest`가 입력 검증을 또 트리거하니, 내부 호출 시 검증 실패하지 않도록 forecast값들이 PredictionRequest 스키마 범위 안에 있어야 함 (이미 ForecastPoint에서 동일한 범위로 검증되니 OK)
- 또는 별도의 내부용 dataclass를 만들어 우회해도 됨 — 어느 쪽이 깔끔한지 판단해서 선택
- `lag_24h = history_powers[-24]`는 첫 step에선 24시간 전 실측이지만, step≥25면 그것도 예측값이 됨. 다만 horizon ≤ 48이므로 step 25~48에선 lag_24h가 step (n-24)의 예측값. 자기참조 정상 동작.

### (c) app/main.py에 엔드포인트 추가

```python
@app.post(
    "/predict_horizon",
    response_model=HorizonResponse,
    tags=["prediction"],
    summary="N시간 발전량 예측 (재귀 multi-step)",
    description=(
        "start_time부터 horizon(1~48)시간의 발전량을 예측한다. "
        "재귀적 multi-step 방식으로 t+1 예측값을 t+2의 lag로 사용. "
        "MPC 등 미래 시간 구간을 한 번에 받아야 하는 다운스트림 정책 솔버용. "
        "X-API-Key 헤더 필수."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def predict_horizon_endpoint(
    request: HorizonRequest,
) -> HorizonResponse:
    loop = asyncio.get_event_loop()
    values = await loop.run_in_executor(
        None,
        predict_horizon,
        app.state.model,
        app.state.region_encoder,
        app.state.feature_order,
        request,
    )

    predictions = [
        HorizonPrediction(
            timestamp=request.forecast[i].timestamp,
            predicted_power_mwh=float(v),
            step=i + 1,
        )
        for i, v in enumerate(values)
    ]
    return HorizonResponse(
        region=request.region,
        start_time=request.start_time,
        horizon=request.horizon,
        predictions=predictions,
    )
```

**중요**:
- 재귀 루프 안에서 XGBoost를 horizon번 호출하므로 horizon=48이면 충분히 무거움 (수십 ms ~ 수백 ms). 반드시 `run_in_executor`로 감싸서 이벤트 루프 막지 않게.
- 단순 batch와 달리 step 간 순차 의존성이 있어 vectorize 불가 — 재귀 루프 그대로 유지.

### (d) app/tests/test_api.py에 회귀 테스트 6개 추가

기존 8개 테스트는 건드리지 말고 다음을 추가:

```python
def _make_valid_horizon_payload(horizon: int = 24) -> dict:
    """검증 통과하는 정상 페이로드 생성 헬퍼."""
    start = datetime(2023, 7, 15, 13, 0, 0)
    history = [
        {"timestamp": (start - timedelta(hours=24-i)).isoformat(),
         "power_mwh": 100.0 + i}
        for i in range(24)
    ]
    forecast = [
        {"timestamp": (start + timedelta(hours=i)).isoformat(),
         "irradiance": 2.5, "기온": 28.0, "강수량": 0.0,
         "습도": 60.0, "일조": 0.8, "전운량": 3.0}
        for i in range(horizon)
    ]
    return {
        "region": "전라남도",
        "start_time": start.isoformat(),
        "horizon": horizon,
        "history": history,
        "forecast": forecast,
    }


def test_predict_horizon_valid_24():
    """정상 24시간 요청 → 200, predictions 길이 24, 모두 ≥ 0, step 1~24."""
    payload = _make_valid_horizon_payload(horizon=24)
    r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 24
    assert all(p["predicted_power_mwh"] >= 0 for p in body["predictions"])
    assert [p["step"] for p in body["predictions"]] == list(range(1, 25))
    assert body["method"] == "recursive_multistep"


def test_predict_horizon_valid_48():
    """horizon=48 정상 동작."""
    payload = _make_valid_horizon_payload(horizon=48)
    r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 48


def test_predict_horizon_horizon_out_of_range():
    """horizon=0, 49 → 422."""
    for bad_horizon in [0, 49, 100]:
        payload = _make_valid_horizon_payload(horizon=24)
        payload["horizon"] = bad_horizon
        # forecast 길이는 그대로 두면 cross-field validator도 같이 걸림. horizon 자체 검증 확인용이라 둘 중 하나라도 422면 통과
        r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
        assert r.status_code == 422


def test_predict_horizon_history_length_mismatch():
    """history가 23개 또는 25개 → 422."""
    for bad_len in [23, 25]:
        payload = _make_valid_horizon_payload(horizon=24)
        if bad_len < 24:
            payload["history"] = payload["history"][:bad_len]
        else:
            payload["history"].append(payload["history"][-1])
        r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
        assert r.status_code == 422


def test_predict_horizon_forecast_length_mismatch():
    """horizon=24인데 forecast 23개 → 422."""
    payload = _make_valid_horizon_payload(horizon=24)
    payload["forecast"] = payload["forecast"][:23]
    r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 422


def test_predict_horizon_history_timestamp_gap():
    """history에 1시간 갭이 있으면 → 422."""
    payload = _make_valid_horizon_payload(horizon=24)
    # 중간 시점 하나를 2시간 뒤로 밀어서 갭 발생시키기
    h = payload["history"]
    h[10]["timestamp"] = (datetime.fromisoformat(h[10]["timestamp"]) + timedelta(hours=2)).isoformat()
    r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 422


def test_predict_horizon_invalid_region():
    """가짜 region → 422 + 에러 메시지에 유효 region 목록 포함."""
    payload = _make_valid_horizon_payload()
    payload["region"] = "화성시"
    r = client.post("/predict_horizon", json=payload, headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 422
    assert "전라남도" in r.text  # 유효 region 목록이 노출되는지


def test_predict_horizon_requires_api_key():
    """API Key 없으면 → 401."""
    payload = _make_valid_horizon_payload()
    r = client.post("/predict_horizon", json=payload)  # 헤더 없음
    assert r.status_code == 401
```

## 검증 방법

구현 후 `uvicorn app.main:app --reload`로 띄우고 다음 5개 케이스를 `curl` 또는 `requests`로 직접 실행해서 결과 보여줘:

1. **정상 horizon=24**: 전라남도, 2023-07-15 13:00 시작, history 24개 + forecast 24개 → 200 OK + predictions 24개, step 1~24, 모두 양수
2. **정상 horizon=48**: 위와 동일한 시작이지만 forecast 48개 → 200 OK + predictions 48개
3. **horizon=49**: → 422
4. **history 23개**: → 422
5. **forecast의 timestamp 갭**: forecast[5]를 2시간 뒤로 밀기 → 422 (에러 메시지에 "기대 X, 실제 Y" 형태로 어긋난 지점 명시 확인)
6. **API Key 없음**: → 401

각 케이스의 요청 페이로드 + 응답 본문(또는 응답 본문 일부)을 함께 출력해줘. **horizon=24와 horizon=48의 응답 시간(ms)**도 같이 측정해서 보여주면 좋음 (재귀 multi-step의 비용 감각 잡기용).

회귀 테스트도 `pytest app/tests/test_api.py -v`로 기존 8개 + 신규 8개 모두 통과 확인.

## 주의사항

- **feature_engineering_national.py와 lag/rolling 계산 공식 완전 일치**: 특히 `rolling_std_3h`는 학습 시 pandas의 기본값(`ddof=1`인지 `ddof=0`인지)을 확인하고 동일하게 적용. 학습 코드의 `df.rolling(3).std()`는 pandas 기본이라 `ddof=1`임. 위 예시 코드의 `ddof=0`은 잘못된 예시일 수 있으니 학습 코드를 다시 확인하고 맞춰줘.
- **재귀 루프 안의 PredictionRequest 재검증 비용**: 매 step마다 Pydantic 검증을 또 트리거하는 게 부담된다면, 내부용 dataclass를 만들어 `build_feature_vector`를 그쪽으로도 받게 시그니처 확장 가능. 다만 기존 함수 시그니처를 깨지 않는 선에서.
- **음수 클립은 step 누적값에도 적용됨**: `predict_single`에서 음수 → 0 처리하므로 history_powers에 들어가는 값도 0 이상. 야간 시간대 예측이 음수로 떨어져도 자동 처리.
- **회귀 테스트의 `TEST_API_KEY`, `client`, `datetime`, `timedelta` import**는 기존 테스트 파일의 imports에 맞춰 추가.
- **README의 "엔드포인트" 섹션 업데이트**: `/predict_horizon` 한 줄 설명 추가. 호출 예시 코드(`requests`)도 한 블록 추가.

## 작업 끝나면 알려줘야 할 것

1. 추가/수정한 파일 목록
2. 위 5개 케이스 + 응답시간 측정 결과
3. `pytest` 전체 통과 로그 (16개 테스트)
4. README 업데이트 diff
