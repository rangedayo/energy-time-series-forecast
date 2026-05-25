"""FastAPI entry point for the solar power forecast service."""
from __future__ import annotations

import asyncio
import logging
import pickle
import sys
from contextlib import asynccontextmanager

import numpy as np
import xgboost as xgb
from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import ENCODER_PATH, FEATURE_ORDER, MODEL_PATH
from app.exceptions import (
    unhandled_exception_handler,
    validation_error_handler,
    value_error_handler,
)
from app.inference import (
    build_feature_vector,
    predict_batch,
    predict_horizon,
    predict_single,
)
from app.middleware import RequestLoggingMiddleware
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HorizonPrediction,
    HorizonRequest,
    HorizonResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.security import verify_api_key

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading artifacts")

    feature_order = list(FEATURE_ORDER)
    app.state.feature_order = feature_order
    logger.info(
        "Feature order taken from app.config.FEATURE_ORDER (%d features)",
        len(feature_order),
    )

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    app.state.model = booster
    logger.info("XGBoost model loaded from %s", MODEL_PATH)

    booster_names = booster.feature_names
    if booster_names is None:
        raise RuntimeError(
            "XGBoost booster has no feature_names; cannot verify against "
            "FEATURE_ORDER. Model artifact may be corrupted."
        )
    if list(booster_names) != feature_order:
        raise RuntimeError(
            "FEATURE_ORDER mismatch with booster.feature_names.\n"
            f"  config:  {feature_order}\n"
            f"  booster: {list(booster_names)}"
        )
    logger.info("Feature-name check passed: booster matches FEATURE_ORDER")

    with open(ENCODER_PATH, "rb") as f:
        region_encoder = pickle.load(f)
    app.state.region_encoder = region_encoder
    region_map = {str(name): int(code) for code, name in enumerate(region_encoder.classes_)}
    app.state.region_map = region_map
    logger.info(
        "Region encoder loaded from %s (%d regions: %s)",
        ENCODER_PATH,
        len(region_map),
        ", ".join(region_map.keys()),
    )

    logger.info("Startup complete")
    try:
        yield
    finally:
        logger.info("Shutdown: releasing artifacts")
        app.state.model = None
        app.state.region_encoder = None
        app.state.feature_order = None
        app.state.region_map = None
        logger.info("Shutdown complete")


app = FastAPI(
    title="태양광 발전량 예측 API",
    description="전국 17개 시도의 시간당 태양광 발전량(MWh) 예측. XGBoost 모델 기반.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/", tags=["meta"], summary="서비스 식별 정보")
async def root():
    return {
        "service": "solar-power-forecast",
        "model": "national_xgboost",
        "status": "ok",
    }


@app.get(
    "/health",
    tags=["meta"],
    summary="헬스 체크",
    description="모델/인코더/피처 리스트가 모두 로드되었는지 확인. 누락 시 503.",
)
async def health():
    loaded = (
        getattr(app.state, "model", None) is not None
        and getattr(app.state, "region_encoder", None) is not None
        and getattr(app.state, "feature_order", None) is not None
    )
    if not loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "model artifacts not loaded"},
        )
    return {
        "status": "healthy",
        "feature_count": len(app.state.feature_order),
        "region_count": len(app.state.region_map),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="단일 시점 발전량 예측",
    description=(
        "단일 (region, timestamp) 시점의 발전량(MWh)을 예측한다. "
        "X-API-Key 헤더 필수."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    vector = build_feature_vector(
        request, app.state.region_encoder, app.state.feature_order
    )
    value = predict_single(app.state.model, vector, app.state.feature_order)
    return PredictionResponse(
        predicted_power_mwh=value,
        region=request.region,
        timestamp=request.timestamp,
    )


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    summary="배치 예측 (최대 1000건)",
    description=(
        "여러 (region, timestamp) 시점을 한 번에 예측한다. 항목 수: 1~1000. "
        "XGBoost 추론은 별도 스레드에서 실행되어 이벤트 루프를 막지 않는다. "
        "X-API-Key 헤더 필수."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def predict_batch_endpoint(
    batch: BatchPredictionRequest,
) -> BatchPredictionResponse:
    rows = [
        build_feature_vector(
            item, app.state.region_encoder, app.state.feature_order
        )
        for item in batch.items
    ]
    feature_matrix = np.vstack(rows)

    loop = asyncio.get_event_loop()
    values = await loop.run_in_executor(
        None,
        predict_batch,
        app.state.model,
        feature_matrix,
        app.state.feature_order,
    )

    predictions = [
        PredictionResponse(
            predicted_power_mwh=float(value),
            region=item.region,
            timestamp=item.timestamp,
        )
        for item, value in zip(batch.items, values)
    ]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


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
