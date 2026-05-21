"""Pydantic v2 request/response models for the prediction API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import REGION_LIST_SORTED, VALID_REGIONS


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "timestamp": "2023-07-15T13:00:00",
                "region": "전라남도",
                "기온": 28.5,
                "강수량": 0.0,
                "습도": 65.0,
                "일조": 0.9,
                "irradiance": 3.0,
                "전운량": 2.0,
                "lag_1h": 120.0,
                "lag_2h": 110.0,
                "lag_3h": 95.0,
                "lag_24h": 130.0,
                "power_diff_1h": 10.0,
                "power_diff_2h": 25.0,
                "rolling_mean_3h": 108.3,
                "rolling_mean_6h": 95.5,
                "rolling_std_3h": 12.7,
            }
        },
    )

    timestamp: datetime = Field(
        ...,
        description="예측 대상 시점 (ISO 8601, 한국 표준시 가정)",
        examples=["2023-07-15T13:00:00"],
    )
    region: str = Field(..., description="17개 시도 중 하나")

    temperature: float = Field(
        ...,
        ge=-30,
        le=45,
        alias="기온",
        description="기상청 ASOS 시간별 관측 — 기온 (°C). 한반도 합리적 범위 -30..45",
    )
    precipitation: float = Field(
        ...,
        ge=0,
        le=200,
        alias="강수량",
        description="기상청 ASOS 시간별 관측 — 강수량 (mm/h). 0..200 (극한 호우 상한)",
    )
    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        alias="습도",
        description="기상청 ASOS 시간별 관측 — 상대 습도 (%). 0..100",
    )
    sunshine: float = Field(
        ...,
        ge=0,
        le=1,
        alias="일조",
        description="기상청 ASOS 시간별 관측 — 시간당 일조시간 (hr). 0..1",
    )
    irradiance: float = Field(
        ...,
        ge=0,
        le=5.0,
        description=(
            "기상청 ASOS 시간별 관측 — 일사량 (MJ/m²). 학습 컬럼명 '일사량'. "
            "음수 불가, 물리적 상한 5.0"
        ),
    )
    cloud_cover: float = Field(
        ...,
        ge=0,
        le=10,
        alias="전운량",
        description="기상청 ASOS 시간별 관측 — 전운량 (10분위). 0(맑음)..10(흐림)",
    )

    lag_1h: float = Field(..., ge=0, description="1시간 전 발전량 (MWh)")
    lag_2h: float = Field(..., ge=0, description="2시간 전 발전량 (MWh)")
    lag_3h: float = Field(..., ge=0, description="3시간 전 발전량 (MWh)")
    lag_24h: float = Field(..., ge=0, description="24시간 전 발전량 (MWh)")
    power_diff_1h: float = Field(..., description="1시간 전 대비 변화량 (음수 가능)")
    power_diff_2h: float = Field(..., description="2시간 전 대비 변화량 (음수 가능)")
    rolling_mean_3h: float = Field(..., ge=0, description="직전 3시간 평균 발전량")
    rolling_mean_6h: float = Field(..., ge=0, description="직전 6시간 평균 발전량")
    rolling_std_3h: float = Field(..., ge=0, description="직전 3시간 발전량 표준편차")

    @field_validator("region")
    @classmethod
    def _region_must_be_known(cls, v: str) -> str:
        if v not in VALID_REGIONS:
            allowed = ", ".join(REGION_LIST_SORTED)
            raise ValueError(
                f"Unknown region {v!r}. Allowed regions: [{allowed}]."
            )
        return v


class PredictionResponse(BaseModel):
    predicted_power_mwh: float
    region: str
    timestamp: datetime
    model_version: str = "national_xgboost_v1"


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
