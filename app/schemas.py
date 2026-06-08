"""Pydantic v2 request/response models for the prediction API."""
from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import REGION_LIST_SORTED, VALID_REGIONS

_MIN_VALID_DATE = datetime(1900, 1, 1)
_MAX_VALID_DATE = datetime(2100, 12, 31, 23, 59, 59)


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


class HistoryPoint(BaseModel):
    timestamp: datetime = Field(..., description="과거 시점 (ISO 8601)")
    power_mwh: float = Field(
        ...,
        ge=0,
        le=50000,
        description="해당 시점의 실측 발전량 (MWh). 음수 불가, 일괄 상한 50000.",
    )


class ForecastPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime = Field(..., description="예측 대상 시점 (ISO 8601)")
    irradiance: float = Field(..., ge=0, le=5.0, description="일사량 (MJ/m²)")
    temperature: float = Field(..., alias="기온", description="기온 (°C)")
    precipitation: float = Field(..., ge=0, alias="강수량", description="강수량 (mm)")
    humidity: float = Field(..., ge=0, le=100, alias="습도", description="습도 (%)")
    sunshine: float = Field(..., ge=0, le=1.0, alias="일조", description="일조시간 비율 (0~1)")
    cloud_cover: float = Field(..., ge=0, le=10, alias="전운량", description="전운량 (0~10)")


class HorizonRequest(BaseModel):
    region: str = Field(..., description="17개 시도 또는 '전국합산'")
    start_time: datetime = Field(
        ...,
        description="예측 시작 시점 (ISO 8601). predictions[0]의 timestamp가 됨.",
    )
    horizon: int = Field(
        ..., ge=1, le=48, description="예측 길이 (시간 단위). 1~48."
    )
    history: list[HistoryPoint] = Field(
        ...,
        min_length=24,
        max_length=24,
        description="start_time 직전 24시간의 실측 발전량. 정확히 24개.",
    )
    forecast: list[ForecastPoint] = Field(
        ...,
        min_length=1,
        max_length=48,
        description="start_time부터 horizon시간의 기상 예보. 정확히 horizon개.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "region": "전라남도",
                "start_time": "2023-07-15T13:00:00",
                "horizon": 24,
                "history": [
                    {"timestamp": "2023-07-14T13:00:00", "power_mwh": 185.0},
                    # ... 24개
                ],
                "forecast": [
                    {
                        "timestamp": "2023-07-15T13:00:00",
                        "irradiance": 2.95,
                        "기온": 29.1,
                        "강수량": 0.0,
                        "습도": 60.0,
                        "일조": 0.9,
                        "전운량": 2.0,
                    },
                    # ... horizon개
                ],
            }
        }
    )

    @field_validator("region")
    @classmethod
    def _region_must_be_known(cls, v: str) -> str:
        if v not in VALID_REGIONS:
            allowed = ", ".join(REGION_LIST_SORTED)
            raise ValueError(
                f"Unknown region {v!r}. Allowed regions: [{allowed}]."
            )
        return v

    @field_validator("start_time")
    @classmethod
    def _start_time_within_sane_range(cls, v: datetime) -> datetime:
        if v < _MIN_VALID_DATE or v > _MAX_VALID_DATE:
            raise ValueError(
                f"start_time {v.isoformat()} out of sane range "
                f"[{_MIN_VALID_DATE.isoformat()}, {_MAX_VALID_DATE.isoformat()}]"
            )
        return v

    @model_validator(mode="after")
    def _check_lengths_and_timestamps(self) -> "HorizonRequest":
        if len(self.forecast) != self.horizon:
            raise ValueError(
                f"forecast length must equal horizon. "
                f"horizon={self.horizon}, forecast 길이={len(self.forecast)}"
            )

        for i, hp in enumerate(self.history):
            expected = self.start_time - timedelta(hours=24 - i)
            if hp.timestamp != expected:
                raise ValueError(
                    f"history[{i}] timestamp 연속성 위반: "
                    f"기대 {expected.isoformat()}, 실제 {hp.timestamp.isoformat()}"
                )

        for i, fp in enumerate(self.forecast):
            expected = self.start_time + timedelta(hours=i)
            if fp.timestamp != expected:
                raise ValueError(
                    f"forecast[{i}] timestamp 연속성 위반: "
                    f"기대 {expected.isoformat()}, 실제 {fp.timestamp.isoformat()}"
                )

        return self


class SimulateRequest(BaseModel):
    """운영 시뮬레이션(/simulate) 요청. React 프론트엔드 입력 패널과 1:1 대응."""

    region: str = Field(..., description="17개 시도 중 하나")
    start_time: datetime = Field(..., description="시뮬 시작 시각 (ISO 8601)")
    initial_soc: float = Field(
        0.5, ge=0.0, le=1.0, description="배터리 시작 충전 상태 (0=공, 1=만충)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "region": "전라남도",
                "start_time": "2022-06-15T09:00:00",
                "initial_soc": 0.5,
            }
        }
    )

    @field_validator("region")
    @classmethod
    def _region_must_be_known(cls, v: str) -> str:
        if v not in VALID_REGIONS:
            allowed = ", ".join(REGION_LIST_SORTED)
            raise ValueError(f"Unknown region {v!r}. Allowed regions: [{allowed}].")
        return v


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
