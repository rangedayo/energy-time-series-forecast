옵션 A로 진행해. 발견사항을 바탕으로 다음과 같이 보강해줘.

## 1. 학습 코드에서 정확한 피처 셋 재확인

`src/models/train_xgboost_national.py`와 `src/tests/behavioral_tests_national.py`를 직접 열어서:
- NON_FEAT 상수의 정확한 값
- `feature_cols = [c for c in test.columns if c not in NON_FEAT]`로 만들어지는 컬럼의 **정확한 순서**

이 두 가지를 코드 인용해서 보고서로 먼저 보여줘. 학습 시점 컬럼 순서를 그대로 따라야 학습/추론이 일치하므로, 이 순서가 진실의 원천이야.

## 2. PredictionRequest에 원시 기상 5개 추가

기존 11개 필드에 다음 5개 추가 (필드명은 학습 코드와 동일하게 한글 유지하되, alias로 영문 이름도 받을 수 있게):

- `기온` (temperature): float, 합리적 범위 검증. 한반도 기준 -30 <= x <= 45 (°C)
- `강수량` (precipitation): float, ge=0, 합리적 상한 le=200 (mm/h, 극한 호우 상한)
- `습도` (humidity): float, ge=0, le=100 (%)
- `일조` (sunshine): float, ge=0, le=1 (hr, 시간당 일조시간은 0~1)
- `전운량` (cloud_cover): float, ge=0, le=10 (10분위)

기존 `irradiance` 필드는 학습 코드의 `일사량`에 매핑되므로 그대로 유지.

Pydantic v2 `Field(..., alias="기온")` 패턴으로 영문 변수명 + 한글 alias 둘 다 허용. 이유는 (a) Python 변수명은 영문이 안전하고, (b) 학습 코드와 한글 컬럼명을 매칭해야 하기 때문.

각 필드의 `description`에 단위와 출처 명시 (예: "기상청 ASOS 시간별 관측 — 기온 °C").

## 3. inference.build_feature_vector 재작성

피처 순서는 **학습 코드에서 산출된 정확한 순서**를 따라야 함. 위 (1) 보고에서 확인된 컬럼 순서를 그대로 사용.

dump한 24개 피처 순서가 클로드 코드 보고와 일치하는지 직접 검증:
0..5:  기온, 강수량, 습도, 일조, 일사량, 전운량
6:     region_code
7..12: hour, month, day_of_week, is_weekend, season, solar_altitude_proxy
13..19: lag_1h, lag_2h, lag_3h, lag_24h, power_diff_1h, power_diff_2h, rolling_mean_3h
20..21: rolling_mean_6h, rolling_std_3h
22..23: irrad_x_solar, is_daytime

이 순서를 `app/config.py`에 `FEATURE_ORDER: list[str]` 상수로 박아 (하드코딩이긴 하지만, 학습 모델이 바뀌지 않는 한 변하지 않음). `feature_list_national.json`의 engineered_features는 더 이상 권위 있는 소스가 아니므로 **참조 금지**. 대신 config.py 상단 주석에 다음 문구 명시:

```python
# 이 순서는 학습 시점의 train.columns - NON_FEAT 결과를 그대로 반영한다.
# feature_list_national.json의 engineered_features는 "엔지니어링된 피처 18개"의 목록일 뿐
# 학습 모델 입력(24개)이 아니다. 원시 기상 6개가 학습에 포함되어 있음을 주의.
```

검증: `model.get_booster().feature_names`가 None이거나, 위 24개 순서와 일치하는지 startup 시 확인 → 불일치 시 lifespan에서 명확한 에러로 fail-fast.

## 4. Swagger 예시값 보강

기존 1개 시점 예시에 5개 기상 필드 추가. 전라남도 2023-07-15 13:00 시나리오:
- 기온: 28.5
- 강수량: 0.0
- 습도: 65.0
- 일조: 0.9
- 일사량: 3.0 (기존)
- 전운량: 2.0

(맑은 여름 한낮 가정. 실제 학습 데이터에 비슷한 값이 많아 합리적 예시.)

## 5. README에 사항 메모

README의 "API 서빙" 섹션에 짧게 명시:
"본 API는 학습 모델이 24개 피처를 입력으로 사용함에 따라, 클라이언트가 원시 기상 6개(기온/강수량/습도/일조/일사량/전운량)와 시간/지역, 그리고 직전 시점들의 lag/rolling 통계를 함께 제공해야 한다. 실서비스에서는 기상청 API + 자체 발전 이력 DB와 결합해서 사용하는 것을 가정한다."

## 6. 6개 검증 케이스 실행

작업 끝나면 이전 프롬프트의 6개 케이스를 실행하되, 검증 케이스 1번의 응답값이 **합리적 범위(전라남도 한낮 발전량은 보통 수백 MWh 단위)**인지 함께 보고.