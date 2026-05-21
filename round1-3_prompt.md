# 작업: 태양광 발전량 예측 FastAPI (3/3단계) — 운영 안정성

## 전제
1, 2단계 완료. /predict, /predict_batch 모두 정상 동작.
2단계에서 발견된 사항:
- curl @file의 chunked transfer encoding과 FastAPI ValidationError가 충돌해 400을 반환하는 케이스 있음 (클라이언트 측 이슈, 서버 정상)
- booster.feature_names == FEATURE_ORDER 검증이 lifespan에 있음 (이번 단계에서 깨지 말 것)

## 3단계 목표
프로덕션 수준의 안정성 장치를 추가한다: 로깅, 예외 처리, API Key 본구현, 회귀 테스트.

## 작업 내역

### (a) app/middleware.py — 로깅 미들웨어

BaseHTTPMiddleware를 상속해서 RequestLoggingMiddleware 구현:
- 요청 들어올 때마다: method, path, client IP, request_id (uuid4) 로깅
- 응답 나갈 때: status_code, 처리 시간 (ms) 로깅
- request_id를 response header `X-Request-ID`에 실어 보내기
- 형식: `[req_id] METHOD path -> status (Xms)`

main.py에 `app.add_middleware(RequestLoggingMiddleware)` 등록.

### (b) app/exceptions.py — Global Exception Handler

다음 핸들러 정의 후 main.py에서 `app.add_exception_handler(...)`로 등록:

1. `ValueError` 핸들러 → 400, `{"error": "invalid_input", "detail": str(exc)}`
   - inference.py의 NaN/Inf, feature_order 길이 불일치에서 발생
2. `RequestValidationError` 핸들러 → 422, 어떤 필드가 왜 틀렸는지 간결하게 (Case 2의 region 목록 노출은 유지)
3. 모든 `Exception` 핸들러 (최후방어선) → 500, `{"error": "internal_error", "detail": "예측 처리 중 오류가 발생했습니다."}`
   - **스택트레이스는 로그에만 남기고 클라이언트에는 노출 금지.**

### (c) app/security.py — API Key 검증 본구현

현재 2단계의 stub을 `APIKeyHeader` 클래스 기반으로 재작성:

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from app.config import API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key
```

엔드포인트에 적용되어 있는 `Depends(verify_api_key)`는 그대로 유지. Case 6의 401 동작이 계속 통과해야 함.

### (d) app/tests/test_api.py — 회귀 테스트

`src/tests/behavioral_tests_national.py`의 API 버전. pytest + `fastapi.testclient.TestClient` 사용.

테스트 케이스:
1. `test_health_ok` — /health가 200
2. `test_predict_requires_api_key` — API Key 없으면 401
3. `test_predict_valid_input` — 정상 입력 → 200, predicted_power_mwh >= 0, NaN/Inf 아님
4. `test_predict_invalid_region` — 가짜 지역명 → 422, 에러 메시지에 유효 region 목록 포함
5. `test_predict_negative_irradiance` — 일사량 -1 → 422
6. `test_predict_batch_under_limit` — 100개 → 200, 응답 길이 일치
7. `test_predict_batch_over_limit` — 1001개 → 422
8. `test_predict_snapshot` — 가장 중요한 회귀 테스트:
   - `data/processed/national_test_features.csv`에서 임의의 5개 행 추출
   - 각 행에 대해 (a) 직접 `model.predict()` 호출한 결과와 (b) API `/predict` 호출 결과를 비교
   - **|diff| < 1e-4** 안에 들어와야 함
   - 이게 통과하면 "API 경로로 예측해도 학습 모델과 같은 답을 낸다"는 보증
   - 단, API는 음수 클립 적용이고 raw model은 아니므로, raw model 출력도 동일한 클립 적용 후 비교

### (e) README.md 보강

`## API 서빙 (FastAPI)` 섹션에 추가:
- 실행 방법 (`uvicorn app.main:app --host 0.0.0.0 --port 8000`)
- Swagger UI 접속 (`http://localhost:8000/docs`)
- API Key 사용 예시 (환경변수 `SOLAR_API_KEY` 설정)
- curl 예제 1개 (urllib/requests 권장 — 2단계 Case 5의 chunked encoding 이슈 메모)

`## 운영 모니터링 회고` 섹션 신설 (실제 모니터링은 안 하지만 설계 사고를 보여주는 용도):

- 실서비스라면 추가할 항목:
  - 응답 시간 p95/p99 모니터링
  - 예측 분포 드리프트 (학습 시점 발전량 분포와 비교)
  - 입력 피처 분포 드리프트 (특히 기상 변수)
  - feature_names 검증 실패 알림 (이번 24-피처 발견 같은 사고 자동 감지)
- 실패 사례 수집 → 재학습 데이터로 활용 (선생님 자료의 "4단계 → 2단계" 사이클)
- 본 프로젝트에서는 입력이 사용자 직접 입력이 아니라 기상청 API 등에서 오는 구조이므로 사용자 피드백 수집은 해당 없음
- **본 단계에서 학습한 교훈: 모델 입출력 명세의 진실의 원천은 학습 스크립트와 모델 파일이지, 별도 JSON 문서가 아니다.** 이 교훈을 CLAUDE.md에도 한 줄 추가.

## 검증 방법
1. `pytest app/tests/test_api.py -v` 결과 전체 출력 (모두 PASS)
2. 잘못된 API Key로 /predict 호출 → 401 응답 (Case 6 회귀 확인)
3. 로깅 미들웨어가 한 요청에 어떻게 로그를 찍는지 실제 출력 한 줄 보여주기
4. /openapi.json에서 두 엔드포인트가 잘 문서화됐는지 출력 확인
5. **2단계 회귀**: Case 1(전라남도 13:00) 다시 호출해서 여전히 129 MWh 근처(±1) 나오는지 확인