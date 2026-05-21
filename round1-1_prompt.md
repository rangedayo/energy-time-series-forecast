# 작업: 태양광 발전량 예측 모델용 FastAPI 서비스 구축 (1/3단계)

## 프로젝트 컨텍스트
- 이 프로젝트는 전국 17개 시도의 시간별 태양광 발전량을 예측하는 XGBoost 모델 (`models/national_xgboost_model.json`)을 학습 완료한 상태야.
- 학습된 모델은 18개 피처를 입력받아 `power_mwh`(시간당 발전량, MWh)를 예측해.
- 피처 목록은 `src/features/feature_list_national.json`에 정의되어 있어. 반드시 이 파일을 먼저 읽고 작업해.
- 지역 인코더는 `models/region_encoder.pkl`에 LabelEncoder로 저장되어 있어 (17개 시도 → region_code 0~16).
- 입력 데이터는 **이미지가 아니라 숫자 피처**야. UploadFile, multipart, 이미지 처리는 일절 필요 없음.

## 1단계 목표
FastAPI 앱의 뼈대를 만든다. 모델은 서버 시작 시 한 번만 로드되어야 한다.

## 작업 내역

### (a) 디렉토리 생성
프로젝트 루트에 `app/` 디렉토리를 만들고 아래 구조로 파일을 생성해:
app/
├── init.py
├── main.py              # FastAPI 앱 entry point + lifespan
├── schemas.py           # Pydantic 입력/출력 모델
├── inference.py         # XGBoost 예측 로직 (run_in_executor 대상)
├── middleware.py        # 로깅 미들웨어
├── exceptions.py        # Global Exception Handler
├── security.py          # API Key 검증
└── config.py            # 환경 변수 / 상수

### (b) requirements.in에 추가
다음 패키지를 `requirements.in`에 추가하되, 기존 항목은 건드리지 마:
- fastapi
- uvicorn[standard]
- pydantic

추가 후 `pip-compile requirements.in`을 실행할 필요는 없어 (사용자가 직접 함).

### (c) app/config.py 작성
- `MODEL_PATH = "models/national_xgboost_model.json"`
- `ENCODER_PATH = "models/region_encoder.pkl"`
- `FEATURE_LIST_PATH = "src/features/feature_list_national.json"`
- `API_KEY`는 환경변수 `SOLAR_API_KEY`에서 읽어오되, 미설정 시 개발용 기본값 `"dev-key-change-me"`로 fallback. 이 fallback 시 경고 로그 출력.

### (d) app/main.py — lifespan 패턴으로 모델 로드
다음을 반드시 지켜:
- `@asynccontextmanager`로 `lifespan` 함수 정의 후 `FastAPI(lifespan=lifespan)`에 전달
- `on_event("startup")`/`on_event("shutdown")`은 절대 쓰지 말 것 (deprecated)
- lifespan 안에서:
  1. `feature_list_national.json` 로드 → `engineered_features` 리스트를 `app.state.feature_order`에 저장 (예측 시 컬럼 순서 보장용)
  2. XGBoost 모델 로드 → `app.state.model`
  3. region encoder 로드 → `app.state.region_encoder`. 17개 region 이름과 코드 매핑을 `app.state.region_map`에 dict로도 저장 (Swagger 문서에 노출할 용도)
  4. 시작/종료 시 로그 출력 (어떤 모델/피처 수가 로드됐는지)
- 루트 엔드포인트 `GET /` : `{"service": "solar-power-forecast", "model": "national_xgboost", "status": "ok"}` 반환
- `GET /health` : 모델 로드 여부 확인. 로드 안 됐으면 503

이 단계에서는 `/predict` 엔드포인트는 아직 만들지 마. 2단계에서 만들 예정.

## 작업 끝나면 알려줘야 할 것
1. 만든 파일 목록
2. `uvicorn app.main:app --reload`로 띄웠을 때 startup 로그가 어떻게 찍히는지 (실제로 실행해보고 결과 보여줘)
3. `curl http://localhost:8000/health` 결과

