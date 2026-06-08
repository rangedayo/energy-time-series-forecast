# ESS 운영 도구 — React 프론트엔드

`design/ui_mockup_dark.html` 시안을 기반으로 만든 다크 테마 운영 대시보드.
기존 FastAPI 백엔드(`app/`)의 `/sim/meta`·`/simulate` 엔드포인트를 호출한다.
Streamlit 앱(`app_streamlit/`)과 **병행**하며, 같은 orchestrator 로직을 공유한다.

## 스택

- Vite + React 18 + TypeScript
- Plotly (react-plotly.js + plotly.js-dist-min) — 인터랙티브 차트 4종
- Framer Motion — 카드 등장/호버 애니메이션

## 실행

두 개의 터미널이 필요하다.

### 1) 백엔드 (프로젝트 루트에서)

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

`/simulate`, `/sim/meta` 가 추가돼 있다. CORS 는 `http://localhost:5173` 을 허용.

### 2) 프론트엔드 (frontend/ 에서)

```powershell
cd frontend
npm install   # 최초 1회
npm run dev    # http://localhost:5173
```

브라우저에서 `http://localhost:5173` 접속 → 좌측 입력 설정 후 **실행**.

## 환경 변수 (`.env`)

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `VITE_API_BASE` | `http://localhost:8000` | 백엔드 주소 |
| `VITE_API_KEY` | `dev-key-change-me` | `X-API-Key` 헤더 값 (백엔드 `SOLAR_API_KEY` 와 일치해야 함) |

## 구조

```
src/
  api.ts              # /sim/meta·/simulate 클라이언트 + 응답 타입 + 정책 라벨/색
  App.tsx             # 상태(meta/result/loading) + 레이아웃
  index.css           # 다크 + 에너지 액센트 테마 (시안 이식)
  components/
    Sidebar.tsx       # 지역/날짜/시각/SOC 입력 + 실행(리플)
    Panels.tsx        # Header · KPI 카드(카운트업) · 핵심 지표 표
    Charts.tsx        # 예측vs실측 / SOC / 순수익vs자급률 / 시간대별 매매(탭)
```

## 빌드

```powershell
npm run build   # tsc 타입체크 + vite 프로덕션 번들 → dist/
```
