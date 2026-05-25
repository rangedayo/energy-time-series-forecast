# Round 2-3: Streamlit 운영 모드 화면

## 배경

Round 2-2 리팩토링 완료 — `app_streamlit/orchestrator.py::run_mpc_simulation()`이 단일 진입점으로 작동. 이제 운영자가 실제로 쓸 **Streamlit UI**를 만든다.

화면은 탭 2개로 구성하되 round 2-3에서는 **운영 모드 탭만 구현**한다 (분석 모드는 round 2-4). 분석 모드 탭은 placeholder만 둔다.

## 목표

운영자가 region, initial_soc, start_time을 선택하고 "실행" 버튼을 누르면 3개 정책(naive/xgb_lookahead/mpc_xgb)의 24시간 시뮬 결과를 한 화면에서 비교할 수 있는 Streamlit 앱.

## 디렉토리 구조

```
app_streamlit/
├── app.py                     ← Streamlit 진입점 (신규)
├── ui/                        ← UI 컴포넌트 (신규)
│   ├── __init__.py
│   ├── inputs.py              ← 입력 위젯
│   ├── charts.py              ← 4개 그래프
│   ├── metrics_table.py       ← 핵심 지표 표
│   └── naming.py              ← 정책 내부명 ↔ 표시명 매핑
├── orchestrator.py            ← 기존 (수정 없음)
├── data_loader.py             ← 기존 (수정 없음)
└── tests/
    └── test_orchestrator.py   ← 기존 (수정 없음)
```

## 작업 내역

### (a) `app_streamlit/ui/naming.py`

정책 내부명 ↔ 운영자 친화 표시명 매핑.

```python
POLICY_DISPLAY_NAMES = {
    "naive": "기본 운영",
    "xgb_lookahead": "단기 예측 기반",
    "mpc_xgb": "수익 최적화 (MPC)",
}

POLICY_DESCRIPTIONS = {
    "naive": "예측 없이 SOC를 0.20~0.80 범위로 고정 유지",
    "xgb_lookahead": "다음 1시간 발전량 예측을 보고 SOC 목표 조정",
    "mpc_xgb": "24시간 예측 기반 LP 최적화 (Rolling Horizon)",
}

POLICY_COLORS = {
    "naive": "#1D9E75",          # teal-600
    "xgb_lookahead": "#BA7517",  # amber-600
    "mpc_xgb": "#534AB7",        # purple-600
}

def display_name(policy: str) -> str:
    return POLICY_DISPLAY_NAMES.get(policy, policy)
```

### (b) `app_streamlit/ui/inputs.py`

입력 위젯 컴포넌트.

```python
def render_input_panel(csv_path: str) -> dict | None:
    """
    좌측 입력 패널 렌더링.
    Returns:
        실행 버튼을 눌렀을 때만 dict 반환. 아니면 None.
        {"region": str, "start_time": datetime, "initial_soc": float}
    """
```

요구사항:
- **region**: `st.selectbox`로 17개 시도 선택. 학습 CSV에서 unique region 동적 로드.
- **시뮬 시작 날짜**: `st.date_input`. min/max는 학습 CSV 범위에서 history 24h + forecast 48h 여유 빼고 계산. 디폴트는 `2022-06-15`.
- **시뮬 시작 시각**: `st.time_input`. 디폴트 `09:00`. step=1시간.
- **initial_soc**: `st.slider`. min=0.0, max=1.0, step=0.05, 디폴트=0.5.
- **실행 버튼**: `st.button("실행", type="primary")`.
- 입력 영역 하단에 "사용 가능 범위: YYYY-MM-DD ~ YYYY-MM-DD" 안내.

### (c) `app_streamlit/ui/metrics_table.py`

3개 정책 × 5개 핵심 지표 표.

```python
def render_metrics_table(results: dict) -> None:
    """
    3개 정책 비교 표. best 값에 하이라이트.
    """
```

지표 (정책별):
- 순수익 (원) — net_revenue_krw, `{:,}` 포맷
- 자급률 (%) — self_sufficiency_rate_pct, 소수 2자리
- 총 매입 (MWh) — total_import_mwh, 소수 2자리
- 총 판매 (MWh) — total_export_mwh, 소수 2자리
- 배터리 사이클 — battery_cycles, 소수 3자리

best 표시:
- 순수익 max → 초록 배경
- 자급률 max → 초록 배경
- 총 매입 min → 초록 배경 (적을수록 좋음)
- 나머지는 강조 없음

Streamlit `st.dataframe` 사용. 정책명은 `display_name()`으로 변환.

### (d) `app_streamlit/ui/charts.py`

4개 차트. 모두 matplotlib + `st.pyplot()` 사용 (Streamlit 기본 의존성).

차트 각각:

**(d-1) `render_prediction_vs_actual(predictions, actuals)`**
- 48시간 라인 차트 (forecast 전체 보여줌, 시뮬 24h 외 추가 24h 점선)
- x축: timestamp, y축: 발전량 MWh
- 예측 곡선 파랑 실선, 실측 곡선 빨강 점선
- 시뮬 구간(0~23h) 음영으로 강조

**(d-2) `render_soc_curves(results)`**
- 24시간 라인 차트
- 3개 정책의 SOC를 한 그래프에 겹쳐서. 색상은 `POLICY_COLORS`.
- y축 0.0~1.0, SOC_MIN(0.10) / SOC_MAX(0.90) 수평선 점선으로 표시.
- 범례에 `display_name()` 사용.

**(d-3) `render_revenue_sufficiency_bars(results)`**
- 그룹 막대: 3개 정책 × 2개 지표(net_revenue, self_sufficiency)
- 좌측 y축: 순수익(원, 백만 단위 변환), 우측 y축: 자급률(%)
- twin axis 사용. 같은 정책의 두 막대는 인접 배치, 정책 간 간격 둠.
- 색상은 `POLICY_COLORS`로 정책 구분.

**(d-4) `render_hourly_trading(results, selected_policy)`**
- 24시간 스택드 바 + 라인
- selected_policy 하나만. 라디오로 선택.
- 양수 막대: grid_sell (판매, 초록), 음수 막대: grid_buy (매입, 빨강)
- SOC를 보조 라인으로 overlay (오른쪽 y축)
- TOU 가격대별 배경 음영 (off/mid/max 구분)

차트 한국어 폰트: `from src.utils.font_setting import apply as _apply_font; _apply_font()` 최상단에서 호출.

### (e) `app_streamlit/app.py` — 진입점

```python
"""Streamlit 운영 도구 진입점.

실행: streamlit run app_streamlit/app.py
사전 조건: uvicorn app.main:app 별도 실행 중일 것.
"""
```

레이아웃:
1. 페이지 설정: `st.set_page_config(page_title="ESS 운영 도구", layout="wide")`
2. 헤더: `st.title("ESS 운영 도구")`
3. API 헬스 체크 (앱 시작 시 1회):
   - `/health` 호출 시도, 실패 시 화면 상단에 `st.error("API 서버를 켜주세요: `uvicorn app.main:app`")`
   - 실패해도 앱은 계속 렌더링 (탭 구조까지는 보이게)
4. **탭 구조**: `st.tabs(["운영 모드", "분석 모드"])`
   - 운영 모드: 아래 (f) 진행
   - 분석 모드: `st.info("분석 모드는 round 2-4에서 구현 예정입니다.")` placeholder만

### (f) 운영 모드 탭 내부

`st.columns([1, 3])`로 좌/우 분할:

**좌측 (1/4 폭)**: 입력 패널 (`render_input_panel()`)

**우측 (3/4 폭)**:
- 결과 없는 초기 상태: `st.info("좌측에서 입력을 설정한 후 실행 버튼을 눌러주세요.")`
- 결과 있는 경우:
  1. 핵심 지표 표 (`render_metrics_table()`)
  2. 4개 차트 2×2 그리드:
     ```python
     col1, col2 = st.columns(2)
     with col1:
         render_prediction_vs_actual(...)
         render_revenue_sufficiency_bars(...)
     with col2:
         render_soc_curves(...)
         render_hourly_trading(...)
     ```
  3. 4번째 차트(시간대별 매매) 위에 정책 선택 라디오:
     ```python
     selected = st.radio(
         "시간대별 매매 — 정책 선택",
         options=["naive", "xgb_lookahead", "mpc_xgb"],
         format_func=display_name,
         horizontal=True,
     )
     ```

### (g) 캐싱 — `@st.cache_data`

```python
@st.cache_data(show_spinner=False)
def _cached_run(region: str, start_time_iso: str, initial_soc: float) -> dict:
    from datetime import datetime
    return run_mpc_simulation(
        start_time=datetime.fromisoformat(start_time_iso),
        region=region,
        initial_soc=initial_soc,
    )
```

- 실행 버튼 눌렀을 때만 호출. 슬라이더 만지는 것만으로는 재실행 안 됨.
- 동일 (region, start_time, initial_soc) 조합은 캐시에서 즉시 반환.
- spinner는 별도로 표시: `with st.spinner("3개 정책 시뮬레이션 실행 중... (~5초)"):`.

`start_time`을 datetime 객체로 직접 cache key에 넣으면 hashable 이슈 가능 → ISO 문자열로 변환해서 전달.

### (h) 에러 처리

`run_mpc_simulation()`이 던지는 두 예외 처리:

```python
try:
    result = _cached_run(...)
except ValueError as e:
    st.error(f"입력 검증 실패: {e}")
    return
except RuntimeError as e:
    if "unreachable" in str(e) or "timeout" in str(e):
        st.error(
            "API 서버에 연결할 수 없습니다.\n"
            "터미널에서 `uvicorn app.main:app` 실행 중인지 확인해주세요."
        )
    else:
        st.error(f"API 오류: {e}")
    return
```

## 검증

### (i) 수동 검증 체크리스트

다음 명령으로 앱 실행:
```bash
# 터미널 1
uvicorn app.main:app

# 터미널 2
streamlit run app_streamlit/app.py
```

체크 항목:
1. 첫 화면에 탭 2개("운영 모드", "분석 모드") 보임
2. 분석 모드 탭 클릭 → placeholder 메시지 보임
3. 운영 모드 탭 좌측: region/날짜/시각/SOC 슬라이더/실행 버튼 보임
4. **정상 케이스** (region=전라남도, 2022-06-15 09:00, soc=0.5): 실행 → spinner ~5초 → 결과 표시
5. **표 확인**: 3개 정책 5개 지표 모두 출력, 정책명은 한글 표시명, best 값 하이라이트
6. **차트 4개 모두 렌더링**: 예측/SOC/수익바/매매바
7. 시간대별 매매 정책 라디오 토글 → 차트 갱신
8. **캐시 동작**: 동일 입력으로 재실행 → 즉시 결과 (~100ms)
9. **다른 입력으로 재실행**: 슬라이더만 만지고 실행 안 누르면 결과 그대로. 실행 누르면 갱신.
10. **에러 케이스 1**: API 서버 죽인 상태에서 실행 → 친절한 에러 메시지 + 앱 자체는 안 깨짐
11. **에러 케이스 2**: 잘못된 region이나 범위 밖 시점 직접 시도하면 → `st.error`로 메시지 출력 (UI에서 막혀있어 발생 어려움)

스크린샷 4장 저장:
- `outputs/streamlit_screenshots/initial.png` — 첫 진입
- `outputs/streamlit_screenshots/operational_result.png` — 정상 결과
- `outputs/streamlit_screenshots/api_down.png` — API 다운 에러
- `outputs/streamlit_screenshots/analytics_tab.png` — 분석 모드 placeholder

## 주의사항

- **기존 파일 수정 금지**: `orchestrator.py`, `data_loader.py`, `tests/` 손대지 말 것.
- **Streamlit 캐싱 함수는 단순한 인자만 받게**: datetime 객체 직접 인자로 받으면 hashable 이슈. ISO 문자열로 변환해서 캐시 키 만들 것.
- **차트 한국어 폰트**: `src/utils/font_setting::apply()` 호출 안 하면 한글 깨짐.
- **차트는 matplotlib + st.pyplot()**: plotly 등 추가 라이브러리 도입 금지 (이미 학습/시뮬에서 matplotlib 쓰고 있음).
- **API 헬스 체크는 한 번만**: `@st.cache_resource`로 캐시. 매 rerun마다 호출하면 앱이 느려짐.
- **requirements.txt 갱신**: `streamlit`이 없으면 추가. 버전은 최신 안정판 (>= 1.30).
- 4번째 차트(시간대별 매매)에 SOC overlay 넣을 때 좌/우 y축 단위 명확히 (왼쪽 MWh, 오른쪽 SOC 0~1).

## 끝나면 알려줄 것

1. 새로 만든 파일 목록 (5개: `app.py`, `ui/__init__.py`, `ui/inputs.py`, `ui/charts.py`, `ui/metrics_table.py`, `ui/naming.py`)
2. requirements.txt 변경 사항 (streamlit 추가 여부)
3. 수동 검증 체크리스트 결과 (11개 항목 통과/실패)
4. 스크린샷 4장 경로
5. 실행 시 stdout/stderr에 이상한 메시지 있었는지
6. 4번째 차트(시간대별 매매)의 SOC overlay가 시각적으로 잘 보이는지 (좌/우 축 스케일 차이 때문에 둘 다 보이는 좋은 비율 찾기 어려울 수 있음 — 어려우면 SOC overlay 빼도 됨)
7. 막힌 부분 / 의외였던 점
