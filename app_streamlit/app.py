"""Streamlit 운영 도구 진입점 (round 2-3).

실행:
    streamlit run app_streamlit/app.py

사전 조건:
    uvicorn app.main:app --host 0.0.0.0 --port 8000  (별도 터미널)
"""
from __future__ import annotations

import json
import sys as _sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

import streamlit as st

_sys.path.insert(0, ".")

from app_streamlit.orchestrator import run_mpc_simulation  # noqa: E402
from app_streamlit.ui.charts import (  # noqa: E402
    render_hourly_trading,
    render_prediction_vs_actual,
    render_revenue_sufficiency_bars,
    render_soc_curves,
)
from app_streamlit.ui.inputs import render_input_panel  # noqa: E402
from app_streamlit.ui.metrics_table import render_metrics_table  # noqa: E402
from app_streamlit.ui.naming import POLICY_DESCRIPTIONS, display_name  # noqa: E402

_API_BASE_URL = "http://localhost:8000"
_API_KEY = "dev-key-change-me"
_CSV_PATH = "data/processed/national_train_features.csv"
_POLICY_OPTIONS: tuple[str, ...] = ("naive", "xgb_lookahead", "mpc_xgb")


@st.cache_resource(show_spinner=False)
def _api_healthy(api_base_url: str, timeout: float = 2.0) -> tuple[bool, str]:
    """앱 시작 시 한 번만 호출. 결과는 프로세스 수명 동안 캐시."""
    url = f"{api_base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            body = json.loads(resp.read().decode("utf-8"))
        status = body.get("status", "")
        if status != "healthy":
            return False, f"status={status}"
        return True, ""
    except urllib.error.URLError as e:
        return False, f"unreachable ({getattr(e, 'reason', e)})"
    except (TimeoutError, ConnectionError, OSError) as e:
        return False, f"unreachable ({e})"
    except Exception as e:  # noqa: BLE001
        return False, f"unexpected ({e})"


@st.cache_data(show_spinner=False)
def _cached_run(
    region: str,
    start_time_iso: str,
    initial_soc: float,
    api_base_url: str,
    api_key: str,
    csv_path: str,
) -> dict[str, Any]:
    """orchestrator 호출. 동일 (region, start_time, initial_soc) 조합 캐시."""
    return run_mpc_simulation(
        start_time=datetime.fromisoformat(start_time_iso),
        region=region,
        initial_soc=initial_soc,
        api_base_url=api_base_url,
        api_key=api_key,
        csv_path=csv_path,
    )


def _render_stored_result(
    result: dict[str, Any], last_inputs: dict[str, Any]
) -> None:
    """session_state 에 저장된 결과 + 입력 dict 을 받아 우측 영역 렌더링."""
    elapsed_ms = float(result.get("elapsed_ms", 0.0))
    st.success(
        f"실행 완료 — region={last_inputs['region']} | "
        f"start={last_inputs['start_time'].isoformat()} | "
        f"initial_soc={last_inputs['initial_soc']:.2f} | "
        f"소요 {elapsed_ms / 1000.0:.2f}s"
    )

    with st.expander("정책 설명", expanded=False):
        for policy in _POLICY_OPTIONS:
            st.markdown(
                f"- **{display_name(policy)}** — {POLICY_DESCRIPTIONS[policy]}"
            )

    st.markdown("### 핵심 지표")
    render_metrics_table(result["results"])

    st.markdown("### 차트")
    meta = result.get("meta", {})
    col1, col2 = st.columns(2)
    with col1:
        render_prediction_vs_actual(
            result["predictions"],
            result["actuals"],
            sim_length=int(meta.get("sim_length", 24)),
        )
        render_revenue_sufficiency_bars(result["results"])
    with col2:
        render_soc_curves(result["results"])
        # key 인자로 session_state 에 자동 보존 → 다른 위젯 만져도 유지
        selected = st.radio(
            "시간대별 매매 — 정책 선택",
            options=list(_POLICY_OPTIONS),
            format_func=display_name,
            horizontal=True,
            key="trading_policy_radio",
        )
        render_hourly_trading(result["results"], selected)


def _run_and_store(inputs: dict[str, Any]) -> None:
    """orchestrator 호출 → 성공 시 session_state 저장, 실패 시 클리어."""
    try:
        with st.spinner("3개 정책 시뮬레이션 실행 중... (~5초)"):
            result = _cached_run(
                region=inputs["region"],
                start_time_iso=inputs["start_time"].isoformat(),
                initial_soc=inputs["initial_soc"],
                api_base_url=_API_BASE_URL,
                api_key=_API_KEY,
                csv_path=_CSV_PATH,
            )
    except ValueError as e:
        st.error(f"입력 검증 실패: {e}")
        st.session_state.last_result = None
        st.session_state.last_inputs = None
        return
    except RuntimeError as e:
        msg = str(e)
        if "unreachable" in msg.lower() or "timeout" in msg.lower():
            st.error(
                "API 서버에 연결할 수 없습니다.\n\n"
                "터미널에서 `uvicorn app.main:app` 실행 중인지 확인해주세요."
            )
        else:
            st.error(f"API 오류: {e}")
        st.session_state.last_result = None
        st.session_state.last_inputs = None
        return

    st.session_state.last_result = result
    st.session_state.last_inputs = inputs


def _render_operational_tab(api_ok: bool) -> None:
    # session_state 초기화 (탭 진입 시 1회)
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_inputs" not in st.session_state:
        st.session_state.last_inputs = None

    left, right = st.columns([1, 3])

    with left:
        inputs = render_input_panel(_CSV_PATH)

    with right:
        if not api_ok:
            st.warning(
                "API 서버가 응답하지 않습니다. 실행해도 호출 단계에서 실패합니다."
            )

        # 실행 버튼이 눌렸을 때만 새 시뮬 호출 → session_state 갱신
        if inputs is not None:
            _run_and_store(inputs)

        # 저장된 결과가 있으면 항상 렌더링 (위젯 토글에도 살아남음)
        stored_result = st.session_state.last_result
        stored_inputs = st.session_state.last_inputs
        if stored_result is not None and stored_inputs is not None:
            _render_stored_result(stored_result, stored_inputs)
        else:
            st.info(
                "좌측에서 입력을 설정한 후 **실행** 버튼을 눌러주세요."
            )


def main() -> None:
    st.set_page_config(page_title="ESS 운영 도구", layout="wide")
    st.title("ESS 운영 도구")
    st.caption(
        "태양광 발전 예측 기반 ESS 운영 정책 비교 (naive / xgb_lookahead / mpc_xgb)"
    )

    api_ok, api_err = _api_healthy(_API_BASE_URL)
    if not api_ok:
        st.error(
            f"API 서버를 켜주세요: `uvicorn app.main:app`  \n"
            f"_상세: {api_err}_"
        )

    _render_operational_tab(api_ok)


main()
