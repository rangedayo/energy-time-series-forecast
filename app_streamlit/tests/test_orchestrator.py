"""run_mpc_simulation() 검증 테스트.

이 테스트는 `uvicorn app.main:app` 실행 중일 때만 통과한다 (API unreachable 테스트 제외).

CSV 시간 범위: 2017-01-02 01:00 ~ 2022-12-31 00:00 (학습용).
정상 시점: 2022-06-15T09:00:00 / 전라남도 (큰 발전량 → MPC 분기 잘 발생).
"""
from __future__ import annotations

import urllib.error
import urllib.request
from datetime import datetime

import pytest

from app_streamlit.orchestrator import run_mpc_simulation

NORMAL_START = datetime(2022, 6, 15, 9, 0, 0)
NORMAL_REGION = "전라남도"
API_BASE = "http://localhost:8000"


def _api_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{API_BASE}/health", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


_NEEDS_API = pytest.mark.skipif(
    not _api_alive(),
    reason="uvicorn 서버가 localhost:8000 에서 실행 중이 아님",
)


@_NEEDS_API
def test_normal_case():
    """1. 정상 입력 → 반환 dict 키 + 3개 정책 모두 존재."""
    out = run_mpc_simulation(
        start_time=NORMAL_START, region=NORMAL_REGION, initial_soc=0.5
    )
    for k in ("meta", "predictions", "actuals", "results", "api_calls"):
        assert k in out, f"missing key: {k}"
    for p in ("naive", "xgb_lookahead", "mpc_xgb"):
        assert p in out["results"], f"missing policy: {p}"
        assert "hourly" in out["results"][p]
        assert "summary" in out["results"][p]
        assert len(out["results"][p]["hourly"]) == 24


def test_invalid_region():
    """2. 존재하지 않는 region → ValueError."""
    with pytest.raises(ValueError, match="region not found"):
        run_mpc_simulation(
            start_time=NORMAL_START, region="없는도", initial_soc=0.5
        )


def test_start_time_out_of_range():
    """3. CSV 범위 밖 (2023년) → ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        run_mpc_simulation(
            start_time=datetime(2023, 1, 1, 0, 0, 0),
            region=NORMAL_REGION,
            initial_soc=0.5,
        )


def test_insufficient_window():
    """4. CSV 끝 근처 → +48h 안 잡힘 → ValueError.

    CSV max = 2022-12-31 00:00. start=2022-12-29 12:00 이면 +47h = 2022-12-31 11:00,
    CSV 범위 안에 있지만 마지막 timestamp 이후를 요구 → insufficient window.
    """
    with pytest.raises(ValueError, match="insufficient continuous window"):
        run_mpc_simulation(
            start_time=datetime(2022, 12, 29, 12, 0, 0),
            region=NORMAL_REGION,
            initial_soc=0.5,
        )


@_NEEDS_API
def test_initial_soc_boundary():
    """5. initial_soc 0.0/1.0 정상, 그 밖이면 ValueError."""
    out_low = run_mpc_simulation(
        start_time=NORMAL_START, region=NORMAL_REGION, initial_soc=0.0
    )
    assert out_low["meta"]["initial_soc"] == 0.0
    out_high = run_mpc_simulation(
        start_time=NORMAL_START, region=NORMAL_REGION, initial_soc=1.0
    )
    assert out_high["meta"]["initial_soc"] == 1.0
    with pytest.raises(ValueError, match="initial_soc"):
        run_mpc_simulation(
            start_time=NORMAL_START, region=NORMAL_REGION, initial_soc=-0.1
        )
    with pytest.raises(ValueError, match="initial_soc"):
        run_mpc_simulation(
            start_time=NORMAL_START, region=NORMAL_REGION, initial_soc=1.1
        )


def test_api_unreachable():
    """6. 잘못된 포트 → RuntimeError. (서버 띄울 필요 없음)"""
    with pytest.raises(RuntimeError, match=r"unreachable|timeout|error"):
        run_mpc_simulation(
            start_time=NORMAL_START,
            region=NORMAL_REGION,
            initial_soc=0.5,
            api_base_url="http://localhost:9999",
            timeout=2.0,
        )


@_NEEDS_API
def test_predictions_shared():
    """7. predictions 48개, api_calls.predict_horizon 정확히 1회.

    3개 정책이 같은 predictions 를 공유했는지 — api_calls 가 단일 entry 인 것으로
    간접 확인 (정책별로 호출했다면 list 였을 것).
    """
    out = run_mpc_simulation(
        start_time=NORMAL_START, region=NORMAL_REGION, initial_soc=0.5
    )
    assert len(out["predictions"]) == 48
    assert len(out["actuals"]) == 48
    assert "predict_horizon" in out["api_calls"]
    assert len(out["api_calls"]) == 1
    pred_meta = out["api_calls"]["predict_horizon"]
    assert pred_meta["status"] == 200
    assert pred_meta["horizon"] == 48
    steps = [p["step"] for p in out["predictions"]]
    assert steps == list(range(1, 49))
