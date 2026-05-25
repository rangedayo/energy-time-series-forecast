"""Round 2-2 MPC 오케스트레이터.

run_mpc_simulation() — Streamlit 운영자 화면이 호출할 단일 진입점.
(start_time, region, initial_soc) 입력 → 3개 정책 결과 dict 반환.

흐름:
  1. data_loader.load_history_and_forecast()로 history 24h + forecast 48h 슬라이스
  2. /predict_horizon 1번 호출 (horizon=48)
  3. 받은 48개 예측을 3개 정책이 공유 (naive/xgb_lookahead/mpc_xgb)
  4. 정책마다 src.simulation.ess_simulation_v2.run_simulation(include_hourly=True) 24h 실행
  5. 결과 dict 조립
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

import numpy as np

from app_streamlit.data_loader import _load_csv_cached, load_history_and_forecast
from src.simulation.ess_config_v2 import build_region_params
from src.simulation.ess_policy_v2 import (
    policy_lookahead,
    policy_mpc_xgb,
    policy_naive,
)
from src.simulation.ess_simulation_v2 import run_simulation

_POLICIES: dict[str, Any] = {
    "naive": policy_naive,
    "xgb_lookahead": policy_lookahead,
    "mpc_xgb": policy_mpc_xgb,
}

_SIM_LENGTH = 24       # 운영자 화면이 보여줄 시뮬 길이
_HORIZON = 48          # /predict_horizon 호출 시 사용
_MPC_LP_HORIZON = 48   # mpc_xgb LP가 매 시점 보는 lookahead (round 2-2-pre PASS 근거)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def _post_predict_horizon(
    url: str, api_key: str, payload: dict, timeout: float
) -> tuple[list[float], dict[str, Any]]:
    """/predict_horizon POST. 에러는 RuntimeError로 정규화."""
    body = json.dumps(payload, default=_json_default).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"API error {e.code}: {err_body[:500]}") from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise RuntimeError(
                f"API server timeout after {timeout}s ({url})"
            ) from e
        raise RuntimeError(f"API server unreachable at {url} ({reason})") from e
    except TimeoutError as e:
        raise RuntimeError(
            f"API server timeout after {timeout}s ({url})"
        ) from e
    except (ConnectionError, OSError) as e:
        raise RuntimeError(f"API server unreachable at {url} ({e})") from e
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if status != 200:
        raise RuntimeError(f"API error {status}: {resp_body[:500]}")

    parsed = json.loads(resp_body)
    predictions = [float(p["predicted_power_mwh"]) for p in parsed["predictions"]]
    meta = {
        "url": url,
        "elapsed_ms": round(elapsed_ms, 2),
        "horizon": int(parsed.get("horizon", len(predictions))),
        "status": status,
    }
    return predictions, meta


def _payload_from_window(
    region: str,
    start_time: datetime,
    history: list[dict],
    forecast: list[dict],
) -> dict:
    return {
        "region": region,
        "start_time": start_time.isoformat(),
        "horizon": _HORIZON,
        "history": [
            {"timestamp": h["timestamp"].isoformat(), "power_mwh": h["power_mwh"]}
            for h in history
        ],
        "forecast": [
            {
                "timestamp": f["timestamp"].isoformat(),
                "irradiance": f["irradiance"],
                "기온": f["기온"],
                "강수량": f["강수량"],
                "습도": f["습도"],
                "일조": f["일조"],
                "전운량": f["전운량"],
            }
            for f in forecast
        ],
    }


def _region_params_for(csv_path: str, region: str) -> dict[str, float]:
    """학습 CSV 전체로 weight 정규화 후 region 파라미터 추출."""
    df = _load_csv_cached(csv_path)
    return build_region_params(df)[region]


def _run_policy(
    policy_name: str,
    actual_arr: np.ndarray,
    predicted_arr: np.ndarray,
    hours_arr: np.ndarray,
    months_arr: np.ndarray,
    params: dict[str, Any],
    initial_soc: float,
) -> dict[str, Any]:
    policy_fn = _POLICIES[policy_name]
    if policy_name == "mpc_xgb":
        policy_kwargs = {
            "months_arr": months_arr,
            "hours_arr": hours_arr,
            "horizon": _MPC_LP_HORIZON,
        }
    elif policy_name == "xgb_lookahead":
        policy_kwargs = {"horizon": 1}
    else:  # naive
        policy_kwargs = None

    result = run_simulation(
        actual=actual_arr,
        predicted=predicted_arr,
        hours=hours_arr,
        params=params,
        policy_fn=policy_fn,
        policy_kwargs=policy_kwargs,
        months=months_arr,
        initial_soc=initial_soc,
        include_hourly=True,
    )
    # 외부 노출 형식 유지: {"hourly": [...], "summary": {...}}
    hourly = result.pop("hourly")
    return {"hourly": hourly, "summary": result}


def run_mpc_simulation(
    start_time: datetime,
    region: str,
    initial_soc: float = 0.5,
    policies: tuple[str, ...] = ("naive", "xgb_lookahead", "mpc_xgb"),
    api_base_url: str = "http://localhost:8000",
    api_key: str = "dev-key-change-me",
    csv_path: str = "data/processed/national_train_features.csv",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """MPC 오케스트레이터 — Streamlit 운영자 화면용 단일 진입점.

    Raises:
        ValueError: 입력 검증 실패 (region/start_time/initial_soc).
        RuntimeError: API 호출 실패 (unreachable/timeout/non-200).
    """
    t_total_start = time.perf_counter()

    if not (0.0 <= initial_soc <= 1.0):
        raise ValueError(
            f"initial_soc must be in [0.0, 1.0], got {initial_soc}"
        )
    unknown = [p for p in policies if p not in _POLICIES]
    if unknown:
        raise ValueError(
            f"unknown policies: {unknown}. Allowed: {list(_POLICIES)}"
        )

    history, forecast, actuals_48 = load_history_and_forecast(
        csv_path=csv_path, region=region, start_time=start_time
    )

    payload = _payload_from_window(region, start_time, history, forecast)
    predict_url = f"{api_base_url.rstrip('/')}/predict_horizon"
    predictions_48, api_meta = _post_predict_horizon(
        predict_url, api_key, payload, timeout
    )
    if len(predictions_48) != _HORIZON:
        raise RuntimeError(
            f"API returned {len(predictions_48)} predictions, expected {_HORIZON}"
        )

    actual_arr = np.asarray(actuals_48, dtype=float)
    predicted_arr = np.asarray(predictions_48, dtype=float)
    forecast_times = [f["timestamp"] for f in forecast]
    hours_arr = np.asarray([t.hour for t in forecast_times], dtype=int)
    months_arr = np.asarray([t.month for t in forecast_times], dtype=int)

    params = _region_params_for(csv_path, region)

    # 시뮬은 24h. actual 만 자르고, predicted/hours/months 는 48 그대로 유지.
    # → mpc_xgb 가 매 시점 t∈[0,23] 에서 predicted[t:t+_MPC_LP_HORIZON] 까지 안전 접근.
    sim_actual = actual_arr[:_SIM_LENGTH]

    results: dict[str, Any] = {}
    for policy_name in policies:
        if policy_name == "naive":
            policy_predicted = sim_actual.copy()  # naive 는 예측 무관
        else:
            policy_predicted = predicted_arr
        results[policy_name] = _run_policy(
            policy_name=policy_name,
            actual_arr=sim_actual,
            predicted_arr=policy_predicted,
            hours_arr=hours_arr,
            months_arr=months_arr,
            params=params,
            initial_soc=initial_soc,
        )

    elapsed_ms = (time.perf_counter() - t_total_start) * 1000.0

    return {
        "meta": {
            "start_time": start_time.isoformat(),
            "region": region,
            "initial_soc": float(initial_soc),
            "policies": list(policies),
            "api_base_url": api_base_url,
            "horizon_used": _HORIZON,
            "sim_length": _SIM_LENGTH,
        },
        "predictions": [
            {"timestamp": forecast_times[i].isoformat(),
             "predicted_power_mwh": predictions_48[i],
             "step": i + 1}
            for i in range(_HORIZON)
        ],
        "actuals": [
            {"timestamp": forecast_times[i].isoformat(),
             "actual_power_mwh": actuals_48[i]}
            for i in range(_HORIZON)
        ],
        "results": results,
        "api_calls": {"predict_horizon": api_meta},
        "elapsed_ms": round(elapsed_ms, 2),
    }
