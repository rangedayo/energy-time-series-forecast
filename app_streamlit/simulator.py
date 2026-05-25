"""run_simulation()의 헬퍼 복제 — initial_soc + hourly 시계열 추가.

원본 src/simulation/ess_simulation_v2.py:run_simulation()의 SOC 동역학을 그대로 옮긴다.
원본은 (1) initial_soc 인자가 없고 SOC_INIT 고정, (2) hourly 시계열 미반환이므로
Streamlit 운영자 화면용으로는 부적합. 원본 보존 + 신규 헬퍼.

원본과 동일성 보장 — 동일 입력 + initial_soc=SOC_INIT 이면 aggregate 지표 1:1 일치.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.simulation.ess_config_v2 import (
    EFFICIENCY,
    get_demand_at_hour,
    get_load_period,
    get_tou_price_krw_per_mwh,
)

_PERIODS = ("off_peak", "mid_peak", "max_peak")


def run_simulation_with_hourly(
    actual: np.ndarray,
    predicted: np.ndarray,
    hours: np.ndarray,
    months: np.ndarray,
    params: dict[str, Any],
    policy_fn: Callable[..., dict[str, float]],
    initial_soc: float,
    policy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """원본 run_simulation 의 1:1 복제 + initial_soc + hourly 트래커.

    Returns:
        {"hourly": [...n개...], "summary": {...aggregate 지표...}}
        hourly[i] = {step, soc, generation_mwh, demand_mwh, charge_mwh,
                     discharge_mwh, grid_buy_mwh, grid_sell_mwh,
                     price_krw_per_mwh, period, soc_target_high, soc_target_low}
    """
    if not (0.0 <= initial_soc <= 1.0):
        raise ValueError(
            f"initial_soc must be in [0.0, 1.0], got {initial_soc}"
        )

    policy_kwargs = policy_kwargs or {}

    n = len(actual)
    if len(hours) < n or len(months) < n:
        raise ValueError(
            f"hours/months length must be >= len(actual) ({n}); "
            f"got hours={len(hours)}, months={len(months)}"
        )

    soc = float(initial_soc)
    total_curtailment = 0.0
    total_shortage_mwh = 0.0
    total_demand_mwh = 0.0
    shortage_list: list[float] = []
    charge_cycles = 0.0
    discharge_cycles = 0.0

    total_import_mwh = 0.0
    total_export_mwh = 0.0
    total_cost_krw = 0.0
    total_revenue_krw = 0.0

    import_by_period = {p: 0.0 for p in _PERIODS}
    export_by_period = {p: 0.0 for p in _PERIODS}
    cost_by_period = {p: 0.0 for p in _PERIODS}
    revenue_by_period = {p: 0.0 for p in _PERIODS}

    cap = float(params["ess_capacity_mwh"])
    base_demand = float(params["demand_mwh_per_h"])
    chg_max = float(params["charge_rate_max"])
    dis_max = float(params["discharge_rate_max"])

    hourly: list[dict[str, Any]] = []

    for i in range(n):
        gen = float(actual[i])
        h = int(hours[i])
        m = int(months[i])
        demand_t = get_demand_at_hour(base_demand, h)
        total_demand_mwh += demand_t

        targets = policy_fn(
            i, actual, predicted, soc, demand_t, params, **policy_kwargs
        )
        soc_target_high = float(targets["soc_target_high"])
        soc_target_low = float(targets["soc_target_low"])

        actual_net = gen - demand_t
        price_t = get_tou_price_krw_per_mwh(m, h)
        period_t = get_load_period(m, h)

        charge_amount = 0.0
        discharge_amount = 0.0
        export_mwh_t = 0.0
        import_mwh_t = 0.0

        if actual_net > 0:
            max_storable = max(0.0, (soc_target_high - soc) * cap / EFFICIENCY)
            charge_amount = min(actual_net, chg_max, max_storable)
            soc += charge_amount * EFFICIENCY / cap
            charge_cycles += charge_amount / cap
            export_mwh_t = actual_net - charge_amount
            total_curtailment += export_mwh_t
            if export_mwh_t > 0:
                total_export_mwh += export_mwh_t
                total_revenue_krw += export_mwh_t * price_t
                export_by_period[period_t] += export_mwh_t
                revenue_by_period[period_t] += export_mwh_t * price_t
        else:
            needed = -actual_net
            max_dischargeable = max(0.0, (soc - soc_target_low) * cap * EFFICIENCY)
            discharge_amount = min(needed, dis_max, max_dischargeable)
            soc -= discharge_amount / (cap * EFFICIENCY)
            discharge_cycles += discharge_amount / cap

            shortfall = max(0.0, demand_t - (gen + discharge_amount))
            if shortfall > 0:
                shortage_list.append(shortfall)
                total_shortage_mwh += shortfall
                import_mwh_t = shortfall
                total_import_mwh += shortfall
                total_cost_krw += shortfall * price_t
                import_by_period[period_t] += shortfall
                cost_by_period[period_t] += shortfall * price_t

        hourly.append({
            "step": i,
            "soc": round(soc, 6),
            "generation_mwh": round(gen, 4),
            "demand_mwh": round(demand_t, 4),
            "charge_mwh": round(charge_amount, 4),
            "discharge_mwh": round(discharge_amount, 4),
            "grid_buy_mwh": round(import_mwh_t, 4),
            "grid_sell_mwh": round(export_mwh_t, 4),
            "price_krw_per_mwh": round(price_t, 2),
            "period": period_t,
            "soc_target_high": round(soc_target_high, 4),
            "soc_target_low": round(soc_target_low, 4),
        })

    total_gen = float(np.sum(actual[:n]))
    curtailment_rate = total_curtailment / max(total_gen, 1e-10) * 100.0
    self_consumption_rate = 100.0 - curtailment_rate
    self_sufficiency_rate = (
        1.0 - total_shortage_mwh / max(total_demand_mwh, 1e-10)
    ) * 100.0
    battery_cycles = (charge_cycles + discharge_cycles) / 2.0
    shortage_count = len(shortage_list)
    ess_score = (
        (1.0 - curtailment_rate / 100.0)
        * (1.0 - shortage_count / max(n, 1))
        * 100.0
    )
    net_revenue_krw = total_revenue_krw - total_cost_krw
    avg_import_price = (
        total_cost_krw / total_import_mwh if total_import_mwh > 0 else 0.0
    )
    avg_export_price = (
        total_revenue_krw / total_export_mwh if total_export_mwh > 0 else 0.0
    )

    summary = {
        "self_consumption_rate_pct": round(self_consumption_rate, 2),
        "self_sufficiency_rate_pct": round(self_sufficiency_rate, 2),
        "total_shortage_mwh": round(total_shortage_mwh, 4),
        "mean_shortage_mwh": (
            round(float(np.mean(shortage_list)), 4) if shortage_list else 0.0
        ),
        "max_shortage_mwh": (
            round(float(np.max(shortage_list)), 4) if shortage_list else 0.0
        ),
        "curtailment_rate_pct": round(curtailment_rate, 2),
        "shortage_count": int(shortage_count),
        "battery_cycles": round(battery_cycles, 4),
        "ess_score": round(ess_score, 2),
        "total_curtailment_mwh": round(total_curtailment, 4),
        "total_demand_mwh": round(total_demand_mwh, 4),
        "total_gen_mwh": round(total_gen, 4),
        "n_hours": int(n),
        "total_import_mwh": round(total_import_mwh, 4),
        "total_export_mwh": round(total_export_mwh, 4),
        "total_cost_krw": round(total_cost_krw, 0),
        "total_revenue_krw": round(total_revenue_krw, 0),
        "net_revenue_krw": round(net_revenue_krw, 0),
        "import_mwh_by_period": {
            p: round(import_by_period[p], 4) for p in _PERIODS
        },
        "export_mwh_by_period": {
            p: round(export_by_period[p], 4) for p in _PERIODS
        },
        "cost_krw_by_period": {p: round(cost_by_period[p], 0) for p in _PERIODS},
        "revenue_krw_by_period": {
            p: round(revenue_by_period[p], 0) for p in _PERIODS
        },
        "avg_import_price_krw_per_mwh": round(avg_import_price, 2),
        "avg_export_price_krw_per_mwh": round(avg_export_price, 2),
        "initial_soc": round(float(initial_soc), 4),
        "final_soc": round(soc, 4),
    }

    return {"hourly": hourly, "summary": summary}
