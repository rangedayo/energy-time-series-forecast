"""
ESS 결정 정책 함수 (v2)

각 정책은 동일한 시그니처를 따르며 SOC 목표만 결정한다.
실제 충방전 실행은 시뮬레이터 본체(ess_simulation_v2.run_simulation)가
actual(실측) 기준으로 수행한다.
→ "충방전은 항상 실측 기준, 예측은 강도 조절에만 사용"의 구현.

정책 시그니처:
    policy_xxx(t, actual, predicted, soc, demand_t, params, **kwargs) -> dict
        {
            "soc_target_high": float,  # 충전 분기 도달 목표 SOC (SOC_MAX 이하)
            "soc_target_low":  float,  # 방전 분기 유지 하한 SOC (SOC_MIN 이상)
        }
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import linprog

import sys as _sys
_sys.path.insert(0, ".")
from src.simulation.ess_config_v2 import (
    SOC_MIN, SOC_MAX, EFFICIENCY,
    get_demand_at_hour, get_tou_price_krw_per_mwh,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ── LP 통계 (MPC 디버그용) ────────────────────────────────────────────────────
# run_simulation 한 번 호출마다 reset_lp_stats() → MPC 정책 내부에서 누적
LP_STATS = {"total": 0, "infeasible": 0}


def reset_lp_stats():
    LP_STATS["total"] = 0
    LP_STATS["infeasible"] = 0


def get_lp_stats():
    return dict(LP_STATS)


# ── 정책 4가지 ────────────────────────────────────────────────────────────────
def policy_naive(t, actual, predicted, soc, demand_t, params):
    """
    가장 단순한 베이스라인. "다음 시점도 지금과 같다"는 persistence 가정.
    예측을 전혀 사용하지 않음. 사실상 lag1 가정.
    항상 (high=0.80, low=0.20) 반환.
    """
    return {
        "soc_target_high": 0.80,
        "soc_target_low":  0.20,
    }


def policy_lookahead(t, actual, predicted, soc, demand_t, params, horizon=1):
    """
    예측을 기반으로 다음 N시점을 보고 SOC 목표를 동적 조정.

    - 다음 시점이 잉여 예상 → 지금 덜 채워서 공간 남김
    - 다음 시점이 부족 예상 → 지금 덜 빼서 비축

    horizon=1이 기본. 멀티스텝은 후속 과제.
    """
    n = len(predicted)
    next_t = min(t + horizon, n - 1)
    forecast_next = float(predicted[next_t])
    forecast_net = forecast_next - demand_t

    if forecast_net > 0:
        # 다음 시점 잉여 예상 → 지금 충전 목표를 낮춰 공간 확보
        return {"soc_target_high": min(0.80, soc + 0.10), "soc_target_low": 0.20}
    else:
        # 다음 시점 부족 예상 → 방전 하한을 높여 비축
        return {"soc_target_high": 0.80, "soc_target_low": min(0.80, 0.20 + 0.10)}


def policy_perfect_foresight(t, actual, predicted, soc, demand_t, params, horizon=1):
    """
    Oracle. 예측 대신 실측값을 lookahead로 사용 → 예측 오차 0인 가상 케이스.
    ESS 운영 효율의 이론 상한을 정의한다.
    """
    return policy_lookahead(t, actual, actual, soc, demand_t, params, horizon)


def policy_lookahead_noisy(t, actual, predicted, soc, demand_t, params,
                           noise_level=0.0, horizon=1, rng=None):
    """
    Sensitivity 분석용. 실측값에 합성 노이즈를 주입한 예측으로 lookahead.
    noise_level=0.0 이면 oracle 과 동일.
    """
    rng = rng or np.random.default_rng(42)
    n = len(actual)
    next_t = min(t + horizon, n - 1)
    sigma = float(np.std(actual)) * noise_level
    noisy = actual[next_t] + rng.normal(0, sigma)
    fake_pred = np.asarray(predicted, dtype=float).copy()
    fake_pred[next_t] = noisy
    return policy_lookahead(t, actual, fake_pred, soc, demand_t, params, horizon)


def policy_xgb_no_lookahead(t, actual, predicted, soc, demand_t, params):
    """
    기존 v1 시뮬 재현용. t 시점 예측값으로 결정하되 lookahead 없음.
    pred_net 부호에 관계없이 SOC 범위 (0.80, 0.20)를 반환한다.
    → 사실상 naive와 같은 SOC 범위. 결정 분기 자체에는 영향 없음.
       차이는 시뮬레이터 본체가 actual_net 으로 충방전을 실행하는 데서 나타난다.

    주의: 새 시뮬 구조 안에서 기존 동작을 근사한 재현이며 100% 동일하지 않다.
    """
    pred_net = float(predicted[t]) - demand_t
    if pred_net > 0:
        return {"soc_target_high": 0.80, "soc_target_low": 0.20}
    else:
        return {"soc_target_high": 0.80, "soc_target_low": 0.20}


# ── MPC (LP) 정책 ────────────────────────────────────────────────────────────
# 시뮬 본체(ess_simulation_v2.run_simulation)의 SOC 동역학을 그대로 LP 에 옮긴다.
#
# 시뮬 본체 SOC 갱신식 (확인 결과: 충전·방전 양쪽에 효율이 모두 적용되는
# split-loss 형태):
#   - 충전 분기(actual_net>0):  soc <- soc + EFFICIENCY * charge_amount / cap
#   - 방전 분기(actual_net<0):  soc <- soc - discharge_amount / (EFFICIENCY * cap)
#   여기서 charge_amount = "잉여 측에서 본 충전량(MWh)" (≤ surplus, ≤ chg_max)
#         discharge_amount = "부하 측으로 공급된 방전량(MWh)" (≤ deficit, ≤ dis_max)
#
# 또한 시뮬 본체는 "발전 잉여만 충전 / 부하 부족만 방전"인 부하 추종 구조라
# 매수(import)는 부족분 잔여로만, 매도(export)는 잉여 잔여로만 발생한다.
# 따라서 LP 도 c[h] <= surplus[h], d[h] <= deficit[h] 로 묶어 동일 구조를 가짐.
# (LP 가 그리드 차익거래를 계획해도 시뮬이 실행하지 못해 SOC 궤적이 어긋남.)


def _solve_mpc_lp(t, gen_forecast, soc, params, months_arr, hours_arr,
                  horizon=24):
    """
    시점 t 에서 horizon 시점을 보고 LP 로 최적 (c[0], d[0]) 도출.

    Decision vars (총 2N개): x = [c[0..N-1], d[0..N-1]]
      c[h] : 시점 t+h 의 ESS 충전량 (interface MWh, 잉여 측)
      d[h] : 시점 t+h 의 ESS 방전량 (load MWh, 부하 측)

    Constraints
      c[h] ∈ [0, min(surplus[h], chg_max)]   ← 잉여로만 충전
      d[h] ∈ [0, min(deficit[h], dis_max)]   ← 부족 메우려 방전
      SOC_MIN ≤ SOC[h+1] ≤ SOC_MAX   ∀ h ∈ [0, N-1]
        SOC[h+1] = SOC[0] + Σ_{j≤h} ( EFFICIENCY * c[j] / cap
                                      - d[j] / (EFFICIENCY * cap) )

    Objective
      revenue = Σ price[h] * (surplus[h] - c[h])
      cost    = Σ price[h] * (deficit[h] - d[h])
      max (revenue - cost)  ↔  min Σ price[h] * (c[h] - d[h])
      (surplus/deficit 항은 상수이므로 목적함수에서 무시 가능)

    Returns
      (c0, d0): 시점 t 의 최적 충방전량. infeasible 이면 (0, 0).
    """
    n_total = len(gen_forecast)
    N = min(horizon, n_total - t)
    if N <= 0:
        return 0.0, 0.0

    cap = params["ess_capacity_mwh"]
    base_demand = params["demand_mwh_per_h"]
    chg_max = params["charge_rate_max"]
    dis_max = params["discharge_rate_max"]

    gen = np.asarray(gen_forecast[t:t + N], dtype=float)
    h_arr = hours_arr[t:t + N]
    m_arr = months_arr[t:t + N]
    demand = np.array(
        [get_demand_at_hour(base_demand, int(h)) for h in h_arr], dtype=float)
    price = np.array(
        [get_tou_price_krw_per_mwh(int(m), int(h))
         for m, h in zip(m_arr, h_arr)], dtype=float)

    surplus = np.maximum(0.0, gen - demand)
    deficit = np.maximum(0.0, demand - gen)

    c_ub = np.minimum(surplus, chg_max)
    d_ub = np.minimum(deficit, dis_max)
    bounds = ([(0.0, float(c_ub[h])) for h in range(N)]
              + [(0.0, float(d_ub[h])) for h in range(N)])

    # 목적: min Σ price * (c - d)
    c_obj = np.concatenate([price, -price])

    # SOC 누적용 하삼각 행렬 (L[h,j] = 1 if j ≤ h)
    L = np.tril(np.ones((N, N)))
    a_c = EFFICIENCY / cap          # c 의 SOC 증가 계수
    a_d = 1.0 / (EFFICIENCY * cap)  # d 의 SOC 감소 계수

    # SOC[h+1] ≤ SOC_MAX  ⇔   a_c·Σc - a_d·Σd ≤ SOC_MAX - soc
    A_upper = np.hstack([a_c * L, -a_d * L])
    b_upper = np.full(N, SOC_MAX - soc)

    # SOC[h+1] ≥ SOC_MIN  ⇔   -a_c·Σc + a_d·Σd ≤ soc - SOC_MIN
    A_lower = np.hstack([-a_c * L, a_d * L])
    b_lower = np.full(N, soc - SOC_MIN)

    A_ub = np.vstack([A_upper, A_lower])
    b_ub = np.concatenate([b_upper, b_lower])

    LP_STATS["total"] += 1
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        LP_STATS["infeasible"] += 1
        return 0.0, 0.0

    c0 = float(max(0.0, res.x[0]))
    d0 = float(max(0.0, res.x[N]))
    return c0, d0


def _mpc_targets_from_action(c0, d0, soc, params):
    """LP 의 (c0, d0) → 다음 시점 SOC 목표 (방식 A: 좁은 밴드)."""
    cap = params["ess_capacity_mwh"]
    soc_next = soc + EFFICIENCY * c0 / cap - d0 / (EFFICIENCY * cap)
    soc_next = max(SOC_MIN, min(SOC_MAX, soc_next))
    return {"soc_target_high": soc_next, "soc_target_low": soc_next}


def policy_mpc_xgb(t, actual, predicted, soc, demand_t, params,
                   months_arr=None, hours_arr=None, horizon=24):
    """
    MPC (XGBoost 예측 기반). horizon=24 시점, LP 매 시점 재풀이.

    예측 정확도가 MPC 가치로 얼마나 전환되는지 측정하기 위한 '현실적' MPC.
    SOC 목표는 LP 가 산출한 c[0], d[0] 으로부터 다음 시점 SOC 를 좁은 밴드로
    되돌려주는 방식(=Method A)을 채택한다. 시뮬 본체와 SOC 갱신식이 동일하므로
    SOC 궤적이 LP 내부 SOC 와 일치한다.
    """
    if months_arr is None or hours_arr is None:
        raise ValueError("policy_mpc_xgb 는 months_arr, hours_arr 가 필요합니다.")
    c0, d0 = _solve_mpc_lp(t, predicted, soc, params,
                           months_arr, hours_arr, horizon=horizon)
    return _mpc_targets_from_action(c0, d0, soc, params)


def policy_mpc_oracle(t, actual, predicted, soc, demand_t, params,
                      months_arr=None, hours_arr=None, horizon=24):
    """
    MPC (실측값 기반). 예측 오차 0 인 이론적 상한 MPC.
    policy_mpc_xgb 와 동일한 LP 를 actual 로 풀이.
    """
    if months_arr is None or hours_arr is None:
        raise ValueError("policy_mpc_oracle 는 months_arr, hours_arr 가 필요합니다.")
    c0, d0 = _solve_mpc_lp(t, actual, soc, params,
                           months_arr, hours_arr, horizon=horizon)
    return _mpc_targets_from_action(c0, d0, soc, params)


# ── 검증 (__main__) ──────────────────────────────────────────────────────────
SHARE_DIR = Path("claude_share")


def _validate_targets(name, targets):
    """SOC_MIN ≤ low ≤ high ≤ SOC_MAX 범위 검증."""
    low = targets["soc_target_low"]
    high = targets["soc_target_high"]
    ok = (SOC_MIN - 1e-9 <= low <= high <= SOC_MAX + 1e-9)
    status = "✓" if ok else "✗"
    print(f"   {status} {name:<34} low={low:.3f}  high={high:.3f}")
    assert ok, f"{name} SOC 범위 위반: low={low}, high={high}"
    return ok


def _main():
    print(f"[{ts()}] [TASK G-2] ESS 정책 함수 검증 시작")

    rng = np.random.default_rng(42)
    n = 48
    actual = np.abs(rng.normal(100, 50, n))
    predicted = actual + rng.normal(0, 20, n)
    demand_t = 80.0
    params = {"ess_capacity_mwh": 500.0, "demand_mwh_per_h": 50.0,
              "charge_rate_max": 100.0, "discharge_rate_max": 100.0,
              "weight": 0.1, "is_noise_region": False}

    print(f"\n[{ts()}] 임의 입력에 대한 SOC 범위 검증 (여러 시점 × 여러 soc)")
    checked = 0
    for t in [0, 5, 23, n - 1]:
        for soc in [0.20, 0.50, 0.80]:
            _validate_targets(f"naive(t={t},soc={soc})",
                              policy_naive(t, actual, predicted, soc, demand_t, params))
            _validate_targets(f"lookahead(t={t},soc={soc})",
                              policy_lookahead(t, actual, predicted, soc, demand_t, params))
            _validate_targets(f"perfect_foresight(t={t},soc={soc})",
                              policy_perfect_foresight(t, actual, predicted, soc, demand_t, params))
            _validate_targets(f"lookahead_noisy(t={t},soc={soc})",
                              policy_lookahead_noisy(t, actual, predicted, soc, demand_t,
                                                     params, noise_level=0.3, rng=rng))
            _validate_targets(f"xgb_no_lookahead(t={t},soc={soc})",
                              policy_xgb_no_lookahead(t, actual, predicted, soc, demand_t, params))
            checked += 5

    # policy_naive 는 항상 (0.80, 0.20)
    print(f"\n[{ts()}] policy_naive 항상 (0.80, 0.20) 반환 검증")
    for t in range(n):
        r = policy_naive(t, actual, predicted, 0.5, demand_t, params)
        assert r == {"soc_target_high": 0.80, "soc_target_low": 0.20}, \
            f"naive 가 t={t}에서 다른 값 반환: {r}"
    print(f"   ✓ 48개 시점 전부 (0.80, 0.20) 확인")

    print(f"\n[{ts()}] ✓ 5개 정책 전부 SOC 범위 검증 통과 (총 {checked}건)")

    print(f"\n[{ts()}] claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    dst = SHARE_DIR / Path(__file__).name
    shutil.copy2(__file__, dst)
    print(f"   → {dst}")

    print(f"\n[{ts()}] [TASK G-2] 완료")


if __name__ == "__main__":
    _main()
