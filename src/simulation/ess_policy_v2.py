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

import sys as _sys
_sys.path.insert(0, ".")
from src.simulation.ess_config_v2 import SOC_MIN, SOC_MAX

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ts():
    return datetime.now().strftime("%H:%M:%S")


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
