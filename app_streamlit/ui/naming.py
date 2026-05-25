"""정책 내부명 ↔ 운영자 친화 표시명 매핑 (round 2-3).

orchestrator.run_mpc_simulation() 의 results 키와 1:1 대응한다.
"""
from __future__ import annotations

POLICY_DISPLAY_NAMES: dict[str, str] = {
    "naive": "기본 운영",
    "xgb_lookahead": "단기 예측 기반",
    "mpc_xgb": "수익 최적화 (MPC)",
}

POLICY_DESCRIPTIONS: dict[str, str] = {
    "naive": "예측 없이 SOC를 0.20~0.80 범위로 고정 유지",
    "xgb_lookahead": "다음 1시간 발전량 예측을 보고 SOC 목표 조정",
    "mpc_xgb": "24시간 예측 기반 LP 최적화 (Rolling Horizon)",
}

POLICY_COLORS: dict[str, str] = {
    "naive": "#1D9E75",          # teal-600
    "xgb_lookahead": "#BA7517",  # amber-600
    "mpc_xgb": "#534AB7",        # purple-600
}


def display_name(policy: str) -> str:
    return POLICY_DISPLAY_NAMES.get(policy, policy)
