"""
ESS 시뮬레이션 파라미터 (v2)

본 시뮬레이션은 통제된 모델 비교 환경이며 실제 운영값 추정이 목적이 아니다.
파라미터는 산업 통상 범위 내에서 선정했으며, 절대값 해석이 아닌
정책(naive/lookahead/oracle) 간 상대 비교에만 유효하다.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ts():
    return datetime.now().strftime("%H:%M:%S")


# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
# 전국 합산 기준 ESS 파라미터 (17개 광역 단위 시뮬용)
# 근거: 한국에너지공단 신재생 REC 가이드라인상 ESS 방전출력은 태양광 설비용량의
# 70% 이내. 산업부 2025년 호남·제주 ESS 입찰 평균 70 MW/곳, 4시간 저장 기준.
TOTAL_ESS_CAPACITY_MWH = 500.0 * 17     # 17개 광역 합산 ESS 용량
TOTAL_DEMAND_MWH_PER_H = 50.0 * 17      # 17개 광역 합산 평균 시간당 수요
TOTAL_CHARGE_RATE_MAX = 100.0 * 17      # 합산 충전 속도 상한
TOTAL_DISCHARGE_RATE_MAX = 100.0 * 17   # 합산 방전 속도 상한

SOC_MIN = 0.20      # DOD 60% 보호. LFP 배터리 수명-가용량 균형의 산업 통상값.
SOC_MAX = 0.80
SOC_INIT = 0.50
EFFICIENCY = 0.90   # 시스템 RTE (Round-Trip Efficiency) 90%, 산업 통상값
                    # (셀 단 95%가 아닌 시스템 단 90%로 보수 조정)

# weight 가 이 임계 미만인 지역은 시뮬 노이즈로 플래그한다.
# (울산시 weight≈0.0002 → ESS 용량 1.4 MWh. 단순 평균에 1/17 비중으로
#  들어가 평균을 왜곡할 수 있으므로 단순 평균을 17개/16개 두 기준으로 병기.)
# 임계 0.001 은 울산시만 잡는다. 서울·세종·대전(weight≈0.003)은 ESS 용량
# 25~29 MWh 로 0/inf 위험이 없어 정상 지역으로 둔다.
WEIGHT_NOISE_THRESHOLD = 0.001  # 0.1% 미만은 노이즈로 플래그 (울산시만 해당)

# ── 시간대별 수요 패턴 ────────────────────────────────────────────────────────
# 한국 일반 전력 부하의 정성적 패턴 (KPX 통계 정성적 참조)
# 절대값 정확도 아닌 시간 변동성 도입이 목적.
# 정성적 패턴 (모양 우선 정의 — 새벽 저점·한낮 피크·저녁 둔덕)
_RAW_PATTERN = np.array([
    0.70, 0.65, 0.60, 0.60, 0.65, 0.75,
    0.85, 0.95, 1.05, 1.15, 1.20, 1.25,
    1.25, 1.20, 1.15, 1.15, 1.10, 1.15,
    1.25, 1.20, 1.10, 1.00, 0.90, 0.80,
])
# 정규화: 평균을 정확히 1.0으로 (원본 평균 약 0.9854에서 보정)
HOURLY_LOAD_FACTOR = _RAW_PATTERN / _RAW_PATTERN.mean()
assert abs(HOURLY_LOAD_FACTOR.mean() - 1.0) < 1e-10


# ── 지역별 파라미터 빌더 ──────────────────────────────────────────────────────
def build_region_params(train_df: pd.DataFrame) -> dict:
    """
    train 데이터의 지역별 평균 발전량 비중으로 ESS 파라미터를 차등 분배.

    근거: 발전 인프라가 큰 지역은 ESS 용량도 크고 수요도 크다는 자연스러운 가정.
    너희 데이터에서 직접 도출되므로 외부 자료 의존성 없음.

    Returns:
        dict[region_name -> dict[param_name -> float]]
    """
    if "region" not in train_df.columns or "power_mwh" not in train_df.columns:
        sys.exit("ERROR: train_df 에 'region' 또는 'power_mwh' 컬럼이 없습니다.")

    region_mean_gen = train_df.groupby("region")["power_mwh"].mean()
    total_mean_gen = region_mean_gen.sum()
    if total_mean_gen <= 0:
        sys.exit("ERROR: 지역별 평균 발전량 합이 0 이하입니다.")

    params = {}
    for region, mean_gen in region_mean_gen.items():
        weight = float(mean_gen / total_mean_gen)
        params[region] = {
            "ess_capacity_mwh":   TOTAL_ESS_CAPACITY_MWH * weight,
            "demand_mwh_per_h":   TOTAL_DEMAND_MWH_PER_H * weight,
            "charge_rate_max":    TOTAL_CHARGE_RATE_MAX * weight,
            "discharge_rate_max": TOTAL_DISCHARGE_RATE_MAX * weight,
            "weight":             weight,
            "is_noise_region":    weight < WEIGHT_NOISE_THRESHOLD,
        }
    return params


def get_demand_at_hour(base_demand: float, hour: int) -> float:
    """시간대별 수요 = 평균 수요 × 부하 패턴."""
    return base_demand * HOURLY_LOAD_FACTOR[hour]


# ── 검증 출력 ─────────────────────────────────────────────────────────────────
TRAIN_FEATURES = "data/processed/national_train_features.csv"
SHARE_DIR = Path("claude_share")


def _main():
    print(f"[{ts()}] [TASK G-1] ESS 파라미터 모듈 검증 시작")

    train_path = Path(TRAIN_FEATURES)
    if not train_path.exists():
        sys.exit(f"ERROR: 파일 없음 → {TRAIN_FEATURES}")

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    region_params = build_region_params(train_df)

    print(f"\n[{ts()}] 지역별 ESS 파라미터 표 ({len(region_params)}개 지역)")
    print("─" * 72)
    print(f"{'region':<12}{'weight':>10}{'ess_capacity':>16}{'demand/h':>16}  flag")
    print("─" * 80)
    weight_sum = 0.0
    noise_regions = []
    for region, p in sorted(region_params.items(), key=lambda kv: -kv[1]["weight"]):
        weight_sum += p["weight"]
        flag = "NOISE" if p["is_noise_region"] else ""
        if p["is_noise_region"]:
            noise_regions.append(region)
        print(f"{region:<12}{p['weight']:>10.4f}"
              f"{p['ess_capacity_mwh']:>16.1f}{p['demand_mwh_per_h']:>16.2f}  {flag}")
    print("─" * 80)
    print(f"{'합계':<12}{weight_sum:>10.4f}"
          f"{TOTAL_ESS_CAPACITY_MWH:>16.1f}{TOTAL_DEMAND_MWH_PER_H:>16.2f}")
    print(f"\n[{ts()}] 노이즈 플래그 지역 (weight < {WEIGHT_NOISE_THRESHOLD}): "
          f"{noise_regions if noise_regions else '없음'}")

    # 검증
    assert abs(weight_sum - 1.0) < 0.001, f"weight 합이 1.0 이 아님: {weight_sum}"
    print(f"\n[{ts()}] ✓ weight 합 = {weight_sum:.6f} (1.0 ± 0.001 통과)")

    load_mean = float(HOURLY_LOAD_FACTOR.mean())
    assert abs(load_mean - 1.0) < 0.01, f"부하 패턴 평균이 1.0 이 아님: {load_mean}"
    print(f"[{ts()}] ✓ 부하 패턴 평균 = {load_mean:.6f} (1.0 ± 0.01 통과)")

    # claude_share 복사 (순수 모듈 — 자기 자신만)
    print(f"\n[{ts()}] claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    dst = SHARE_DIR / Path(__file__).name
    shutil.copy2(__file__, dst)
    print(f"   → {dst}")

    print(f"\n[{ts()}] [TASK G-1] 완료")


if __name__ == "__main__":
    _main()
