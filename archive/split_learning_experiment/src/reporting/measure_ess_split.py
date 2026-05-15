"""
분리 학습 후 ESS 점수 측정 (3개 지표 체계).

입력:
- outputs/split_ess_simulation_results.json
- outputs/extended_metrics_results.json (가중치 출처)

출력:
- outputs/ess_split_results.json
"""
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.reporting.extended_metrics import (
    compute_weighted_ess_score,
    compute_non_jeonnam_avg_ess_score,
)

ESS_JSON     = Path("outputs/split_ess_simulation_results.json")
METRICS_JSON = Path("outputs/extended_metrics_results.json")
OUT_JSON     = Path("outputs/ess_split_results.json")

JEONNAM = "전라남도"


def ts() -> str:
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print(f"{ts()} ESS 시뮬 결과 로드: {ESS_JSON}")
    ess = load_json(ESS_JSON)

    if "region_ess_scores_xgb_split" not in ess:
        sys.exit(f"키 없음: region_ess_scores_xgb_split in {ESS_JSON}")
    if "xgb_split_strategy" not in ess or "ess_score" not in ess["xgb_split_strategy"]:
        sys.exit(f"키 없음: xgb_split_strategy.ess_score in {ESS_JSON}")

    region_ess_scores: dict = ess["region_ess_scores_xgb_split"]
    national_ess_score: float = float(ess["xgb_split_strategy"]["ess_score"])

    print(f"{ts()} 가중치 로드: {METRICS_JSON}")
    metrics = load_json(METRICS_JSON)

    if "weights" not in metrics:
        sys.exit(f"키 없음: weights in {METRICS_JSON}")
    weights: dict = metrics["weights"]

    missing = set(region_ess_scores.keys()) - set(weights.keys())
    if missing:
        sys.exit(f"가중치 누락 지역: {missing}")

    weighted_ess  = compute_weighted_ess_score(region_ess_scores, weights)
    non_jeonnam   = compute_non_jeonnam_avg_ess_score(region_ess_scores)
    jeonnam_score = float(region_ess_scores.get(JEONNAM, 0.0))

    result = {
        "model":                     "xgb_split",
        "measured_at":               datetime.datetime.now().isoformat(),
        "region_ess_scores":         region_ess_scores,
        "jeonnam_ess_score":         round(jeonnam_score, 4),
        "non_jeonnam_avg_ess_score": round(non_jeonnam, 4),
        "weighted_ess_score":        round(weighted_ess, 4),
        "national_ess_score":        round(national_ess_score, 4),
        "weights":                   weights,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{ts()} 저장 완료: {OUT_JSON}")
    print()
    print("=" * 50)
    print("[분리 학습 ESS 점수 요약]")
    print(f"  전남 ESS 점수         : {jeonnam_score:.2f}")
    print(f"  16개 평균 ESS 점수    : {non_jeonnam:.2f}")
    print(f"  가중 평균 ESS 점수    : {weighted_ess:.2f}")
    print(f"  전국 통합 ESS 점수    : {national_ess_score:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
