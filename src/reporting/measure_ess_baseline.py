"""
분리 학습 전 baseline ESS 점수 측정.

새 시뮬레이션을 돌리지 않고, 기존 결과에서 4가지 집계값을 계산해 저장한다.

입력:
- outputs/national_ess_simulation_results.json
    키: region_ess_scores_xgb  → 지역별 ESS 점수 (17개)
    키: xgb_strategy.ess_score → 전국 통합 ESS 점수
- outputs/extended_metrics_results.json
    키: weights → 지역별 발전량 비중

출력:
- outputs/ess_baseline_results.json
    {
        "model": "national_xgb_unified",
        "measured_at": ISO timestamp,
        "region_ess_scores": {17개 지역: score},
        "weighted_ess_score": float,
        "non_jeonnam_avg_ess_score": float,
        "jeonnam_ess_score": float,
        "national_ess_score": float,
        "weights": {17개 지역: weight}
    }
"""
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

ESS_JSON     = Path("outputs/national_ess_simulation_results.json")
METRICS_JSON = Path("outputs/extended_metrics_results.json")
OUT_JSON     = Path("outputs/ess_baseline_results.json")

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

    if "region_ess_scores_xgb" not in ess:
        sys.exit(f"키 없음: region_ess_scores_xgb in {ESS_JSON}")
    if "xgb_strategy" not in ess or "ess_score" not in ess["xgb_strategy"]:
        sys.exit(f"키 없음: xgb_strategy.ess_score in {ESS_JSON}")

    region_ess_scores: dict = ess["region_ess_scores_xgb"]
    national_ess_score: float = float(ess["xgb_strategy"]["ess_score"])

    print(f"{ts()} 가중치 로드: {METRICS_JSON}")
    metrics = load_json(METRICS_JSON)

    if "weights" not in metrics:
        sys.exit(f"키 없음: weights in {METRICS_JSON}")
    weights: dict = metrics["weights"]

    missing = set(region_ess_scores.keys()) - set(weights.keys())
    if missing:
        sys.exit(f"가중치 누락 지역: {missing}")

    def _weighted(scores: dict, w: dict) -> float:
        total, total_w = 0.0, 0.0
        for region, score in scores.items():
            ww = w.get(region, 0.0)
            total += score * ww
            total_w += ww
        return float(total / total_w) if total_w > 0 else 0.0

    def _non_jeonnam_avg(scores: dict) -> float:
        vals = [v for k, v in scores.items() if k != JEONNAM]
        return float(sum(vals) / len(vals)) if vals else 0.0

    weighted_ess  = _weighted(region_ess_scores, weights)
    non_jeonnam   = _non_jeonnam_avg(region_ess_scores)
    jeonnam_score = float(region_ess_scores.get(JEONNAM, 0.0))

    result = {
        "model":                     "national_xgb_unified",
        "measured_at":               datetime.datetime.now().isoformat(),
        "region_ess_scores":         region_ess_scores,
        "weighted_ess_score":        round(weighted_ess, 4),
        "non_jeonnam_avg_ess_score": round(non_jeonnam, 4),
        "jeonnam_ess_score":         round(jeonnam_score, 4),
        "national_ess_score":        round(national_ess_score, 4),
        "weights":                   weights,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{ts()} 저장 완료: {OUT_JSON}")
    print()
    print("=" * 50)
    print("[Baseline ESS 점수 요약]")
    print(f"  전남 ESS 점수         : {jeonnam_score:.2f}")
    print(f"  16개 평균 ESS 점수    : {non_jeonnam:.2f}")
    print(f"  가중 평균 ESS 점수    : {weighted_ess:.2f}")
    print(f"  전국 통합 ESS 점수    : {national_ess_score:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
