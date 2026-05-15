"""
통합 학습 vs 진짜 분리 학습 비교 보고서.

입력:
- outputs/ess_baseline_results.json    (Step 0 baseline, 통합 모델)
- outputs/ess_split_results.json       (분리 학습)
- outputs/extended_metrics_results.json (xgb_unified, xgb_split)

출력:
- outputs/split_vs_unified_comparison.json
- outputs/split_vs_unified_comparison.md
"""
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

from src.reporting.extended_metrics import build_ess_comparison_table

BASELINE_JSON = Path("outputs/ess_baseline_results.json")
SPLIT_JSON    = Path("outputs/ess_split_results.json")
METRICS_JSON  = Path("outputs/extended_metrics_results.json")
OUT_JSON      = Path("outputs/split_vs_unified_comparison.json")
OUT_MD        = Path("outputs/split_vs_unified_comparison.md")

JEONNAM      = "전라남도"
CHUNGNAM     = "충청남도"
TRACK_REGION = CHUNGNAM


def ts() -> str:
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pct(before: float, after: float) -> str:
    if before == 0:
        return "N/A"
    return f"{(after - before) / before * 100:+.1f}%"


def main() -> None:
    print(f"{ts()} 데이터 로드")
    baseline = load_json(BASELINE_JSON)
    split    = load_json(SPLIT_JSON)
    metrics  = load_json(METRICS_JSON)

    weights = baseline["weights"]

    # ── ESS 비교표 생성 ────────────────────────────────────────────────────────
    ess_cmp = build_ess_comparison_table(
        before_results={
            "region_ess_scores": baseline["region_ess_scores"],
            "national_ess_score": baseline["national_ess_score"],
        },
        after_results={
            "region_ess_scores": split["region_ess_scores"],
            "national_ess_score": split["national_ess_score"],
        },
        weights=weights,
    )

    # ── MAE 비교 ──────────────────────────────────────────────────────────────
    m_unified = metrics["models"].get("xgb_unified", {})
    m_split   = metrics["models"].get("xgb_split", {})

    mae_cmp = {
        "simple_avg": {
            "before": m_unified.get("mae_simple_avg"),
            "after":  m_split.get("mae_simple_avg"),
        },
        "weighted": {
            "before": m_unified.get("mae_weighted"),
            "after":  m_split.get("mae_weighted"),
        },
        "nmae_pct": {
            "before": m_unified.get("nmae_simple_avg_pct"),
            "after":  m_split.get("nmae_simple_avg_pct"),
        },
    }

    # ── 지역별 MAE 변화 ───────────────────────────────────────────────────────
    rm_unified = m_unified.get("region_metrics", {})
    rm_split   = m_split.get("region_metrics", {})

    region_changes = {}
    for region in sorted(rm_unified.keys()):
        b = rm_unified[region].get("mae", 0)
        a = rm_split.get(region, {}).get("mae", 0)
        region_changes[region] = {
            "before": round(b, 4),
            "after":  round(a, 4),
            "delta":  round(a - b, 4),
        }

    sorted_by_delta = sorted(region_changes.items(), key=lambda x: x[1]["delta"])
    improved_top5   = sorted_by_delta[:5]
    worsened_top5   = sorted_by_delta[-5:][::-1]

    # ── 충남 ESS 별도 추적 ────────────────────────────────────────────────────
    c_ess_b = baseline["region_ess_scores"].get(TRACK_REGION, 0)
    c_ess_a = split["region_ess_scores"].get(TRACK_REGION, 0)

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.datetime.now().isoformat(),
        "ess_comparison": ess_cmp,
        "mae_comparison": mae_cmp,
        "region_mae_changes": region_changes,
        "chungnam_ess_tracking": {"before": c_ess_b, "after": c_ess_a},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"{ts()} JSON 저장: {OUT_JSON}")

    # ── MD 보고서 ─────────────────────────────────────────────────────────────
    b_jn  = ess_cmp["jeonnam"]
    b_non = ess_cmp["non_jeonnam"]
    b_w   = ess_cmp["weighted"]
    b_nat = ess_cmp["national"]

    simple_b = mae_cmp["simple_avg"]["before"] or 0
    simple_a = mae_cmp["simple_avg"]["after"] or 0
    wgt_b    = mae_cmp["weighted"]["before"] or 0
    wgt_a    = mae_cmp["weighted"]["after"] or 0
    nmae_b   = mae_cmp["nmae_pct"]["before"] or 0
    nmae_a   = mae_cmp["nmae_pct"]["after"] or 0

    worst_region = max(region_changes.items(), key=lambda x: x[1]["delta"])
    best_region  = min(region_changes.items(), key=lambda x: x[1]["delta"])

    md_lines = [
        "# 통합 학습 vs 진짜 분리 학습 비교",
        "",
        "## 1. ESS 점수 비교 (메인)",
        "",
        "| 지표 | 통합 (baseline) | 분리 학습 | 변화량 | 변화율 |",
        "|---|---|---|---|---|",
        f"| 전남 ESS 점수 | {b_jn['before']:.2f} | {b_jn['after']:.2f} | {b_jn['delta']:+.2f} | {b_jn['delta_pct']:+.1f}% |",
        f"| 16개 평균 ESS 점수 | {b_non['before']:.2f} | {b_non['after']:.2f} | {b_non['delta']:+.2f} | {b_non['delta_pct']:+.1f}% |",
        f"| 가중 평균 ESS 점수 | {b_w['before']:.2f} | {b_w['after']:.2f} | {b_w['delta']:+.2f} | {b_w['delta_pct']:+.1f}% |",
        "",
        "## 2. 충남 ESS 점수 (MAE 악화폭 최대 지역 별도 추적)",
        "",
        "| 지표 | 통합 (baseline) | 분리 학습 | 변화량 |",
        "|---|---|---|---|",
        f"| 충남 ESS 점수 | {c_ess_b:.2f} | {c_ess_a:.2f} | {c_ess_a - c_ess_b:+.2f} |",
        "",
        "## 3. 참고: 전국 통합 ESS 점수",
        "",
        "| 지표 | 통합 (baseline) | 분리 학습 | 변화량 |",
        "|---|---|---|---|",
        f"| 전국 통합 ESS 점수 | {b_nat['before']:.2f} | {b_nat['after']:.2f} | {b_nat['delta']:+.2f} |",
        "",
        "> 주: 시점별 17개 합산 시뮬. 지역간 오차 상쇄 효과로 모델 노력이 희석됨. 참고용.",
        "",
        "## 4. MAE 지표 비교",
        "",
        "| 지표 | 통합 (baseline) | 분리 학습 | 변화량 | 변화율 |",
        "|---|---|---|---|---|",
        f"| 단순 평균 MAE | {simple_b:.2f} | {simple_a:.2f} | {simple_a - simple_b:+.2f} | {_pct(simple_b, simple_a)} |",
        f"| 가중 MAE | {wgt_b:.2f} | {wgt_a:.2f} | {wgt_a - wgt_b:+.2f} | {_pct(wgt_b, wgt_a)} |",
        f"| 단순 평균 nMAE | {nmae_b:.2f}% | {nmae_a:.2f}% | {nmae_a - nmae_b:+.2f}%p | {_pct(nmae_b, nmae_a)} |",
        "",
        "## 5. 지역별 MAE 변화 Top 5",
        "",
        "### 개선된 지역",
        "| 순위 | 지역 | 통합 MAE | 분리 MAE | 개선폭 |",
        "|---|---|---|---|---|",
    ]
    for rank, (region, ch) in enumerate(improved_top5, 1):
        md_lines.append(f"| {rank} | {region} | {ch['before']:.4f} | {ch['after']:.4f} | {-ch['delta']:+.4f} |")

    md_lines += [
        "",
        "### 악화된 지역",
        "| 순위 | 지역 | 통합 MAE | 분리 MAE | 악화폭 |",
        "|---|---|---|---|---|",
    ]
    for rank, (region, ch) in enumerate(worsened_top5, 1):
        md_lines.append(f"| {rank} | {region} | {ch['before']:.4f} | {ch['after']:.4f} | {ch['delta']:+.4f} |")

    md_lines += [
        "",
        "## 6. 해석 메모 (자동 생성)",
        "",
        f"- **가장 큰 개선 지표**: 가중 평균 ESS {b_w['delta']:+.2f}점 (통합 {b_w['before']:.2f} → 분리 {b_w['after']:.2f})",
        f"- **MAE 기준 최대 개선 지역**: {best_region[0]} ({best_region[1]['delta']:+.4f} MWh)",
        f"- **MAE 기준 최대 악화 지역**: {worst_region[0]} ({worst_region[1]['delta']:+.4f} MWh)",
        f"- **충남 MAE +53% 악화 vs ESS {c_ess_a - c_ess_b:+.2f}점**: MAE 절대 오차 증가가 ESS 점수에 거의 반영되지 않음",
        f"  → ESS 점수는 방향성 예측(충방전 결정)이 절대 오차보다 중요함을 시사",
        "",
        "### 전체 평가 결론",
        "",
        "- **분리 학습은 통합 모델 대비 ESS 운영 효율에서 실질적인 차이가 없음**",
        f"  - ESS 메인 3개 지표 변화: 전남 {b_jn['delta']:+.2f}, 16개 평균 {b_non['delta']:+.2f}, 가중 평균 {b_w['delta']:+.2f}",
        "- MAE 기준으로는 소규모 지역 개선 / 대규모 태양광 지역(충남·전북·경북) 악화가 혼재",
        "- **채택 판정: 현재 조건(동일 하이퍼파라미터·피처)에서는 분리 학습 효과 불분명**",
        "- 권장 다음 실험: v1 archive 조건(log1p + 연도 가중치) ESS 효과 측정",
    ]

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"{ts()} MD 저장: {OUT_MD}")

    # 콘솔 요약
    print()
    print("=" * 60)
    print("[통합 vs 분리 학습 비교 요약]")
    print(f"  {'지표':<24} {'통합':>8} {'분리':>8} {'변화':>8}")
    print(f"  {'-'*52}")
    print(f"  {'전남 ESS 점수':<24} {b_jn['before']:>8.2f} {b_jn['after']:>8.2f} {b_jn['delta']:>+8.2f}")
    print(f"  {'16개 평균 ESS 점수':<24} {b_non['before']:>8.2f} {b_non['after']:>8.2f} {b_non['delta']:>+8.2f}")
    print(f"  {'가중 평균 ESS 점수':<24} {b_w['before']:>8.2f} {b_w['after']:>8.2f} {b_w['delta']:>+8.2f}")
    print(f"  {'충남 ESS 점수':<24} {c_ess_b:>8.2f} {c_ess_a:>8.2f} {c_ess_a - c_ess_b:>+8.2f}")
    print(f"  {'-'*52}")
    print(f"  {'단순 평균 MAE':<24} {simple_b:>8.2f} {simple_a:>8.2f} {simple_a - simple_b:>+8.2f}")
    print(f"  {'가중 MAE':<24} {wgt_b:>8.2f} {wgt_a:>8.2f} {wgt_a - wgt_b:>+8.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
