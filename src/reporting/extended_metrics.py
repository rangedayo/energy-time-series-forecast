"""
평가 지표 확장 스크립트
MAE / nMAE / 가중 MAE를 Naive + 정의된 XGBoost 모델들에 대해 계산.

모델 추가 방법: MODELS 딕셔너리에 항목 추가만 하면 됨.
"""
import sys as _sys
_sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font
_apply_font()

import sys
import json
import datetime
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

BASELINE_JSON = Path("outputs/national_baseline_results.json")
TEST_FEATURES = Path("data/processed/national_test_features.csv")

OUT_JSON  = Path("outputs/extended_metrics_results.json")
OUT_MD    = Path("outputs/extended_metrics_report.md")
OUT_PNG   = Path("outputs/extended_metrics_comparison.png")
SHARE_DIR = Path("claude_share")

# 평가 대상 모델 정의 — 모델 추가 시 여기에만 추가
MODELS = {
    "xgb_unified": {
        "label": "XGBoost 통합",
        "csv": Path("outputs/national_xgb_predictions.csv"),
    },
}


def ts() -> str:
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def load_predictions(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["timestamp"])
    for col in ["region", "actual", "predicted"]:
        if col not in df.columns:
            sys.exit(f"컬럼 없음 ({col}): {csv_path}")
    return df


def compute_region_metrics(df: pd.DataFrame, actual_sum_by_region: dict, total_actual_sum: float) -> dict:
    region_metrics = {}
    for region in sorted(df["region"].unique()):
        sub = df[df["region"] == region]
        actual    = sub["actual"].values
        predicted = sub["predicted"].values

        mae = float(np.mean(np.abs(actual - predicted)))

        mask = actual > 0
        if mask.sum() > 0:
            nmae_pct = float(np.mean(np.abs(actual[mask] - predicted[mask])) / np.mean(actual[mask]) * 100)
        else:
            nmae_pct = None

        weight = float(actual_sum_by_region.get(region, 0) / total_actual_sum) if total_actual_sum > 0 else 0.0
        region_metrics[region] = {"mae": mae, "nmae_pct": nmae_pct, "weight": weight}
    return region_metrics


def aggregate(region_metrics: dict) -> dict:
    mae_list  = [v["mae"] for v in region_metrics.values()]
    nmae_list = [v["nmae_pct"] for v in region_metrics.values() if v["nmae_pct"] is not None]
    mae_weighted = sum(v["mae"] * v["weight"] for v in region_metrics.values())
    return {
        "mae_simple_avg":      float(np.mean(mae_list)),
        "nmae_simple_avg_pct": float(np.mean(nmae_list)) if nmae_list else None,
        "mae_weighted":        float(mae_weighted),
    }


def compute_baseline_metrics(baseline_json: dict, actual_mean_pos_by_region: dict,
                              actual_sum_by_region: dict, total_actual_sum: float) -> dict:
    region_mae = baseline_json["lag1"]["region_MAE"]
    region_metrics = {}
    for region, mae in region_mae.items():
        w        = float(actual_sum_by_region.get(region, 0) / total_actual_sum) if total_actual_sum > 0 else 0.0
        mean_pos = actual_mean_pos_by_region.get(region)
        nmae_pct = float(mae / mean_pos * 100) if mean_pos and mean_pos > 0 else None
        region_metrics[region] = {"mae": float(mae), "nmae_pct": nmae_pct, "weight": w}
    return region_metrics


def build_md_report(result: dict, actual_sum_by_region: dict, total_actual_sum: float) -> str:
    lines = ["# 평가 지표 확장 보고서 (MAE / nMAE / 가중 MAE)\n"]

    # naive + 정의된 모델 순서로 레이블 구성
    model_labels = {"naive": "Naive (lag1)"}
    for key, cfg in MODELS.items():
        model_labels[key] = cfg["label"]

    # 표 1 — 전국 집계 비교
    lines.append("## 표 1 — 전국 집계 비교\n")
    lines.append("| 모델 | 단순 평균 MAE | 단순 평균 nMAE(%) | 가중 MAE |")
    lines.append("|---|---|---|---|")
    for key, label in model_labels.items():
        if key not in result["models"]:
            continue
        m        = result["models"][key]
        nmae_str = f"{m['nmae_simple_avg_pct']:.2f}" if m["nmae_simple_avg_pct"] is not None else "N/A"
        lines.append(f"| {label} | {m['mae_simple_avg']:.2f} | {nmae_str} | {m['mae_weighted']:.2f} |")
    lines.append("")

    # 표 2 — 지역별 가중치
    lines.append("## 표 2 — 지역별 가중치 (가중치 큰 순)\n")
    lines.append("| 지역 | 발전량 합계(MWh) | 가중치(%) |")
    lines.append("|---|---|---|")
    for region in sorted(actual_sum_by_region, key=lambda r: -actual_sum_by_region[r]):
        s     = actual_sum_by_region[region]
        w_pct = s / total_actual_sum * 100 if total_actual_sum > 0 else 0
        lines.append(f"| {region} | {s:,.0f} | {w_pct:.2f} |")
    lines.append("")

    # 표 3 — 지역별 3개 지표 (xgb_unified 기준)
    ref_key = "xgb_unified"
    lines.append(f"## 표 3 — 지역별 3개 지표 ({model_labels.get(ref_key, ref_key)}, 가중치 큰 순)\n")
    lines.append("| 지역 | MAE | nMAE(%) | 가중치(%) | MAE × 가중치 |")
    lines.append("|---|---|---|---|---|")
    ref_rm = result["models"].get(ref_key, {}).get("region_metrics", {})
    for region in sorted(ref_rm, key=lambda r: -ref_rm[r].get("weight", 0)):
        m        = ref_rm[region]
        nmae_str = f"{m['nmae_pct']:.2f}" if m["nmae_pct"] is not None else "N/A"
        w_pct    = m["weight"] * 100
        lines.append(f"| {region} | {m['mae']:.2f} | {nmae_str} | {w_pct:.2f} | {m['mae'] * m['weight']:.2f} |")
    lines.append("")

    # 관찰 메모
    lines.append("## 관찰 메모\n")
    lines.append("### 단순 평균 vs 가중 평균 차이\n")
    for key, label in model_labels.items():
        if key not in result["models"]:
            continue
        m    = result["models"][key]
        diff = m["mae_simple_avg"] - m["mae_weighted"]
        lines.append(f"- {label}: 단순 {m['mae_simple_avg']:.2f} vs 가중 {m['mae_weighted']:.2f} (차이: {diff:+.2f})")
    lines.append("")

    lines.append(f"### nMAE 기준 정확도 ({model_labels.get(ref_key, ref_key)} 기준)\n")
    nmae_valid = {r: m["nmae_pct"] for r, m in ref_rm.items() if m["nmae_pct"] is not None}
    if nmae_valid:
        best_r  = min(nmae_valid, key=lambda r: nmae_valid[r])
        worst_r = max(nmae_valid, key=lambda r: nmae_valid[r])
        lines.append(f"- 가장 정확한 지역 (nMAE 최소): {best_r} ({nmae_valid[best_r]:.2f}%)")
        lines.append(f"- 가장 부정확한 지역 (nMAE 최대): {worst_r} ({nmae_valid[worst_r]:.2f}%)")
    lines.append("")

    lines.append("### 전라남도가 가중 MAE에서 차지하는 실제 영향\n")
    jn = "전라남도"
    for key, label in model_labels.items():
        if key not in result["models"]:
            continue
        rm = result["models"][key]["region_metrics"]
        if jn in rm:
            contrib     = rm[jn]["mae"] * rm[jn]["weight"]
            total_wmae  = result["models"][key]["mae_weighted"]
            pct         = contrib / total_wmae * 100 if total_wmae > 0 else 0
            lines.append(f"- {label}: 전남 기여분 {contrib:.2f} (가중 MAE {total_wmae:.2f}의 {pct:.1f}%)")

    return "\n".join(lines) + "\n"


def compute_weighted_ess_score(region_ess_scores: dict, weights: dict) -> float:
    """지역별 ESS 점수를 발전량 비중으로 가중평균."""
    total = 0.0
    total_weight = 0.0
    for region, score in region_ess_scores.items():
        w = weights.get(region, 0.0)
        total += score * w
        total_weight += w
    return float(total / total_weight) if total_weight > 0 else 0.0


def compute_non_jeonnam_avg_ess_score(region_ess_scores: dict) -> float:
    """전남을 제외한 16개 지역의 ESS 점수 단순 평균."""
    scores = [v for k, v in region_ess_scores.items() if k != "전라남도"]
    return float(np.mean(scores)) if scores else 0.0


def build_ess_comparison_table(
    before_results: dict,
    after_results: dict,
    weights: dict,
) -> dict:
    """
    분리 학습 전후 ESS 점수 비교 표 생성.

    Args:
        before_results: {"region_ess_scores": {...}, "national_ess_score": float}
        after_results:  같은 구조
        weights:        지역별 발전량 비중 (합 = 1.0)

    Returns:
        jeonnam / non_jeonnam / weighted / national 각각 before·after·delta·delta_pct
    """
    def _delta(b: float, a: float) -> dict:
        d     = a - b
        d_pct = (d / b * 100.0) if b != 0 else None
        return {
            "before":    round(b, 4),
            "after":     round(a, 4),
            "delta":     round(d, 4),
            "delta_pct": round(d_pct, 2) if d_pct is not None else None,
        }

    b_scores = before_results["region_ess_scores"]
    a_scores = after_results["region_ess_scores"]

    return {
        "jeonnam": _delta(
            b_scores.get("전라남도", 0.0),
            a_scores.get("전라남도", 0.0),
        ),
        "non_jeonnam": _delta(
            compute_non_jeonnam_avg_ess_score(b_scores),
            compute_non_jeonnam_avg_ess_score(a_scores),
        ),
        "weighted": _delta(
            compute_weighted_ess_score(b_scores, weights),
            compute_weighted_ess_score(a_scores, weights),
        ),
        "national": _delta(
            before_results["national_ess_score"],
            after_results["national_ess_score"],
        ),
    }


def plot_comparison(result: dict, out_path: Path):
    model_keys   = ["naive"] + list(MODELS.keys())
    model_labels = ["Naive"] + [cfg["label"] for cfg in MODELS.values()]
    colors       = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: len(model_keys)]

    mae_simple   = [result["models"].get(k, {}).get("mae_simple_avg", 0)          for k in model_keys]
    nmae         = [result["models"].get(k, {}).get("nmae_simple_avg_pct") or 0   for k in model_keys]
    mae_weighted = [result["models"].get(k, {}).get("mae_weighted", 0)             for k in model_keys]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("모델별 평가 지표 비교", fontsize=14)

    x = np.arange(len(model_keys))
    for ax, title, vals in zip(
        axes,
        ["단순 평균 MAE", "단순 평균 nMAE (%)", "가중 MAE"],
        [mae_simple, nmae, mae_weighted],
    ):
        bars = ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=15, ha="right")
        ax.set_ylabel("값")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    for p in [BASELINE_JSON, TEST_FEATURES]:
        if not p.exists():
            sys.exit(f"필수 파일 없음: {p}")

    print(f"{ts()} baseline JSON 로드: {BASELINE_JSON}")
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        baseline_json = json.load(f)

    print(f"{ts()} test_features 로드: {TEST_FEATURES}")
    test_df = pd.read_csv(TEST_FEATURES, encoding="utf-8-sig", parse_dates=["timestamp"])
    if "region" not in test_df.columns or "power_mwh" not in test_df.columns:
        sys.exit("test_features에 region 또는 power_mwh 컬럼 없음")

    actual_sum_by_region     = test_df.groupby("region")["power_mwh"].sum().to_dict()
    total_actual_sum         = sum(actual_sum_by_region.values())
    mask_pos                 = test_df["power_mwh"] > 0
    actual_mean_pos_by_region = test_df[mask_pos].groupby("region")["power_mwh"].mean().to_dict()

    models_data = {}

    print(f"{ts()} baseline 지표 계산...")
    baseline_rm       = compute_baseline_metrics(baseline_json, actual_mean_pos_by_region, actual_sum_by_region, total_actual_sum)
    models_data["naive"] = {**aggregate(baseline_rm), "region_metrics": baseline_rm}

    # MODELS 딕셔너리 기반 루프 — 모델 추가 시 위 MODELS에만 추가
    for key, cfg in MODELS.items():
        print(f"{ts()} {cfg['label']} 예측 로드: {cfg['csv']}")
        df = load_predictions(cfg["csv"])
        if df is None:
            sys.exit(f"파일 없음: {cfg['csv']}")
        rm = compute_region_metrics(df, actual_sum_by_region, total_actual_sum)
        models_data[key] = {**aggregate(rm), "region_metrics": rm}

    weights = {r: float(actual_sum_by_region.get(r, 0) / total_actual_sum) for r in sorted(actual_sum_by_region)}
    result  = {"weights": weights, "models": models_data}

    print(f"{ts()} JSON 저장: {OUT_JSON}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{ts()} MD 보고서 생성: {OUT_MD}")
    OUT_MD.write_text(build_md_report(result, actual_sum_by_region, total_actual_sum), encoding="utf-8")

    print(f"{ts()} 시각화 생성: {OUT_PNG}")
    plot_comparison(result, OUT_PNG)

    print(f"{ts()} claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    for src, dst in [
        (__file__,  SHARE_DIR / "extended_metrics.py"),
        (OUT_JSON,  SHARE_DIR / "extended_metrics_results.json"),
        (OUT_MD,    SHARE_DIR / "extended_metrics_report.md"),
        (OUT_PNG,   SHARE_DIR / "extended_metrics_comparison.png"),
    ]:
        shutil.copy2(src, dst)

    print(f"\n{'='*60}")
    print("[평가 지표 확장 결과]")
    print(f"{'='*60}")
    all_labels = {"naive": "Naive (lag1)"}
    all_labels.update({k: cfg["label"] for k, cfg in MODELS.items()})
    print(f"  {'모델':22s} | {'단순 MAE':>10s} | {'가중 MAE':>10s} | {'nMAE(%)':>10s}")
    print(f"  {'-'*58}")
    for key, label in all_labels.items():
        if key not in result["models"]:
            continue
        m        = result["models"][key]
        nmae_str = f"{m['nmae_simple_avg_pct']:.2f}" if m["nmae_simple_avg_pct"] is not None else "N/A"
        print(f"  {label:22s} | {m['mae_simple_avg']:>10.2f} | {m['mae_weighted']:>10.2f} | {nmae_str:>10s}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
