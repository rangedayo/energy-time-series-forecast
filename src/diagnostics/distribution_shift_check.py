"""
분포 차이 진단 스크립트
train 마지막 20% vs test 분포 비교 (시간 순 분리 기준)
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
from scipy.stats import ks_2samp

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_FEATURES = Path("data/processed/national_train_features.csv")
TEST_FEATURES = Path("data/processed/national_test_features.csv")
OUT_JSON = Path("outputs/distribution_shift_diagnosis.json")
OUT_MD = Path("outputs/distribution_shift_diagnosis.md")
OUT_PNG_OVERALL = Path("outputs/distribution_shift_power_overall.png")
OUT_PNG_JEONNAM = Path("outputs/distribution_shift_power_jeonnam.png")
OUT_PNG_HOURLY = Path("outputs/distribution_shift_hourly.png")
SHARE_DIR = Path("claude_share")

FEATURES_OF_INTEREST = ["lag_1h", "일사량", "기온", "rolling_mean_3h"]
TARGET = "power_mwh"
JEONNAM = "전라남도"


def ts() -> str:
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def load_data():
    for p in [TRAIN_FEATURES, TEST_FEATURES]:
        if not p.exists():
            sys.exit(f"파일 없음: {p}")
    train = pd.read_csv(TRAIN_FEATURES, encoding="utf-8-sig", parse_dates=["timestamp"])
    test = pd.read_csv(TEST_FEATURES, encoding="utf-8-sig", parse_dates=["timestamp"])
    required = [TARGET, "hour", "region"] + FEATURES_OF_INTEREST
    for col in required:
        if col not in train.columns:
            sys.exit(f"train에 컬럼 없음: {col}")
        if col not in test.columns:
            sys.exit(f"test에 컬럼 없음: {col}")
    return train, test


def split_train_late(train: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    train_sorted = train.sort_values("timestamp")
    cut = int(len(train_sorted) * (1 - frac))
    return train_sorted.iloc[cut:].copy()


def period_info(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "rows": len(df),
    }


def power_stats(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "nonzero_ratio": float((series > 0).mean()),
    }


def ks_result(a: pd.Series, b: pd.Series) -> dict:
    stat, pvalue = ks_2samp(a.values, b.values)
    return {"ks_stat": round(float(stat), 4), "p_value": round(float(pvalue), 6), "shift_flag": bool(pvalue < 0.01)}


def hourly_avg(df: pd.DataFrame, region: str | None = None) -> pd.Series:
    sub = df if region is None else df[df["region"] == region]
    return sub.groupby("hour")[TARGET].mean()


def feature_stats(df: pd.DataFrame, feat: str) -> dict:
    s = df[feat].dropna()
    return {"mean": float(s.mean()), "std": float(s.std()), "p95": float(s.quantile(0.95)), "max": float(s.max())}


def run_diagnosis(train_late: pd.DataFrame, test: pd.DataFrame) -> dict:
    regions = sorted(test["region"].unique())

    result: dict = {
        "period": {
            "train_late": period_info(train_late, "train_late"),
            "test": period_info(test, "test"),
        },
        "power_overall": {},
        "power_by_region": {},
        "hourly": {},
        "features": {},
    }

    result["power_overall"]["train_late"] = power_stats(train_late[TARGET])
    result["power_overall"]["test"] = power_stats(test[TARGET])
    result["power_overall"]["ks"] = ks_result(train_late[TARGET], test[TARGET])

    for region in regions:
        tl = train_late[train_late["region"] == region][TARGET]
        te = test[test["region"] == region][TARGET]
        result["power_by_region"][region] = {
            "train_late": power_stats(tl),
            "test": power_stats(te),
            "ks": ks_result(tl, te),
        }

    hourly_tl_all = hourly_avg(train_late)
    hourly_te_all = hourly_avg(test)
    hourly_tl_jn = hourly_avg(train_late, JEONNAM)
    hourly_te_jn = hourly_avg(test, JEONNAM)

    hourly_rows_all = []
    hourly_rows_jn = []
    for h in range(24):
        tl_v = float(hourly_tl_all.get(h, 0))
        te_v = float(hourly_te_all.get(h, 0))
        diff_pct = (te_v - tl_v) / tl_v * 100 if tl_v > 0 else None
        hourly_rows_all.append({"hour": h, "train_late": tl_v, "test": te_v, "diff_pct": diff_pct})

        tl_v_jn = float(hourly_tl_jn.get(h, 0))
        te_v_jn = float(hourly_te_jn.get(h, 0))
        diff_pct_jn = (te_v_jn - tl_v_jn) / tl_v_jn * 100 if tl_v_jn > 0 else None
        hourly_rows_jn.append({"hour": h, "train_late": tl_v_jn, "test": te_v_jn, "diff_pct": diff_pct_jn})

    result["hourly"]["overall"] = hourly_rows_all
    result["hourly"]["jeonnam"] = hourly_rows_jn

    for feat in FEATURES_OF_INTEREST:
        result["features"][feat] = {
            "overall": {
                "train_late": feature_stats(train_late, feat),
                "test": feature_stats(test, feat),
                "ks": ks_result(train_late[feat].dropna(), test[feat].dropna()),
            },
            "jeonnam": {
                "train_late": feature_stats(train_late[train_late["region"] == JEONNAM], feat),
                "test": feature_stats(test[test["region"] == JEONNAM], feat),
                "ks": ks_result(
                    train_late[train_late["region"] == JEONNAM][feat].dropna(),
                    test[test["region"] == JEONNAM][feat].dropna(),
                ),
            },
        }

    return result


def plot_power_hist(tl_series: pd.Series, te_series: pd.Series, title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)
    for ax, (tl, te, subtitle) in zip(
        axes,
        [
            (tl_series, te_series, "전체 (0 포함)"),
            (tl_series[tl_series > 0], te_series[te_series > 0], "발전 시간대 (0 제외)"),
        ],
    ):
        ax.hist(tl, bins=60, alpha=0.5, label="train_late", density=True)
        ax.hist(te, bins=60, alpha=0.5, label="test", density=True)
        ax.set_title(subtitle)
        ax.set_xlabel("발전량 (MWh)")
        ax.set_ylabel("밀도")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_hourly(result: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("시간대별 평균 발전량 비교 (train_late vs test)", fontsize=14)
    for ax, (key, label) in zip(axes, [("overall", "전국 전체"), ("jeonnam", "전라남도")]):
        rows = result["hourly"][key]
        hours = [r["hour"] for r in rows]
        tl_vals = [r["train_late"] for r in rows]
        te_vals = [r["test"] for r in rows]
        ax.plot(hours, tl_vals, marker="o", label="train_late", markersize=4)
        ax.plot(hours, te_vals, marker="s", label="test", markersize=4)
        ax.set_title(label)
        ax.set_xlabel("시간 (hour)")
        ax.set_ylabel("평균 발전량 (MWh)")
        ax.legend()
        ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def build_md_report(result: dict, regions: list) -> str:
    lines = ["# 분포 차이 진단 보고서\n"]

    p = result["period"]
    lines.append("## 1. 기간 정보\n")
    lines.append("| 구분 | 시작 | 종료 | 행수 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| train_late | {p['train_late']['start']} | {p['train_late']['end']} | {p['train_late']['rows']:,} |")
    lines.append(f"| test | {p['test']['start']} | {p['test']['end']} | {p['test']['rows']:,} |\n")

    lines.append("## 2. 전체 발전량 분포 비교\n")
    lines.append("| 통계량 | train_late | test |")
    lines.append("|---|---|---|")
    for k in ["mean", "std", "min", "max", "median", "p95", "p99", "nonzero_ratio"]:
        tl_v = result["power_overall"]["train_late"][k]
        te_v = result["power_overall"]["test"][k]
        lines.append(f"| {k} | {tl_v:.4f} | {te_v:.4f} |")
    ks = result["power_overall"]["ks"]
    shift_label = "SHIFT 의심" if ks["shift_flag"] else "정상"
    lines.append(f"\n- KS 통계량: {ks['ks_stat']}, p-value: {ks['p_value']} → **{shift_label}**\n")

    lines.append("## 3. 지역별 발전량 분포 비교\n")
    lines.append("| 지역 | train_late mean | test mean | train_late std | test std | KS 통계량 | p-value | 판정 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    shift_regions = []
    for region in regions:
        rb = result["power_by_region"][region]
        ks = rb["ks"]
        flag = "SHIFT 의심" if ks["shift_flag"] else ""
        if ks["shift_flag"]:
            shift_regions.append(region)
        lines.append(
            f"| {region} | {rb['train_late']['mean']:.2f} | {rb['test']['mean']:.2f} | "
            f"{rb['train_late']['std']:.2f} | {rb['test']['std']:.2f} | "
            f"{ks['ks_stat']} | {ks['p_value']} | {flag} |"
        )
    lines.append("")

    lines.append("## 4. 시간대별 평균 발전량 비교\n")
    lines.append("### 4-1. 전국 전체\n")
    lines.append("| 시간 | train_late | test | 차이(%) |")
    lines.append("|---|---|---|---|")
    big_diff_hours = []
    for row in result["hourly"]["overall"]:
        diff_str = f"{row['diff_pct']:.1f}%" if row["diff_pct"] is not None else "N/A"
        lines.append(f"| {row['hour']:02d}:00 | {row['train_late']:.3f} | {row['test']:.3f} | {diff_str} |")
        if row["diff_pct"] is not None and abs(row["diff_pct"]) > 20:
            big_diff_hours.append(row["hour"])

    lines.append("\n### 4-2. 전라남도\n")
    lines.append("| 시간 | train_late | test | 차이(%) |")
    lines.append("|---|---|---|---|")
    for row in result["hourly"]["jeonnam"]:
        diff_str = f"{row['diff_pct']:.1f}%" if row["diff_pct"] is not None else "N/A"
        lines.append(f"| {row['hour']:02d}:00 | {row['train_late']:.3f} | {row['test']:.3f} | {diff_str} |")
    lines.append("")

    lines.append("## 5. 핵심 피처 분포 비교\n")
    for feat in FEATURES_OF_INTEREST:
        lines.append(f"### {feat}\n")
        lines.append("| 범위 | train_late mean | test mean | train_late std | test std | train_late p95 | test p95 | KS 통계량 | p-value | 판정 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for scope_key, scope_label in [("overall", "전국"), ("jeonnam", "전라남도")]:
            fd = result["features"][feat][scope_key]
            ks = fd["ks"]
            flag = "SHIFT 의심" if ks["shift_flag"] else ""
            lines.append(
                f"| {scope_label} | {fd['train_late']['mean']:.4f} | {fd['test']['mean']:.4f} | "
                f"{fd['train_late']['std']:.4f} | {fd['test']['std']:.4f} | "
                f"{fd['train_late']['p95']:.4f} | {fd['test']['p95']:.4f} | "
                f"{ks['ks_stat']} | {ks['p_value']} | {flag} |"
            )
        lines.append("")

    lines.append("## 진단 결론\n")
    lines.append(f"### SHIFT 의심 지역 (KS p < 0.01): {len(shift_regions)}개\n")
    lines.append(", ".join(shift_regions) if shift_regions else "없음")
    lines.append("")

    lines.append(f"### 시간대별 차이 큰 시간 (전국 평균 |%| > 20%): {len(big_diff_hours)}개\n")
    lines.append(", ".join([f"{h:02d}:00" for h in big_diff_hours]) if big_diff_hours else "없음")
    lines.append("")

    jn_ks = result["power_by_region"].get(JEONNAM, {}).get("ks", {})
    jn_flag = "분포 차이 유의 (SHIFT 의심)" if jn_ks.get("shift_flag") else "유의한 분포 차이 없음"
    lines.append("### 전라남도 특화 결론\n")
    lines.append(f"- 발전량 KS p-value: {jn_ks.get('p_value', 'N/A')} → {jn_flag}")
    jn_tl = result["power_by_region"].get(JEONNAM, {}).get("train_late", {})
    jn_te = result["power_by_region"].get(JEONNAM, {}).get("test", {})
    if jn_tl and jn_te:
        lines.append(f"- train_late 평균: {jn_tl['mean']:.2f} MWh, test 평균: {jn_te['mean']:.2f} MWh")

    return "\n".join(lines) + "\n"


def main():
    print(f"{ts()} 데이터 로드 중...")
    train, test = load_data()

    print(f"{ts()} train 마지막 20% 분리 중...")
    train_late = split_train_late(train, frac=0.2)
    regions = sorted(test["region"].unique())

    print(f"{ts()} train_late 기간: {train_late['timestamp'].min()} ~ {train_late['timestamp'].max()} ({len(train_late):,}행)")
    print(f"{ts()} test 기간: {test['timestamp'].min()} ~ {test['timestamp'].max()} ({len(test):,}행)")

    print(f"{ts()} 분포 분석 실행 중...")
    result = run_diagnosis(train_late, test)

    shift_regions = [r for r in regions if result["power_by_region"][r]["ks"]["shift_flag"]]
    big_diff_hours = [
        row["hour"]
        for row in result["hourly"]["overall"]
        if row["diff_pct"] is not None and abs(row["diff_pct"]) > 20
    ]

    print(f"{ts()} 시각화 생성 중...")
    plot_power_hist(
        train_late[TARGET], test[TARGET],
        "전체 발전량 분포 비교 (train_late vs test)",
        OUT_PNG_OVERALL,
    )
    plot_power_hist(
        train_late[train_late["region"] == JEONNAM][TARGET],
        test[test["region"] == JEONNAM][TARGET],
        "전라남도 발전량 분포 비교 (train_late vs test)",
        OUT_PNG_JEONNAM,
    )
    plot_hourly(result, OUT_PNG_HOURLY)

    print(f"{ts()} JSON 저장: {OUT_JSON}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{ts()} MD 보고서 저장: {OUT_MD}")
    md_text = build_md_report(result, regions)
    OUT_MD.write_text(md_text, encoding="utf-8")

    print(f"{ts()} claude_share 복사 중...")
    SHARE_DIR.mkdir(exist_ok=True)
    for src, dst in [
        (__file__, SHARE_DIR / "distribution_shift_check.py"),
        (OUT_JSON, SHARE_DIR / "distribution_shift_diagnosis.json"),
        (OUT_MD, SHARE_DIR / "distribution_shift_diagnosis.md"),
        (OUT_PNG_OVERALL, SHARE_DIR / "distribution_shift_power_overall.png"),
        (OUT_PNG_JEONNAM, SHARE_DIR / "distribution_shift_power_jeonnam.png"),
        (OUT_PNG_HOURLY, SHARE_DIR / "distribution_shift_hourly.png"),
    ]:
        shutil.copy2(src, dst)

    print(f"\n{'='*60}")
    print(f"[작업 1 요약] 분포 차이 진단 결과")
    print(f"{'='*60}")
    print(f"  SHIFT 의심 지역 수: {len(shift_regions)}개")
    if shift_regions:
        print(f"  SHIFT 의심 지역: {', '.join(shift_regions)}")
    jn_ks = result["power_by_region"].get(JEONNAM, {}).get("ks", {})
    print(f"  전남 KS p-value: {jn_ks.get('p_value', 'N/A')}")
    print(f"  시간대 차이 큰 시간 (|%| > 20%): {len(big_diff_hours)}개")
    if big_diff_hours:
        print(f"  해당 시간: {', '.join([f'{h:02d}:00' for h in big_diff_hours])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
