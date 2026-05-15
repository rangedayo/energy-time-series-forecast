"""
Phase 1 데이터 진단 스크립트

충남(-0.1% 개선)과 전남(MAE 90.42 이상치)의 근본 원인을 데이터 레벨에서 진단.
모델 학습/예측 코드 없음. scipy.stats만 허용.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ── 경로 상수 ─────────────────────────────────────────────
TRAIN_READY = "data/processed/national_train_ready.csv"
TEST_READY  = "data/processed/national_test_ready.csv"
TRAIN_FEAT  = "data/processed/national_train_features.csv"
TEST_FEAT   = "data/processed/national_test_features.csv"
OUT_DIR     = "outputs/diagnostics"
RESULT_JSON = f"{OUT_DIR}/phase1_diagnosis_results.json"
REPORT_MD   = f"{OUT_DIR}/phase1_diagnosis_report.md"

# ── 진단 임계값 상수 ─────────────────────────────────────
CORR_MIN_THRESHOLD   = 0.70   # 일사량-발전량 상관 미만 시 매핑 의심
CORR_BELOW_MEAN_GAP  = 0.10   # 전국 평균 대비 이 값 이상 낮으면 의심
CAPACITY_GROWTH_RATE = 0.50   # 연간 최대값 50% 이상 급증 시 설비 급증 의심
IQR_OUTLIER_MULTIPLE = 1.50   # 전국 평균의 이 배수 이상이면 이상치 의심
KS_PVAL_THRESHOLD    = 0.01   # KS p-value 미만이면 분포 유의하게 다름
DRIFT_CHANGE_PCT     = 15.0   # 평균 변화율 절댓값 초과 시 drift 경고

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────

def _ensure_utf8_stdout() -> None:
    """Windows CP949 콘솔에서 UTF-8 출력 가능하게 재설정."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def log(task: str, msg: str) -> None:
    print(f"{ts()} [{task}] {msg}")


# ─────────────────────────────────────────────────────────
# TASK P1-A — 환경 점검 및 데이터 로드
# ─────────────────────────────────────────────────────────

def task_a_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("TASK P1-A", "환경 점검 시작")

    for path in [TRAIN_READY, TEST_READY, TRAIN_FEAT, TEST_FEAT]:
        if not os.path.exists(path):
            log("TASK P1-A", f"ERROR: 파일 없음 — {path}")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    log("TASK P1-A", f"출력 디렉토리 준비: {OUT_DIR}")

    enc = "utf-8-sig"
    train      = pd.read_csv(TRAIN_READY, encoding=enc, parse_dates=["timestamp"])
    test       = pd.read_csv(TEST_READY,  encoding=enc, parse_dates=["timestamp"])
    train_feat = pd.read_csv(TRAIN_FEAT,  encoding=enc, parse_dates=["timestamp"])
    test_feat  = pd.read_csv(TEST_FEAT,   encoding=enc, parse_dates=["timestamp"])

    for name, df in [
        ("train_ready", train),
        ("test_ready",  test),
        ("train_feat",  train_feat),
        ("test_feat",   test_feat),
    ]:
        period    = f"{df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}"
        n_regions = df["region"].nunique()
        log("TASK P1-A", f"{name}: {len(df):,}행 | 기간 {period} | 지역 {n_regions}개")

    return train, test, train_feat, test_feat


# ─────────────────────────────────────────────────────────
# TASK P1-B — 충남·전남 raw 데이터 연도별 추이 분석
# ─────────────────────────────────────────────────────────

def task_b_yearly_trend(
    train: pd.DataFrame,
    test: pd.DataFrame,
    diagnosis: dict,
) -> None:
    log("TASK P1-B", "충남·전남 연도별 추이 분석 시작")

    full_df = pd.concat([train, test], ignore_index=True)
    diagnosis["yearly_trend"]  = {}
    diagnosis["설비_급증_의심"] = {}

    for region in ["충청남도", "전라남도"]:
        rdf = full_df[full_df["region"] == region].copy()
        rdf["year"] = rdf["timestamp"].dt.year

        yearly_stats = rdf.groupby("year").agg(
            평균=("power_mwh", "mean"),
            중앙값=("power_mwh", "median"),
            최대=("power_mwh", "max"),
            표준편차=("power_mwh", "std"),
            결측비율=("power_mwh", lambda x: x.isna().mean()),
            영값비율=("power_mwh", lambda x: (x == 0).mean()),
        )

        peak_df = rdf[rdf["timestamp"].dt.hour.isin(range(10, 15))]
        peak_yearly = peak_df.groupby("year")["power_mwh"].mean().rename("피크시간_평균")
        yearly_stats = yearly_stats.join(peak_yearly)

        # 설비 급증 신호 (최대값 기준 50% 이상 급증)
        growth_flags: dict[int, float] = {}
        yearly_max = yearly_stats["최대"]
        for y1, y2 in zip(yearly_max.index[:-1], yearly_max.index[1:]):
            if yearly_max[y1] > 0:
                growth = (yearly_max[y2] - yearly_max[y1]) / yearly_max[y1]
                if growth > CAPACITY_GROWTH_RATE:
                    growth_flags[int(y2)] = round(growth * 100, 1)

        diagnosis["설비_급증_의심"][region] = growth_flags
        diagnosis["yearly_trend"][region]  = {
            "stats":     yearly_stats.round(3).to_dict(),
            "급증_연도": growth_flags,
        }

        # ── 시각화 (4-subplot) ──────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{region} 연도별 추이", fontsize=14, fontweight="bold")
        years  = yearly_stats.index.tolist()
        colors = ["#2196F3" if y < 2023 else "#F44336" for y in years]

        ax = axes[0, 0]
        bars = ax.bar(years, yearly_stats["평균"], color=colors)
        ax.set_title("연도별 평균 발전량 (MWh)")
        ax.set_xlabel("연도")
        ax.set_ylabel("MWh")
        for bar, val in zip(bars, yearly_stats["평균"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
            )

        ax = axes[0, 1]
        ax.bar(years, yearly_stats["최대"], color=colors)
        ax.set_title("연도별 최대값 (MWh)")
        ax.set_xlabel("연도")
        ax.set_ylabel("MWh")
        for yr, pct in growth_flags.items():
            if yr in years:
                idx = years.index(yr)
                ax.annotate(
                    f"급증+{pct}%",
                    xy=(years[idx], yearly_stats["최대"].iloc[idx]),
                    fontsize=8, color="red",
                    xytext=(0, 5), textcoords="offset points", ha="center",
                )

        ax = axes[1, 0]
        peak_vals = yearly_stats["피크시간_평균"].fillna(0).values
        ax.bar(years, peak_vals, color=colors)
        ax.set_title("연도별 피크 시간대(10~14시) 평균 (MWh)")
        ax.set_xlabel("연도")
        ax.set_ylabel("MWh")

        ax = axes[1, 1]
        w = 0.35
        ax.bar(
            [y - w / 2 for y in years],
            yearly_stats["영값비율"] * 100,
            width=w, label="영값 비율 (%)", color="#FF9800", alpha=0.85,
        )
        ax.bar(
            [y + w / 2 for y in years],
            yearly_stats["결측비율"] * 100,
            width=w, label="결측 비율 (%)", color="#9E9E9E", alpha=0.85,
        )
        ax.set_title("연도별 영값/결측 비율 (%)")
        ax.set_xlabel("연도")
        ax.set_ylabel("%")
        ax.legend()

        from matplotlib.patches import Patch
        fig.legend(
            handles=[
                Patch(facecolor="#2196F3", label="Train (2023 이전)"),
                Patch(facecolor="#F44336", label="Test (2023)"),
            ],
            loc="upper right",
        )
        plt.tight_layout()

        short    = "충남" if "충청" in region else "전남"
        png_path = f"{OUT_DIR}/phase1_{short}_연도별_추이.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("TASK P1-B", f"{region} → {png_path}")

        if growth_flags:
            for yr, pct in growth_flags.items():
                log("TASK P1-B", f"[설비 급증 의심] {region} {yr}년 최대값 전년 대비 +{pct}%")
        else:
            log("TASK P1-B", f"{region}: 설비 급증 신호 없음")

    log("TASK P1-B", "완료")


# ─────────────────────────────────────────────────────────
# TASK P1-C — 17개 지역 전체 이상치 비율 분석
# ─────────────────────────────────────────────────────────

def task_c_outlier_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame,
    diagnosis: dict,
) -> None:
    log("TASK P1-C", "17개 지역 이상치 비율 분석 시작")

    full_df = pd.concat([train, test], ignore_index=True)
    records = []
    missing_peak_regions: list[str] = []

    for region in sorted(full_df["region"].unique()):
        rdf  = full_df[full_df["region"] == region]
        peak = rdf[rdf["timestamp"].dt.hour.isin(range(10, 15))]

        # 피크 시간대 데이터 결측 감지 (전체의 1% 미만이면 결측으로 판단)
        expected_peak_ratio = len(peak) / len(rdf) if len(rdf) > 0 else 0
        peak_missing = len(peak) < 100 or expected_peak_ratio < 0.01

        if peak_missing:
            missing_peak_regions.append(region)
            hour_dist = rdf["timestamp"].dt.hour.unique().tolist()
            log(
                "TASK P1-C",
                f"[데이터 결측] {region}: 피크 시간대 행수={len(peak)} "
                f"(전체 {len(rdf)}행의 {expected_peak_ratio*100:.1f}%) "
                f"존재 시간대={sorted(hour_dist)[:10]}",
            )
            iqr_outlier_pct = float("nan")
            daytime_zero_pct = float("nan")
        else:
            q1, q3 = peak["power_mwh"].quantile([0.25, 0.75])
            iqr = q3 - q1
            iqr_outlier_pct = (
                (peak["power_mwh"] > q3 + 1.5 * iqr)
                | (peak["power_mwh"] < q1 - 1.5 * iqr)
            ).mean() * 100
            daytime_zero_pct = (peak["power_mwh"] == 0).mean() * 100

        mean_v, std_v = rdf["power_mwh"].mean(), rdf["power_mwh"].std()
        z_outlier_pct = (
            (np.abs((rdf["power_mwh"] - mean_v) / std_v) > 3).mean() * 100
            if std_v > 0 else 0.0
        )

        records.append({
            "region":           region,
            "IQR_이상치_pct":    round(iqr_outlier_pct, 2) if not np.isnan(iqr_outlier_pct) else None,
            "Z3_이상치_pct":     round(z_outlier_pct, 2),
            "주간_영값_pct":     round(daytime_zero_pct, 2) if not np.isnan(daytime_zero_pct) else None,
            "평균_발전량":       round(mean_v, 2),
            "최대_발전량":       round(rdf["power_mwh"].max(), 2),
            "피크행수":          len(peak),
            "피크시간_결측여부": peak_missing,
        })

    diagnosis["missing_peak_regions"] = missing_peak_regions

    outlier_df = pd.DataFrame(records).sort_values("IQR_이상치_pct", ascending=False)
    csv_path   = f"{OUT_DIR}/phase1_지역별_이상치_비율.csv"
    outlier_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log("TASK P1-C", f"이상치 비율 CSV → {csv_path}")

    diagnosis["outlier_summary"] = records

    # 자동 해석
    nat_iqr_mean  = outlier_df["IQR_이상치_pct"].mean()
    nat_zero_mean = outlier_df["주간_영값_pct"].mean()

    for region, short in [("충청남도", "충남"), ("전라남도", "전남")]:
        row = outlier_df[outlier_df["region"] == region]
        if row.empty:
            continue
        row = row.iloc[0]

        # 피크 시간대 결측 체크
        if row.get("피크시간_결측여부", False):
            print(
                f"[해석] {short} 피크 시간대(10~14시) 데이터 완전 결측!"
                f" (행수={row.get('피크행수', 0)}) ← 데이터 수집/전처리 오류 확정"
            )
            continue

        iqr_v = row["IQR_이상치_pct"]
        if pd.isna(iqr_v):
            print(f"[해석] {short} IQR 이상치 비율: 계산 불가 (피크 데이터 부족)")
        else:
            ratio = iqr_v / nat_iqr_mean if nat_iqr_mean > 0 else 0
            print(
                f"[해석] {short} IQR 이상치 비율: {iqr_v}%"
                f" (전국 평균 {nat_iqr_mean:.1f}%의 {ratio:.1f}배)"
                f"  {'← 의심' if ratio >= IQR_OUTLIER_MULTIPLE else '← 정상'}"
            )

        zero_v = row["주간_영값_pct"]
        if not pd.isna(zero_v):
            zratio = zero_v / nat_zero_mean if nat_zero_mean > 0 else 0
            print(
                f"[해석] {short} 주간 영값 비율: {zero_v}%"
                f" (전국 평균 {nat_zero_mean:.1f}%)"
                f"  {'← 의심' if zratio >= IQR_OUTLIER_MULTIPLE else '← 정상'}"
            )

    top2 = outlier_df.nlargest(2, "최대_발전량")
    if len(top2) >= 2:
        t1, t2 = top2.iloc[0], top2.iloc[1]
        r = t1["최대_발전량"] / t2["최대_발전량"] if t2["최대_발전량"] > 0 else 0
        print(
            f"[해석] {t1['region']} 최대값 {t1['최대_발전량']}"
            f" (2위 지역의 {r:.1f}배)"
            f"  {'← 이상치 확정' if r > 3 else '← 정상 범주'}"
        )

    # 시각화 — 17개 지역 boxplot (log scale)
    fig, ax = plt.subplots(figsize=(16, 7))
    peak_full   = pd.concat([train, test], ignore_index=True)
    peak_full   = peak_full[peak_full["timestamp"].dt.hour.isin(range(10, 15))]
    reg_ordered = outlier_df["region"].tolist()
    box_data    = [
        peak_full[peak_full["region"] == r]["power_mwh"].dropna().values
        for r in reg_ordered
    ]
    ax.boxplot(
        box_data, labels=reg_ordered, vert=True, patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#90CAF9", color="#1565C0"),
        medianprops=dict(color="#F44336", linewidth=2),
    )
    ax.set_yscale("log")
    ax.set_title("17개 지역 피크 시간대(10~14시) 발전량 분포 비교 (log scale)", fontsize=13)
    ax.set_ylabel("발전량 (MWh, log scale)")
    ax.set_xlabel("지역")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png_path = f"{OUT_DIR}/phase1_지역별_분포_비교.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("TASK P1-C", f"지역별 분포 비교 PNG → {png_path}")
    log("TASK P1-C", "완료")


# ─────────────────────────────────────────────────────────
# TASK P1-D — 기상 관측소 매핑 품질 진단
# ─────────────────────────────────────────────────────────

def task_d_mapping_quality(train: pd.DataFrame, diagnosis: dict) -> None:
    log("TASK P1-D", "기상 관측소 매핑 품질 진단 시작")

    required_cols = ["일사량", "기온", "전운량"]
    missing = [c for c in required_cols if c not in train.columns]
    if missing:
        log("TASK P1-D", f"WARNING: 기상 컬럼 없음 — {missing}")
        diagnosis["correlation_quality"] = []
        diagnosis["매핑품질_의심"] = {}
        return

    correlations = []
    for region in sorted(train["region"].unique()):
        rdf  = train[train["region"] == region]
        peak = rdf[rdf["timestamp"].dt.hour.isin(range(10, 15))]
        if len(peak) < 100:
            continue
        row: dict = {"region": region}
        for col in required_cols:
            v = peak[col].corr(peak["power_mwh"])
            row[f"corr_{col}_발전량"] = round(v, 3) if not np.isnan(v) else None
        correlations.append(row)

    corr_df  = pd.DataFrame(correlations).sort_values("corr_일사량_발전량")
    csv_path = f"{OUT_DIR}/phase1_지역별_기상상관.csv"
    corr_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log("TASK P1-D", f"기상 상관계수 CSV → {csv_path}")

    diagnosis["correlation_quality"] = correlations
    diagnosis["매핑품질_의심"] = {}

    valid = corr_df["corr_일사량_발전량"].dropna()
    mean_corr = valid.mean()

    for region in ["충청남도", "전라남도"]:
        rows = corr_df[corr_df["region"] == region]
        if rows.empty:
            continue
        rc = rows["corr_일사량_발전량"].iloc[0]
        if rc is not None and rc < mean_corr - CORR_BELOW_MEAN_GAP:
            diagnosis["매핑품질_의심"][region] = {
                "corr":   rc,
                "전국평균": round(mean_corr, 3),
                "결론":   "기상 관측소 매핑 재검토 필요 — 다중 관측소 평균 도입 검토",
            }

    # 시각화 — 충남/전남 각 4-subplot
    for focus, short in [("충청남도", "충남"), ("전라남도", "전남")]:
        if focus not in corr_df["region"].values:
            continue
        others = corr_df[corr_df["region"] != focus].dropna(subset=["corr_일사량_발전량"])
        if others.empty:
            continue

        median_r = others.loc[
            (others["corr_일사량_발전량"] - others["corr_일사량_발전량"].median()).abs().idxmin(),
            "region",
        ]
        best_r  = others.loc[others["corr_일사량_발전량"].idxmax(), "region"]
        worst_r = others.loc[others["corr_일사량_발전량"].idxmin(), "region"]

        panels = [(focus, "대상"), (median_r, "중간"), (best_r, "최고"), (worst_r, "최저")]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{focus} 기상 관측소 매핑 품질 (일사량-발전량 산점도)", fontsize=13)

        for ax, (r, label) in zip(axes.flatten(), panels):
            peak = (
                train[train["region"] == r]
                .loc[train["timestamp"].dt.hour.isin(range(10, 15))]
                .dropna(subset=["일사량", "power_mwh"])
            )
            if len(peak) > 0:
                cv    = peak["일사량"].corr(peak["power_mwh"])
                color = "#F44336" if cv < CORR_MIN_THRESHOLD else "#4CAF50"
                ax.scatter(peak["일사량"], peak["power_mwh"], alpha=0.15, s=5, color=color)
                if len(peak) > 10:
                    z_fit = np.polyfit(peak["일사량"], peak["power_mwh"], 1)
                    x_rng = np.linspace(peak["일사량"].min(), peak["일사량"].max(), 100)
                    ax.plot(x_rng, np.poly1d(z_fit)(x_rng), "k--", linewidth=1.5)
                warn = " [!] 매핑 의심" if cv < CORR_MIN_THRESHOLD else ""
                ax.set_title(f"{r} ({label})\nr={cv:.3f}{warn}", fontsize=10)
            else:
                ax.set_title(f"{r} ({label})\n데이터 없음")
            ax.set_xlabel("일사량 (MJ/m²)")
            ax.set_ylabel("발전량 (MWh)")

        plt.tight_layout()
        png_path = f"{OUT_DIR}/phase1_{short}_관측소매핑_품질.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("TASK P1-D", f"{focus} → {png_path}")

    log("TASK P1-D", "완료")


# ─────────────────────────────────────────────────────────
# TASK P1-E — Train/Test 분포 차이 진단
# ─────────────────────────────────────────────────────────

def task_e_distribution_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    diagnosis: dict,
) -> None:
    log("TASK P1-E", "Train/Test 분포 차이 진단 시작")

    train_late_2022 = train[train["timestamp"] >= "2022-07-01"]
    test_2023       = test

    # (1) 지역별 평균 비교 CSV
    records = []
    for region in sorted(train["region"].unique()):
        a      = train_late_2022[train_late_2022["region"] == region]["power_mwh"]
        b      = test_2023[test_2023["region"] == region]["power_mwh"]
        a_mean = a.mean()
        b_mean = b.mean()
        records.append({
            "region":         region,
            "2022하반기_평균": round(a_mean, 2),
            "2023_평균":      round(b_mean, 2),
            "변화율_pct":     round((b_mean - a_mean) / a_mean * 100, 1) if a_mean > 0 else None,
            "2022하반기_최대": round(a.max(), 2),
            "2023_최대":      round(b.max(), 2),
        })

    drift_df = pd.DataFrame(records)
    csv_path = f"{OUT_DIR}/phase1_지역별_평균발전량_train_vs_test.csv"
    drift_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log("TASK P1-E", f"분포 비교 CSV → {csv_path}")

    # (2) KS 검정
    ks_results: dict[str, dict] = {}
    for region in train["region"].unique():
        a = train_late_2022[train_late_2022["region"] == region]["power_mwh"].dropna().values
        b = test_2023[test_2023["region"] == region]["power_mwh"].dropna().values
        if len(a) > 30 and len(b) > 30:
            stat, pval = ks_2samp(a, b)
            chg_series = drift_df.loc[drift_df["region"] == region, "변화율_pct"]
            chg = float(chg_series.iloc[0]) if not chg_series.empty and chg_series.iloc[0] is not None else 0.0
            ks_results[region] = {
                "ks_stat":    round(stat, 4),
                "pvalue":     round(pval, 4),
                "change_pct": chg,
            }

    diagnosis["train_test_drift"] = ks_results

    drift_regions = [
        (r, info["change_pct"])
        for r, info in ks_results.items()
        if info["pvalue"] < KS_PVAL_THRESHOLD and abs(info["change_pct"]) > DRIFT_CHANGE_PCT
    ]
    if drift_regions:
        diagnosis["distribution_drift"] = {
            "결론":          "유의한 분포 변화 감지 — validation 전략 재검토 필요",
            "drift_regions": drift_regions,
            "권장":          "TimeSeriesSplit 도입 또는 2023년 1~3월을 별도 val로 분리하는 방안 검토",
        }
        for r, chg in sorted(drift_regions, key=lambda x: abs(x[1]), reverse=True):
            log("TASK P1-E", f"[DRIFT] {r}: 평균 변화 {chg:+.1f}% (KS p={ks_results[r]['pvalue']:.4f})")
    else:
        diagnosis["distribution_drift"] = {"결론": "유의한 분포 변화 없음"}
        log("TASK P1-E", "유의한 분포 drift 없음")

    # (3) 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("2022 하반기 Train vs 2023 Test 발전량 분포 비교", fontsize=13)

    for ax, (region, short) in zip(axes, [("전라남도", "전남"), ("충청남도", "충남")]):
        a = train_late_2022[train_late_2022["region"] == region]["power_mwh"].dropna()
        b = test_2023[test_2023["region"] == region]["power_mwh"].dropna()
        if len(a) == 0 or len(b) == 0:
            ax.set_title(f"{region} ({short})\n데이터 없음")
            continue

        bins = np.linspace(0, max(a.max(), b.max()), 50)
        ax.hist(a, bins=bins, alpha=0.6, color="#2196F3", label="2022 하반기 Train", density=True)
        ax.hist(b, bins=bins, alpha=0.6, color="#F44336", label="2023 Test",        density=True)
        ax.legend()
        ax.set_xlabel("발전량 (MWh)")
        ax.set_ylabel("밀도")

        ks = ks_results.get(region, {})
        chg_s = drift_df.loc[drift_df["region"] == region, "변화율_pct"]
        chg_l = f"{float(chg_s.iloc[0]):+.1f}%" if not chg_s.empty and chg_s.iloc[0] is not None else "N/A"
        ax.set_title(
            f"{region} ({short})\nKS p={ks.get('pvalue', 'N/A')}, 변화율={chg_l}"
            if ks else f"{region} ({short})"
        )

    plt.tight_layout()
    png_path = f"{OUT_DIR}/phase1_2022하반기_vs_2023_분포.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("TASK P1-E", f"분포 비교 PNG → {png_path}")
    log("TASK P1-E", "완료")


# ─────────────────────────────────────────────────────────
# TASK P1-F — 종합 진단 리포트 생성
# ─────────────────────────────────────────────────────────

def _mean(lst: list) -> float:
    valid = [v for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return sum(valid) / len(valid) if valid else 0.0


def task_f_report(diagnosis: dict) -> None:
    log("TASK P1-F", "종합 진단 리포트 생성 시작")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_iqr   = [r["IQR_이상치_pct"] for r in diagnosis.get("outlier_summary", [])]
    mean_iqr  = _mean(all_iqr)
    all_corrs = [
        r.get("corr_일사량_발전량")
        for r in diagnosis.get("correlation_quality", [])
        if r.get("corr_일사량_발전량") is not None
    ]
    mean_corr_val = _mean(all_corrs)

    auto_diag: dict[str, dict] = {}
    for region in ["충청남도", "전라남도"]:
        growth       = diagnosis.get("설비_급증_의심", {}).get(region, {})
        o_rows       = [r for r in diagnosis.get("outlier_summary", []) if r["region"] == region]
        iqr_pct      = o_rows[0]["IQR_이상치_pct"] if o_rows else 0.0
        map_susp     = region in diagnosis.get("매핑품질_의심", {})
        c_rows       = [r for r in diagnosis.get("correlation_quality", []) if r["region"] == region]
        r_corr       = c_rows[0].get("corr_일사량_발전량") if c_rows else None
        ks           = diagnosis.get("train_test_drift", {}).get(region, {})
        drift_sig    = ks.get("pvalue", 1.0) < KS_PVAL_THRESHOLD and abs(ks.get("change_pct", 0.0)) > DRIFT_CHANGE_PCT

        # 피크 시간대 결측 최우선 확인
        peak_missing = region in diagnosis.get("missing_peak_regions", [])
        if peak_missing:
            cause = "피크 시간대(10~14시) 데이터 완전 결측 — 데이터 수집/전처리 오류"
            rec   = "원천 데이터 재수집 또는 해당 지역 분리 처리 (피크 시간 결측 보완 필수)"
        elif map_susp and r_corr is not None:
            cause = f"기상 관측소 매핑 품질 미흡 (일사량 상관 {r_corr:.3f}, 전국 평균 {mean_corr_val:.3f})"
            rec   = "다중 관측소 매핑 또는 지역 내 여러 관측소 평균 피처 도입"
        elif iqr_pct > mean_iqr * IQR_OUTLIER_MULTIPLE:
            cause = f"IQR 이상치 비율 과다 ({iqr_pct:.1f}%, 전국 평균 {mean_iqr:.1f}%의 {iqr_pct / mean_iqr:.1f}배)"
            rec   = "raw 데이터 정제 후 재학습"
        elif drift_sig:
            cause = f"Train/Test 분포 이동 (KS p={ks['pvalue']:.4f}, 평균 변화 {ks['change_pct']:+.1f}%)"
            rec   = "TimeSeriesSplit 도입 또는 2023년 초를 별도 validation으로 분리"
        elif growth:
            yrs   = ", ".join(f"{y}년(+{v}%)" for y, v in sorted(growth.items()))
            cause = f"설비 급증 의심: {yrs}"
            rec   = "설비 용량 정규화 피처 도입 또는 분리 학습 검토"
        else:
            cause = "데이터 레벨 이상 없음 — 모델/피처 레벨 원인 가능성 높음"
            rec   = "지역별 정규화(Phase 2) 또는 추가 피처 엔지니어링 진행"

        auto_diag[region] = {"primary_cause": cause, "recommended_action": rec}

    diagnosis["auto_diagnosis"] = auto_diag

    # JSON 저장
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, ensure_ascii=False, indent=2, default=str)
    log("TASK P1-F", f"JSON 결과 → {RESULT_JSON}")

    # ── 마크다운 리포트 ──────────────────────────────────
    lines: list[str] = [
        "# Phase 1 데이터 진단 리포트",
        f"생성일시: {now_str}",
        "분석 대상: 충청남도 (-0.1% 개선), 전라남도 (MAE 90.42)",
        "",
        "---",
        "",
        "## 1. 핵심 결론 (자동 생성)",
        "",
    ]

    for region, short in [("충청남도", "충남"), ("전라남도", "전남")]:
        lines.append(f"### {region} ({short})")

        growth = diagnosis.get("설비_급증_의심", {}).get(region, {})
        peak_miss = region in diagnosis.get("missing_peak_regions", [])
        if peak_miss:
            lines.append("- **연도별 추이**: [CRITICAL] 피크 시간대(10~14시) 데이터 완전 결측")
        elif growth:
            yrs = ", ".join(f"{y}년 +{v}%" for y, v in sorted(growth.items()))
            lines.append(f"- **연도별 추이**: 급변 의심 — {yrs}")
        else:
            lines.append("- **연도별 추이**: 정상 (급격한 변화 없음)")

        o_r = [r for r in diagnosis.get("outlier_summary", []) if r["region"] == region]
        if o_r:
            iv = o_r[0]["IQR_이상치_pct"]
            if iv is None:
                lines.append("- **이상치 비율**: 피크 시간대 데이터 결측으로 계산 불가")
            else:
                ratio = iv / mean_iqr if mean_iqr > 0 else 0
                lines.append(
                    f"- **이상치 비율**: IQR {iv}%"
                    f" (전국 평균 {mean_iqr:.1f}%) — {'의심' if ratio >= IQR_OUTLIER_MULTIPLE else '정상'}"
                )

        c_r = [r for r in diagnosis.get("correlation_quality", []) if r["region"] == region]
        if c_r and c_r[0].get("corr_일사량_발전량") is not None:
            cv   = c_r[0]["corr_일사량_발전량"]
            susp = "의심" if cv < mean_corr_val - CORR_BELOW_MEAN_GAP else "정상"
            lines.append(
                f"- **기상 매핑 품질**: 일사량-발전량 상관 {cv:.3f}"
                f" (전국 평균 {mean_corr_val:.3f}) — {susp}"
            )

        ks_i = diagnosis.get("train_test_drift", {}).get(region, {})
        if ks_i:
            pv  = ks_i.get("pvalue", 1.0)
            chg = ks_i.get("change_pct", 0.0)
            sig = "있음" if pv < KS_PVAL_THRESHOLD and abs(chg) > DRIFT_CHANGE_PCT else "없음"
            lines.append(f"- **Train/Test drift**: KS p-value={pv:.4f}, 평균 변화 {chg:+.1f}% — {sig}")

        ad = auto_diag.get(region, {})
        lines.append(f"- **추정 원인**: {ad.get('primary_cause', '-')}")
        lines.append(f"- **권장 다음 단계**: {ad.get('recommended_action', '-')}")
        lines.append("")

    lines += ["---", "", "## 2. 17개 지역 이상치 비율 표", ""]
    if os.path.exists(f"{OUT_DIR}/phase1_지역별_이상치_비율.csv"):
        df_o = pd.read_csv(f"{OUT_DIR}/phase1_지역별_이상치_비율.csv", encoding="utf-8-sig")
        lines.append(df_o.to_markdown(index=False))
    lines.append("")

    lines += ["---", "", "## 3. 17개 지역 기상 상관계수 표", ""]
    if os.path.exists(f"{OUT_DIR}/phase1_지역별_기상상관.csv"):
        df_c = pd.read_csv(f"{OUT_DIR}/phase1_지역별_기상상관.csv", encoding="utf-8-sig")
        df_c = df_c.sort_values("corr_일사량_발전량")
        lines.append(df_c.to_markdown(index=False))
    lines.append("")

    lines += ["---", "", "## 4. Train(2022 하반기) vs Test(2023) 분포 변화 표", ""]
    if os.path.exists(f"{OUT_DIR}/phase1_지역별_평균발전량_train_vs_test.csv"):
        df_d = pd.read_csv(f"{OUT_DIR}/phase1_지역별_평균발전량_train_vs_test.csv", encoding="utf-8-sig")
        df_d = df_d.assign(_abs=df_d["변화율_pct"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
        lines.append(df_d.to_markdown(index=False))
    lines.append("")

    lines += ["---", "", "## 5. 권장 다음 작업 (자동 추론)", "", "### 충청남도"]

    cn_map   = bool(diagnosis.get("매핑품질_의심", {}).get("충청남도"))
    cn_iqr   = any(
        r["region"] == "충청남도"
        and r["IQR_이상치_pct"] is not None
        and r["IQR_이상치_pct"] > mean_iqr * IQR_OUTLIER_MULTIPLE
        for r in diagnosis.get("outlier_summary", [])
    )
    cn_drift = any(
        r == "충청남도"
        for r, _ in diagnosis.get("distribution_drift", {}).get("drift_regions", [])
    )
    cn_peak_missing = "충청남도" in diagnosis.get("missing_peak_regions", [])
    if cn_peak_missing:
        lines.append("- **[CRITICAL] 충남 피크 시간대 데이터 완전 결측** — 원천 데이터 재수집 필수")
    elif cn_map:
        lines.append("- **충남 다중 관측소 평균 매핑 우선** — 기상-발전량 상관이 전국 평균보다 낮음")
    elif cn_iqr:
        lines.append("- **충남 raw 데이터 정제 우선** — IQR 이상치 비율 과다")
    elif cn_drift:
        lines.append("- **validation 전략 변경 우선 (TimeSeriesSplit)** — Train/Test 분포 이동 감지")
    else:
        lines.append("- **데이터 레벨 문제 없음** — 6번(피처 추가) 또는 2번(지역별 정규화)으로 진행")
    lines.append(f"  - 세부: {auto_diag.get('충청남도', {}).get('recommended_action', '-')}")
    lines.append("")

    lines.append("### 전라남도")
    jn_growth = diagnosis.get("설비_급증_의심", {}).get("전라남도", {})
    jn_drift  = any(
        r == "전라남도"
        for r, _ in diagnosis.get("distribution_drift", {}).get("drift_regions", [])
    )
    if jn_growth:
        lines.append("- **전남 분리 학습 또는 설비용량 정규화 피처 도입 검토** — 설비 급증 감지")
    elif jn_drift:
        lines.append("- **전남 단독 시계열 분포 정렬 필요** — 분포 drift 감지")
    else:
        lines.append("- **지역별 정규화(2번) 우선 적용** — 이상치/drift 없으나 MAE 과대 추정")
    lines.append(f"  - 세부: {auto_diag.get('전라남도', {}).get('recommended_action', '-')}")
    lines.append("")

    lines += ["---", "", "## 6. 첨부 시각화 목록", ""]
    for pf in sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".png") and f.startswith("phase1")):
        lines.append(f"- `{OUT_DIR}/{pf}`")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("TASK P1-F", f"리포트 → {REPORT_MD}")
    log("TASK P1-F", "완료")


# ─────────────────────────────────────────────────────────
# 검증 체크리스트
# ─────────────────────────────────────────────────────────

def print_verification() -> None:
    print("\n[검증]")

    def check(path: str) -> None:
        if os.path.exists(path):
            if path.endswith(".md"):
                with open(path, "r", encoding="utf-8") as f:
                    n = len(f.readlines())
                print(f"  [OK] {path} 생성됨 ({n}줄)")
            else:
                kb = os.path.getsize(path) // 1024
                print(f"  [OK] {path} 생성됨 ({kb} KB)")
        else:
            print(f"  [X] {path} 없음!")

    check(RESULT_JSON)
    check(REPORT_MD)

    png_count = len([f for f in os.listdir(OUT_DIR) if f.endswith(".png") and f.startswith("phase1")])
    csv_count = len([f for f in os.listdir(OUT_DIR) if f.endswith(".csv") and f.startswith("phase1")])
    print(f"  [OK] PNG 파일 {png_count}개 생성됨")
    print(f"  [OK] CSV 파일 {csv_count}개 생성됨")
    print("")
    print("[다음 단계]")
    print(f"  → {REPORT_MD} 의 5번 섹션을 확인하여 우선 작업을 결정하라.")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    _ensure_utf8_stdout()
    log("MAIN", "Phase 1 데이터 진단 시작")

    sys.path.insert(0, os.path.abspath("."))
    from src.utils.font_setting import apply
    font = apply()
    log("MAIN", f"한글 폰트 적용: {font}")

    diagnosis: dict = {}

    train, test, train_feat, test_feat = task_a_load()
    task_b_yearly_trend(train, test, diagnosis)
    task_c_outlier_analysis(train, test, diagnosis)
    task_d_mapping_quality(train, diagnosis)
    task_e_distribution_drift(train, test, diagnosis)
    task_f_report(diagnosis)
    print_verification()

    log("MAIN", "Phase 1 데이터 진단 완료")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
