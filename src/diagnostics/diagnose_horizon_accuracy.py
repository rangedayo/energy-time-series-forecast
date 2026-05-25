"""
48h 재귀 multi-step 예측 정확도 진단.

목적: round 2-2의 (b-1) 48h 슬라이스 방식이 정당한지 확인.
즉, 우리 XGBoost 모델이 t+24~t+47 구간에서도 사용 가능한 정확도를 내는가?

방법:
1. 학습 CSV에서 (region, start_time) 페어 N개 샘플링
2. 각 페어에 대해 /predict_horizon (horizon=48) 호출
3. 응답의 48개 예측을 실측과 비교
4. 단기(step 1~23) vs 장기(step 24~48) 오차 통계 비교
5. JSON + Markdown 보고서 출력
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import urllib.error
import urllib.request

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font  # noqa: E402

_apply_font()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ─── 설정 ──────────────────────────────────────────────────────────────────────
N_SAMPLES = 200
SHORT_STEPS = list(range(1, 24))      # step 1~23
LONG_STEPS = list(range(24, 49))      # step 24~48
API_URL = "http://localhost:8000/predict_horizon"
API_KEY = "dev-key-change-me"
TRAIN_CSV = "data/processed/national_train_features.csv"
OUT_DIR = Path("outputs/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
REQUEST_TIMEOUT_SEC = 30
REQUEST_SLEEP_SEC = 0.0    # dev 서버라 sleep 불필요. 필요시 증가
MAPE_FLOOR_MWH = 1.0       # 실측 |power| < 1.0 MWh 인 시점은 MAPE 계산 제외

WEATHER_COLS = ["기온", "강수량", "습도", "일조", "일사량", "전운량"]
POWER_COL = "power_mwh"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ─── (1) 데이터 로딩 및 샘플링 ─────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    log(f"CSV 로딩: {TRAIN_CSV}")
    if not Path(TRAIN_CSV).exists():
        sys.exit(f"ERROR: 파일 없음 → {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig", parse_dates=["timestamp"])
    required = ["timestamp", "region", POWER_COL] + WEATHER_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: CSV에 컬럼 누락: {missing}")
    df = df.sort_values(["region", "timestamp"]).reset_index(drop=True)
    log(f"  로드 완료: {len(df):,}행, {df['region'].nunique()}개 region")
    return df


def sample_start_points(
    df: pd.DataFrame, n_total: int, seed: int
) -> list[tuple[str, datetime]]:
    """region별로 균등하게 (region, start_time) 페어를 샘플링.

    조건: start_time-24h ~ start_time+47h 의 72시간 윈도우가
    해당 region 데이터에 1시간 간격으로 모두 존재해야 함.
    """
    rng = random.Random(seed)
    regions = sorted(df["region"].unique())
    per_region = max(1, n_total // len(regions))

    log(f"  region별 {per_region}개씩 후보 샘플링 (총 목표 ~{per_region * len(regions)})")

    samples: list[tuple[str, datetime]] = []
    for region in regions:
        sub = df[df["region"] == region].sort_values("timestamp").reset_index(drop=True)
        timestamps = sub["timestamp"].tolist()
        ts_set = set(timestamps)
        candidates: list[datetime] = []
        for t in timestamps:
            history_ok = all(
                (t - timedelta(hours=24 - i)) in ts_set for i in range(24)
            )
            if not history_ok:
                continue
            forecast_ok = all((t + timedelta(hours=i)) in ts_set for i in range(48))
            if not forecast_ok:
                continue
            candidates.append(t)
        if not candidates:
            log(f"  [WARN] {region}: 윈도우 후보 0개 → 스킵")
            continue
        k = min(per_region, len(candidates))
        chosen = rng.sample(candidates, k)
        for start_time in chosen:
            samples.append((region, start_time))

    rng.shuffle(samples)
    if len(samples) > n_total:
        samples = samples[:n_total]
    log(f"  최종 샘플 수: {len(samples)}")
    return samples


# ─── (2) API 호출 ──────────────────────────────────────────────────────────────
def build_payload(df: pd.DataFrame, region: str, start_time: datetime) -> dict:
    sub = df[df["region"] == region].set_index("timestamp")

    history = []
    for i in range(24):
        t = start_time - timedelta(hours=24 - i)
        history.append({
            "timestamp": t.isoformat(),
            "power_mwh": float(sub.at[t, POWER_COL]),
        })

    forecast = []
    for i in range(48):
        t = start_time + timedelta(hours=i)
        row = sub.loc[t]
        forecast.append({
            "timestamp": t.isoformat(),
            "irradiance": float(row["일사량"]),
            "기온": float(row["기온"]),
            "강수량": float(row["강수량"]),
            "습도": float(row["습도"]),
            "일조": float(row["일조"]),
            "전운량": float(row["전운량"]),
        })

    return {
        "region": region,
        "start_time": start_time.isoformat(),
        "horizon": 48,
        "history": history,
        "forecast": forecast,
    }


def call_api(payload: dict) -> list[float] | None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            if resp.status != 200:
                return None
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None
    return [float(p["predicted_power_mwh"]) for p in body["predictions"]]


def get_actuals(df: pd.DataFrame, region: str, start_time: datetime) -> list[float]:
    sub = df[df["region"] == region].set_index("timestamp")
    return [float(sub.at[start_time + timedelta(hours=i), POWER_COL]) for i in range(48)]


# ─── (3) 오차 집계 ────────────────────────────────────────────────────────────
def aggregate_errors(
    step_indices: list[int],
    preds_matrix: np.ndarray,
    actuals_matrix: np.ndarray,
) -> dict:
    idx = [s - 1 for s in step_indices]
    p = preds_matrix[:, idx].ravel()
    a = actuals_matrix[:, idx].ravel()
    err = p - a
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))

    mape_mask = np.abs(a) >= MAPE_FLOOR_MWH
    if mape_mask.any():
        mape = float(np.mean(np.abs(err[mape_mask] / a[mape_mask])) * 100.0)
        mape_n = int(mape_mask.sum())
    else:
        mape = float("nan")
        mape_n = 0

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "n_points": int(len(p)),
        "n_points_mape": mape_n,
    }


# ─── (4) 시각화 ───────────────────────────────────────────────────────────────
def plot_step_wise_rmse(step_wise_rmse: list[float], out_path: Path) -> None:
    steps = list(range(1, 49))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, step_wise_rmse, marker="o", markersize=4, linewidth=1.6,
            color="#1f77b4")
    ax.axvline(23.5, color="red", linestyle="--", alpha=0.6,
               label="단기/장기 경계 (step 23↔24)")
    ax.set_xlabel("step (hour)")
    ax.set_ylabel("RMSE (MWh)")
    ax.set_title("step별 예측 RMSE (재귀 multi-step, horizon=48)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_error_distribution(
    short_abs_err: np.ndarray, long_abs_err: np.ndarray, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [short_abs_err, long_abs_err]
    bp = ax.boxplot(
        data,
        labels=[
            f"단기 (step 1~23)\nn={len(short_abs_err):,}",
            f"장기 (step 24~48)\nn={len(long_abs_err):,}",
        ],
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["#2ca02c", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("절대 오차 |pred - actual| (MWh)")
    ax.set_title("단기 vs 장기 절대오차 분포 (outlier 미표시)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_sample_trajectories(
    samples_meta: list[tuple[str, datetime]],
    preds_matrix: np.ndarray,
    actuals_matrix: np.ndarray,
    out_path: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)
    n_total = len(samples_meta)
    if n_total == 0:
        return
    pick = rng.sample(range(n_total), min(6, n_total))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    steps = list(range(1, 49))
    for ax, idx in zip(axes.ravel(), pick):
        region, start_time = samples_meta[idx]
        ax.plot(steps, actuals_matrix[idx], color="black", linewidth=1.4,
                label="실측")
        ax.plot(steps, preds_matrix[idx], color="#1f77b4", linewidth=1.4,
                linestyle="--", label="예측")
        ax.axvline(23.5, color="red", linestyle=":", alpha=0.5)
        ax.set_title(f"{region}  start={start_time.strftime('%Y-%m-%d %H:%M')}",
                     fontsize=10)
        ax.set_xlabel("step (h)")
        ax.set_ylabel("MWh")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("샘플 trajectory: 실측 vs 재귀 48h 예측", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── (5) 평결 ─────────────────────────────────────────────────────────────────
def decide_verdict(ratio_rmse: float) -> tuple[str, str]:
    if ratio_rmse < 2.0:
        return "PASS", (
            f"장기/단기 RMSE 비율 {ratio_rmse:.2f} → (b-1) 48h 슬라이스 방식이 "
            f"정당함. round 2-2 진행 가능."
        )
    if ratio_rmse < 3.0:
        return "MARGINAL", (
            f"장기/단기 RMSE 비율 {ratio_rmse:.2f} → 주의가 필요하나 (b-1) 가능. "
            f"round 2-2 결과에서 끝쪽 시점의 액션 합리성을 추가 점검 권고."
        )
    return "FAIL", (
        f"장기/단기 RMSE 비율 {ratio_rmse:.2f} → (b-2) 24h 슬라이스로 후퇴 권고. "
        f"round 2-2 설계 재논의 필요."
    )


# ─── (6) MAIN ─────────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.time()
    log("=" * 70)
    log("48h 재귀 multi-step 예측 정확도 진단 시작")
    log(f"  API={API_URL}, N_SAMPLES={N_SAMPLES}, seed={RANDOM_SEED}")
    log("=" * 70)

    df = load_data()
    samples = sample_start_points(df, N_SAMPLES, RANDOM_SEED)

    n_target = len(samples)
    preds_rows: list[list[float]] = []
    actuals_rows: list[list[float]] = []
    meta_succ: list[tuple[str, datetime]] = []
    n_fail = 0

    log(f"API 호출 시작 ({n_target}건)")
    for i, (region, start_time) in enumerate(samples, 1):
        payload = build_payload(df, region, start_time)
        preds = call_api(payload)
        if preds is None or len(preds) != 48:
            n_fail += 1
            if n_fail <= 5:
                log(f"  [FAIL {n_fail}] {region} @ {start_time}")
            if REQUEST_SLEEP_SEC > 0:
                time.sleep(REQUEST_SLEEP_SEC)
            continue
        actuals = get_actuals(df, region, start_time)
        preds_rows.append(preds)
        actuals_rows.append(actuals)
        meta_succ.append((region, start_time))
        if i % 25 == 0 or i == n_target:
            log(f"  진행: {i}/{n_target}  (성공 {len(preds_rows)}, 실패 {n_fail})")
        if REQUEST_SLEEP_SEC > 0:
            time.sleep(REQUEST_SLEEP_SEC)

    n_succ = len(preds_rows)
    if n_succ == 0:
        sys.exit("ERROR: 성공한 샘플이 0개. API 서버 상태 확인.")

    preds_matrix = np.asarray(preds_rows, dtype=np.float64)
    actuals_matrix = np.asarray(actuals_rows, dtype=np.float64)

    log(f"API 호출 완료: 성공 {n_succ}, 실패 {n_fail}")

    short_stats = aggregate_errors(SHORT_STEPS, preds_matrix, actuals_matrix)
    long_stats = aggregate_errors(LONG_STEPS, preds_matrix, actuals_matrix)

    step_wise_rmse: list[float] = []
    step_wise_mae: list[float] = []
    for s in range(1, 49):
        idx = s - 1
        err = preds_matrix[:, idx] - actuals_matrix[:, idx]
        step_wise_rmse.append(float(np.sqrt(np.mean(err ** 2))))
        step_wise_mae.append(float(np.mean(np.abs(err))))

    ratio_rmse = (
        long_stats["rmse"] / short_stats["rmse"] if short_stats["rmse"] > 0 else float("inf")
    )
    ratio_mae = (
        long_stats["mae"] / short_stats["mae"] if short_stats["mae"] > 0 else float("inf")
    )
    if (
        not np.isnan(short_stats["mape"]) and short_stats["mape"] > 0
        and not np.isnan(long_stats["mape"])
    ):
        ratio_mape = long_stats["mape"] / short_stats["mape"]
    else:
        ratio_mape = float("nan")

    verdict, verdict_msg = decide_verdict(ratio_rmse)

    log("시각화 생성")
    short_idx = [s - 1 for s in SHORT_STEPS]
    long_idx = [s - 1 for s in LONG_STEPS]
    short_abs = np.abs(
        preds_matrix[:, short_idx] - actuals_matrix[:, short_idx]
    ).ravel()
    long_abs = np.abs(
        preds_matrix[:, long_idx] - actuals_matrix[:, long_idx]
    ).ravel()

    fig_step = OUT_DIR / "step_wise_rmse.png"
    fig_dist = OUT_DIR / "error_distribution_compare.png"
    fig_traj = OUT_DIR / "sample_trajectories.png"
    plot_step_wise_rmse(step_wise_rmse, fig_step)
    plot_error_distribution(short_abs, long_abs, fig_dist)
    plot_sample_trajectories(
        meta_succ, preds_matrix, actuals_matrix, fig_traj, seed=RANDOM_SEED
    )

    json_path = OUT_DIR / "horizon_accuracy.json"
    payload_json = {
        "config": {
            "n_samples": N_SAMPLES,
            "short_steps": [SHORT_STEPS[0], SHORT_STEPS[-1]],
            "long_steps": [LONG_STEPS[0], LONG_STEPS[-1]],
            "seed": RANDOM_SEED,
            "mape_floor_mwh": MAPE_FLOOR_MWH,
        },
        "n_samples_succeeded": n_succ,
        "n_samples_failed": n_fail,
        "short_term": short_stats,
        "long_term": long_stats,
        "ratio_long_to_short": {
            "rmse": ratio_rmse,
            "mae": ratio_mae,
            "mape": ratio_mape,
        },
        "step_wise_rmse": step_wise_rmse,
        "step_wise_mae": step_wise_mae,
        "verdict": verdict,
        "verdict_criteria": "ratio_rmse < 2.0 = PASS, 2.0~3.0 = MARGINAL, >=3.0 = FAIL",
        "verdict_message": verdict_msg,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload_json, f, ensure_ascii=False, indent=2)

    md_path = OUT_DIR / "horizon_accuracy_report.md"
    elapsed = time.time() - t_start

    def _fmt(x: float, nd: int = 4) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        return f"{x:.{nd}f}"

    md = [
        "# 48h 재귀 multi-step 예측 정확도 진단 보고서",
        "",
        f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ",
        f"_실행 시간: {elapsed:.1f}s_",
        "",
        "## 목적",
        "",
        "round 2-2 운영자 대시보드용 MPC 오케스트레이터가 매 시점 24h lookahead 윈도우를",
        "사용한다. 시뮬 끝 시점에서도 lookahead가 가능하려면 미래 데이터 48h가 필요한데,",
        "재귀 multi-step의 본질상 t+24~t+47 구간 예측 품질이 t+1~t+23보다 떨어질 수 있다.",
        "본 진단은 단기/장기 오차 비율을 측정하여 (b-1) 48h 슬라이스 방식의 정당성을 판정한다.",
        "",
        "## 방법",
        "",
        f"- 데이터: `{TRAIN_CSV}` (학습 CSV)",
        f"- 샘플링: region별 균등, 총 {N_SAMPLES}개 (seed={RANDOM_SEED})",
        f"- 호출: `{API_URL}` (horizon=48)",
        "- forecast 기상값: 학습 CSV의 ex-post 실측 (모델 자체 오차만 분리 측정)",
        f"- MAPE 계산: |actual| ≥ {MAPE_FLOOR_MWH} MWh 시점만 사용",
        "",
        "## 결과 요약",
        "",
        f"- 성공 샘플: **{n_succ} / {N_SAMPLES}** (실패 {n_fail}, "
        f"성공률 {n_succ / N_SAMPLES * 100:.1f}%)",
        f"- 최종 평결: **{verdict}**",
        "",
        f"> {verdict_msg}",
        "",
        "### 단기 vs 장기 오차",
        "",
        "| 구간 | RMSE (MWh) | MAE (MWh) | MAPE (%) | n_points | n_points_MAPE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 단기 (step 1~23) | {_fmt(short_stats['rmse'])} | "
        f"{_fmt(short_stats['mae'])} | {_fmt(short_stats['mape'], 2)} | "
        f"{short_stats['n_points']:,} | {short_stats['n_points_mape']:,} |",
        f"| 장기 (step 24~48) | {_fmt(long_stats['rmse'])} | "
        f"{_fmt(long_stats['mae'])} | {_fmt(long_stats['mape'], 2)} | "
        f"{long_stats['n_points']:,} | {long_stats['n_points_mape']:,} |",
        f"| **장기/단기 비율** | **{_fmt(ratio_rmse, 3)}** | "
        f"**{_fmt(ratio_mae, 3)}** | **{_fmt(ratio_mape, 3)}** | — | — |",
        "",
        "## 시각화",
        "",
        "![step별 RMSE](step_wise_rmse.png)",
        "",
        "![단기/장기 오차 분포](error_distribution_compare.png)",
        "",
        "![샘플 trajectory](sample_trajectories.png)",
        "",
        "## 결론",
        "",
        f"**[{verdict}]** {verdict_msg}",
        "",
        "## 생성물",
        "",
        f"- JSON: `{json_path}`",
        f"- step별 RMSE: `{fig_step}`",
        f"- 오차 분포: `{fig_dist}`",
        f"- 샘플 trajectory: `{fig_traj}`",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    log("=" * 70)
    log("최종 결과")
    log("=" * 70)
    log(f"  실행 시간: {elapsed:.1f}s")
    log(f"  성공 / 실패 / 목표: {n_succ} / {n_fail} / {N_SAMPLES}")
    log("")
    log(f"  {'구간':16s}  {'RMSE':>10s}  {'MAE':>10s}  {'MAPE(%)':>10s}")
    log(f"  {'단기 step 1~23':16s}  {short_stats['rmse']:>10.4f}  "
        f"{short_stats['mae']:>10.4f}  {short_stats['mape']:>10.2f}")
    log(f"  {'장기 step 24~48':16s}  {long_stats['rmse']:>10.4f}  "
        f"{long_stats['mae']:>10.4f}  {long_stats['mape']:>10.2f}")
    log(f"  {'비율 (장/단)':16s}  {ratio_rmse:>10.3f}  {ratio_mae:>10.3f}  "
        f"{ratio_mape:>10.3f}")
    log("")
    log(f"  평결: {verdict}")
    log(f"  → {verdict_msg}")
    log("")
    log(f"  JSON:   {json_path}")
    log(f"  보고서: {md_path}")
    log(f"  그래프: {fig_step.name}, {fig_dist.name}, {fig_traj.name}")
    log("=" * 70)


if __name__ == "__main__":
    main()
