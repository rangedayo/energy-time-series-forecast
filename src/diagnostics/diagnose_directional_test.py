"""
방향성 테스트 FAIL 58% + 지역 불변성 FAIL 원인 진단

5가지 진단을 한 번에 실행:
  1. 지역별 방향성 비율 (17개 지역 각각, 지역당 100 샘플)
  2. 시간대별 방향성 비율 (아침/오전/한낮/오후/저녁)
  3. 샘플링 안정성 (random_state 10개)
  4. Feature importance (gain/weight/cover)
  5. Perturbation 크기 sensitivity (+0.1, +0.5, +1.0, +2.0, +3.0, +5.0)
     - 5-A: 일사량만 변경 (기존 테스트 방식)
     - 5-B: 일사량 + irrad_x_solar 동시 변경 (파생 피처 동기화)

기존 행동 테스트(behavioral_tests_national.py)와 완전히 동일한 환경 가정.
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 한글 폰트 (프로젝트 컨벤션)
try:
    from font_setting import set_korean_font
    set_korean_font()
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def ts():
    return datetime.now().strftime("%H:%M:%S")


TEST_FEAT  = "data/processed/national_test_features.csv"
MODEL_PATH = "models/national_xgboost_model.json"
OUTPUT_DIR = "outputs"

COL_POWER = "power_mwh"
COL_IRRAD = "일사량"
NON_FEAT  = [COL_POWER, "timestamp", "region"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[{ts()}] 방향성 테스트 진단 시작")

for path in [TEST_FEAT, MODEL_PATH]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: 파일 없음 → {path}")

import xgboost as xgb

test = pd.read_csv(TEST_FEAT, encoding="utf-8-sig", parse_dates=["timestamp"])
print(f"[{ts()}]   Test 로드: {len(test):,}행  {test['region'].nunique()}개 지역")

feature_cols = [c for c in test.columns if c not in NON_FEAT]
print(f"[{ts()}]   피처 {len(feature_cols)}개")

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
print(f"[{ts()}]   모델 로드 완료")

daytime_test = test[test["is_daytime"] == 1].reset_index(drop=True)
daytime_test["hour"] = daytime_test["timestamp"].dt.hour
print(f"[{ts()}]   daytime 행수: {len(daytime_test):,}")

results = {"meta": {"timestamp": datetime.now().isoformat(),
                    "test_rows": int(len(test)),
                    "daytime_rows": int(len(daytime_test)),
                    "n_regions": int(test["region"].nunique()),
                    "n_features": int(len(feature_cols))}}


# ─── 진단 1: 지역별 ─────────────────────────────────────────────────────────
print(f"\n[{ts()}] [진단 1] 지역별 방향성 비율 (지역당 100 샘플)")

region_results = {}
for region in sorted(daytime_test["region"].unique()):
    r_df = daytime_test[daytime_test["region"] == region]
    n_sample = min(100, len(r_df))
    sample = r_df.sample(n=n_sample, random_state=42)

    X_orig = sample[feature_cols].copy()
    X_p    = X_orig.copy()
    X_p[COL_IRRAD] = X_p[COL_IRRAD] + 0.5

    pred_orig = model.predict(X_orig)
    pred_p    = model.predict(X_p)
    ratio     = float((pred_p > pred_orig).mean())
    mean_diff = float((pred_p - pred_orig).mean())
    region_results[region] = {"n_samples": int(n_sample),
                              "increase_ratio": ratio,
                              "mean_diff": mean_diff}
    print(f"  {region:12s}  n={n_sample:3d}  ratio={ratio:.1%}  mean_diff={mean_diff:+.4f}")
results["diagnosis1_by_region"] = region_results

fig, ax = plt.subplots(figsize=(12, 6))
regions_sorted = sorted(region_results.keys(),
                        key=lambda r: region_results[r]["increase_ratio"])
ratios = [region_results[r]["increase_ratio"] * 100 for r in regions_sorted]
colors = ["#d62728" if r < 50 else "#ff7f0e" if r < 70 else "#2ca02c" if r >= 90 else "#1f77b4"
          for r in ratios]
ax.barh(regions_sorted, ratios, color=colors)
ax.axvline(90, color="green", linestyle="--", alpha=0.5, label="통과 기준 90%")
ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="동전 던지기 50%")
ax.set_xlabel("일사량 +0.5 perturbation 시 예측 증가 비율 (%)")
ax.set_title("진단 1: 지역별 방향성 비율 (지역당 100 샘플)")
ax.set_xlim(0, 105)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/diagnosis_by_region.png", dpi=120, bbox_inches="tight")
plt.close()


# ─── 진단 2: 시간대별 ───────────────────────────────────────────────────────
print(f"\n[{ts()}] [진단 2] 시간대별 방향성 비율")

time_buckets = {
    "아침 (6-9시)":   (6, 9),
    "오전 (10-11시)": (10, 11),
    "한낮 (12-14시)": (12, 14),
    "오후 (15-17시)": (15, 17),
    "저녁 (18-19시)": (18, 19),
}

hour_results = {}
for bucket_name, (h_start, h_end) in time_buckets.items():
    b_df = daytime_test[(daytime_test["hour"] >= h_start) &
                        (daytime_test["hour"] <= h_end)]
    if len(b_df) == 0:
        print(f"  {bucket_name}: (데이터 없음)")
        hour_results[bucket_name] = {"n_samples": 0, "increase_ratio": None}
        continue

    n_sample = min(500, len(b_df))
    sample = b_df.sample(n=n_sample, random_state=42)
    X_orig = sample[feature_cols].copy()
    X_p    = X_orig.copy()
    X_p[COL_IRRAD] = X_p[COL_IRRAD] + 0.5
    pred_orig = model.predict(X_orig)
    pred_p    = model.predict(X_p)
    ratio = float((pred_p > pred_orig).mean())
    mean_diff = float((pred_p - pred_orig).mean())
    mean_irrad = float(sample[COL_IRRAD].mean())
    hour_results[bucket_name] = {"n_samples": int(n_sample),
                                 "increase_ratio": ratio,
                                 "mean_diff": mean_diff,
                                 "mean_irrad_scaled": mean_irrad}
    print(f"  {bucket_name:18s}  n={n_sample:4d}  ratio={ratio:.1%}  "
          f"mean_diff={mean_diff:+.4f}  mean_일사량(scaled)={mean_irrad:+.3f}")
results["diagnosis2_by_hour"] = hour_results

fig, ax = plt.subplots(figsize=(10, 5))
valid_buckets = [b for b, v in hour_results.items() if v.get("increase_ratio") is not None]
valid_ratios  = [hour_results[b]["increase_ratio"] * 100 for b in valid_buckets]
colors = ["#d62728" if r < 50 else "#ff7f0e" if r < 70 else "#2ca02c" if r >= 90 else "#1f77b4"
          for r in valid_ratios]
ax.bar(valid_buckets, valid_ratios, color=colors)
ax.axhline(90, color="green", linestyle="--", alpha=0.5, label="통과 기준 90%")
ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="동전 던지기 50%")
ax.set_ylabel("방향성 비율 (%)")
ax.set_title("진단 2: 시간대별 방향성 비율")
ax.set_ylim(0, 105)
ax.legend()
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/diagnosis_by_hour.png", dpi=120, bbox_inches="tight")
plt.close()


# ─── 진단 3: 샘플링 안정성 ───────────────────────────────────────────────────
print(f"\n[{ts()}] [진단 3] 샘플링 안정성 (random_state 10개)")

stability = []
for seed in range(10):
    frames = []
    for region in daytime_test["region"].unique():
        r_df = daytime_test[daytime_test["region"] == region]
        frames.append(r_df.sample(n=min(10, len(r_df)), random_state=seed))
    sample = pd.concat(frames, ignore_index=True)
    X_orig = sample[feature_cols].copy()
    X_p    = X_orig.copy()
    X_p[COL_IRRAD] = X_p[COL_IRRAD] + 0.5
    pred_orig = model.predict(X_orig)
    pred_p    = model.predict(X_p)
    ratio = float((pred_p > pred_orig).mean())
    stability.append({"seed": seed, "increase_ratio": ratio})
    print(f"  seed={seed:2d}  ratio={ratio:.1%}")
ratios = [s["increase_ratio"] for s in stability]
results["diagnosis3_sampling_stability"] = {
    "seeds":  stability,
    "mean":   float(np.mean(ratios)),
    "std":    float(np.std(ratios)),
    "min":    float(np.min(ratios)),
    "max":    float(np.max(ratios)),
}
print(f"  → 평균 {np.mean(ratios):.1%}, std {np.std(ratios):.3f}, "
      f"min {np.min(ratios):.1%}, max {np.max(ratios):.1%}")


# ─── 진단 4: Feature importance ──────────────────────────────────────────────
print(f"\n[{ts()}] [진단 4] Feature importance")

booster = model.get_booster()
gain   = booster.get_score(importance_type="gain")
weight = booster.get_score(importance_type="weight")
cover  = booster.get_score(importance_type="cover")

all_imp = []
for feat in feature_cols:
    all_imp.append({"feature": feat,
                    "gain":   float(gain.get(feat, 0.0)),
                    "weight": float(weight.get(feat, 0.0)),
                    "cover":  float(cover.get(feat, 0.0))})
all_imp.sort(key=lambda x: x["gain"], reverse=True)

print(f"\n  {'rank':4s}  {'feature':25s}  {'gain':>14s}  {'weight':>8s}")
for i, imp in enumerate(all_imp, 1):
    marker = ""
    if imp["feature"] == COL_IRRAD:           marker = "  ★ 일사량"
    elif imp["feature"] == "irrad_x_solar":   marker = "  ★ 일사량 파생"
    elif imp["feature"] == "region_code":     marker = "  ★ 지역"
    print(f"  {i:4d}  {imp['feature']:25s}  {imp['gain']:>14.2f}  "
          f"{imp['weight']:>8.0f}{marker}")

def rank_of(feat):
    return next((i for i, x in enumerate(all_imp, 1) if x["feature"] == feat), None)

results["diagnosis4_feature_importance"] = {
    "by_gain":            all_imp,
    "irrad_rank":         rank_of(COL_IRRAD),
    "irrad_x_solar_rank": rank_of("irrad_x_solar"),
    "region_code_rank":   rank_of("region_code"),
    "lag_1h_gain":        gain.get("lag_1h", 0.0),
    "irrad_gain":         gain.get(COL_IRRAD, 0.0),
    "lag_to_irrad_ratio": (gain.get("lag_1h", 0.0) / max(gain.get(COL_IRRAD, 1e-9), 1e-9)),
}


# ─── 진단 5: Perturbation 크기 ───────────────────────────────────────────────
print(f"\n[{ts()}] [진단 5] Perturbation 크기 sensitivity (1000 샘플)")

big_sample = daytime_test.sample(n=min(1000, len(daytime_test)), random_state=42)
X_orig_big = big_sample[feature_cols].copy()
pred_orig_big = model.predict(X_orig_big)

perturb_sizes = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
size_results = {"irrad_only": {}, "irrad_and_interaction": {}}

print(f"\n  [5-A] 일사량만 변경:")
for delta in perturb_sizes:
    X_p = X_orig_big.copy()
    X_p[COL_IRRAD] = X_p[COL_IRRAD] + delta
    pred_p = model.predict(X_p)
    ratio = float((pred_p > pred_orig_big).mean())
    mean_diff = float((pred_p - pred_orig_big).mean())
    size_results["irrad_only"][f"+{delta}"] = {"increase_ratio": ratio, "mean_diff": mean_diff}
    print(f"    +{delta:4.1f}  ratio={ratio:.1%}  mean_diff={mean_diff:+.4f}")

if "irrad_x_solar" in feature_cols and "solar_altitude_proxy" in feature_cols:
    print(f"\n  [5-B] 일사량 + irrad_x_solar 동시 변경 (파생 피처 동기화):")
    for delta in perturb_sizes:
        X_p = X_orig_big.copy()
        X_p[COL_IRRAD] = X_p[COL_IRRAD] + delta
        X_p["irrad_x_solar"] = X_p["irrad_x_solar"] + delta * X_p["solar_altitude_proxy"]
        pred_p = model.predict(X_p)
        ratio = float((pred_p > pred_orig_big).mean())
        mean_diff = float((pred_p - pred_orig_big).mean())
        size_results["irrad_and_interaction"][f"+{delta}"] = {
            "increase_ratio": ratio, "mean_diff": mean_diff}
        print(f"    +{delta:4.1f}  ratio={ratio:.1%}  mean_diff={mean_diff:+.4f}")
else:
    print(f"\n  [5-B] SKIP: irrad_x_solar 또는 solar_altitude_proxy 없음")

results["diagnosis5_perturbation_size"] = size_results

fig, ax = plt.subplots(figsize=(10, 5))
sizes = perturb_sizes
ratios_a = [size_results["irrad_only"][f"+{d}"]["increase_ratio"] * 100 for d in sizes]
ax.plot(sizes, ratios_a, marker="o", linewidth=2, label="5-A: 일사량만 변경")
if size_results["irrad_and_interaction"]:
    ratios_b = [size_results["irrad_and_interaction"][f"+{d}"]["increase_ratio"] * 100
                for d in sizes]
    ax.plot(sizes, ratios_b, marker="s", linewidth=2, label="5-B: 일사량 + irrad_x_solar 동기화")
ax.axhline(90, color="green", linestyle="--", alpha=0.5, label="통과 기준 90%")
ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="동전 던지기 50%")
ax.set_xlabel("일사량 perturbation 크기 (scaled value)")
ax.set_ylabel("방향성 비율 (%)")
ax.set_title("진단 5: Perturbation 크기 sensitivity")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/diagnosis_perturbation_size.png", dpi=120, bbox_inches="tight")
plt.close()


# ─── 저장 ────────────────────────────────────────────────────────────────────
JSON_PATH = f"{OUTPUT_DIR}/diagnosis_directional_test.json"
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


# ─── Markdown 요약 ───────────────────────────────────────────────────────────
def fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "N/A"

r1 = results["diagnosis1_by_region"]
worst_region = min(r1.items(), key=lambda x: x[1]["increase_ratio"])
best_region  = max(r1.items(), key=lambda x: x[1]["increase_ratio"])
n_pass = sum(1 for v in r1.values() if v["increase_ratio"] >= 0.90)
n_fail = sum(1 for v in r1.values() if v["increase_ratio"] < 0.50)
overall_r1 = float(np.mean([v["increase_ratio"] for v in r1.values()]))

r2 = results["diagnosis2_by_hour"]
r2_valid = {k: v for k, v in r2.items() if v.get("increase_ratio") is not None}
r3 = results["diagnosis3_sampling_stability"]
r4 = results["diagnosis4_feature_importance"]
r5 = results["diagnosis5_perturbation_size"]

md_lines = [
    "# 방향성 테스트 FAIL 58% 진단 보고서",
    "",
    f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
    "",
    "## 요약",
    "",
    "- 기존 행동 테스트 결과: 방향성 비율 **57.6%** (170 샘플)",
    f"- 재측정 (지역당 100 샘플, 17지역 = 1,700 샘플): **{fmt_pct(overall_r1)}**",
    f"- 가장 나쁜 지역: **{worst_region[0]}** ({fmt_pct(worst_region[1]['increase_ratio'])})",
    f"- 가장 좋은 지역: **{best_region[0]}** ({fmt_pct(best_region[1]['increase_ratio'])})",
    f"- 통과 기준(90%) 충족 지역: **{n_pass}/{len(r1)}개**",
    f"- 동전던지기 미만(<50%) 지역: **{n_fail}/{len(r1)}개**",
    f"- 일사량 feature importance 순위: **{r4['irrad_rank']}위 / {len(r4['by_gain'])}**",
    f"- lag_1h gain / 일사량 gain 비율: **{r4['lag_to_irrad_ratio']:.1f}배**",
    "",
    "## 진단 1: 지역별 방향성 비율",
    "",
    "| 지역 | 샘플 | 방향성 비율 | 평균 예측 변화 |",
    "|---|---:|---:|---:|",
]
for region in sorted(r1.keys(), key=lambda r: r1[r]["increase_ratio"]):
    v = r1[region]
    md_lines.append(f"| {region} | {v['n_samples']} | {fmt_pct(v['increase_ratio'])} | {v['mean_diff']:+.4f} |")

md_lines += ["", "## 진단 2: 시간대별 방향성 비율", "",
             "| 시간대 | 샘플 | 방향성 비율 | 평균 예측 변화 | 평균 일사량(scaled) |",
             "|---|---:|---:|---:|---:|"]
for bucket, v in r2_valid.items():
    md_lines.append(f"| {bucket} | {v['n_samples']} | {fmt_pct(v['increase_ratio'])} | "
                    f"{v['mean_diff']:+.4f} | {v.get('mean_irrad_scaled', 0):+.3f} |")

md_lines += ["", "## 진단 3: 샘플링 안정성 (random_state 10개)", "",
             f"- 평균: **{fmt_pct(r3['mean'])}**",
             f"- 표준편차: {r3['std']:.4f}",
             f"- 최소: {fmt_pct(r3['min'])} / 최대: {fmt_pct(r3['max'])}",
             f"- 폭: {(r3['max']-r3['min'])*100:.1f}%p",
             "", "| seed | 방향성 비율 |", "|---:|---:|"]
for s in r3["seeds"]:
    md_lines.append(f"| {s['seed']} | {fmt_pct(s['increase_ratio'])} |")

md_lines += ["", "## 진단 4: Feature importance (gain 기준)", "",
             f"- **일사량 순위: {r4['irrad_rank']}위 / {len(r4['by_gain'])}**",
             f"- **irrad_x_solar 순위: {r4['irrad_x_solar_rank']}위**",
             f"- **region_code 순위: {r4['region_code_rank']}위**",
             f"- lag_1h gain / 일사량 gain 비율: **{r4['lag_to_irrad_ratio']:.1f}배**",
             "", "| 순위 | 피처 | gain | weight |", "|---:|---|---:|---:|"]
for i, imp in enumerate(r4["by_gain"], 1):
    mark = " ★" if imp["feature"] in [COL_IRRAD, "irrad_x_solar", "region_code"] else ""
    md_lines.append(f"| {i} | {imp['feature']}{mark} | {imp['gain']:.0f} | {imp['weight']:.0f} |")

md_lines += ["", "## 진단 5: Perturbation 크기 sensitivity", "",
             "### 5-A: 일사량만 변경", "",
             "| Perturbation | 방향성 비율 | 평균 예측 변화 |",
             "|---:|---:|---:|"]
for delta in perturb_sizes:
    v = r5["irrad_only"].get(f"+{delta}", {})
    md_lines.append(f"| +{delta} | {fmt_pct(v.get('increase_ratio'))} | {v.get('mean_diff', 0):+.4f} |")

if r5["irrad_and_interaction"]:
    md_lines += ["", "### 5-B: 일사량 + irrad_x_solar 동시 변경", "",
                 "| Perturbation | 방향성 비율 | 평균 예측 변화 |",
                 "|---:|---:|---:|"]
    for delta in perturb_sizes:
        v = r5["irrad_and_interaction"].get(f"+{delta}", {})
        md_lines.append(f"| +{delta} | {fmt_pct(v.get('increase_ratio'))} | {v.get('mean_diff', 0):+.4f} |")

MD_PATH = f"{OUTPUT_DIR}/diagnosis_directional_test.md"
with open(MD_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n[{ts()}] 진단 완료")
print(f"  결과:   {JSON_PATH}")
print(f"  보고서: {MD_PATH}")
print(f"  그래프: diagnosis_by_region.png, diagnosis_by_hour.png, diagnosis_perturbation_size.png")
