import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
TEST_FEAT        = "data/processed/national_test_features.csv"
MODEL_PATH       = "models/national_xgboost_model.json"
BASELINE_RESULTS = "outputs/national_baseline_results.json"
OUTPUT_PATH      = "outputs/national_behavioral_test_results.json"

COL_POWER = "power_mwh"
COL_IRRAD = "일사량"
NON_FEAT  = [COL_POWER, "timestamp", "region"]

print(f"[{ts()}] [TASK E] 전국 행동 테스트 시작")

for path in [TEST_FEAT, MODEL_PATH, BASELINE_RESULTS]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: 파일 없음 → {path}")

import xgboost as xgb

test = pd.read_csv(TEST_FEAT, encoding="utf-8-sig", parse_dates=["timestamp"])
print(f"[{ts()}]   Test 로드: {len(test):,}행  {test['region'].nunique()}개 지역")

feature_cols = [c for c in test.columns if c not in NON_FEAT]
X_test = test[feature_cols]
y_test = test[COL_POWER]

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
print(f"[{ts()}]   모델 로드 완료")

with open(BASELINE_RESULTS, "r", encoding="utf-8") as f:
    baseline = json.load(f)

results = {}

# ── 테스트 1: NaN/Inf 검증 ─────────────────────────────────────────────────────
print(f"\n[{ts()}] [테스트 1] NaN/Inf 검증")
try:
    pred = model.predict(X_test)
    assert np.isfinite(pred).all(), \
        f"NaN/Inf 감지: NaN={np.isnan(pred).sum()}건, Inf={np.isinf(pred).sum()}건"
    print(f"  [테스트 1] 통과 ✓")
    results["test1_nan_inf"] = {"status": "PASS"}
except AssertionError as e:
    print(f"  [테스트 1] FAIL: {e}")
    results["test1_nan_inf"] = {"status": "FAIL", "detail": str(e)}

# ── 테스트 2: 방향성 테스트 (각 지역 10개 × 17 = 170개) ──────────────────────
print(f"\n[{ts()}] [테스트 2] 방향성 테스트 (각 지역 10개 샘플)")
try:
    daytime_test = test[test["is_daytime"] == 1]
    sample_frames = []
    for region in daytime_test["region"].unique():
        r_df = daytime_test[daytime_test["region"] == region]
        sample_frames.append(r_df.sample(n=min(10, len(r_df)), random_state=42))
    sample = pd.concat(sample_frames, ignore_index=True)

    X_orig    = sample[feature_cols].copy()
    X_perturb = X_orig.copy()
    X_perturb[COL_IRRAD] = X_perturb[COL_IRRAD] + 0.5

    pred_orig    = model.predict(X_orig)
    pred_perturb = model.predict(X_perturb)

    increase_ratio = float((pred_perturb > pred_orig).mean())
    passed = increase_ratio >= 0.90

    if passed:
        print(f"  [테스트 2] 통과 ✓  (증가 비율: {increase_ratio:.1%})")
        results["test2_directional"] = {"status": "PASS", "increase_ratio": increase_ratio}
    else:
        print(f"  [테스트 2] FAIL: 증가 비율 {increase_ratio:.1%} < 90%")
        results["test2_directional"] = {"status": "FAIL", "increase_ratio": increase_ratio}
except Exception as e:
    print(f"  [테스트 2] FAIL: {e}")
    results["test2_directional"] = {"status": "FAIL", "detail": str(e)}

# ── 테스트 3: 불변성 테스트 ────────────────────────────────────────────────────
print(f"\n[{ts()}] [테스트 3] 불변성 테스트 (5회 반복)")
try:
    sample_row = X_test.iloc[[0]]
    preds_repeat = [float(model.predict(sample_row)[0]) for _ in range(5)]
    std_val = float(np.std(preds_repeat))
    if std_val == 0:
        print(f"  [테스트 3] 통과 ✓  (std: {std_val})")
        results["test3_invariance"] = {"status": "PASS", "std": std_val}
    else:
        print(f"  [테스트 3] FAIL: std {std_val} ≠ 0")
        results["test3_invariance"] = {"status": "FAIL", "std": std_val}
except Exception as e:
    print(f"  [테스트 3] FAIL: {e}")
    results["test3_invariance"] = {"status": "FAIL", "detail": str(e)}

# ── 테스트 4: 정확성 테스트 ────────────────────────────────────────────────────
print(f"\n[{ts()}] [테스트 4] 정확성 테스트")
try:
    peak_mask = test["timestamp"].dt.hour.isin(range(10, 15)).values
    pred_peak = np.clip(model.predict(X_test[peak_mask]), 0, None)
    mae_peak  = float(np.mean(np.abs(pred_peak - y_test.values[peak_mask])))
    baseline_mae_peak = float(baseline["lag1"]["MAE_peak"])
    passed = mae_peak < baseline_mae_peak
    if passed:
        print(f"  [테스트 4] 통과 ✓  (XGB: {mae_peak:.4f} < baseline: {baseline_mae_peak:.4f})")
        results["test4_accuracy"] = {"status": "PASS", "xgb_mae_peak": mae_peak, "baseline_mae_peak": baseline_mae_peak}
    else:
        print(f"  [테스트 4] FAIL: {mae_peak:.4f} ≥ {baseline_mae_peak:.4f}")
        results["test4_accuracy"] = {"status": "FAIL", "xgb_mae_peak": mae_peak, "baseline_mae_peak": baseline_mae_peak}
except Exception as e:
    print(f"  [테스트 4] FAIL: {e}")
    results["test4_accuracy"] = {"status": "FAIL", "detail": str(e)}

# ── 테스트 5: 지역 불변성 테스트 (신규) ────────────────────────────────────────
print(f"\n[{ts()}] [테스트 5] 지역 불변성 테스트 (region_code만 변경 시 예측값 달라지는지)")
try:
    base_row = X_test.iloc[[0]].copy()
    n_regions = test["region_code"].nunique()
    region_preds = {}
    for code in range(n_regions):
        row = base_row.copy()
        row["region_code"] = code
        region_preds[code] = float(model.predict(row)[0])

    unique_preds = len(set(round(v, 6) for v in region_preds.values()))
    passed = unique_preds == n_regions
    if passed:
        print(f"  [테스트 5] 통과 ✓  ({n_regions}개 지역 모두 다른 예측값)")
        results["test5_region_invariance"] = {"status": "PASS", "n_unique_preds": unique_preds}
    else:
        print(f"  [테스트 5] FAIL: {unique_preds}/{n_regions}개만 다른 예측값")
        results["test5_region_invariance"] = {"status": "FAIL", "n_unique_preds": unique_preds}
except Exception as e:
    print(f"  [테스트 5] FAIL: {e}")
    results["test5_region_invariance"] = {"status": "FAIL", "detail": str(e)}

# ── 저장 ─────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[{ts()}]   결과 저장: {OUTPUT_PATH}")

pass_count = sum(1 for v in results.values() if v.get("status") == "PASS")
print(f"[{ts()}] [TASK E] 행동 테스트 완료: {pass_count}/5 통과")
print(json.dumps(results, indent=2, ensure_ascii=False))
