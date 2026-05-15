import sys
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import sys as _sys
_sys.path.insert(0, ".")
from src.utils.font_setting import apply as _apply_font
_apply_font()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
INTEGRATED_PRED = "outputs/national_xgb_predictions.csv"
JEONNAM_PRED    = "outputs/jeonnam_xgb_predictions.csv"
HYBRID_PRED     = "outputs/hybrid_xgb_predictions.csv"
HYBRID_RESULTS  = "outputs/hybrid_xgb_results.json"

REGION_NAME            = "전라남도"
PEAK_HOURS             = range(10, 15)
INTEGRATED_MAE         = 9.66
JEONNAM_INTEGRATED_MAE = 90.42
JEONNAM_STANDALONE_MAE = 86.68

print(f"[{ts()}] [하이브리드 예측 생성 시작]")

for path in [INTEGRATED_PRED, JEONNAM_PRED]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: 파일 없음 → {path}")

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print(f"[{ts()}]   통합 모델 예측 로드: {INTEGRATED_PRED}")
integrated_df = pd.read_csv(INTEGRATED_PRED, encoding="utf-8-sig", parse_dates=["timestamp"])

print(f"[{ts()}]   전남 단독 예측 로드: {JEONNAM_PRED}")
jeonnam_df = pd.read_csv(JEONNAM_PRED, encoding="utf-8-sig", parse_dates=["timestamp"])

print(f"[{ts()}]   통합: {len(integrated_df):,}행 / {integrated_df['region'].nunique()}개 지역")
print(f"[{ts()}]   전남 단독: {len(jeonnam_df):,}행")

# ── 하이브리드 결합 ───────────────────────────────────────────────────────────
print(f"\n[{ts()}] [1] 하이브리드 결합")
non_jeonnam = integrated_df[integrated_df["region"] != REGION_NAME].copy()
hybrid = pd.concat([non_jeonnam, jeonnam_df], ignore_index=True)
hybrid = hybrid.sort_values(["region", "timestamp"]).reset_index(drop=True)

# ── 검증 ──────────────────────────────────────────────────────────────────────
print(f"[{ts()}]   검증 중...")
assert len(hybrid) == len(integrated_df), \
    f"행 수 불일치: hybrid {len(hybrid)} != integrated {len(integrated_df)}"
assert hybrid["region"].nunique() == 17, \
    f"지역 수 불일치: {hybrid['region'].nunique()} != 17"
size_match = (
    hybrid.groupby("region").size() == integrated_df.groupby("region").size()
).all()
assert size_match, "지역별 행 수 불일치"
print(f"[{ts()}]   검증 통과 ✓  ({len(hybrid):,}행, {hybrid['region'].nunique()}개 지역)")

# ── 메트릭 계산 ───────────────────────────────────────────────────────────────
print(f"\n[{ts()}] [2] 메트릭 계산")
actual    = hybrid["actual"].values
predicted = hybrid["predicted"].values
hour      = hybrid["timestamp"].dt.hour
peak_mask = hour.isin(PEAK_HOURS).values

mae       = float(np.mean(np.abs(actual - predicted)))
rmse      = float(np.sqrt(np.mean((actual - predicted) ** 2)))
mae_peak  = float(np.mean(np.abs(actual[peak_mask] - predicted[peak_mask])))
rmse_peak = float(np.sqrt(np.mean((actual[peak_mask] - predicted[peak_mask]) ** 2)))

imp_vs_integrated = (INTEGRATED_MAE - mae) / INTEGRATED_MAE * 100

region_mae = {}
for region in sorted(hybrid["region"].unique()):
    mask = (hybrid["region"] == region).values
    region_mae[region] = round(float(np.mean(np.abs(actual[mask] - predicted[mask]))), 4)

# ── 출력 ──────────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] [하이브리드 결과]")
print(f"  {'전국 MAE':<20}: {mae:.4f}  (통합 {INTEGRATED_MAE} → 변화 {imp_vs_integrated:+.1f}%)")
print(f"  {'전국 피크 MAE':<20}: {mae_peak:.4f}")
print(f"  {'전남 MAE':<20}: {region_mae.get(REGION_NAME, 0):.4f}  (단독 모델 적용)")
print(f"  {'RMSE':<20}: {rmse:.4f}")
print(f"\n[{ts()}] [지역별 MAE 변화 요약]")
print(f"  전남 외 16개 지역: 변화 없음 (통합 모델 그대로)")
print(f"  전남: {JEONNAM_INTEGRATED_MAE} → {region_mae.get(REGION_NAME, 0):.4f}")

# ── 저장 ──────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
hybrid.to_csv(HYBRID_PRED, index=False, encoding="utf-8-sig")
print(f"\n[{ts()}]   하이브리드 예측 저장: {HYBRID_PRED}")

hybrid_results = {
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "MAE_peak": round(mae_peak, 4),
    "RMSE_peak": round(rmse_peak, 4),
    "improvement_vs_integrated_pct": round(imp_vs_integrated, 2),
    "region_MAE": region_mae,
    "comparison": {
        "integrated_total_MAE": INTEGRATED_MAE,
        "hybrid_total_MAE": round(mae, 4),
        "jeonnam_integrated_MAE": JEONNAM_INTEGRATED_MAE,
        "jeonnam_standalone_MAE": JEONNAM_STANDALONE_MAE,
    },
}
with open(HYBRID_RESULTS, "w", encoding="utf-8") as f:
    json.dump(hybrid_results, f, indent=2, ensure_ascii=False)
print(f"[{ts()}]   결과 JSON 저장: {HYBRID_RESULTS}")

print(f"\n[{ts()}] [하이브리드 예측 생성 완료]")
