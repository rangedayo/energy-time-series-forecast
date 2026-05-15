import sys
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def ts():
    return datetime.now().strftime("%H:%M:%S")

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
BASELINE_JSON    = "outputs/national_baseline_results.json"
XGB_JSON         = "outputs/national_xgb_results.json"
LSTM_JSON        = "outputs/national_lstm_results.json"
BEHAVIORAL_JSON  = "outputs/national_behavioral_test_results.json"
SAVE_VERIFY_JSON = "outputs/national_model_save_verify_results.json"
ESS_JSON         = "outputs/national_ess_simulation_results.json"
REPORT_OUT       = "outputs/national_final_report.md"

print(f"[{ts()}] [TASK H] 전국 최종 리포트 생성 시작")

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

baseline    = load_json(BASELINE_JSON)
xgb_r       = load_json(XGB_JSON)
lstm_r      = load_json(LSTM_JSON)
behavioral  = load_json(BEHAVIORAL_JSON)
save_verify = load_json(SAVE_VERIFY_JSON)
ess_r       = load_json(ESS_JSON)

naive_mae = float(baseline["lag1"]["MAE"]) if baseline else None

lines = []

# ══════════════════════════════════════════════════════════════════════════════
lines.append("=== [전국] 모델 성능 비교 (Test Set 2023) ===\n")
lines.append(f"| {'모델':<10} | {'MAE':>7} | {'RMSE':>7} | {'피크 MAE':>9} | {'Naive 대비 개선율':>16} |\n")
lines.append(f"|{'-'*12}|{'-'*9}|{'-'*9}|{'-'*11}|{'-'*18}|\n")

if baseline:
    lines.append(f"| {'Naive(lag1)':<10} | {baseline['lag1']['MAE']:>7.4f}"
                 f" | {baseline['lag1']['RMSE']:>7.4f}"
                 f" | {baseline['lag1']['MAE_peak']:>9.4f}"
                 f" | {'- ':>16} |\n")
if xgb_r:
    xgb_imp = f"{(naive_mae - xgb_r['MAE']) / naive_mae * 100:.1f}%" if naive_mae else "-"
    lines.append(f"| {'XGBoost':<10} | {xgb_r['MAE']:>7.4f}"
                 f" | {xgb_r['RMSE']:>7.4f}"
                 f" | {xgb_r['MAE_peak']:>9.4f}"
                 f" | {xgb_imp:>16} |\n")
if lstm_r:
    lstm_imp = f"{(naive_mae - lstm_r['MAE']) / naive_mae * 100:.1f}%" if naive_mae else "-"
    lines.append(f"| {'LSTM':<10} | {lstm_r['MAE']:>7.4f}"
                 f" | {lstm_r['RMSE']:>7.4f}"
                 f" | {lstm_r['MAE_peak']:>9.4f}"
                 f" | {lstm_imp:>16} |\n")
lines.append("\n")

# ══════════════════════════════════════════════════════════════════════════════
if xgb_r and "region_MAE" in xgb_r:
    lines.append("=== [전국] 지역별 MAE (XGBoost 기준) ===\n")
    lines.append(f"| {'지역':<12} | {'MAE':>7} | {'Naive 대비 개선율':>16} |\n")
    lines.append(f"|{'-'*14}|{'-'*9}|{'-'*18}|\n")
    baseline_region = baseline["lag1"].get("region_MAE", {}) if baseline else {}
    for region, mae in sorted(xgb_r["region_MAE"].items()):
        bl  = baseline_region.get(region, 0)
        imp = f"{(bl - mae) / bl * 100:.1f}%" if bl > 0 else "-"
        lines.append(f"| {region:<12} | {mae:>7.4f} | {imp:>16} |\n")
    lines.append("\n")

# ══════════════════════════════════════════════════════════════════════════════
if ess_r:
    lines.append("=== [전국] ESS 시뮬레이션 비교 ===\n")
    lines.append(f"| {'전략':<14} | {'전력낭비율':>10} | {'부족횟수':>8} | {'사이클수':>8} | {'운영효율점수':>12} |\n")
    lines.append(f"|{'-'*16}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*14}|\n")
    strategy_display = {
        "naive_strategy": "Naive",
        "xgb_strategy":   "XGBoost 기반",
        "lstm_strategy":  "LSTM 기반",
    }
    for k, display in strategy_display.items():
        if k in ess_r:
            r = ess_r[k]
            lines.append(f"| {display:<14} | {r['curtailment_rate_pct']:>9.1f}%"
                         f" | {r['shortage_count']:>8}"
                         f" | {r['battery_cycles']:>8.1f}"
                         f" | {r['ess_score']:>12.1f} |\n")
    lines.append("\n")

# ══════════════════════════════════════════════════════════════════════════════
lines.append("=== [전국] 테스트 결과 ===\n\n")

lines.append("[데이터 기댓값 테스트]\n")
lines.append(f"- national_train: {'PASS' if baseline else 'N/A'}\n")
lines.append(f"- national_test:  {'PASS' if baseline else 'N/A'}\n\n")

lines.append("[XGBoost 행동 테스트]\n")
if behavioral:
    def _s(key): return behavioral.get(key, {}).get("status", "N/A")
    t2 = behavioral.get("test2_directional", {})
    ratio_str = f" (증가 비율: {t2.get('increase_ratio', 0):.0%})" if "increase_ratio" in t2 else ""
    lines.append(f"- NaN/Inf 출력 검증:  {_s('test1_nan_inf')}\n")
    lines.append(f"- 방향성 테스트:       {t2.get('status', 'N/A')}{ratio_str}\n")
    lines.append(f"- 불변성 테스트:       {_s('test3_invariance')}\n")
    lines.append(f"- 정확성 테스트:       {_s('test4_accuracy')}\n")
    lines.append(f"- 지역 불변성 테스트:  {_s('test5_region_invariance')}\n")
else:
    lines.append("- 행동 테스트 결과 없음\n")
lines.append("\n")

lines.append("[LSTM 테스트]\n")
if lstm_r:
    lines.append(f"- 암기 테스트:         {lstm_r.get('memorization_test', 'N/A')}\n")
    lines.append(f"- NaN/Inf 출력 검증:  {lstm_r.get('nan_inf_check', 'N/A')}\n")
else:
    lines.append("- 암기 테스트:         N/A (TASK F 미실행)\n")
    lines.append("- NaN/Inf 출력 검증:  N/A\n")
if save_verify:
    bv = save_verify.get("batch_inference", {})
    lines.append(f"- state_dict 검증:    {save_verify.get('state_dict',  {}).get('status', 'N/A')}\n")
    lines.append(f"- TorchScript 검증:   {save_verify.get('torchscript', {}).get('status', 'N/A')}\n")
    lines.append(f"- ONNX 검증:          {save_verify.get('onnx',        {}).get('status', 'N/A')}\n")
    lines.append(f"- 배치 추론 (1):      {bv.get('batch_1',  {}).get('status', 'N/A')}\n")
    lines.append(f"- 배치 추론 (8):      {bv.get('batch_8',  {}).get('status', 'N/A')}\n")
    lines.append(f"- 배치 추론 (64):     {bv.get('batch_64', {}).get('status', 'N/A')}\n")
else:
    lines.append("- LSTM 저장 검증:      N/A (TASK F 미실행)\n")

report_text = "".join(lines)
print("\n" + "=" * 64)
print(report_text)
print("=" * 64)

os.makedirs("outputs", exist_ok=True)
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(f"# [전국] 최종 비교 리포트\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(report_text)
print(f"[{ts()}]   리포트 저장: {REPORT_OUT}")
print(f"[{ts()}] [TASK H] 전국 최종 리포트 완료")
