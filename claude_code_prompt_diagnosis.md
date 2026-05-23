# Claude Code 프롬프트 — Phase 1 데이터 진단

## 🎯 목적

XGBoost v2 결과(MAE 9.66, 충남 25.53 / 전남 90.42)에서 **충남이 Naive 대비 -0.1%로 전혀 개선되지 않은 원인**과 **전남이 전국 평균을 9배 끌어올리는 이상치인 원인**을 데이터 레벨에서 진단한다. **모델은 절대 건드리지 않는다.** 이 진단 결과에 따라 이후 개선 방향(지역별 정규화 / 다중 관측소 매핑 / 분리 학습 / 피처 추가)이 결정된다.

---

## ⚠️ 절대 규칙

1. **모델 학습/예측 코드 작성 금지.** XGBoost, LSTM, 어떤 모델도 만들지 않는다.
2. **기존 피처 엔지니어링 결과(`national_train_features.csv` 등)는 참고만 하고 수정하지 않는다.**
3. **시각화는 한국어 폰트 적용** — 기존 `src/utils/font_setting.py`의 `apply()` 사용.
4. **로그**: 각 단계 타임스탬프 출력. `[HH:MM:SS] [TASK X-N] 메시지` 형식.
5. **에러 처리**: 파일/컬럼 없으면 명확한 메시지와 함께 종료.
6. **재현성**: random_state=42 고정.

---

## 📁 출력 파일 구조

```
src/diagnostics/
└── phase1_data_diagnosis.py        ← 새로 작성

outputs/diagnostics/
├── phase1_충남_연도별_추이.png
├── phase1_전남_연도별_추이.png
├── phase1_지역별_분포_비교.png
├── phase1_충남_관측소매핑_품질.png
├── phase1_전남_관측소매핑_품질.png
├── phase1_2022하반기_vs_2023_분포.png
├── phase1_지역별_이상치_비율.csv
├── phase1_지역별_평균발전량_train_vs_test.csv
├── phase1_diagnosis_results.json   ← 모든 수치 결과
└── phase1_diagnosis_report.md      ← 사람이 읽을 진단 리포트
```

---

## 🚀 실행 명령

```bash
.venv/bin/python src/diagnostics/phase1_data_diagnosis.py
```

---

## 📋 TASK 구성

### TASK P1-A — 환경 점검 및 데이터 로드

```python
# ── 경로 상수 ─────────────────────────────────────────────
TRAIN_READY  = "data/processed/national_train_ready.csv"   # 전처리만 끝난 raw 수준
TEST_READY   = "data/processed/national_test_ready.csv"
TRAIN_FEAT   = "data/processed/national_train_features.csv"
TEST_FEAT    = "data/processed/national_test_features.csv"
OUT_DIR      = "outputs/diagnostics"
RESULT_JSON  = f"{OUT_DIR}/phase1_diagnosis_results.json"
REPORT_MD    = f"{OUT_DIR}/phase1_diagnosis_report.md"
# ─────────────────────────────────────────────────────────
```

- 위 4개 파일 존재 여부 확인. 없으면 명확한 메시지로 종료.
- `OUT_DIR` 자동 생성.
- 모든 데이터를 `parse_dates=["timestamp"]`로 로드.
- 각 데이터프레임의 행수, 기간, 지역 수를 로그로 출력.

---

### TASK P1-B — 충남·전남 raw 데이터 연도별 추이 분석

#### 목적
충남이 -0.1% 개선에 그친 이유와 전남이 90.42 MAE 이상치인 이유가 **데이터 자체의 시간적 변화**에 있는지 확인.

#### 실행 사항

train + test를 `pd.concat`으로 합쳐서 2017~2023 전 기간을 본다. 각 지역(충남, 전남)에 대해 **연도별 다음 통계**를 계산:

```python
for region in ["충청남도", "전라남도"]:
    yearly_stats = full_df[full_df["region"] == region].groupby(
        full_df["timestamp"].dt.year
    ).agg(
        평균=("power_mwh", "mean"),
        중앙값=("power_mwh", "median"),
        최대=("power_mwh", "max"),
        표준편차=("power_mwh", "std"),
        피크시간_평균=("power_mwh", lambda x: x[
            full_df.loc[x.index, "timestamp"].dt.hour.isin(range(10, 15))
        ].mean()),
        결측비율=("power_mwh", lambda x: x.isna().mean()),
        영값비율=("power_mwh", lambda x: (x == 0).mean()),
    )
```

#### 시각화 (지역마다 1장)
- `phase1_충남_연도별_추이.png`, `phase1_전남_연도별_추이.png`
- 4-subplot: (1) 연도별 평균 발전량 라인 (2) 연도별 최대값 라인 (3) 연도별 피크 시간대 평균 (4) 연도별 영값/결측 비율
- 2023년(test 구간)을 다른 색으로 강조 — train과 분포가 다른지 시각적으로 확인

#### 자동 진단 로직
```python
# 설비 급증 신호 자동 검출
yearly_max = yearly_stats["최대"]
for y1, y2 in zip(yearly_max.index[:-1], yearly_max.index[1:]):
    growth = (yearly_max[y2] - yearly_max[y1]) / yearly_max[y1]
    if growth > 0.5:  # 50% 이상 급증
        diagnosis["설비_급증_의심"][region][int(y2)] = round(growth * 100, 1)
```

#### 출력
`diagnosis["yearly_trend"][region]`에 위 통계와 급증 의심 연도를 저장.

---

### TASK P1-C — 17개 지역 전체 이상치 비율 분석

#### 목적
충남/전남만이 아니라 **모든 지역의 이상치 비율을 비교**해서 충남/전남이 정말 특이한지, 아니면 다른 지역도 비슷한지 확인. 이상치 비율이 비슷하다면 충남 문제는 이상치가 아니라 다른 곳에 있음.

#### 이상치 정의
- **방법1 — IQR**: `q75 + 1.5 * IQR`을 초과하는 비율 (피크 시간대 10~14시만)
- **방법2 — Z-score**: 지역 내 z-score > 3인 비율 (전 시간대)
- **방법3 — 영값 폭주**: `power_mwh == 0` 비율 (피크 시간대 10~14시만 — 낮인데 0이면 이상)

```python
records = []
for region in sorted(full_df["region"].unique()):
    rdf = full_df[full_df["region"] == region]
    peak = rdf[rdf["timestamp"].dt.hour.isin(range(10, 15))]
    
    q1, q3 = peak["power_mwh"].quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_outlier_pct = ((peak["power_mwh"] > q3 + 1.5 * iqr) |
                       (peak["power_mwh"] < q1 - 1.5 * iqr)).mean() * 100
    
    z = (rdf["power_mwh"] - rdf["power_mwh"].mean()) / rdf["power_mwh"].std()
    z_outlier_pct = (np.abs(z) > 3).mean() * 100
    
    daytime_zero_pct = (peak["power_mwh"] == 0).mean() * 100
    
    records.append({
        "region": region,
        "IQR_이상치_pct": round(iqr_outlier_pct, 2),
        "Z3_이상치_pct": round(z_outlier_pct, 2),
        "주간_영값_pct": round(daytime_zero_pct, 2),
        "평균_발전량": round(rdf["power_mwh"].mean(), 2),
        "최대_발전량": round(rdf["power_mwh"].max(), 2),
    })
```

#### 출력
- `phase1_지역별_이상치_비율.csv` — 17개 지역 전체 표
- `phase1_지역별_분포_비교.png` — 17개 지역 boxplot 한 장에 (피크 시간대만, log scale y축)
- `diagnosis["outlier_summary"]`에 표 데이터 저장

#### 자동 해석
콘솔에 다음 형식으로 출력:
```
[해석] 충남 IQR 이상치 비율: 4.2% (전국 평균 2.1%의 2배)  ← 의심
[해석] 충남 주간 영값 비율: 8.5% (전국 평균 5.1%)         ← 정상
[해석] 전남 최대값 989.2 (2위 지역의 4.6배)               ← 이상치 확정
```

---

### TASK P1-D — 기상 관측소 매핑 품질 진단

#### 목적
충남이 보령 1개 관측소로 대표되는데, 이게 **충남 전체와 얼마나 상관이 있는지** 확인. 만약 보령 일사량과 충남 발전량의 상관이 다른 지역(예: 제주-제주, 부산-부산) 대비 약하다면, 매핑 품질이 근본 원인.

#### 분석 항목

**(1) 지역별 기상↔발전량 상관계수 비교**
```python
correlations = []
for region in sorted(train["region"].unique()):
    rdf = train[train["region"] == region]
    peak = rdf[rdf["timestamp"].dt.hour.isin(range(10, 15))]
    if len(peak) < 100:
        continue
    correlations.append({
        "region": region,
        "corr_일사량_발전량": round(peak["일사량"].corr(peak["power_mwh"]), 3),
        "corr_기온_발전량":   round(peak["기온"].corr(peak["power_mwh"]), 3),
        "corr_전운량_발전량": round(peak["전운량"].corr(peak["power_mwh"]), 3),
    })
corr_df = pd.DataFrame(correlations).sort_values("corr_일사량_발전량")
```

**(2) 충남·전남 시각화**
지역별로 (일사량 x축, 발전량 y축) scatter plot.
- `phase1_충남_관측소매핑_품질.png` — 충남 + 비교군 3개(평균적인 지역, 매핑 좋은 지역, 매핑 나쁜 지역)을 4-subplot으로
- `phase1_전남_관측소매핑_품질.png` — 동일 구조

산점도 위에 상관계수와 회귀선을 표시. **상관계수가 0.7 미만이면 매핑 품질 의심**으로 자동 표시.

**(3) 자동 진단**
```python
mean_corr = corr_df["corr_일사량_발전량"].mean()
충남_corr = corr_df[corr_df["region"] == "충청남도"]["corr_일사량_발전량"].iloc[0]
전남_corr = corr_df[corr_df["region"] == "전라남도"]["corr_일사량_발전량"].iloc[0]

if 충남_corr < mean_corr - 0.1:
    diagnosis["매핑품질_의심"]["충청남도"] = {
        "corr": 충남_corr,
        "전국평균": mean_corr,
        "결론": "기상 관측소 매핑 재검토 필요 — 다중 관측소 평균 도입 검토",
    }
```

#### 출력
- `outputs/diagnostics/phase1_지역별_기상상관.csv`
- `diagnosis["correlation_quality"]`에 17개 지역 전체 상관계수 저장

---

### TASK P1-E — Train/Test 분포 차이 진단 (3번 항목)

#### 목적
LSTM val/test 성능 괴리(train loss 0.01 / val loss 0.22)의 원인이 **2022년 말 ~ 2023년 분포 변화**에 있는지 확인. XGBoost에도 동일한 영향이 있는지 검증.

#### 분석 구간 정의
```python
train_late_2022 = train[train["timestamp"] >= "2022-07-01"]   # 2022 하반기
test_2023       = test                                         # 2023 전체
train_early     = train[train["timestamp"] <  "2022-07-01"]   # 2022 상반기 이전
```

#### 비교 항목 (지역별 + 전국)

**(1) 평균 발전량 train_late_2022 vs test_2023**
지역별로 평균/표준편차/최대값을 비교. CSV로 저장.
```python
records = []
for region in sorted(train["region"].unique()):
    a = train_late_2022[train_late_2022["region"] == region]["power_mwh"]
    b = test_2023[test_2023["region"] == region]["power_mwh"]
    records.append({
        "region": region,
        "2022하반기_평균": round(a.mean(), 2),
        "2023_평균":      round(b.mean(), 2),
        "변화율_pct":     round((b.mean() - a.mean()) / a.mean() * 100, 1) if a.mean() > 0 else None,
        "2022하반기_최대": round(a.max(), 2),
        "2023_최대":      round(b.max(), 2),
    })
```
→ `phase1_지역별_평균발전량_train_vs_test.csv`

**(2) Kolmogorov-Smirnov 검정 (지역별)**
```python
from scipy.stats import ks_2samp
for region in train["region"].unique():
    a = train_late_2022[train_late_2022["region"] == region]["power_mwh"].values
    b = test_2023[test_2023["region"] == region]["power_mwh"].values
    if len(a) > 30 and len(b) > 30:
        stat, pval = ks_2samp(a, b)
        # p < 0.01 이면 분포가 유의하게 다름 → drift 경고
```

**(3) 시각화**
`phase1_2022하반기_vs_2023_분포.png` — 2-subplot:
- 좌: 전남 분포 비교 (히스토그램 겹쳐 그리기) — 가장 큰 변화 예상
- 우: 충남 분포 비교

**(4) 자동 진단 로직**
```python
drift_regions = []
for r, info in ks_results.items():
    if info["pvalue"] < 0.01 and abs(info["change_pct"]) > 15:
        drift_regions.append((r, info["change_pct"]))

if drift_regions:
    diagnosis["distribution_drift"] = {
        "결론": "유의한 분포 변화 감지 — validation 전략 재검토 필요",
        "drift_regions": drift_regions,
        "권장": "TimeSeriesSplit 도입 또는 2023년 1~3월을 별도 val로 분리하는 방안 검토",
    }
```

#### 출력
`diagnosis["train_test_drift"]`에 KS 결과 저장.

---

### TASK P1-F — 종합 진단 리포트 생성

#### 출력 파일: `outputs/diagnostics/phase1_diagnosis_report.md`

다음 형식으로 자동 생성:

```markdown
# Phase 1 데이터 진단 리포트
생성일시: YYYY-MM-DD HH:MM:SS
분석 대상: 충청남도 (-0.1% 개선), 전라남도 (MAE 90.42)

---

## 1. 핵심 결론 (자동 생성)

### 충청남도
- **연도별 추이**: [정상 / 급변 의심 — 20XX년 ±NN%]
- **이상치 비율**: IQR NN% (전국 평균 NN%) — [정상 / 의심]
- **기상 매핑 품질**: 일사량-발전량 상관 0.NN (전국 평균 0.NN) — [정상 / 의심]
- **Train/Test drift**: KS p-value=N.NNN, 평균 변화 NN% — [없음 / 있음]
- **추정 원인**: [자동 추론한 한 줄]
- **권장 다음 단계**: [자동 추론한 우선순위]

### 전라남도
(동일 형식)

---

## 2. 17개 지역 이상치 비율 표
(CSV 내용 인라인)

## 3. 17개 지역 기상 상관계수 표
(CSV 내용 인라인, 상관계수 오름차순)

## 4. Train(2022 하반기) vs Test(2023) 분포 변화 표
(CSV 내용 인라인, 변화율 절댓값 내림차순)

## 5. 권장 다음 작업 (자동 추론)

진단 결과 조합에 따른 분기:

- IF 충남 매핑 품질 < 평균 - 0.1 → "충남 다중 관측소 평균 매핑 우선"
- ELIF 충남 IQR 이상치 > 평균 × 1.5 → "충남 raw 데이터 정제 우선"
- ELIF 충남 drift 유의 → "validation 전략 변경 우선 (TimeSeriesSplit)"
- ELSE → "데이터 레벨 문제 없음 — 6번(피처 추가) 또는 2번(지역별 정규화)으로 진행"

- IF 전남 설비 급증 감지 → "전남 분리 학습 또는 설비용량 정규화 피처 도입 검토"
- ELIF 전남 drift 유의 → "전남 단독 시계열 분포 정렬 필요"
- ELSE → "지역별 정규화(2번) 우선 적용"

## 6. 첨부 시각화 목록
(생성된 PNG 파일 경로 리스트)
```

#### 출력 파일: `outputs/diagnostics/phase1_diagnosis_results.json`
모든 수치를 구조화된 JSON으로 저장:
```python
{
  "yearly_trend": {"충청남도": {...}, "전라남도": {...}},
  "outlier_summary": [...],
  "correlation_quality": [...],
  "train_test_drift": {...},
  "auto_diagnosis": {
    "충청남도": {"primary_cause": "...", "recommended_action": "..."},
    "전라남도": {"primary_cause": "...", "recommended_action": "..."}
  }
}
```

---

## 🔍 검증 체크리스트

스크립트 종료 시 콘솔 마지막에 다음을 출력:

```
[검증]
  ✓ outputs/diagnostics/phase1_diagnosis_results.json 생성됨 (NN KB)
  ✓ outputs/diagnostics/phase1_diagnosis_report.md 생성됨 (NN줄)
  ✓ PNG 파일 6개 생성됨
  ✓ CSV 파일 3개 생성됨

[다음 단계]
  → outputs/diagnostics/phase1_diagnosis_report.md 의 5번 섹션을 확인하여 우선 작업을 결정하라.
```

---

## ⚠️ 작성 시 주의사항

1. **모델 학습 코드 작성 금지.** sklearn/xgboost/torch import도 하지 말 것 (scipy.stats만 허용).
2. **시각화 시 한국어 폰트** — `from src.utils.font_setting import apply; apply()` 호출 후 plot.
3. **figure 닫기** — 매 plot 후 `plt.close(fig)`로 메모리 누수 방지.
4. **DataFrame 출력 시** — `to_csv(... encoding="utf-8-sig")`로 한글 깨짐 방지.
5. **콘솔 출력은 모든 단계마다 타임스탬프 포함**, 결과 수치를 표 형태로 깔끔하게.
6. **자동 진단 로직의 임계값**(0.7 상관, 1.5 IQR, p<0.01 등)은 스크립트 상단에 상수로 분리하여 추후 조정 가능하게 할 것.

---

## 📦 최종 산출물 요약

이 작업이 끝나면 사용자는 다음 한 가지 질문에 답할 수 있어야 한다:

> **"충남과 전남 문제의 근본 원인은 무엇이며, 다음에 무엇을 해야 하는가?"**

리포트의 1번 섹션과 5번 섹션만 읽어도 답이 나오도록 작성할 것.
