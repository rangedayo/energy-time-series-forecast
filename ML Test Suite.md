# ML Test Suite (테스트 수트)

테스트 수트는 특정 목적을 위해 모아둔 테스트 케이스들의 집합을 의미한다.

**Fixtures** : Pytest에서 여러 테스트 사이에 공통적으로 필요한 '요리 재료(데이터베이스 연결, 모델 로드 등)'를 미리 준비해두는 기능

**Marks** : Pytest에서 특정 테스트에 '느림보'나 '빨리빨리' 같은 라벨을 붙여 그룹화하는 기능





### 데이터 통로 확인, 수치적 폭주 감지, 모델 loss 검사

+ 코드

   ```python
   # tests/test_model.py
   import torch
   import pytest
   from my_project.model import LitClassifier
   
   @pytest.fixture
   def model():
       """테스트 전체에서 공유하는 모델 인스턴스"""
       return LitClassifier(hidden_dim=64)
   
   @pytest.fixture
   def sample_batch():
       """테스트용 가짜 배치"""
       x = torch.randn(8, 1, 28, 28)
       y = torch.randint(0, 10, (8,))
       return x, y
   
   def test_output_shape(model, sample_batch):
       """모델 출력의 shape이 올바른지 확인"""
       x, y = sample_batch
       logits = model(x)
       assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"
   
   def test_output_range(model, sample_batch):
       """출력 logits에 NaN이나 Inf가 없는지 확인"""
       x, y = sample_batch
       logits = model(x)
       assert torch.isfinite(logits).all(), "NaN or Inf detected in model output"
   
   def test_loss_decreases(model, sample_batch):
       """단일 배치에서 1 step 학습 후 loss가 감소하는지 확인"""
       x, y = sample_batch
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
   
       logits = model(x)
       loss_before = torch.nn.functional.cross_entropy(logits, y).item()
   
       loss = torch.nn.functional.cross_entropy(model(x), y)
       loss.backward()
       optimizer.step()
   
       logits_after = model(x)
       loss_after = torch.nn.functional.cross_entropy(logits_after, y).item()
   
       assert loss_after < loss_before, "Loss did not decrease after one training step"
   ```



### **데이터 품질을 위한 기댓값 테스트 (Expectation Testing)**

1. **스모크 테스트부터 시작해라.**

   모든 버그를 잡으려 하기보다, 데이터에 대해 가지고 있는 기본적인 가정을 명시적 코드로 만들어라.

   + 코드

      ```python
      def smoke_test_training_data(df):
          """데이터 파이프라인에 '불이 났는지' 확인"""
          # 1. 결측치 — 필수 컬럼에 Null이 있으면 학습 자체가 실패할 수 있음
          assert df['user_id'].isnull().sum() == 0, "user_id에 결측치 발견"
          assert df['label'].isnull().sum() == 0, "label에 결측치 발견"
      
          # 2. 논리적 선후 관계
          if 'end_date' in df.columns and 'start_date' in df.columns:
              assert (df['end_date'] >= df['start_date']).all(), \
                  "종료일이 시작일보다 빠른 데이터 존재"
      
          # 3. 값의 범위 — 느슨하게! (오탐 방지가 핵심)
          if 'height' in df.columns:
              assert df['height'].min() >= 0, "키는 음수일 수 없음"
              assert df['height'].max() <= 30, "키 30피트 초과 — 단위 변환 오류 의심"
      
          # 4. 레이블 분포 — 클래스가 극단적으로 불균형하지 않은지
          label_counts = df['label'].value_counts(normalize=True)
          assert label_counts.min() > 0.01, \
              f"클래스 '{label_counts.idxmin()}'가 전체의 1% 미만 — 극단적 불균형"
      
          # 5. 데이터 크기 — 예상 범위 내인지
          assert len(df) > 100, f"데이터가 너무 적음: {len(df)}행"
      
          print(f"✅ 스모크 테스트 통과 ({len(df)}행, {df['label'].nunique()}개 클래스)")
      ```



2. **"느슨한 경계" 원칙이 핵심**

   오탐을 줄이기 위해 물리적으로 불가능한 범위 위주로 먼저 설정해라.

   사람의 키가 보통 4\~8피트라고 해서 그 범위로 잡으면, 정상 데이터인데도 테스트가 실패(오탐)한다. "키는 음수일 수 없고 30피트를 넘을 수 없다" 정도로 느슨하게 잡아라.

   

3. **기댓값을 점진적으로 강화하기**

   처음에는 느슨한 기댓값으로 시작하여, 데이터에 대한 이해가 깊어지면 통계적 범위를 활용한 더 정교한 기댓값(예: "99%의 데이터가 평균으로부터 3표준편차 이내에 있어야 함")을 추가하자.

   

4. **데이터 품질의 6대 기준 고려하기**

   "정확성(Accuracy), 완전성(Completeness: Null 체크), 일관성(Consistency: 시스템 간 모순 여부), 적시성(Timeliness: 최신 데이터 공급), 유효성(Validity: 형식/범위), 고유성(Uniqueness: 중복 여부)을 체크하되, 특히 완전성과 유효성을 중점적으로 체크해줘."

   

5. **Great Expectations 전문 도구 사용하기**

   "처음엔 assert문으로 시작하고, 나중에 Great Expectations로 확장 가능한 구조면 좋겠어."

   Great Expectations는\
   자동 문서화, 시각적 품질 보고서, 알림 시스템을 내장하고 있어, 데이터 파이프라인의 각 단계(수집 → 전처리 → 학습 직전)에 검증 게이트를 배치할 수 있다.

   1. GX을 쓸 때는 파이프라인에 3개 이상의 데이터 소스가 연결되는 시점에서 전환을 고려하자. 이 시점에서 GX를 쓰면, 각 소스가 합쳐지기 직전에 '합격 통지서'를 요구하게 된다.

   + 예시) 배달 앱 추천 모델 파이프라인

      이 모델이 잘 돌아가려면 최소한 3가지 다른 종류의 데이터(소스)가 필요하다.

      - 사용자 정보 (DB 1): 사용자의 나이, 거주 지역, 선호 음식 카테고리 (정적인 데이터)

         → 검문소 A (사용자 데이터): "지역 코드가 비어있지 않은가?" (성공 시 통과)

      - 주문 이력 (DB 2): 어제 무엇을 시켜 먹었는지, 최근 한 달간 결제 금액은 얼마인지 (동적인 트랜잭션 데이터)

         → 검문소 B (주문 데이터): "최근 1시간 내 데이터가 생성되었는가? (적시성)" (성공 시 통과)

      - 가게 정보 (외부 API): 현재 영업 중인 가게 리스트, 실시간 배달 예상 시간, 현재 할인 이벤트 여부 (외부 연동 데이터)

         → 검문소 C (가게 API): "현재 배달 가능 시간이 음수로 나오지는 않는가?" (성공 시 통과)

   2. "데이터 특성이 달라졌다"는 것은 결국 AI의 지식이 낡은 유통기한 지난 지식이 되었다는 신호이다. 그래서 Great Expectations나 Evidently AI 같은 도구로 이 변화를 계속 감시하다가, 분포가 너무 많이 차이 나면 이제 새로운 데이터로 다시 학습(Retrain)시킬 때가 됐다고 판단해야 한다.



6. **파이프라인 단계별 기댓값 배치 전략**

   "데이터 수집 직후와 학습 직전 저장 단계, 이 두 곳에 배치할 코드를 짜줘."

   데이터 파이프라인은 보통 수집 → 전처리 → 피처 엔지니어링 → 학습 직전 저장의 단계를 거친다. 각 단계의 출력에 기댓값 테스트를 배치하면, 어느 단계에서 데이터가 오염되었는지 정확히 파악할 수 있다.

   + 코드

      ```python
      # 파이프라인 단계별 기댓값 테스트 설계 예시
      def validate_raw_data(df):
          """수집 직후 — 가장 느슨한 검증"""
          assert len(df) > 0, "데이터가 비어 있음"
          assert 'user_id' in df.columns, "user_id 컬럼 누락"
          assert 'timestamp' in df.columns, "timestamp 컬럼 누락"
      
      def validate_preprocessed_data(df):
          """전처리 후 — 형식과 범위 검증"""
          assert df['user_id'].isnull().sum() == 0, "user_id 결측치"
          assert df['age'].between(0, 150).all(), "비현실적인 나이 값"
          assert df['price'].min() >= 0, "음수 가격"
          # 전처리로 생성된 새로운 컬럼이 존재하는지
          assert 'normalized_price' in df.columns, "정규화 컬럼 누락 — 전처리 실패"
      
      def validate_training_ready(df):
          """학습 직전 — 가장 엄격한 검증"""
          assert df.duplicated().sum() == 0, "중복 레코드 발견"
          label_dist = df['label'].value_counts(normalize=True)
          assert label_dist.min() > 0.01, f"극단적 클래스 불균형: {label_dist.to_dict()}"
          # 이전 학습 데이터와 통계적 분포 비교 (드리프트 감지의 첫걸음)
          assert abs(df['feature_1'].mean() - EXPECTED_MEAN) < TOLERANCE, "피처 분포 변화 감지"
      ```



7. 모델 개발자가 정기적으로 '온콜(On-call) 로테이션'을 통해 실제 운영 데이터를 직접 레이블링하고, 에지 케이스를 체험하는 단계까지 가면 베스트다.





### 학습 코드 검증을 위한 암기 테스트 (Memorization Testing)

"우리 모델이 아주 작은 데이터를 완벽하게 100점 맞을 지능(용량)이 있는가?"를 확인하는 테스트.

모델이 단 1개의 배치조차 암기하지 못한다면, 이는 모델의 학습 능력 문제가 아니라 코드의 물리적 연결이 끊어져 있다는 의미이다. 암기 테스트는 `DataLoader → Model → Loss → Optimizer → GPU` 파이프라인의 모든 배관이 새지 않고 연결되어 있는지 증명하기 위해 만든 테스트다.



***어떻게 하는가?***

- 아주 적은 양(예: 8\~16개 샘플)만 골라낸다.

- 모델이 공부하는 걸 방해하는 설정(Dropout, augmentation 등)을 모두 OFF.

- 손실(Loss) 값이 거의 0에 가깝게 떨어지는지 또는 정해놓은 기준 아래인지 확인한다.

   - 목표 손실에 도달하기까지의 에포크 수나 실행 시간을 기록

- 테스트는 무조건 10분 이내에 끝나야 한다. (너무 오래 걸리면 개발자가 테스트를 안 하게 됨)

   - 모델 크기를 살짝 줄이거나, 가장 계산이 오래 걸리는 부분은 잠시 빼고 테스트하자.



실전 구현은 “Lightning으로 2줄”이면 충분하다.

```python
# 모든 규제를 끄세요 — 암기는 의도적인 오버피팅입니다
trainer = pl.Trainer(
    overfit_batches=1,              # 단 1개 배치만 사용
    max_epochs=50,                  # 충분한 에포크
    accelerator="auto",
    enable_checkpointing=False,     # 테스트이므로 불필요
    logger=False,                   # 로깅도 불필요
)
trainer.fit(model, train_loader)
# 결과: Loss가 0에 가까워지면 통과 ✅, 그렇지 않으면 코드에 버그 ❌
```





### 행동 테스트와 메타모픽 테스트

행동 테스트가 더 큰 범위의 개념이고 메타모픽 테스팅은 그 안에 포함될 수 있는 구체적인 기법 중 하나라고 보면 된다.



**행동 테스트 (Behavioral Testing) → 불변성**

"이 모델이 특정 상황에서 상식적으로 행동하는가?"

**메타모픽 테스트 (Metamorphic Testing) → 방향성**

"입력값을 살짝 바꿨을 때, 출력값도 그에 맞춰 '논리적'으로 변하는가?"




| **테스트 유형** | **설명 (의미)** | **예시** | 
|---|---|---|
| 불변성 (Invariance) | 입력을 바꿔도 결과가 안 변해야 함 | "서울 날씨 좋아" → "부산 날씨 좋아" (지역만 바꿔도 긍정인 건 변함없어야 함) | 
| 방향성 (Directional) | 입력을 바꾸면 결과가 예상대로 변해야 함 | "이 영화 볼만해" → "이 영화 진짜 최고로 볼만해" (긍정 점수가 더 높아져야 함) | 
| 최소 기능 (MFT) | 아주 기초적인 능력 확인 | "이것은 사과다"를 "사과가 아니다"라고 판단하지 않는지 확인 | 



```python
# (불변성) 행동 테스트

def test_sentiment_invariance(model, tokenizer):
    """어미 변경에 모델이 불변해야 함"""
    texts = [
        ("이 영화는 정말 좋다", "이 영화는 정말 좋습니다"),
        ("서비스가 최악이다", "서비스가 최악입니다"),
    ]
    for text_a, text_b in texts:
        score_a = model.predict(tokenizer(text_a))
        score_b = model.predict(tokenizer(text_b))
        assert abs(score_a - score_b) < 0.1, \
            f"Invariance violated: '{text_a}'={score_a:.2f}, '{text_b}'={score_b:.2f}"
```



***어떻게 이 테스트를 만들 것인가?***

+ 코드 예시

   ```python
   def test_negation_handling(model):
       """부정 표현을 올바르게 처리하는지 확인"""
       positive = "이 서비스는 정말 편리합니다"
       negated = "이 서비스는 정말 편리하지 않습니다"
   
       score_pos = model.predict_sentiment(positive)
       score_neg = model.predict_sentiment(negated)
   
       assert score_pos > 0.5, f"긍정 문장이 긍정으로 분류되지 않음: {score_pos}"
       assert score_neg < 0.5, f"부정 문장이 부정으로 분류되지 않음: {score_neg}"
       assert score_pos > score_neg + 0.3, "긍정/부정 간 점수 차이가 충분하지 않음"
   
   def test_robustness_to_typos(model):
       """오타에 강건한지 확인 — 실사용자 데이터에는 오타가 많음"""
       original = "이 영화는 정말 재미있었다"
       with_typo = "이 영화는 정말 재미있엇다"  # 오타 포함
   
       score_orig = model.predict_sentiment(original)
       score_typo = model.predict_sentiment(with_typo)
   
       assert abs(score_orig - score_typo) < 0.15, \
           f"오타에 민감: 원본={score_orig:.2f}, 오타={score_typo:.2f}"
   ```

1. 어휘 + 구문 이해 (Vocabulary + POS)

   부정 표현: "좋다" → "좋지 않다" → 감성 반전 확인

   동의어: "훌륭하다" → "탁월하다" → 감성 유지 확인

   비교급: "A가 B보다 낫다" → A에 대한 감성이 더 높아야 함



2. 강건성 (Robustness)

   오타 삽입: "이 영화는 정말 좋앗다" → 결과가 크게 바뀌면 안 됨

   축약어/비속어: 실제 사용자 데이터에는 정제되지 않은 표현이 포함됨

   무관한 정보 추가: "참고로 오늘 날씨가 좋은데, 이 영화는 최고다" → "날씨" 언급이 감성에 영향을 주면 안 됨



3. NER / 고유명사

   사람 이름 교체: "김철수의 연기가 좋다" → "이영희의 연기가 좋다" → 감성 유지

   이것이 실패하면 모델이 특정 이름에 대한 편향을 학습한 것일 수 있음


