# 모델 트러블슈팅 3단계

트러블슈팅 : 발생한 문제의 원인을 논리적으로 찾아내어 해결하는 과정을 말한다.



## 1단계 : Make it Run

1. **Shape 오류**

   1. 코드 주석에 예상하는 shape 기록하기

   2. 자동 맞춤(Broadcasting) 조심

      Broadcasting은 편리하지만 위험하다. (batch, 1) 텐서와 (1, classes) 텐서를 더하면 (batch, classes) 결과가 나오는데, 이것이 의도한 것인지 아닌지 코드만 봐서는 알기 어렵다. 나중에 값이 이상하게 나와도 트러블슈팅하기가 정말 어려워진다. `unsqueeze`나 `view`로 차원을 명시적으로 제어하자.

   3. Breakpoint 적극 활용

      에러가 발생하면 print(tensor.shape) 대신 VS Code의 Breakpoint을 사용하자. 중단점에서 모든 텐서의 shape, dtype, device를 한눈에 확인할 수 있다.

   + shape 오류 예시 코드

      ```python
      # ❌ 흔한 실수 1: 배치 차원을 잃어버림
      x = torch.randn(32, 3, 224, 224)  # (batch, channels, H, W)
      x = x.mean(dim=(2, 3))            # (32, 3) — 의도: 공간 평균
      x = x.mean(dim=0)                  # (3,) — 실수! 배치 차원도 평균냄
      # classifier(x)를 호출하면 shape 불일치 발생
      
      # ✅ 수정: dim을 명시적으로 확인
      x = x.mean(dim=(2, 3))            # (32, 3)
      assert x.dim() == 2, f"Expected 2D, got {x.dim()}D"  # 차원 검증 추가
      
      # ❌ 흔한 실수 2: 브로드캐스팅 함정
      labels = torch.tensor([0, 1, 2])        # shape: (3,)
      predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # shape: (3, 2)
      # labels + predictions → 브로드캐스팅으로 (3, 2)가 됨 — 에러 없이 잘못된 결과!
      
      # ✅ 수정: 연산 전에 shape를 명시적으로 확인
      assert labels.shape == (3,) and predictions.shape == (3, 2)
      loss = torch.nn.functional.cross_entropy(predictions, labels)  # 올바른 사용
      ```



2. **OOM (Out of Memory)**

   OOM의 원인이 모델 파라미터 자체가 아니라 Adam 옵티마이저의 상태(파라미터의 3\~4배 메모리)인 경우가 매우 흔하다. 모델 크기가 GPU 메모리의 1/4도 안 되는데 OOM이 발생한다면, 옵티마이저를 의심해봐야 함.

   1. 정밀도 낮추기

      precision="16-mixed". 메모리를 거의 절반으로 줄이면서 속도도 빨라진다.

   2. 배치 사이즈 줄이기

      가장 직관적이지만, 너무 작아지면 학습이 불안정해진다.

   3. 그래디언트 축적

      여러 번의 작은 배치 계산 결과를 합쳤다가, 한꺼번에 모델을 업데이트하는 기술. GPU 메모리가 부족해도 큰 배치 사이즈를 쓰는 효과를 냄. 배치 사이즈를 줄여서 발생하는 학습 불안정을 보상할 수 있다.

   4. 그래디언트 체크포인팅

      순전파 중간 결과(활성화 값)를 저장하지 않고, 역전파 시 필요할 때 재계산합니다. PyTorch에서는 torch.utils.checkpoint.checkpoint 함수로, 또는 Lightning에서 Trainer(strategy=...)의 일부로 활용 가능.

   ※ 배치 사이즈 팁 : 2의 거듭제곱(32, 64, 128, 256...)으로 설정하면 GPU 메모리 활용 효율이 더 좋다. GPU의 연산 유닛이 2의 거듭제곱 크기에 최적화되어 있기 때문

   

3. **NaN/Inf : 보이지 않는 위협**

   모델이 멈추지는 않지만, 값이 조용히 NaN이나 무한대로 변해버리는 현상이다. 학습 곡선을 보다가 갑자기 Loss가 `nan`으로 표시되면, 이전 몇 스텝에서 이미 그래디언트가 폭주하고 있었던 것이다.

   

   *진단 방법*

   1. `gradient_clip_val=1.0`으로 그래디언트 크기를 제한하고 다시 실행

   2. W&B나 TensorBoard에서 그래디언트 Norm을 모니터링. NaN 직전에 스파이크가 보이는지 확인

   3. 스파이크가 보인다면 → 학습률을 낮추거나, 그래디언트 클리핑 값을 더 낮게 설정

   4. 스파이크 없이 갑자기 NaN → 모델 아키텍처나 초기화에 문제

   

   *대표적인 원인*

   - BatchNorm/LayerNorm

      정규화 레이어는 수치적 불안정의 대표적 원인이다. 배치가 너무 작으면 BatchNorm의 분산 추정이 불안정해진다.

   - 16비트 정밀도의 한계

      FP16은 표현 가능한 값의 범위가 좁아, 매우 작은 값이 0으로 수렴(Underflow)하거나 큰 값이 무한대로 발산(Overflow)하기 쉽다. 임시로 32비트로 전환해본 후, 문제가 해결되면 수치 정밀도 이슈다. 

      BF16은 FP16보다 Dynamic Range가 넓어 이 문제가 덜하다.

   - 학습률이 너무 높음

      가장 단순하고 흔한 원인이다. 학습률을 10배 낮춰보자.



## 2단계 : Make it Fast

모델이 구동되면, 다음은 실험 주기를 단축하는 것이다. 최적화 전에, 나의 직감이 얼마나 틀릴 수 있는지 먼저 인식해야 한다.

```python
# Lightning 프로파일러 — 가장 쉬운 시작점
trainer = pl.Trainer(profiler="simple", max_epochs=1)
trainer.fit(model, datamodule=data)
```

```python
Action                  | Mean (s)  | Total (s) | Percentage
---------------------------------------------------------------
run_training_epoch      | 12.5      | 12.5      | 100%
  training_step         | 0.08      | 8.0       | 64%
  get_train_batch       | 0.04      | 4.0       | 32%
  optimizer_step        | 0.005     | 0.5       | 4%
```

▶ 이 결과를 읽는 법:

- `get_train_batch`(데이터 로딩)가 32%? 

   → 모델을 아무리 빠르게 해도 32%의 한계가 있음. `num_workers`부터 늘려야 한다.

- `training_step`이 64%? 

   → 모델 연산 자체가 주 병목. 혼합 정밀도, `torch.compile`, 또는 모델 경량화를 시도하기.

- `optimizer_step`이 40% 이상? 

   → 옵티마이저가 병목. AdamW에서 SGD로의 전환이나, 분산 학습 시 All-Reduce 통신 최적화를 검토하자.

- 더 정밀한 분석이 필요하면 `profiler="advanced"`(Python cProfile 기반)이나 PyTorch의 `torch.profiler`를 사용하자.



### **속도 최적화 체크리스트 (차례로 실행)**

1. 데이터 로딩 최적화 : `num_workers` 증가, `persistent_workers=True`, `pin_memory=True`, 데이터를 NVMe SSD에 배치

2. 혼합 정밀도 : `precision="16-mixed"` → 거의 항상 적용해야 함.

3. `torch.compile(model)` : 커널 퓨전으로 20\~50% 속도 향상 가능

4. 그래디언트 축적으로 효과적 배치 증가 : 큰 배치는 GPU 활용률을 높인다.

5. 불필요한 로깅/시각화 제거 : 매 스텝마다 텐서를 CPU로 가져오는 로깅은 놀라울 정도로 큰 오버헤드를 발생시킨다. `self.log()`의 `on_step=False, on_epoch=True`를 활용하기.

6. 분산 학습 : 데이터 병렬화(DDP)로 GPU를 추가



### **데이터 굶주림(Data Starvation) 현상의 진단과 해결**

`get_train_batch` 시간이 `training_step`보다 현저히 길다면, `nvidia-smi`로 GPU 활용률을 확인했을 때 30\~50% 수준이라면 높은 확률로 데이터 굶주림이다.



**해결 체크리스트 (효과가 큰 순서대로) :**

1. `num_workers` 증가 : CPU 코어 수의 절반\~전체 사이에서 실험적으로 조정. 너무 높이면 오히려 IPC 오버헤드가 발생

2. `persistent_workers=True` : 에포크 사이에 워커 프로세스를 재사용하여 초기화 비용 제거

3. `pin_memory=True` : CPU → GPU 데이터 전송을 비동기(non-blocking)로 수행

4. 데이터를 NVMe SSD에 배치 : 이전에 배운 스토리지 지연 시간 차이를 떠올리자.

5. 전처리를 사전 캐싱 : 무거운 전처리(토크나이징, 리사이징 등)를 학습 전에 미리 수행하여 `.pt` 또는 `.arrow` 파일로 저장

6. GPU 기반 전처리 : NVIDIA DALI 또는 `torchvision.transforms.v2`(GPU 지원)를 활용



+ 데이터 굶주림 진단 코드

   ```python
   # 데이터 굶주림 진단 코드
   import time
   
   def measure_data_loading_speed(dataloader, num_batches=50):
       """DataLoader의 배치 공급 속도를 측정"""
       start = time.time()
       for i, batch in enumerate(dataloader):
           if i >= num_batches:
               break
       elapsed = time.time() - start
       batches_per_sec = num_batches / elapsed
       print(f"DataLoader 속도: {batches_per_sec:.1f} batches/sec ({elapsed:.1f}s for {num_batches} batches)")
       print(f"→ GPU가 배치당 0.05초 걸린다면, 필요한 공급 속도는 20 batches/sec")
       return batches_per_sec
   ```



## 3단계 : Make it Right

모델이 구동되고 빨라졌으면, 이제 검증/테스트 메트릭을 목표치까지 끌어올린다.

### **성능 저하의 3가지 원인과 처방**

성능이 안 나올 때 첫 번째 할 일은 train_loss와 val_loss를 비교하는 것이다.

![image.png](./모델%20트러블슈팅%203단계-assets/image.png)



### **스케일링 법칙(Scaling Laws)의 실전 활용**

스케일링(Scaling) : 모델의 파라미터 수(두뇌 용량), 데이터의 양, 학습에 투입하는 컴퓨팅 자원(GPU 시간) 자체를 늘리는 것이다. 요리로 치면 재료의 질을 높이고 양을 대폭 늘려 요리의 체급 자체를 바꾸는 걸 말한다.

작은 모델에서 아무리 HPO를 잘해도, 더 큰 모델이 대충 학습한 성능을 넘어서기 어렵다. 성능을 1% 올리기 위해 수백 번의 HPO 실험을 반복하는 것보다, 그 자원을 모아 모델 크기를 2배 키우는 것이 결과적으로 더 높은 성능을 보장하는 경우가 많다.

스케일링을 통해 목표 성능 근처에 도달했을 때, 마지막 '한 끝'을 올리기 위해 하이퍼파라미터를 미세하게 조정하는 게 순서상 맞다.

과거에는 "성능을 올리려면 무엇을 해야 하지?"가 직감의 영역이었다. 이제는 수학적으로 예측할 수 있다.



**실전 활용법**

1. 작은 규모 실험 3\~5개 수행

   파라미터 수나 데이터 크기를 점진적으로 늘리며 각각의 최종 Loss를 기록한다.

2. 로그-로그 차트에 점을 찍고 직선 회귀

   x축은 자원(compute/data/params), y축은 Loss.

3. 직선을 연장하여 필요 자원 추산

   "목표 Loss에 도달하려면 데이터가 10배, GPU 시간이 5배 필요하다"라는 구체적인 숫자를 얻는다.

→ 이것이 바로 OpenAI, Google DeepMind 같은 선도 팀들이 학습을 언제 멈출지, 모델을 얼마나 크게 만들지를 결정하는 방식이다. "실험을 10배 돌리면 성능이 X% 향상될까?"에 대해 데이터 기반으로 답할 수 있게 된다.



DeepMind의 연구에 따르면, 많은 모델이 파라미터 수에 비해 데이터가 부족한 상태로 학습되고 있다고 한다. 성능 최적화를 위해서는 파라미터 수와 데이터 양을 균형 있게 늘려야 한다. 파라미터만 키우고 데이터를 그대로 두면 비용 대비 성능 향상이 매우 비효율적임.



### **비용 한계와 대안: Foundation Model 활용**

모든 팀이 무한정 스케일을 키울 수는 없다. 비용이 문제라면,

1. 이미 대규모로 사전 학습된 Foundation Model을 가져와 Fine-tuning 하자.

2. 검증된 아키텍처와 하이퍼파라미터를 고수하자. Hugging Face Hub이나 Papers with Code에서 이미 작동이 증명된 설정을 가져온다.

3. HPO보다 스케일링이 우선이다. 모델의 성능을 올리기 위해 "자잘한 설정값(Hyperparameter)을 만지는 데 시간을 쓰기보다, 데이터와 모델의 크기(Scaling)를 키우는 것이 훨씬 확실한 필승 전략이다.

4. 체급이 큰 모델을 쓰고 있는데도 성능을 올리고 싶다면, 아래 링크 내용 참고하기.

   1. 학습을 너무 일찍 멈추지 마라. 학습 초기에는 성능이 떨어지는 '딥(Dip)' 구간이 발생하는데, 이를 견디고 오래 학습(Extended Training)해야 비로소 성능이 반등하며 일반화가 일어난다. 또한, 새로운 데이터를 많이 보는 것보다 적은 데이터를 여러 번 반복(Repetition)해서 학습하는 것이 복잡한 추론 패턴을 익히는 데 더 효과적이다.

   2. '무엇'을 배우느냐보다 '어떻게' 풀어나가는지가 중요하다. 정답만 있는 수학 문제보다, 아주 쉬운 게임이라도 단계별 사고 과정(Chain of Thought)이 포함된 데이터를 학습할 때 모델은 문제 해결의 '절차적 패턴'을 더 잘 습득한다.

   <https://www.youtube.com/watch?v=bXi-VnB_H8s&list=LL&index=1>


