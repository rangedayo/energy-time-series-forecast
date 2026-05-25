- 1차 시도
    
    XGBoost는 수동 피처 엔지니어링(lag, rolling, datetime 등)을 거친 9.59. 
    AutoGluon은 각 모델 내부 피처 생성기 사용한다. 따라서 직접 비교가 아닌 '엔지니어링 비용 대비 성능' 비교를 해보려고 한다.
    
    **템플릿 정보**
    
    - solar-ess-portfolio
    - asia-northeast1 (도쿄)
    - g2-standard-4
    - energy-ts-forecast
    
    **주의사항 (실제 실행 시)**
    
    - OOM이 나면 A100 40GB로 런타임 템플릿 다시 만들기
    - 마치고 나면 로컬에 다운받을 핵심 파일 :
        - national_autogluon_results.json (분기 진단 결과)
        - leaderboard.csv (모델별 성능표)
        - national_autogluon_predictions.csv (ESS v2 시뮬 통과용)
        - autogluon_leaderboard.png, autogluon_vs_xgb_region.png
    
    ### 결론 :
    
    | 모델 | MASE | 평가 |
    | --- | --- | --- |
    | **WeightedEnsemble** (LightGBM+Naive) | **0.496** | 압도적 1위 |
    | RecursiveTabular (LightGBM) | 0.546 | 2위 |
    | DirectTabular (LightGBM) | 0.595 | 3위 |
    | SeasonalNaive | 0.721 | 4위 |
    | TFT | 1.938 | 6위 (의미 없음) |
    
    TFT 심화보다 LightGBM 앙상블 심화가 ROI가 좋을 가능성이 크다.
    
    - MPC 먼저 도입 후 > TFT 학습 시간 늘려서 도전
    - Chronos는 검증용으로 쓰기. 어떤 경우로든 다 포트폴리오에 가치 있다.
        - 잘 나오면 → "사전학습 모델로도 충분"이라는 시그널 (가능성 낮음)
        - 별로면 → "사전학습 시계열 모델까지 검증했지만 도메인 특화 모델이 우위"라는 시그널 (가능성 높음)
    
- 2차 시도
    
    ### v1 대비 v2의 핵심 차이 요약
    
    | 항목 | v1 | v2 |
    | --- | --- | --- |
    | 모델 선택 | `preset="best_quality"` | `hyperparameters=` 명시 |
    | RecursiveTabular | 포함 (누적오차 원인) | 제외 |
    | Chronos2/DeepAR | 환경 충돌로 실패 | 명시적 제외 |
    | TFT | 9분만 학습 | max_epochs=100, context_length 168/336 × 2개 |
    | PatchTST | 시도조차 안 함 | 신규 추가 × 2개 |
    | DirectTabular | LightGBM만 | GBM + XGB + CAT (XGBoost 본인 모델과 직접 비교) |
    | num_val_windows | 3 | 5 (분포 shift 완화) |
    | 학습 시간 | 28800 (8시간), 실제 18분 | 10800 (3시간), 시간 정확히 채움 |
    | 갭 처리 | 도중에 추가 | 셀로 명시 분리 |
    | 예측 시 model | 자동 (경고 발생) | 명시 가능 |
    | v1과의 비교 | — | 자동 출력 |
    | 저장 경로 | `outputs/autogluon_v1` | `outputs/autogluon_v2` (v1 보존) |
    
    ### 결과 :
    
    - Ensemble weights: {'DirectTabular': 0.73, 'SeasonalNaive': 0.27}
        
        TFT, PatchTST, AutoETS의 앙상블 비중이 0%이다. AutoGluon이 "이 모델들은 도움이 안 된다"고 판단하고 완전히 제외한 것. 어제 v1 결과(`DirectTabular_2: 0.73, SeasonalNaive: 0.27`)와 동일한 앙상블 구조다.
        
        → 태양광 발전 데이터의 특성을 생각하면 납득이 된다.
        
        1. 주요 신호가 기상 변수에 직접 의존
            
            `일사량`, `전운량` 등 known_covariates가 발전량을 거의 결정적으로 설명. 트리 모델이 이런 특성에 강함.
            
        2. 트랜스포머는 긴 시계열 패턴 학습에 강한데, 태양광은 "어제 발전량보다 오늘 일사량"이 훨씬 중요. 
            - "지금이 오후 2시니까 곧 피크일 것" → 일일 주기, 트리 모델도 OK (시간 feature만 있으면)
            - "작년 같은 주에 한파가 와서 수요가 폭증했다" → 장기 패턴, 트랜스포머가 잘함
            
            비유하자면, TFT는 "복잡한 추리 소설을 읽고 범인 맞히기"에 강한 모델인데, 태양광 예측은 "지금 등 켜져있나?"를 묻는 단순 함수 문제에 가깝다. **추리 능력이 무용지물이고, 오히려 복잡한 구조가 단순한 문제에서는 노이즈로 작용할 수 있다.**
            
            차라리 그 시점의 일사량 값 하나 보는 게 훨씬 정확해서 트랜스포머보다 단순 시계열 모델인 XGBoost가 효과가 더 큰 것. **AI는 시간 더 들여서 TFT를 살려도 DirectTabular를 이길 가능성이 낮아 보인다, 데이터 특성 자체가 트리 모델에 유리한 형태라고 평가하고 있음**
            
        3. PatchTST는 known_covariates 활용도가 낮음
            
            PatchTST의 -3.05는 단순히 일사량을 못 보고 있다는 신호일 수 있음 (모델 구조상 한계)
            
    - **Best Model: WeightedEnsemble (-0.5986)**
        
        **트랜스포머 추가 실험은 "효과 없음"이 확인됨**
        

### 보고서

- 발견 1 : XGBoost가 AutoGluon보다 약 2배 좋다.
    
    AutoGluon을 도입했으나 본인이 직접 튜닝한 XGBoost가 우월함을 확인. 자동화 도구의 한계와 도메인 지식의 가치를 입증한 실험이었다.
    
- 발견 2 : v1 vs v2의 진짜 차이는 거의 없음
    - v1 validation score: 0.4956 (WeightedEnsemble)
    - v2 validation score: 0.5986 (WeightedEnsemble)
- 발견 3 : 트랜스포머는 결국 앙상블에서 0%
    
    ```python
    DirectTabular:  0.73
    SeasonalNaive:  0.27
    TFT, PatchTST, AutoETS: 0.00  ← 완전히 제외됨
    ```
    
    트랜스포머 4개를 추가 학습했지만 AutoGluon이 "도움 안 된다"고 판단해서 가중치 0을 줬다.
    
- 발견 4 : 환경 트러블슈팅 자체가 보고서의 가치 있는 부분
    - v1 로그 — Chronos2, Chronos2SmallFineTuned, ChronosWithRegressor, DeepAR이 전부 `torch.fx.experimental.symbolic_shapes` 에러로 실패.
    - v2 로그 — torch 2.10→2.9.1 다운그레이드로 TFT/PatchTST 살림