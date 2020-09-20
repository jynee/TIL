# 머신러닝(ML)

- K-Means 클러스터링

- H-clustering

- DBSAN

- 앙상블

- 연관규칙 분석

  



## **k-means** 클러스터링

* Grid search: 
  1. Init = ‘k-means++’
  2. n_init=3
  3. max_iter=300
  4. tol=le-04
  5. random_state=0

* k-means 클러스터링은 데이터를 k개의 클러스터(cluster, 무리)로 분류

* 비계층적 군집분석
* 비지도학습
* `EM알고리즘`: 중점을 할당한 후, 각 중점까지의 거리의 합을 최소화하는 알고리즘
* 알고리즘(작동 원리):
  1. 사용자로부터 입력받은 k의 값에 따라, 임의로 클러스터 중심(centroid) k개를 설정해준다.
  2. k개의 클러스터 중심으로부터 모든 데이터가 얼마나 떨어져 있는지 계산한 후에, 가장 가까운 클러스터 중심을 각 데이터의 클러스터로 정해준다. 
  3. 각 클러스터에 속하는 데이터들의 **평균**을 계산함으로 클러스터 중심을 **옮겨준다**. 
  4. 보정된 클러스터 중심을 기준으로 2, 3단계를 반복한다.
  5. 더이상 클러스터 중심이 이동하지 않으면 알고리즘을 종료한다. 

* new data 입력(발생) 시엔 각 중점과의 거리만 비교해서 가장 가까운 곳에 있는 군집에 속한다고 파악

* 초기값에 따라 전역이 아닌 지역 최소 값을 찾을 수 있음

* r에 따라 {0,1} 이면 명목형, 확률이면 연속형(GMM 모델),

> 참고: 심교훈. 2019. 10. 9. “가장 간단한 군집 알고리즘, k-means 클러스터링". https://bskyvision.com/564. b스카이비전
>

 

 

* 적합한 k 개수를 찾고 검증하는 모델들

### K-Means Elbow Method

* “k는 얼마가 적합할까?” <- 적합한 k개수 찾는 성격인듯

* Grid search: 
  1. n_clusters 조절
  2. error: 군집 속 중점과의 거리의 합 <- 이라고 개념을 설정해두고(왜냐면 k-means는 비지도학습이라 정답이 없어서(label이나 target, class 등이 없어서 확인 못함) 군집화가 잘 된 경우라면 error가 작을 것.
* 따라서 k가 증가할 때 줄어드는 ‘폭’이 작아지는 지점의 k값이 최적 군집 개수
* 거리의 합 != 거리가 줄어드는 폭.
* 엘보우는 거리가 줄어드는 폭을 봄

 

### 실루엣(Silhouette)

* “군집화가 잘 됐나?” <- 약간 검증하는 성격인듯
* 잘 된 군집화: 
  1. 군집 간 거리(b) > 군집 내 거리(a) 
  2. cohesion(응집도:군집 내) < separation(분리도:군집 간) 

* 실루엣 계수는 원형 군집이 아닌 경우 잘 맞지 않음

* 0~1값을 가짐

 

## K-Means++군집(clustering)

* local optimum 해결 위해 초기 중점을 좀 더 합리적으로 설정하는 방법

 

 

------------------





## H-clustering

* 계층적 군집분석

* 덴드로그램

* k-menas 와의 차이점:
  * k-means는 **사전에 그룹수(k)** 결정, 
  * H-clutering은 한 개의 그룹이 남을 때까지 **그룹을 다 나눈 후** 몇 개 선택할지 두 개의 feature를 갖는 2차원의 덴드로그램으로 결정

 



 

------------------





 

## DBSCAN

* Grid search: eps(앱실론: 특정 ‘반경’)
  1. eps = 0.2
  2. min_samples=5
     * 0.2 반경에 샘플 5개가 있어야 함
  3. metric = ‘euclidean’ 

* 계층적 군집분석
* 밀집도 기반의 군집 알고리즘: core point(핵심 샘플), border point(경계 샘플), noise point(잡음 샘플)
* noise point는 분류하지 않는다
* K-means or 다른 군집분석과의 차이점: 모든 샘플을 클러스터에 할당하지 않고 잡음 샘플을 구분하는 능력이 있다



 



 

------------------









## 앙상블 기법

* 다수의 결과를 예측하여 종합하고 분류함
* 여러 알고리즘을 사용한 후 결과를 종합하여 정확도, 일반화 특성을 증가시킴
* `classification_report(~~)`



### 배깅(Bagging or Bootstrap Aggregation)

* Hyper parameter : BaggingClassifier(~~)
  1. base_estimatior = m
  2. n_estimators = 100
  3. bootstrap = True

* prob += bag.predict_proba(testX)

* predY = np.argmax(prob, axis=1)  # axis=1 하는 이유: 안 하면 하나의 값인 int가 나와서 밑에 testY랑 mean하려면 array 형태로 나와야 함
* accuracy = (testY == predY).mean()
* BootStrap(단순복원 임의추출)을 통해 샘플 뽑아내 서브 데이터 만듦
  * 각각의 서브 데이터 크기 = 원본 훈련 데이터 크기 
  * why? 데이터 중복 허용해서 분산(변동)이 감소하고 overfitting 방지

 

### 부스팅(Boosting): 증폭, 가속. 
* 잘못 분류된 데이터에 가중치 두어 다시 뽑고 다시 분류하는 알고리즘

  1. 서브 훈련 샘플 만듦

  2. 처음 샘플링은 가중치를 두어 샘플링함 -> 분류 잘못된 데이터는 가중치 높이고 다시 샘플링. 이때 잘못 분류된 패턴이 선택될 확률이 높음. 

     → 분류가 어려운 패턴에 더욱 집중하여 정확도를 높이는 방법

     

#### AdaBoost(Adaptive Boosting) 

* **약한 분류기** 사용. 잘못 분류한 데이터 샘플에 가중치를 두어 더 많이 샘플링하여 정확도 높임
* 서브 데이터로만 만듦(잔차 등으로 만드는 거 아님)
* Hyper parameter: `AdaBoostClassifier(~~) `
  1. base_estimator=svm
  2. n_estimators=100 #100번 재조합한단 뜻(오분류한 거에 가중치둬서)



#### Gradient Boosting(for regression)

* target 데이터의 **잔차**를 줄이도록 학습. 학습할수록 residual(잔차)가 계속 작아짐. 잔차가 더 이상 줄어들지 않을 떄까지 tree 생성하며 학습+추정치 업데이트
* residual 계산법 : 변수-(평균+학습률(알파. 0~1사이 값. 아무렇게나 줘도 됨. 보통 0.1)*tree의 leaf 평균
* 선형회귀라 dataset도 연속형 변수인 boston 집값을 보도록 한다.

*  Hyper parameter: `GradientBoostingRegressor(~~)`
  1. Loss = ‘ls’           #lest square = MSE 사용
  2. Learning_rate = 0.1     #알파. 가중치
  3. n_estimators = 100     #잔차(tree) 100개 만들라
  4. max_depth=3 #얕은 depth

* 선형회귀라 MSE 대신 r2 사용해도 OK
* MSE: 선형, 로지스틱 회귀 둘다 쓰여도 OK. 다만 선형에선 R2를, 로지스틱에선 BCE(바이너리일 떄) 더 잘 쓰임..(?)



#### Gradient Boosting(for classification)

* Regression과 동일하나, 추정치를 위해 odds, logs(odds), probability 개념 사용
* binary cross entropy(BCE)를 loss함수로 사용
* Hyper parameter: `GradientBoostingClassifier(~~)`
  1. loss = ‘devianve’,      #로지스틱 함수 + CE 쓰라는 뜻
  2. learning_rate = 0.1,     #학습률, 가중치, 알파값
  3. n_estimators=100,      #잔차(tree) 수
  4. max_depth=3 



#### XGBoost(Extreme Gradient Boosting)(for regression)

* 정규화와 가지치기를 통해 
  1. overfitting을 줄이고 
  2. 일반화 특성을 좋게 만듦
  3. 특히 대용량 data의 경우에 속도도 SOSO

* Similarity, output 값 사용: Similarity를 사용해서 잔차와 IG 계산하고 마지막에 output 계산해서 최종적으로 잔차 계산
  
* 데이터 大 ~ similarity(유사도) 小 why? 상쇄되는 값이 많아서.
  
* Hyper parameter:`XGBRegressor(~~)`

  1. Objective =’reg:squarederror’     #regression 사용하고 MSE 사용한단 뜻
  2. regression이니 r2 사용

  

#### **XGBoost(Extreme Gradient Boosting)(for classification)**

* 잔차 계산 시, output value를 사용한다는 데서 Gradient Boost랑 차이가 있음

* Hyper parameter:

  (chapter 1) `XGBClassifier(~~)`

  * Objective =’binary:logistic’       #바이너리 변수고 logistic함수(sigmoid) 사용

  (chapter 2) `XGBClassifier(~~)`

  * Param – {‘eta’ : 0.3, 

    ‘max_depth’ : 3, 

    ‘objective’ : ‘multi:softprob’      #softmas 사용한단 뜻

    ‘num_class’ : 3 }             #클래스 개수





### 랜덤포레스트(Random Forest)

* Hyper parameter : `RandomForestClassifier(~~)`
  1. max_depth=5
  2. estimaors=100
* DT(Decision Tree)를 앙상블함 
  * 트리마다 서로 다른 feature 사용
* 샘플링 함

 



### 다수결 알고리즘(Majority Voting)

* p.85 코드 확인
* 배깅/부스팅과의 차이점: 학습데이터를 서브data에 sampling 하느냐/안 하느냐
  * 다수결 알고리즘은 배깅/부스팅처럼 서브데이터로 나눈 게 아니라 그 자체를 쓴다.





### Isolation Forest(iForest):

* Hyper parameter: `Model = IsolationForest`
  * n_estimators = 100   #100개의 트리

* 이상데이터 검출하는 알고리즘 (ex: 카드 불법 사용 탐지에 활용)

* Keyword: 
  * 이진검색트리
  * Anomaly score(이상치 수치)
  * recall, precison 사용
    * 특히 recall 써서 실제 정상(T)인데 비정상(F)으로 예측했다던가, 실제 비정상(F)인데 정상(T)로 예측한 비율 찾아냄

 

 



 

------------------







 

## 연관규칙 분석

* 장바구니 분석. 고객들의 구매 성향을 분석할 수 있음
  1. 지지도: y가 독립변수인지 종속변수인지 불명확함
  2. 신뢰도: y에 대한 영향을 무시한단 단점이 있음
  3. 향상도: 1에 가까우면 X와 Y는 서로 독립적. 1보다 크면 양의 상관성, 1보다 작으면 음의 상관성. 리프트가 1보다 클수록 X→Y 규칙의 의미가 커짐

* 지지도/신뢰도/향상도 특징:
  1. 인과관계가 아닌 상관관계
  2. 얼마나 빈번하게 나타나는지 측정

* 이진행렬 구성
* Hyper parameter:
  1. Item sparse matrix생성: Frequent_itemsets = apriori(df, min_support = 0.6, use_columname==True)
  2. 모델 생성: rules = association_rules(frequent_itemsets, metric=”lift”, min_threshold=0.7)
  3. Lift가 작은 것부터 sort:
     * Rules = association_rules(by=[‘lift]’, axis =0, ascendin=False)

* 지/신/향 모두 임계치 이상인 모든 규칙을 찾기엔 Brute Force 방식을 써서 조합이 너무 많아짐. 따라서 이 조합을 줄일 수 있는 알고리즘이 Apriori 알고리즘.
* 항목을 줄이는 게 관건
* 한 항목 집합이 반발하면, 그것의 모든 부분 집합이 반발한단 뜻에서 지지도 기반 가지치기
* 연관성이 높다 = lift가 높다



 