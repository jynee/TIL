

# 인공신경망

* 입력 데이터, 입력층, 연결, 연결 가중치(w), 은닉층, Bias, 출력층, 활성함수, 출력값
* 알고리즘 순서(계산도 이 순으로 이루어짐):
  1. 입력층
     * 뉴런 pass 
  2. 히든층
     * EX) BCE
  3. 출력단
     * 활성함수: Sigmoid 등

* 출력층의 오차(MSE)가 최소가 되는 `w` 찾기. 

* code를 쓸 땐 데이터 불러오고 **표준화(z-score)를 꼭 시켜줘야함**

  * 이미 되어 있는 경우는 불필요.  

  1.  출력층 개수 = data의 class 개수
  2.  입력층 = data의 feature 개수
  3. 은닉층은 일치시켜 줄 것도, 필요도 없음

* 출력층의 오차를 하위단으로 역전파 시키는 알고리즘 사용

* 경사 하강법으로 w 업데이트

* 은닉층이 1개 이상일 땐 mult-layer : MLP

* **입력 데이터에 대해 원하는 출력이 나오도록 연결 가중치(W)를 결정하는 것이 핵심**
  
  * W를 결정하는 규칙이 필요함(학습 규칙)
* 분류 문제: 출력 결과는 0~1 사이의 값 필요하므로 ‘sigmoid 함수 사용’
* 회귀 문제: 실숫값 그대로 출력해야 하므로 ‘linear 함수 사용’(위와 같음)
* 활성함수:  Cost(w 가중치)는 CE 등
* optimizer: Adam 등
* ReLu: error값이 역전파 잘 되도록 은닉층의 출력값으로 사용
* Bias의 입력값은 항상 1
* 학습할 때마다 동일한 w가 나오는 것은 아님
  
  * unique한 값이 나오진 않음. 따라서 yHat값만 봐도 됨
* 델타 학습 규칙 특징:
  1. 학습률 大 ~ 연결 가중치 大
  2. 오류 大 ~ 가중치 많이 변경
  3. sigmoid : 활성 함수의 기울기. 활성함수 값이 0.5일 때 sigmoid는 0.25로 최대값을 가짐

 







## `Batch normalization(표준화)`

* 은닉층에서 이루어짐
* Keras, tensor 등의 패키지에서 자동 적용되는 입력값을 표준화한 전처리 과정을 지나고, w를 만나 커진 값들이 서로 너무 차이가 나지 않도록 **네트워크 단에서 표준화하는 것**을 Batch normalization이라고 함 
* 순서: 
   입력층에서 입력된 각 값들을 각자 (1) Z-score(표준화) 후 (2) 전부 더함 (3) 그 값을 다른 층으로 넘김
  * 장점 1) 
     내부에 shift 변수가 있어서 자동으로 bias 역할을 함. 따라서 해당 param을 쓸 땐 input Dense에 bias를 빼도 된다
  * 단점 1) 
     test data를 fit할 때 문제가 된다. 왜냐면 Z-score는 train data로 만들어진 거라서.
  * ‘단점 1’의 해결을 위해 감마와 베타 사용하여, *감마와 베타*는 train data의 학습에 쓰고, 이때 계산했던 *표준편차와 평균*은 test data fit(예측)에 쓴다.
  * *감마*, *베타*: 이동평균의 개념(정확히 이동평균은 아님)으로 계산할 때 표준화하는 데 있어 최근에 들어오는 값에 가중치로 사용
  * 해당 값들은 model.summary()를 통해 Param 값으로 확인 가능하다! (캡쳐 참고)





## **`Regularization(정규화)`**

* 특정하게 커지는 값 방지 => 패널티항 느낌
  * Ex: w2값을 loss함수에 넣었을 때 값이 커져서 다음 층에 전달되는 경우, 그 커진 값을 네트워크단에서 **표준화(normalization)** 시켜서 못 올라가도록 막음

* 가중치 감소(Weight Decay): 가중치의 제곱 법칙(`L2` 법칙; 많이 사용된다)를 손실함수에 더해 손실함수 값이 더 커지게 한다. 그만큼 가중치가 커지는 것을 억제하기 되는 것





## `Dropout`

* 뉴런의 연결을 임의로 삭제. 훈련할 때 임의의 뉴런을 골라 삭제하여 신호를 전달하지 않게 한다.

​                               

 

 <br>

## binary classfication 일 때

* 출력층의 노드가 1개인 경우

* *일반화된 델타규칙* > 출력층의 연결 가중치를 변경(업데이트)
  1. activation = sigmoid인 경우, Gradient 
  2. activation = tan 함수인 경우, LSTM
  3. activation = ReLu인 경우, … ReLu를 가장 많이 사용한다! 

* *Vanishing Gradient* > 네트워크가 깊어질수록 (1) 은닉층~은닉층 (2) 은닉층~입력층으로 가는 error(오차)가 줄어드는 현상



 <br>

## multi-class classification일 때

* 출력층의 노드가 여러 개 인 경우
  * Ex: iris data 처럼 feature 3개 중 class 1개로 분류할 때. 따라서 3개 중 1개만 ‘1’이어야 함

* **Activation = ‘softmax’** 함수 사용 (one-hot 인코딩: 하나만 1이 된다)
* Softmax 함수 쓰기 위해 각 row를 0,1로 분류해주는 작업을 해야함 **loss 함수로 CCE 사용**
* 유의: multiple class claasification은 one-hot 인코딩이 아님! {1,1,0}등이 나오기 때문(1이 2개 나와도 OK일 때 사용하는 것). Activation = ‘Sigmoid’ 사용, Loss 함수로 BCE 사용
* 비지도 학습



<br>

## 경쟁학습 모델

* Winner neuron, Winner takes all
* 출력 값 기준
  1. 경쟁에서 이긴 뉴런에 대해서만 연결 가중치(w)를 조절한다 
  2. 한 번의 VS 후 가중치(w) 조절함
  3. 1, 2의 반복 
* Input과 w의 거리 기준: 거리가 짧은 것이 가장 닮은 꼴
* Weight normalization: 가중치 표준화
* INSTA 알고리즘
* 경쟁학습 모델(`SOM`) 알고리즘:
  1. Winner 주변의 이웃 뉴런들도 같이 선택해 winner와 neighborhood 뉴런들의 w를 업뎃함
  2. 처음에는 이웃의 범위 넓게 설정, 학습 반복할수록 최종 한 개의 winner 남을 때까지 이웃의 범위를 점차 좁혀 나감

* K-means와 비슷한 역할: 가중치를 조정해 이긴 뉴런들에게 데이터 입력 받아 군집 할당

 



 