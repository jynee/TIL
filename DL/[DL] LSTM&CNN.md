

# 딥러닝 DL

* RNN
* CNN





## 순환 신경망(RNN)

* hidden 층에서 서로 값을 기억해 순환하는 것
* 지금까진 FNN(feed forward neword) + 순서가 필요 없는 data를 써서 모델이 기억할 필요가 없었지만, 문장 같은 data를 쓸 땐 **순서가 중요한 data(Sequence Data)를 가지고 미래를 예측해야 함**
* 학습(트레이닝) 방법: 순서가 있는 data를 모델이 ‘기억’하게 만드는 것
* RNN 기본 입력은 3d 형태. D1=time, d2=feature, d0=data
* RNN 문제점: 기울기 소실 문제(vanishing gradient): FFN보다 더 심해짐



### LSTM

* RNN의 vanishing gradient 해결을 위해 고안. C 추가

- f와 i의 가중평균 형태

  ```batch
  f: forget date. **이전**의 C를 얼마나 반영할 것인지 조절
  i: input or ignore. **현재 입력값(x)과 이전의 출력값(h)**를 얼마나 반영할 것인지 조절
  
  * h는 위, 왼쪽 / c는 왼쪽으로 전파. 둘다 처음엔 0으로 시작함
  ```

* GRU: LSTM 구조를 간결하게 만듦. 속도도 더 빠르지만 성능도 안 떨어짐 

* 학습 유형: 단방향(FNN,BFN) / 양방향(FNN+BFN) 모두 적용

  1. Many – to – one
  2. Many- to -many
  3. One to many
  4. Many to many

* 단방향은 ‘이후’만 기억, 양방향은 ‘이전’+’이후’ 모두 기억

  * 어떻게? 단방향은 FBN만 사용, 양방향은 FBN+BFN 사용하기 땜에.

* 양방향(FNN+BFN) 진행 순서: 

  1. 모델(code)에서(RNN층 내) FNN 진행 후 정보량 모아두고, 
  2. BFN 진행 후 정보량 모아두고,
  3. CONCAT(합침)해서 
  4. error를 역전파 시킴 

* FNN VS BFN:

  * FNN : TIME이 낮->높
  * BFN : TIME이 높->낮

* `MANY TO ONE`

  둘다 정보량이 TIME이 증가할수록 높아진다. 따라서 높은 정보량끼리 마지막에 합친다

* `MANY TO MANY`: 

  FNN 정보량이 TIME이 증가할수록 높아진다. 

  BFN 정보량이 TIME이 증가할수록 낮아진다.

  따라서 각 뉴런에서 laten layer로 이어지는 정보량(마지막에 FNN정보량+BFN정보량)은 같다

  > 참고: 
  >
  > 아마퀀트. 2019. 7. 19. "Keras LSTM** 유형 정리 (2/5) – 단층-단방향 & many-to-many 유형". http://blog.naver.com/chunjein/221589624838. 아마추어 퀀트 (Amateur Quant).








## CNN

* 사람의 시각인지 과정을 모방해서 피드포워드 신경망에 추가

  * 이미지를 분류한다 치면, 이미지의 부분적 특성에 주목해 분류하여 FFN에 넣는 것 
    * 이미지 분석에 활용: 이미지를 대표할 수 있는 특성들을 도출해서 신경망에 넣어주는 것. 이때 CNN은 특성 도출과정을 자동화시켰다. 

* 순서 

  특성 추출 → 클래스 분류 
  → **컨볼루션** 또는 필터링 과정 
  → **특성지도** 출력 
  → 서브샘플링(subsampling) 또는 **풀링**(pooling) 
  → 다시 컨볼루션, 활성화, 서브샘플링을 수행 
  → 최종 특성지도는 피드포워드 신경망에 입력되어 분류 작업을 시행

  ```batch
  컨볼루션: 
  image data의 경우
  → filter(kernel: image data와의 convolution(cross-correlation) 진행) 
  → feature map 생성(기호에 따라 zero padding, ReLU와 같은 활성함수 적용 가능)
  → (max/mean) pooling 
  → feature map 생성(출력) 
  → flatten(: 1D구조로 만듦) 
  → FNN
  ```



1. 입력된 이미지로부터 이미지의 고유한 특징을 부각시킨 특성지도(feature map)를 새로 만듦
2. 그 이미지는 피드포워드 신경망에 입력되어 이미지가 어떤 클래스 라벨에 속하는지 분류
3. 학습: (grid serch) 수평 엣지 필터, 수직 엣지 필터 컨볼루션
4. ReLU와 같은 활성함수를 거쳐 특성지도 출력
5. 서브샘플링(subsampling) 또는 풀링(pooling) 통해 활성화된 특성지도들의 크기를 줄임
6. (저차원적인 특성부터 시작해서 고차원적인 특성을 도출)이 특성지도들에 다시 컨볼루션, 활성화, 서브샘플링을 수행하여 로컬한 특성지도로부터 글로벌한 특성지도를 만들어간다.
7. 이 과정을 여러번 반복하여 얻어진 최종 특성지도는 fully-connected layer, 즉 피드포워드 신경망에 입력되어 분류 작업을 시행





* 용어

  * `feature map`: 이미지의 부분적 특징을 모아놓은 것의 집합

  * `padding`: ‘원본’(사이즈)을 조정. filter를 거치면 이미지 사이즈가 원본과 달리 작아지는데, 이를 피하기 위해 작아지는 사이즈가 원본 사이즈만큼 되도록 원본 사이즈 크기를 늘림. 이때, Zero padding 기법을 사용. 수치(?)가 없는 부분(가생이..?)을 ‘0’으로 채움(가생이 아니어도 중간 부분 채워도 zero-padding)

  * `pooling`: 원본이미지에서 특징 추출해서 feature map의 크기를 줄여주는 과정. 
    (1) Max pooling, (2) mean pooling이 있음 

  * `convolution layer`: convolution(cross-correlation) 진행되는 곳 

  * `upsampling`: pooling layer와 달리 차원을 줄이는 게 아니라 차원을 늘림. Autoencoder의 decoder와 같은 곳에서 원래 데이터로 복원할 때 사용됨. Zero padding은 가생이를 0으로 채우는데 sampling은 무슨 계산을 해서 채우는 듯

  * `stride`: 필터 적용 시 이동할 칸 수

      



### 코딩 용어 설명

* 컨볼루션 레이어 단계
  * `Filters`: 출력 모양의 깊이(depth) 를 결정
  * `kernel_size`: 
    1. w(연결선, 가중치)이자, filter의 size.
    2. 연산을 수행할 때 윈도우의 크기
       * 2D에서 kernel_size=(8,1)이면 8행+1열(이때 1열은 feature)
       * 1D에서 kernel_sizw=8이면 자동 8행+전체열(1D에서 필터는 아래 방향으로 밖에 이동 못함)
  * `strides`: 한 번에 얼마나 움직일지(이동 크기). 보통 1을 씀. 
  * `padding`: 
    컨볼루션 레이어(합성곱) 혹은 풀링 연산을 수행하는 레이어에 파라미터로 설정
    convolution과 pooling 연산은 파라미터의 수를 줄여나가는 과정이다. 하지만 이러한 과정에서 지나치게 데이터가 축소되어 정보가 소실되는 것을 방지하기 위해 데이터에 0으로 이루어진 패딩을 주는 경우가 있다.
    1. `padding = 'same'`: 원본 사이즈 유지시킴(차원 유지), 
       (원리: 필터의 사이즈가 k이면 사방으로 k/2 만큼의 패딩을 준다.)
    2. `padding = 'valid'`: 패딩 사용하지 않음 
  * `activation: ‘ReLu’`가 default. CNN에선 ReLu 사용을 권장한다고 함 

* `pooling` 단계
  * pool_size: strides가 미리 설정되지 않을 경우 pool_size와 동일하게 설정된다. 
  * strides
  * padding

* `fatten` 단계
  * Flatten

* `outpu` 단계
  * Dense(n, activation = )

 

 

## 차원 참고

* 1차원 벡터: shape(2,)
* 2차원 Matrix : shape(행,열)
* 3차원: shape(면, 행, 열) = D0, D1, D2
* 4차원: shape (samples, rows, cols, channels) = D0, D1, D2 ,D3 

 

 

> 참고: 
>
> * chrisysl. 2018. 9. 10. "3. Convolutional Networks / L2. Convolutional Neural Networks - Convolutional Layers in Keras". https://kevinthegrey.tistory.com/141
> * 심교훈. 2019. 3. 1. "딥러닝 알고리즘의 대세, 컨볼루션 신경망(convolutional neural network, CNN)". https://bskyvision.com/412?category=635506b. 스카이비전
> * Seongyun Byeon. 2018.01.23. 딥러닝에서 사용되는 여러 유형의 Convolution 소개". https://zzsza.github.io/data/2018/02/23/introduction-convolution/. 어쩐지 오늘은

 

 

 



 