 

# 딥러닝 DL

* optimaizer
* kerass

  





## optimizers

* 이차방정식 계수 추정 방법들:
  1. `SGD`: 움직임
  2. `GD` : 미분해서 움직임
  3. `momentum`: 지수이동평균법으로 움직임
  4. `NAG`: 관성 방향으로 이동 후 그 지점에서 GD 방향으로 움직임 

 

* 가중치(알파 or lr) 조정 알고리즘
  1. `Adagrad`: 업데이트 多~작은 알파 / 업데이트 小~큰 알파
  2. `RMSprop`: Adagrad에서 반복할수록 과도하게 작아진다는 알파값 보완 
  3. `Adadelta`: RMSprop와 같은데 알파값이 자동조절됨
  4. `Adam`: RMSprop + momentum

 



## Keras

* Sequential 모델: 

  ```python
  From tensorflow.keras.layes import Dense
  From tensorflow.keras.models import Sequential
  Model = Sequential() #그래프 생성(모델 생성)
  Model.add(Dense(1, input_dim = 2)) #layer(dense), 노드 1개, 그 노드에 2개가 들어온다고 알려줌
  Model.complile(loss=’mse’, optimizer=optimizers.Adam(lr=0.05))
  Model.fit(dataX, y, epochs = 300) #학습
  ```

  

* 더 간단한 모델: 

  ```python
  From tensorflow.keras.layes import Input, Dense
  From tensorflow.keras.models import Model
  xInput = Input(batch_shape=(None,2)) #그래프 생성(모델 생성)
  yInput = Dense(1)(xInput)
  model = Model(xInput, yOutput) #xinput 들어가서 yinput나오는 model
  model.complie(loss=’mse’, optimizer=optimizers.Adam(lr=0.05))
  model.fit(dataX,y,epochs=300) #학습
  ```

 

* 잔차 계산 방법들:

  1. Stochastic GD update: 그때그때 error 계산, a, b, c 업데이트
  2. Batch update: 한꺼번에 error 계산하고 a, b , c 업뎃
  3. Mini-batch update: 일부 error 계산하고 그때마다 a, b, c 를 업데이트. Stochastic GD update, Batch update의 중간 특성

  

* Model 의 기본적인 code:
  * `.fit` : train data를 만들어둔 model로 학습시킴
  * `.predict`: test data를 만들어둔 model로 궁예해봄

 

 

 