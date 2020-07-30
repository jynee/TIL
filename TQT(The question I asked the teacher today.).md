**TQT(The question I asked the teacher today.)**





-------------------





# Dense

* `Dense`: fully connected

* `ANN(FNN)`에서는 여러 `Dense`를 써도 되지만, `RNN(LSTM)`에선 마지막 층에서만 `Dense`를 써야함.
  * `CNN`에서는 여러 `Dense` 써도 됨  ##?#?#??#?#???
  * => 일단... `lstm`에서 `lstm() → Dense → lstm()`은 `lstm 네트워크가 2개` 만들어진다고 보면 됨. `lstm() → Dense` 했을 때, `1개의 네트워크`가 형성된 것
  * => 그리고 `CNN`은 일종의 잘 짜여진 레시피라서 `con1D → pooling → Dense → con1D → pooling`은 위 `lstm`처럼 좀 이상한 네트워크 구조가 되는 거라 생각함...

* Dense(1, activation='sigmoid')

* ==LSTM에서 FNN으로 보내는 마지막 Dense에선 relu 쓰면 안됨==









# FNN(순방향 신경망)

* ↔ RNN
  
* hidden 층에서
  
  * Dense(4, `activation` = 'sigmoid', `kernel_regularizer`=regularizers.l2(0.0001), activation='relu')
  * `Dropout`(rate=0.5)
* `BatchNormalization`(momentum=0.9, epsilon=0.005, center=True, scale=True, moving_variance_initializer='ones')
  
* `predict`까지 끝낸 **연속형** `yHat` 값을, `np.where` 써줘서 **바이너리 형태**로 변환 

  ``` python
  np.where(yHat > 0.5, 1, 0)
  # 딥러닝_파일: 4-4.ANN(Credit_Keras)_직접 해보기_커스텀loss.py
  ```

* `history` 활용

  ```python
  hist.history['loss']
  hist.history['val_loss']
  # 딥러닝_파일: 4-4.ANN(Credit_Keras)_직접 해보기.py
  ```

* 학습/평가/예측용 model로 나누었을 때 **평가 데이터 활용**

  ```python
  model.fit(trainX, trainY, validation_data=(evlX, evlY), epochs=200, batch_size=50)
  ```










# LSTM

* |              | 설명                                                         |
  | ------------ | ------------------------------------------------------------ |
  | 2층          | `lstm()`을 2번 써준다                                        |
  | 양방향       | `bidirectional` + `merge_mode = ‘concat’` <br />FNN, BFN 값을 merge_mode 형태로 합쳐서 list형으로 되돌려줌 |
  | many-to-many | `return-sequences = True`<br />LSTM의 중간 스텝의 출력을 모두 사용 |
  |              | `timedistributed`<br /> FFN으로 가기 전 LSTM 마지막 층에서 각 뉴런의 각 지점에서 계산한 오류를 다음 층으로 전파 |










# CNN

* 이미지를 대표할 수 있는 특성들을 도출해서 FNN에 넣어줌

* | code                                                         | 설명                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | `Input`(batch_shape = (None, nStep, nFeature, nChannel))     |                                                              |
  | `Conv2D`(filters=30, kernel_size=(8,3), strides=1, padding = 'same', activation='relu') |                                                              |
  | `MaxPooling2D`(pool_size=(2,1), strides=1, padding='valid')  | - 경우에 따라 conv2D, pooling 더 써줄 수 있음<br />- `GlobalMaxPooling1D()`도 있음 |
  | `Flatten()`                                                  | 2D는 4차원이라 shape 맞추려고 보통 flatten을 써줌<br />1d는 안 써도 되는 듯(?) |
  | `Dense`(nOutput, activation='linear')                        |                                                              |

  









# activation

* | activation(비선형 함수) | loss                                                         |
  | ----------------------- | ------------------------------------------------------------ |
  | `softmax`               | `sparse_categorical_crossentropy`                            |
  | `sigmoid`               | `binary_crossentropy`                                        |
  | `linear`                | `mse`                                                        |
| `relu`                  | ← Hidden layer에 씀. 기울기가 0이기 때문에 뉴런이 죽을 수 있는 단점 有 |
  |                         |                                                              |
  | Leakly ReLU             | 뉴런이 죽을 수 있는 현상 해결                                |
  | PReLU                   | x<0 에서 학습 가능                                           |
  | granger causality       | 통제된 상황에서 인과관계가 가능하다고 말할 수 있음. 시계열 데이터에서 쓰일 수 있음 |

* 딥러닝 네트워크(DN)의 노드는 입력값을 전부 더한 후, 활성화 함수(Activation function)를 통과시켜 다음 노드에 전달한다.
  
  * 이때 사용하는 활성화 함수는 비선형 함수를 쓴다. 







## ReLu

* 히든층에 자주 쓰임

* 그냥 CNN이든 LSTM이든 출력층 Dense에 Relu 쓰지 말자

  * LSTM에선 Relu 안 쓰는 게 좋음. 특히 출력층엔 쓰면 안 됨.

  





------------------







# 학습(compile), 예측(predict)



## optimizer

* | 종류(빈도순)               |
  | -------------------------- |
  | `adam`                     |
  | Adadelta, RMSprop, Adagrad |
  | `momentum`                 |
  | GD, NAG                    |

* 최적화가 잘 안 되면 글로벌 minmun을 찾지 못하고 로컬 minimum에 빠진다. 이때 로컬 minimum을 **어떻게 빨리** 탈출할 수 있을지 U턴 메소드를 쓸지, 다른 1차 미분방법(GD)를 쓸 지 결정하게 된다. 







## epoch

* `epoch` 수치가 커지면 `optimizer`가 일을 해서 local이 아닌 global을 찾아간다.
* 그런데 너무 크면 overfitting
* 따라서 적당한 `epoch` 설정이 필요 







## Batch_size

* data가 크면 `batch_size`도 크게
  * 25,000개의 raw data라면 `batch_size` = 20 보다 300 이 정도로 설정









-----------------------









# NLP & DL



* SGNS

| 용어        | 설명                      | CODE                                | 참고                             |
| ----------- | ------------------------- | ----------------------------------- | -------------------------------- |
| pre-trained | SKNS에서 학습한 We를 적용 | model.layers[1]**.set_weights**(We) | 해당 code 적용 후 model fit 진행 |
|             |                           |                                     |                                  |
|             |                           |                                     |                                  |



* SGNS에 모델 학습(fit) 시, 학습을 따로 시키는 이유?

  ```python
  # 학습
  hist = model.fit([X[:, 0], X[:, 1]], X[:, 2], 
                   batch_size=BATCH_SIZE,
                   epochs=NUM_EPOCHS)
  ```

  > *  각기 연결된 가중치 선이 구분되어 있기 때문에



* SGNS 모델 만들 때 dot을 한다면, 

  1. **axis=2**    *@2*

     → 후에

  2. reshape**(())**    *@괄호 두 개*



* SGNS로 만든 Embedding의 w(가중치)를 basic한 word data에 적용할 때, load_weights 사용하는 방법도 있다.

  * 근데 이땐 shape을 맞춰줘야 한다.

  ```python
  w = encoder.load_weights('model_w.h5') # 가중치(w) 불러온 후,
  emb = Embedding(max_features, embedding_dims, load_weights = w)(xInput) # embedding layer에 바로 적용
  ```

  * 보통 이런 느낌으로 씀

    ```python
    weights = load_weights()
    embedding_layer = Embedding(input_dim=V,
                                output_dim=embedding_dim,
                                input_length=input_length,
                                trainable=False,
                                weights=weights,
                                name='embedding')
    ```

    







----------------------









# 기타



## 유클리디안 거리

* 거리 계산할 때, 비교하고 싶은 건 `[]`를 쳐서 넣어주기  

  ```python
  euclidean_distances([father, mother])
  ```



## 가중치 저장(Save)

* Embedding (left side) layer의 W를 저장할 때, [2]를 저장한단 사실 알아두기

  ```python
  with open('data/embedding_W.pickle', 'wb') as f:
      pickle.dump(model.layers[2].get_weights(), f, pickle.HIGHEST_PROTOCOL)
  ```



