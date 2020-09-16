

# NLP

* Quora  : 질문 간 텍스트 유사도 분석
* maLSTM : 맨하탄 거리 사용한 LSTM
* GloVe : 빈도 + 맥락(Embedding) 고려한 워드 패키지
* FastText : hash 값을 사용해 어딘가에 저장해두므로, 사전에 없는 단어를 입력해도 비슷한 걸 찾아 vector 값으로 나온다.





## GloVe & maLSTM

* maLSTM = 맨하탄 거리 사용한 LSTM

```python
# Question-1, 2 입력용
K.clear_session()
inputQ1 = Input(batch_shape=(None, trainQ1.shape[1]))
inputQ2 = Input(batch_shape=(None, trainQ2.shape[1]))

# Question-1 처리용 LSTM
embQ1 = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)(inputQ1)
embQ1 = Dropout(rate=0.2)(embQ1)
lstmQ1 = LSTM(HIDDEN_DIM)(embQ1)
lstmQ1 = Dense(FEATURE_DIM, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZER))(lstmQ1)
lstmQ1 = Dropout(rate=0.2)(lstmQ1)

# Question-2 처리용 LSTM
embQ2 = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)(inputQ2)
embQ2 = Dropout(rate=0.2)(embQ2)
lstmQ2 = LSTM(HIDDEN_DIM)(embQ2)
lstmQ2 = Dense(FEATURE_DIM, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZER))(lstmQ2)
lstmQ2 = Dropout(rate=0.2)(lstmQ2)
```

> 위 코드처럼 Embedding을 각각 해줄 땐, 
>
> 1. 한글 사전 / 영어 사전 속 단어의 Vector 값을 각각 확인하는 것 처럼 사전의 근본이 다를 때
>
> 2. 챗봇 Q&A : Q는 형태소 분석을 하고, A는 형태소 분석을 하지 않을 때 처럼 결과값이 달라야 할 때.
>
>    "달라야 할 필요가 있을 때만" 이렇게 사용한다.



* 각각 embedding 값에서 맨하탄 거리 측정

  ```py
  # Question-1, 2의 출력으로 맨하탄 거리를 측정한다.
  # lstmQ1 = lstmQ2 --> mDist = 1
  # lstmQ1 - lstmQ2 = inf --> mDist = 0
  # mDist = 0 ~ 1 사잇값이므로, trainY = [0, 1]과 mse를 측정할 수 있다.
  mDist = K.exp(-K.sum(K.abs(lstmQ1 - lstmQ2), axis=1, keepdims=True))
  ```

  > * ex: 
  >   a=np.array(np.arange(20).reshape(4,5)
  >   np.sum(a,axis=0, keepdims=true)
  >   *keepdims=true쓰면 원래 1차원으로 나와야 할 것이 원래 a의 차원대로 2차원으로 나옴
  > * Backend as k:네트워크 출력 결과에 어떤 연산을 취할 때
  >   keepdims=true:원래 a값 구조대로 줌



* compile

  ```py
  model = Model([inputQ1, inputQ2], mDist)
  model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005))
  model.summary()
  ```

  






## GloVe & maLSTM & Quora  

* 동시발생 확률 고려

* 빈도 기반 문장 , 맥락(context) 고려

* 순서

  1. 전처리가 완료된 학습 데이터를 읽어온다.

  2. Quora 데이터의 Vocabulary를 읽어온다.

  3. maLSTM 모델을 빌드한다.

  4. 저장된 WE를 읽어온다.

     1. Pre-trained GloVe 파일을 읽어와서 GloVe dictionary를 생성한다

        ```python
            GloVe = {} #Vocabulary
            for line in file: # 1라인씩 읽음. # line=['love','~','~'..]. vector가 300개 들어있다.
                wv = line.split()
                word = ''.join(wv[:-300]) # 앞부분 # word=wv[0] 워드만.
                GloVe[word] = np.asarray(wv[-300:], dtype=np.float32) # 뒷부분 vector만.
            file.close() # 뒷부분 300개 
        ```

        > GloVe['love']을  실행하면 아래 vector가 출력된다. shape = (300,)
        > array([ 1.3949e-01,  5.3453e-01, -2.5247e-01, -1.2565e-01,  4.8748e-02,
        >          1.5244e-01,  1.9906e-01, -6.5970e-02,  1.2883e-01,  2.0559e+00, ...

     2. WE = np.zeros((VOCAB_SIZE, EMB_SIZE))

     3. ```python
            for word, i in word2idx.items(): #여기서 word2idx는 quora 용으로 우리가 만든 사전 
                vec = GloVe.get(word) #300개짜리 vector # GloVe.get(word) : 있으면 나오고 없으면 null 값이 나와서 프로그램이 멈추지 않음(get)을 쓰면. # glove[word]라고만 쓸 땐, 있으면 나오고 없으면 error 뜸 
                if vec is not None:
                    WE[i] = vec
        ```

     4. 결과를 저장한다.

  5. 학습 데이터와 시험 데이터로 나눈다.

  6. Question-1, 2 입력용(input)

  7. 공통으로 사용(범용)할 Embedding layer 빌드

  8. Question-1 처리용 LSTM model 빌드

  9. Question-2 처리용 LSTM model 빌드

  10. Question-1, 2의 출력으로 맨하탄 거리를 측정한다.

      ```python
      mDist = K.exp(-K.sum(K.abs(lstmQ1 - lstmQ2), axis=1, keepdims=True))
      ```

      > Question-1, 2의 출력으로 맨하탄 거리를 측정한다.
      > lstmQ1 = lstmQ2 --> mDist = 1
      > lstmQ1 - lstmQ2 = inf --> mDist = 0
      > mDist = 0 ~ 1 사잇값이므로, trainY = [0, 1]과 mse를 측정할 수 있다.

  11. 학습

      ```python
      model = Model([inputQ1, inputQ2], mDist)
      model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005))
      model.summary()
      ```

  12. 예측

      ```python
      trainY = trainY.reshape(-1, 1)
      testY = testY.reshape(-1, 1)
      hist = model.fit([trainQ1, trainQ2], trainY,
                       validation_data = ([testQ1, testQ2], testY),
                       batch_size = 1000, epochs = 10)
      ```

  13. 시험 데이터로 학습 성능을 평가

      ```python
      predicted = model.predict([testQ1, testQ2])
      predY = np.where(predicted > 0.5, 1, 0)
      accuracy = (testY == predY).mean()
      print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))
      ```

      



-------------------





## FastText & maLSTM & Quora  

* subword 덕분에 어떤 단어라도 vector 형태로 만들어준다

  > model.wv.vocab.keys()

* 한글 hash도 있다

* ```python
  from gensim.models import fasttext
  ```

* 순서

  1. 전처리가 완료된 학습 데이터를 읽어온다.

  2. Quora 데이터의 Vocabulary를 읽어온다.

  3. maLSTM 모델을 빌드한다.

     ```python
     # 이때, 
     EMB_SIZE = 300
     # pre-trained FastText의 vector size = 300이므로, 이와 동일하게 맞춘다.
     ```
     1. 저장된 WE를 읽어온다

     ```python
     savedWeightEmbedding = True
     ```
     2. Pre-trained FastText 파일을 읽어와서 dictionary를 생성한다.

     ```python
     model = fasttext.load_facebook_vectors('./dataset/wiki.en.bin')
     ```
     3. 빈(zero) 가중치 파일 생성

     ```python
     WE = np.zeros((VOCAB_SIZE, EMB_SIZE))
     ```

     * Embedding의 w 값을 뽑아낼 수 있게 되었다. 

       ```py
       for word, i in word2idx.items():
       WE[i] = model.wv[word]
       ```

     * 결과 저장

  4. 학습 데이터와 시험 데이터로 나눈다.

  5. Question-1, 2 입력용(input)

  6. 공통으로 사용할 Embedding layer 빌드

  7. Question-1 처리용 LSTM model 빌드

  8. Question-2 처리용 LSTM model 빌드

  9. Question-1, 2의 출력으로 맨하탄 거리를 측정

     ```python
     mDist = K.exp(-K.sum(K.abs(lstmQ1 - lstmQ2), axis=1, keepdims=True))
     ```
     > lstmQ1 = lstmQ2 --> mDist = 1
     > lstmQ1 - lstmQ2 = inf --> mDist = 0
     > mDist = 0 ~ 1 사잇값이므로, trainY = [0, 1]과 mse를 측정할 수 있다.
     
  10. 학습   
  
      ```python
      model = Model([inputQ1, inputQ2], mDist)
      model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0005))
      model.summary()
      ```
      
    11. 예측

          ```python
        trainY = trainY.reshape(-1, 1)
          testY = testY.reshape(-1, 1)
          hist = model.fit([trainQ1, trainQ2], trainY, validation_data = ([testQ1, testQ2], testY), batch_size = 1000, epochs = 10)
          ```
  
    12. 시험 데이터로 학습 성능을 평가

          ```python
        predicted = model.predict([testQ1, testQ2])
          predY = np.where(predicted > 0.5, 1, 0)
          accuracy = (testY == predY).mean()
          print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))
          ```



 


### FastText 연습

* 패키지

  ```python
  from gensim.models import FastText
  from gensim.test.utils import common_texts # 9개 문장
  ```

  > [['human', 'interface', 'computer'],
  >  ['survey', 'user', 'computer', 'system', 'response', 'time'],
  >  ['eps', 'user', 'interface', 'system'],
  >  ['system', 'human', 'system', 'eps'],
  >  ['user', 'response', 'time'],
  >  ['trees'],
  >  ['graph', 'trees'],
  >  ['graph', 'minors', 'trees'],
  >  ['graph', 'minors', 'survey']]

* 모델 빌드

  ```python
  model = FastText(size=5, window=3, min_count=1) 
  model.build_vocab(sentences=common_texts)
  model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)
  ```

  > bucket=hash table. 따라서 100이라 주면, model.wv.vectors_ngrams.shape가 100, n이 됨. hashing trick을 통과시키고 나오는 값들이 들어가는 곳으로, bucket 값이 크면 collision을 방지한다
  > size = EMB_SIZE
  > window = 좌우 context 개수
  > min_count = 빈도 1개 이상(그니까 전부 다)

* word2vec 확인

  ```python
  model.wv['human']
  model.wv['klakasdfsdjf']
  ```

  > model.wv['human']
  > Out[16]: 
  > array([-0.00893843, -0.03338013, -0.0210454 ,  0.03005639, -0.03349178], dtype=float32)
  >
  > model.wv['klakasdfsdjf']
  > Out[17]: 
  > array([ 0.00078089, -0.01366953, -0.00929362,  0.00640602, -0.00282957], dtype=float32)

* ```python
  model.wv.vocab
  model.wv.vocab.keys()
  ```

* ```python
  model.wv.vectors_ngrams.shape
  ```

  > (2000000, 5)











* 참고: 

  >* 아마추어 퀀트, blog.naver.com/chunjein
  >
  >* 전창욱, 최태균, 조중현. 2019.02.15. 텐서플로와 머신러닝으로 시작하는 자연어 처리 - 로지스틱 회귀부터 트랜스포머 챗봇까지. 위키북스
