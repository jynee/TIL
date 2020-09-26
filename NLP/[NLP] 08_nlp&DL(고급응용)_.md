

# NLP 분야에서 딥러닝의 고급 응용



## 텍스트 자동 생성

예제 문장: I love you very much

* 문자 단위의 시계열 데이터 생성

  * 이렇게 되도록 생성: 보통의 시계열 Batch data처럼 생성
  | x          | y(one-hot encoding) |
  | ---------- | ------------------- |
  | I love you | (공백)              |
  | love you   | v                   |
  | ove you v  | e                   |
  | ...        | ...                 |
  
	>  y값을 word 단위가 아니고 characters(ex: a, b, c, ... z) 로 설정

* LSTM으로 학습

  * I love you를 넣어도 v가 나오게끔 신경망 속 activation은 softmax 함수를 쓴다.
    * compile 시, loss 함수는 categorical_crossentropy 사용

* 순서:

  1. 문장으로 이루어진 raw data 불러오기

  2. 전처리:

     1. word 아니고 character 단위로 분류
     2. 시계열 x data 생성. (위 예시의 x 처럼)
     3. Converting indices into vectorized format
        * X, Y를 np.zeros 

  3. Model Building

     * softmax: 출력층의 값이 [0.3, 0.4, 0.8] 등으로 나오면 이 총합이 1이 나오게끔 확률분포 다시 계산. 
       이때 나온 값들의 차이를 더 크게 조작하고 싶을 때, 베타가 들어간 식을 사용하여 계산. 
       이렇게 하면 model.predict(x) 시, 원하는 문자의 수치가 나올 확률이 높아진다
       (역으로 원하지 않는 단어가 나올 확률은 적어진다.)
     * softmax 함수를 쓰는 skip-gram은 계산량이 많단 단점이 있는데, 이를 SGNS가 보완한다.

  4. 예측치를 softmax 확률로 뽑아 다시 역연산(exp) 하는 함수 생성

     * np.random.multinomial로 sampling

       ```python
       def pred_indices(preds, metric=1.0):
           preds = np.asarray(preds).astype('float64')
           preds = np.log(preds) / metric
           exp_preds = np.exp(preds)
           preds = exp_preds/np.sum(exp_preds)
           probs = np.random.multinomial(1, preds, 1)
           return np.argmax(probs)
       ```

     	> * 다항 분포 (Multinomial distribution):
     	>
     	>   다항 분포는 여러 개의 값을 가질 수 있는 독립 확률변수들에 대한 확률분포로, 여러 번의 독립적 시행에서 각각의 값이 특정 횟수가 나타날 확률을 정의한다. 다항 분포에서 차원이 2인 경우 이항 분포가 된다.
     	>
     	>   > 출처: 위키백과. 다항분포. [https://ko.wikipedia.org/wiki/%EB%8B%A4%ED%95%AD_%EB%B6%84%ED%8F%AC](https://ko.wikipedia.org/wiki/다항_분포)

  5. Train & Evaluate the Model

     1. batch
        
        1. randint로 random하게 시작하도록 설정
     2. 확률 임의 설정 후 단어 생성

        * [0.2, 0.7,1.2] 처럼 

          ```python
        for diversity in [0.2, 0.7,1.2]:
          a = np.array([0.9, 0.2, 0.4])
b = 1.0
          e = np.exp(a/b)
           
          print(e/np.sum(e))
          ```
          
          >
          > '0.9'처럼 유독 하나의 값이 클 경우 다른 값들과의 차이가 더 커짐 
          >
          > | print(e/np.sum(e))<br/>[0.40175958 0.2693075  0.32893292]    | print(e/np.sum(e))<br/>[0.47548496 0.23611884 0.2883962 ]    |
          > | ------------------------------------------------------------ | ------------------------------------------------------------ |
          > | a = np.array([0.6, 0.2, 0.4])<br/>b = 1.0<br/>e = np.exp(a/b) | a = np.array([0.9, 0.2, 0.4])<br/>b = 1.0<br/>e = np.exp(a/b) |
        
     3. model.predict(x):

        * 예측(predict): model.predict(x) = > [0.01, 0.005, 0.3, 0.8 ...]

     4. 문자 추출:

        ```python
   sys.stdout.write(pred_char)
        sys.stdout.flush()
        ```
     
        







## `DMN`

* Dynamic Memory Networks

  * 아래 5가지의 N/W가 결합되어 있는 모습

  1. `Input Module`
  2. `Question Module`
  3. `Episodic Memory Module`
  4. `Answer Module`
  5. **`attention score N/W`** (FNN)

### `Ask Me Anything`

* Q → A: Question & Answering

* DL 사용
  
* 논문 저자는 GRU 사용
  
* 특징: Q&A를 기억하는 하나의 경험 단위인 Episode를 기억하는 장치가 있다.

* [x] 순서:

  1. Input 문장(text sequence)과 
  2. attention 연산이 들어간 question 받아 
     * attention 연산: attention score
  3. episodic memory 구성한 후, 
  4. 일반적인 답변을 줄 수 있게끔 구성한 네트워크

  * ![](markdown-images/image-20200926204603184.png)

    > 출처: Ankit Kumar외, 2016.05, Ask Me Anything: Dynamic Memory Networks for Natural Language Processing. 에서 'attention score' 추가

* [x] **attention process**

* attention score 계산

  * attention score
  * 여러 문장에 Epsodic stroy가 있고 이것에 대한 답을 찾을 때 질문과 가장 관련이 높은 (저장된) 문장에 점수를 매기기 위한 계산을 수행함 
  * 즉, 답을 내기 위해 어떤 문장에 attention을 해야 하는지 attention score로 계산하는 알고리즘

* 기계번역, test 분류, part-of-speech tagging, image captioning, Dialog system(chatbot) 가능

* mission: 주어진 Question의 의미를 파악할 수 있도록 네트워크를 구성해야 한다.

  * '의미를 파악할 수 있도록' : 조응어(Anaphora resolution) 해석.

  

* [x] `input module`

  1. Input에 넣을 문장들을 1행으로 붙이고, [EOS] 로 구분

     | 문장 1                          |       | 문장 2                    |       | 문장3                                             |
     | ------------------------------- | ----- | ------------------------- | ----- | ------------------------------------------------- |
     | When I was young, I passed test | <EOS> | But, Now Test is so crazy | <EOS> | Because The test level pretty hard more and more. |

  2. Embedding layer 투입

  3. RNN 거쳐서

  4. Hidden layer 출력은 다시 n개의 문장(c1, c2, c3 등)으로 출력

  5. episodic memory module 투입



* [x] `Question module`
  1. Question 문장 투입
  2. Embedding layer 투입
  3. RNN 거쳐서
  4. episodic memory module 투입



* [x] `episodic memory module`

  * input module(문장마다)+Question module+ atttention mechanism출력된 걸 반복해서 내부의 episodic memory를 반복 update

  * "어떻게 update?"

    1. input module의  embedding value과 atttention score 계산하여 RNN layer에 통과 시키기

       * 이때, `atttention score`: 
         atttention score layer의 출력층에서 나온 w인`g`를 input module의 출력값인 c1,c2,c3 등과 계산한 값

    2. Question module의 embedding value 값을 RNN layer에 통과 시키기

       * 이때, w = `Q`

    3. 2를 answer을 output할 Answer Module의 RNN layer에 통과 시킴

       * 이때, w = `m`

    4. episodic memory module의 RNN layer를 반복할 때마다 `attetion score `계산

    5. 그렇게 해서 나온 attetion score 값 중 가장 높은 것(g) 찾음

       > * `atttention mechanism`
       >
       > 1. 그렇게 해서 나온 attetion score 값 중 가장 높은 것(g) 찾고
       > 2. g로 다시 네트워크 형성
       >    * 2층 구조가 됨 
       >
       > * memory update mechanism
       >   * attention score로 가중 평균



* 용어: 

  * `c`: Input의 출력
  * `m`: episodic memory module 출력값이자 attention score의 입력값
  * `q`: question layer의 출력값
  * `g`: attention score의 출력값

  



### code:

![image-20200926204637030](markdown-images/image-20200926204637030.png)

* 순서

  1. **패키지 불러오기 **
  2. **전처리**
  * 2-1. Document Data processing(raw data)
  3. **데이터 불러오기**
  * 3-1. Raw Document Data 불러오기 
  * 3-2. train/test data split 해서 가져오기
  4. **vocab 만들기**
    * 4-1. Train & Test data를 한꺼번에 묶어서 vocab을 만듦
      * `collections.Counter()`
    * 4-2. word2indx / indx2word 만듦
      * `padding`
  5. **벡터화**
    * 5-1. vocab_size 변수 설정
      * `len(word2indx)`
    * 5-2. story와 question 각각의 max len 변수 설정
      * 뒤에서 padding 맞춰 주려고 max len 설정해줌
    * 5-3. 벡터화 시킴
      * `raw data`와 `word2indx`, 각 모듈(story, question)의 `maxlen`을 함수에 넣어 `padding`, `categorical` 등을 진행함
  6. **모델 빌드**
    * 6-1. train/test data split
    * 이때, Xstrain, Xqtrain, Ytrain = `data_vectorization`(data_train, word2indx, story_maxlen, question_maxlen) 이고,
      data_vectorization의 return 값은 
      * `pad_sequences(Xs, maxlen=story_maxlen)`
      * `pad_sequences(Xq, maxlen=question_maxlen)`
      * `to_categorical(Y, num_classes=len(word2indx))`
    * 6-2. Model Parameters 설정
    * 6-3. Inputs
    * 6-4. Story encoder embedding
    * 6-5. Question encoder embedding
    * 6-6. 모듈 만들어줌
      * Question module는 위에서 만들어준 걸로 사용함
        1. attention score layer
           * **dot**으로 만듦
        2. story module
           * 이 layer는 story layer의 input에서 시작하여 question layer는 건너뛰고 또 다른 embedding layer를 거쳐, 추후에 dot layer와 add를 **해주려고 만듦**
        3. episodic memory module
           * dot한 layer와 바로 위의 story_encoder_c를 **add**해서 만들어지게 됨
        4. answer module
           * episodic memory layer(response) + quetion layer
  7. **compile**

     > model = Model(inputs=**[story_input, question_input]**, outputs=output)
     >
     > * input에 story와 question 두 개 써줬단 것!
  8. **fit **
  9. **loss plot**
  10. **정확도 측정(predict)**
  11. **적용**





* 패키지 불러오기

  ```python
  import collections
  import itertools
  import nltk
  import numpy as np
  import matplotlib.pyplot as plt
  import random
  from tensorflow.keras.layers import Input, Dense, Activation, Dropout
  from tensorflow.keras.layers import LSTM, Permute
  from tensorflow.keras.layers import Embedding
  from tensorflow.keras.layers import Add, Concatenate, Dot
  from tensorflow.keras.models import Model
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  from tensorflow.keras.utils import to_categorical
  ```





#### 전처리

* Raw Document Data processing 

  ```python
  # 문서 내용 예시 : 3문장의 story(episodic story)
  # 현재까지 NLP는 한 문장 안에서 단어들의 의미를 파악하고 이를 통해 한 개의 문장을 분석하는 수준에 그쳐있다(step1 수준).
  # 이때, episodic story는 '한 문장 안'이 아니라 문장 '간'의 단어들의 관계(=문장 간의 관계)를 파악하는 데에 의의가 있으며 따라서 매우 분석이 어렵다(이는 시의 영역이다. step2). 나아가 문단(Paragrape) 간의 관계까지 파악할 필요가 있다(이는 소설의 영역이다. step3).  
  """
  data 생김새
  # 1 Mary moved to the bathroom.\n
  # 2 Daniel went to the garden.\n
  # 3 Where is Mary?\tbathroom\t1 
  """
  ## Question과 answer은 #\t : tab 으로 구분되어 있다.
  # Return: # 3개(Stories, question, answer)를 return 해줌 
  # Stories = ['Mary moved to the bathroom.\n', 'John went to the hallway.\n']
  # questions = 'Where is Mary? '
  # answers = 'bathroom'
  #----------------------------------------------------------------------------
  def get_data(infile):
      stories, questions, answers = [], [], []
      story_text = []
      fin = open(infile, "r") 
      for line in fin:
          lno, text = line.split(" ", 1)
          if "\t" in text: # >data 생김새<에서 \t가 있는 3번을 말하는 것임 
              question, answer, _ = text.split("\t") #\t으로 구분해서 quetion과 answer 구분  # 숫자(ex:1)
              stories.append(story_text) 
              questions.append(question)
              answers.append(answer) # >data 생김새< 에서 3번의 \t 앞의 answer문 
              story_text = []
          else:
              story_text.append(text) # 사실상 해당 함수는 else 부터 시작하는 것. 
      fin.close()
      return stories, questions, answers
  ```





#### 데이터 불러오기

* Raw Document Data 불러오기 

  ```python
  Train_File = "./dataset/qa1_single-supporting-fact_train.txt"
  Test_File = "./dataset/qa1_single-supporting-fact_test.txt"
  ```

* get the data

  ```python
  data_train = get_data(Train_File) # 출력: stories, questions, answers
  data_test = get_data(Test_File)
  print("\n\nTrain observations:",len(data_train[0]),"Test observations:", len(data_test[0]),"\n\n")
  ```

  > Train observations: 10000 Test observations: 1000 

  



#### vocab 만들기

* Building Vocab dictionary from Train & Test data 

  * Train & Test data를 한꺼번에 묶어서 vocab을 만듦 

  ```python
  dictnry = collections.Counter() # collections.Counter() 이용하여 단어들이 사용된 count 조회할 예정 
  for stories, questions, answers in [data_train, data_test]:
      for story in stories:
          for sent in story:
              for word in nltk.word_tokenize(sent):
                  dictnry[word.lower()] +=1
      for question in questions:
          for word in nltk.word_tokenize(question):
              dictnry[word.lower()]+=1
      for answer in answers:
          for word in nltk.word_tokenize(answer):
              dictnry[word.lower()]+=1
  ```

* word2indx / indx2word 만듦

  ```python
  # collections.Counter()과 구조는 같은데, 단어 index는 1부터 시작하게 바꿔줌.  
  word2indx = {w:(i+1) for i,(w,_) in enumerate(dictnry.most_common())} 
  word2indx["PAD"] = 0 # padding
  indx2word = {v:k for k,v in word2indx.items()} 
  # 위에서 word2indx["PAD"] 해줘서 print(indx2word) 하면, 맨 마지막에 ',0: 'PAD'' 가 들어가 있다. 
  ```







#### 벡터화

* `vocab_size` 변수 설정

  ```python
  vocab_size = len(word2indx) # vocab_size = 22 -> 즉 21개 단어만이 쓰인 것(하나는 패딩)
  print("vocabulary size:",len(word2indx))
  print(word2indx)
  ```

  > * vocabulary size: 22
  > * {'to': 1, 'the': 2, '.': 3, 'where': 4, 'is': 5, '?': 6, 'went': 7, 'john': 8, 'sandra': 9, 'mary': 10, 'daniel': 11, 'bathroom': 12, 'office': 13, 'garden': 14, 'hallway': 15, 'kitchen': 16, 'bedroom': 17, 'journeyed': 18, 'travelled': 19, 'back': 20, 'moved': 21, 'PAD': 0}

* story와 question 각각의 `max len` 변수 설정

  ```python
  story_maxlen = 0
  question_maxlen = 0
  
  for stories, questions, answers in [data_train, data_test]:
      for story in stories:
          story_len = 0
          for sent in story:
              swords = nltk.word_tokenize(sent)
              story_len += len(swords)
          if story_len > story_maxlen:
              story_maxlen = story_len # story 중 가장 긴 문장 찾기(=단어가 가장 많은 거)
              
      for question in questions:
          question_len = len(nltk.word_tokenize(question))
          if question_len > question_maxlen: 
              question_maxlen = question_len # question 중 가장 긴 문장 찾기 
              
  print ("Story maximum length:", story_maxlen, "Question maximum length:", question_maxlen)
  ```

  > Story maximum length: 14 Question maximum length: 4

* Converting data into `Vectorized` form 

  * 위의 문장을 수치화함

  ```python
  def data_vectorization(data, word2indx, story_maxlen, question_maxlen):  
      Xs, Xq, Y = [], [], []
      stories, questions, answers = data
      for story, question, answer in zip(stories, questions, answers):
          xs = [[word2indx[w.lower()] for w in nltk.word_tokenize(s)] for s in story] # vocab의 index로 단어를 표시한다(수치화한다)
          xs = list(itertools.chain.from_iterable(xs)) # chain.from_iterable(['ABC', 'DEF']) --> ['A', 'B', 'C', 'D', 'E', 'F']
          xq = [word2indx[w.lower()] for w in nltk.word_tokenize(question)]
          Xs.append(xs)
          Xq.append(xq)
          Y.append(word2indx[answer.lower()]) # Y = answer
      return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen),\
             to_categorical(Y, num_classes=len(word2indx))
             # 가장 긴 문장(maxlen=story_maxlen))을 기준으로 문장의 길이를 통일시킨다. 이것보다 짧은 부분은 padding(0)으로 채움
             # y: anwser이고, 여기선 한 단어로 나온다. 즉, 한 단어 = 숫자 1개 
             # to_categorical: 안 쓰고 sparse categorical을 써도 ok 
  ```

  > * 함수 data_vectorization() 中
  >
  > 1. xs = [[word2indx[w.lower()] for w in nltk.word_tokenize(s)] for s in story]
  >
  >    *xs* >
  >    Out[19]: [[8, 7, 20, 1, 2, 13, 3], [10, 19, 1, 2, 17, 3]]
  >    *story* >
  >    Out[20]: ['John went back to the office.\n', 'Mary travelled to the bedroom.\n']
  >
  > 2. xs = list(itertools.chain.from_iterable(xs))
  >    *xs* >
  >    Out[22]: [8, 7, 20, 1, 2, 13, 3, 10, 19, 1, 2, 17, 3]
  >
  > 3. Xs.append(xs) 해줌으로써 
  >    for문 통해서 output 된 것들을 list 형태로 축적해줌 
  >
  > 4. padding 해줌
  >    pad_sequences(Xs, maxlen=story_maxlen) # story_maxlen = 14 
  >    Out[31]: array([[ 0,  8,  7, 20,  1,  2, 13,  3, 10, 19,  1,  2, 17,  3]])
  >
  > * pad_sequences(Xs, maxlen=story_maxlen)
  >   Out[31]: array([[ 0,  8,  7, 20,  1,  2, 13,  3, 10, 19,  1,  2, 17,  3]])
  >
  > * pad_sequences(Xq, maxlen=question_maxlen)
  >   Out[32]: array([], shape=(0, 4), dtype=int32)
  >
  > * to_categorical(Y, num_classes=len(word2indx))
  > Out[33]: array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)
  >







#### 모델 빌드

* train/test data split 

  ```python
  Xstrain, Xqtrain, Ytrain = data_vectorization(data_train, word2indx, story_maxlen, question_maxlen)
  Xstest, Xqtest, Ytest = data_vectorization(data_test, word2indx, story_maxlen, question_maxlen)
  
  print("Train story",Xstrain.shape,"Train question", Xqtrain.shape,"Train answer", Ytrain.shape)
  print( "Test story",Xstest.shape, "Test question",Xqtest.shape, "Test answer",Ytest.shape)
  ```

  > *print* > 
  > Train story (10000, 14) Train question (10000, 4) Train answer (10000, 22)
  > Test story (1000, 14) Test question (1000, 4) Test answer (1000, 22)

  

* Model Parameters 설정

  ```python
  EMBEDDING_SIZE = 128
  LATENT_SIZE = 64
  BATCH_SIZE = 64
  NUM_EPOCHS = 40
  ```

* Inputs

  ```python
  story_input = Input(shape=(story_maxlen,)) # story_maxlen = 14
  question_input = Input(shape=(question_maxlen,))
  ```

* `Story encoder embedding`

  ```python
  story_encoder = Embedding(input_dim=vocab_size, # vocab_size: 22
                            output_dim=EMBEDDING_SIZE, # EMBEDDING_SIZE* = 128(한 단어를 128개의 vector로 표시, embedding layer의 colum 담당)
                            input_length=story_maxlen)(story_input) # story_maxlen = 14
  story_encoder = Dropout(0.2)(story_encoder)
  ```

* `Question encoder embedding`

  ```python
  question_encoder = Embedding(input_dim=vocab_size,
                               output_dim=EMBEDDING_SIZE,
                               input_length=question_maxlen)(question_input)
  question_encoder = Dropout(0.3)(question_encoder)
  ```





##### `attention score layer`

* attention score layer

  ```python
  match = Dot(axes=[2, 2])([story_encoder, question_encoder]) 
  ```

  > * Match between story and question: story and question를 dot 연산 수행.
  >   * 여기서 dot 연산은 attention score로 사용함 
  > * story_encoder = [None, 14, 128], question_encoder = [None, 4, 128]
  >   * match = [None, 14, 4]
  > * axes=[2, 2]?  story D2(=128 = embedding vector)와 question D2(=128 = embedding vector)를 dot 해라
  >   * 즉, (x, 128)과 (128,y)로 한쪽을 transpose 시켜서 연산 수행
  > * 우선 story input의 embedding layer의 출력은 story_encoder = (None, 한 story에 사용된 최대 단어 개수(=14), embedding vector(128)) 이다. 
  > * question input의 embedding layer의 출력은 question_encoder = (None, 한 question에 사용된 최대 단어 개수(=14), embedding vecotr(128)) 이다. 
  > * dot -> (None)을 빼고 (row, colum)끼리(=14, 128)과 (128, 14)가 연산 수행 



##### `story layer`

* story layer

  ```python
  story_encoder_c = Embedding(input_dim=vocab_size, # vocab_size = 22
                              output_dim=question_maxlen, # question_maxlen = 4 
                              input_length=story_maxlen)(story_input) # story_maxlen = 14 
  
  story_encoder_c = Dropout(0.3)(story_encoder_c) # story_encoder_c.shap=(14, 4)
  ```

  > * 이 layer는 story layer의 input에서 시작하여 question layer는 건너뛰고 또 다른 embedding layer를 거쳐, 추후에 dot layer와 add를 하게 됨  





##### `episodic memory layer`

* episodic memory layer

  ```python
  response = Add()([match, story_encoder_c]) # dot한 layer와 바로 위의 story_encoder_c를 add함 => (14, 4)
  response = Permute((2, 1))(response) # 결론 shape = (4, 14) # Permute((2, 1)): (D2, D1)으로 transpose. permute가 transpose보다 더 축이동이 자유로움 
  ```





##### `answer layer`

* episodic memory layer(response) + quetion layer

  ```python
  answer = Concatenate()([response, question_encoder])
  answer = LSTM(LATENT_SIZE)(answer) # LATENT_SIZE = 64
  answer = Dropout(0.2)(answer)
  answer = Dense(vocab_size)(answer) # shape=(None, 22) # 마지막 dense는 vocab_size=22(단어들의 총 개수)로!
  output = Activation("softmax")(answer) # shape=(None, 22)
  ```






##### compile

* 모델 빌드 마지막

  ```python
  model = Model(inputs=[story_input, question_input], outputs=output) # 합쳤으니 input을 []로 써주는 것 
  model.compile(optimizer="adam", loss="categorical_crossentropy") # 처음에 to_categorial 안 해줬으면 loss="sparse_categorical_crossentropy" 해야함 
  print (model.summary())
  ```






##### fit

* 모델 학습

  ```python
  # Model Training
  history = model.fit([Xstrain, Xqtrain], [Ytrain], # Ytrain: answer
                      batch_size = BATCH_SIZE, 
                      epochs = NUM_EPOCHS,
                      validation_data=([Xstest, Xqtest], [Ytest])) # ytest??? fit 하면 0,0,0 이던 게 13,14 등지로 바뀜
  ```

  > * Ytest.shape
  >   Out[78]: (1000, 22)
  > * Ytest
  >   Out[79]: 
  >   array([[0., 0., 0., ..., 0., 0., 0.],
  >          [0., 0., 0., ..., 0., 0., 0.],
  >          [0., 0., 0., ..., 0., 0., 0.],
  >          ...,
  >          [0., 0., 0., ..., 0., 0., 0.],
  >          [0., 0., 0., ..., 0., 0., 0.],
  >          [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
  >
  > * ytest.shape
  >   Out[87]: (1000,)
  > * ytest
  >   Out[92]: 
  >   array([15, 12, 16, 15, 16, 15, 14 ... ])





#### loss plot

* loss plot

  ```python
  plt.title("Episodic Memory Q & A Loss")
  plt.plot(history.history["loss"], color="g", label="train")
  plt.plot(history.history["val_loss"], color="r", label="validation")
  plt.legend(loc="best")
  plt.show()
  ```






#### 정확도 측정

* get predictions of labels

  ```python
  ytest = np.argmax(Ytest, axis=1)
  Ytest_ = model.predict([Xstest, Xqtest])
  ytest_ = np.argmax(Ytest_, axis=1)
  ```





#### 적용

* 적용

  * Select Random questions and predict answers

  ```python
  NUM_DISPLAY = 10
     
  for i in random.sample(range(Xstest.shape[0]),NUM_DISPLAY):
      story = " ".join([indx2word[x] for x in Xstest[i].tolist() if x != 0])
      question = " ".join([indx2word[x] for x in Xqtest[i].tolist()])
      label = indx2word[ytest[i]]
      prediction = indx2word[ytest_[i]]
      print(story, question, label, prediction)
  ```






* >  코드 출처: 크리슈나 바브사 외. 2019.01.31. 자연어 처리 쿡북 with 파이썬 [파이썬으로 NLP를 구현하는 60여 가지 레시피]. 에이콘

  









## 출력층 

* 출력층이 0 or 1 처럼 하나일 때

  * Binary classification. 따라서 sigmoid - binary-crossentropy 사용

  * | y    | yHat |
    | ---- | ---- |
    | 0    | 0    |
    | 0    | 1    |
    | 1    | 1    |

    > 정확도: 2/3

* 출력층이 두 개 이상 나올 때

  * multi-classification. 따라서 softmax - categorical-crossentropy 사용

  * one-hot 구조

  * | y     | yHat  |
    | ----- | ----- |
    | 0 1 0 | 0 1 0 |
    | 0 0 1 | 0 1 0 |
    | 1 0 0 | 1 0 0 |

    > 정확도: 2/3

* 출력층에 '1'이 여러 개인 구조. one-hot 구조가 아닐 때

  * multi-labeled classification. 따라서 sigmoid - binary-crossentropy 사용

  * 입력 뉴런 각각에 대해 binary-classification 해야 함

  * | y     | yHat  |
    | ----- | ----- |
    | 0 1 0 | 0 1 0 |
    | 0 0 1 | 0 1 0 |
    | 1 0 0 | 1 0 0 |

    > 정확도: 9개 중 7개 맞춤.  따라서 7/9
    >
    > * 위에처럼 row 전체가 다 맞았을 때 맞았다고 보는 게 아니고, row+colum으로 각각 개별로 봄 

