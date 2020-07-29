# NLP & DL

* 특수 목적이 아닌, **범용적(일반적)으로 쓰일 Word Embedding**을 만든다.
* embedding의 방법
  * 따라서 문장 속 단어의 맥락(의미)를 파악할 줄 안다.
  * 즉, semantic 방법을 사용한다.





## Word2Vec

* 특정 목적이 아닌 범용적인 목적으로 사용된다.

  * 방대한 양의 아무문서나 코퍼스를 학습하여 **단어들이 어떤 관계를 갖도록 벡터화(수치화)하는 기술**이다.

* 따라서 단어들의 의미가 범용적이다.

  방법: continuous back of word (CBOW) 등

* 사후적으로 결정되는 Word Embedding 와 다리 사전에 학습하여 단어의 **맥락을 참조**하여 **벡터화**한다.

  * 분포가설 이론 사용: 단어들의 분포를 통해 해당 단어의 의미를 파악한다, 란 뜻

* *CBOW 순서* > 

  1. 문장을 단어로 전처리한다.

  2. 학습 시킨다.

     1. 수치화하고 싶은 단어가 output 되도록 네트워크 구성

     2. 주변 단어를 input

        ex: (input) alic, bit 등 여러 개 ... →  hidden layer → (output) hurt

        따라서  hidden layer: 중간출력. Word2Vec 

* *Skip-Gram 순서* >
  1. 문장을 단어로 전처리한다.
  2. 학습 시킨다.
     * CBOW 를 거꾸로 한 것.
     * input 1개 , output 여러 개
       * (input) hurt → hidden layer → (output) alic, bit 등 여러 개 ...
  3. AE 배울 때, 모델 전체 학습 시킨 후, autoencoding 부문만 따로 빼서 목적에 맞게 돌린 것처럼 Skip-Gram도 그렇게 진행함.

* *원리* > 
  * 예를 들어 (input) hurt를 one-hot encoding해서 hurt의 위치(index) 를 파악한 후, 
  * (output) alic를 넣었을 때 hurt의 위치(index)를 찾도록 함

* 예측 시, 더 편리한 건 Skip-Gram.
  * 단어 하나만 넣으면 output이 나오므로

* *단점* >

  1. **동음이의어를 제대로 파악하지 못한다.**
     * 실제 word2vec의 위치를 계산할 때 가까운 거리에 있는 값들의 평균으로 계산하기 때문에
     * 그림: 63번 참고
     * 해결방법: ELMo
       * embedding 할 때, 문맥에 따라 가변적으로 vector를 만든다. 즉, 맥락화된 단어 임베딩.

  2. 출력층을 **'softmax'**를 사용해서 **계산량이 많다.**
     * softmax를 사용하는 이유: one-hot 인코딩 위해서
     * 그런데 softmax를 사용하기 위해선 전체 단어를 0~1사이의 값으로 표현하기 위해 전부 계산을 진행하는데, 이때 전체 단어가 3만 개 등지가 넘어가는 정도로 큰 vocab일 땐 계산량이 많다.
     * 해결 방법: Skip-Gram Negative Sampling(SGNS)
  3. **OOV**(Out Of Vocbulary)
     * 해결방법: FastText
       * FastText:
         * 빈도수가 적은 단어에 대해서도 OOV 문제에 해결 가능성이 높다
  4. **문서 전체**에 대해선 고려 못한다.
     * 해결방법: GloVe: 
       * 빈도기반(TF-IDF) + 학습기반(Embedding) 방법 혼용
       * TF-IDF: 문서 전체에 대한 통계를 사용하지만, 단어 별 의미는 고려하지 못한다는 단점과
         Word2Vec: 주변 단어만을 사용하기 때문에 문서 전체에 대해서는 고려하지 못한다.





* ### CODE
  * 소설 alice in wonderland에 사용된 단어들을 2차원 feature로 vector화 한다.

  * #### 그림으로 먼저 보기:

  * ![image-20200729173124866](markdown-images/image-20200729173124866.png)

  * ![image-20200729173147483](markdown-images/image-20200729173147483.png)

    > 네트워크에는 center값 넣음

  * ![image-20200729173218124](markdown-images/image-20200729173218124.png)

  * ![image-20200729173227834](markdown-images/image-20200729173227834.png)

    > * x값인 7을 input 했을 때 output이 y값으로 8이 나올 때의 네트워크다.
    >
    > * 2개 뉴런으로 줄였을 때의 latent layer를 전체 학습 후 따로 빼내고,
    > * 이때 나온 x좌표와 y좌표로 2D상의 plt에 그림으로 나타내면, 맥락 상 가까운 의미를 가진 단어들끼리 뭉쳐져 있음을 확인할 수 있다.

  * ![image-20200729173233038](markdown-images/image-20200729173233038.png)

  

  

  * #### code로 확인하기

  * 패키지 불러오기

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    import matplotlib.pyplot as plt
    import nltk
    import numpy as np
    import pandas as pd
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import string
    from nltk import pos_tag
    from nltk.stem import PorterStemmer
    import collections
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.models import Model
    ```

  

  * 전처리

    ```python
    def preprocessing(text): # 한 line(sentence)가 입력됨 
        
        # step1. 특문 제거
        text2 = "".join([" " if ch in string.punctuation else ch for ch in text]) 
        # for ch in text: 한 sentence에서 하나의 character를 보고, string.punctuation:[!@#$% 등]을 공백처리('')=제거 함  
        tokens = nltk.word_tokenize(text2)
        tokens = [word.lower() for word in tokens] # 위 제거에서 살아남은 것들만 .lower() = 소문자로 바꿔서 word에 넣어줌 
    
    	# step2. 불용어 처리(제거)
    	stopwds = stopwords.words('english')
    	tokens = [token for token in tokens if token not in stopwds] # stopword에 없는 것만 token 변수에 저장 
    
    	# step3. 단어의 철자가 3개 이상인 것만 저장 
    	tokens = [word for word in tokens if len(word)>=3] 
    
    	# step4. stemmer: 어간(prefix) 추출(어미(surffix) 제거)  ex: goes -> go / going -> go
    	stemmer = PorterStemmer()
    	tokens = [stemmer.stem(word) for word in tokens]
    
    	# step5. 단어의 품사 태깅(tagging)
    	tagged_corpus = pos_tag(tokens) # ex: (alic, NNP), (love, VB)
    
    	Noun_tags = ['NN','NNP','NNPS','NNS']
    	Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
    
    	# 단어의 원형(표제어,Lemma)을 표시한다 
    	## 표제어(Lemma)는 한글로는 '표제어' 또는 '기본 사전형 단어' 정도의 의미. 동사와 형용사의 활용형 (surfacial form) 을 분석
    	## 참고: https://wikidocs.net/21707
    	## 걍 형용사/동사를 사전형 단어로 만들었다 생각하기.... 
    	# ex: belives -> (stemmer)believe(믿다) // belives -> (lemmatizer)belief(믿음) 
    	# (cooking, N) -> cooking / (cooking, V) -> cook
    	## 한국어 예시:
        """
        lemmatize 함수를 쉽게 만들 수 있습니다. 
        띄어쓰기가 지켜진 단어가 입력되었을 때 Komoran 을 이용하여 형태소 분석을 한 뒤, 
        VV 나 VA 태그를 가진 단어에 '-다'를 붙입니다. 
        단, '쉬고싶다' 와 같은 복합 용언도 '쉬다' 로 복원됩니다.
        출처: https://lovit.github.io/nlp/2019/01/22/trained_kor_lemmatizer/
        """
    	lemmatizer = WordNetLemmatizer()
    	
    	# 품사에 따라 단어의 lemma가 달라진다 
    	# (cooking, N) -> cooking / (cooking, V) -> cook
    	def prat_lemmatize(token,tag):
        	if tag in Noun_tags:
            	return lemmatizer.lemmatize(token,'n')
        	elif tag in Verb_tags:
            	return lemmatizer.lemmatize(token,'v')
        	else:
            	return lemmatizer.lemmatize(token,'n')
    
    	pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])             
    
    	return pre_proc_text
    ```

  * 소설 alice in wonderland를 읽어온다.

    ```python
    lines = []
    fin = open("./dataset/alice_in_wonderland.txt", "r")
    for line in fin:
        if len(line) == 0: 
            continue # 소설 txt내 엔터 없애기
        lines.append(preprocessing(line))
    fin.close()
    ```

  * 단어들이 사용된 횟수를 카운트 한다.

    ```python
    counter = collections.Counter()
    
    for line in lines:
        for word in nltk.word_tokenize(line):
            counter[word.lower()] += 1
    ```

  * 사전을 구축한다.

    * 가장 많이 사용된 단어를 1번으로 시작해서 번호를 부여한다.

    ```python
    word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())} # ex: [(apple:50), (cat: 43), ...]
    idx2word = {v:k for k,v in word2idx.items()} # ex: [(50: apple), (43: cat), ...]
    ```

  * Trigram으로 학습 데이터를 생성한다.

    ```python
    xs = []     # 입력 데이터
    ys = []     # 출력 데이터
    for line in lines:
        # 사전에 부여된 번호로 단어들을 표시한다.
        ## 각 문장을 tokenize해서 소문자로 바꾸고 word2idx로 변환 
        embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)] # word2idx: value값인 index번호가 나옴 
        
        
        # Trigram으로 주변 단어들을 묶는다.
        ## .trigrams(=3)만큼 끊어서 연속된 문장으로 묶기 ex: triples = [(1,2,3), (3,5,3), ...]
        triples = list(nltk.trigrams(embedding))
        
        
        # 왼쪽 단어, 중간 단어, 오른쪽 단어로 분리한다. 
        w_lefts = [x[0] for x in triples]   # [1, 2, ...8]
        w_centers = [x[1] for x in triples] # [2, 8, ...13]
        w_rights = [x[2] for x in triples]  # [8, 13, ...7]
        
        # 입력 (xs)      출력 (xy)
        # ---------    -----------
        # 1. 중간 단어 --> 왼쪽 단어
        # 2. 중간 단어 --> 오른쪽 단어
        xs.extend(w_centers)
        ys.extend(w_lefts)
        xs.extend(w_centers)
        ys.extend(w_rights)
    ```

  * 학습 데이터를 one-hot 형태로 바꾸고, 학습용과 시험용으로 분리한다.

    ```python
    vocab_size = len(word2idx) + 1  # 사전의 크기 # vocab_size = 1787 # + 1 해줘야 밑에 ohe 할 때, vocab 끝까지 전부를 ohe 할 수 있음 
    
    ohe = OneHotEncoder(categories = [range(vocab_size)]) # ohe = OneHotEncoder(categories=[range(0, 1787)])
    X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense() # .todense = .toarray()와 동일함: 결과를 배열 형태로 변환 
    Y = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()
    ## X.shape = (13868, 1787) / y.shape = (13868, 1787)
    ```

  * 학습용/시험용 data로 분리 

    ```python
    Xtrain, Xtest, Ytrain, Ytest, xstr, xsts = train_test_split(X, Y, xs, test_size=0.2) 
    # xs를 또 쓴 이유? => 뒤에서 가까운 단어끼리 그림(plt) 그릴 때 쓰려고
    ```

    > shape 참고 > 
    >
    > np.array(xs).shape
    > Out[19]: (13868,)
    >
    > np.array(xstr).shape
    > Out[20]: (11094,)
    >
    > np.array(xsts).shape
    > Out[21]: (2774,)
    >
    > np.array(Xtrain).shape
    > Out[22]: (11094, 1787)
    >
    > np.array(Xtest).shape
    > Out[23]: (2774, 1787)
    >
    > np.array(Ytrain).shape
    > Out[24]: (11094, 1787)
    >
    > np.array(Ytest).shape
    > Out[25]: (2774, 1787)

  * 딥러닝 모델을 생성한다. 

    ```python
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    
    input_layer = Input(shape = (Xtrain.shape[1],), name="input") # shape = batch(None) 빼고 y feature의 shape만 넣어주면 됨 
    first_layer = Dense(300, activation='relu', name = "first")(input_layer)
    first_dropout = Dropout(0.5, name="firstdout")(first_layer)
    second_layer = Dense(2, activation='relu', name="second")(first_dropout)
    third_layer = Dense(300,activation='relu', name="third")(second_layer)
    third_dropout = Dropout(0.5,name="thirdout")(third_layer)
    fourth_layer = Dense(Ytrain.shape[1], activation='softmax', name = "fourth")(third_dropout)
                        # Ytrain.shape[1] = Xtrain의 shape과 동일해야 함 
                        # activation='softmax': one-hot이 출력되기 때문에 softmax여야 함 
    model = Model(input_layer, fourth_layer)
    model.compile(optimizer = "rmsprop", loss="categorical_crossentropy") 
    # loss="categorical_crossentropy": 만약 one-hot이 아니라, 숫자(vocab의 index)가 출력된다면 loss="sparse_categorical_crossentropy"
    ```

  * 학습

    ```python
    hist = model.fit(Xtrain, Ytrain, 
                     batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCHS,
                     validation_data = (Xtest, Ytest))
    ```

  * Loss history를 그린다

    ```python
    plt.plot(hist.history['loss'], label='Train loss')
    plt.plot(hist.history['val_loss'], label = 'Test loss')
    plt.legend()
    plt.title("Loss history")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    ```

    > ![image-20200729170655602](markdown-images/image-20200729170655602.png)

  

  

  * ### 단어들끼리의 거리를 그림으로 나타내는 code

  * Word2Vec 수치 확인

    ```python
    # Extracting Encoder section of the Model for prediction of latent variables
    # 학습이 완료된 후 중간(hidden layer)의 결과 확인: = Word2Vec layer확인. (word2vec: word를 vec(수치)로 표현. 근까 저번 쉅에서 w의 값을 .get_weight()해서 확인했을 때의 값이 나올 듯)
    encoder = Model(input_layer, second_layer)
    
    # Predicting latent variables with extracted Encoder model
    reduced_X = encoder.predict(Xtest) # Xtest 넣은 것처럼 임의의 단어를 입력하면 reduced_X = 해당 단어의 Word2Vec형태로 출력됨 
    
    
    # 시험 데이터의 단어들에 대한 2차원 latent feature(word2vec 만드는 layer)인 reduced_X를 데이터 프레임(표)으로 정리한다.
    final_pdframe = pd.DataFrame(reduced _X)
    final_pdframe.columns = ["xaxis","yaxis"]
    final_pdframe["word_indx"] = xsts # test 용이므로 train/test split 할 때 같이 나눴던 xstr, xsts 중 y값인 xsts 사용 
    final_pdframe["word"] = final_pdframe["word_indx"].map(idx2word) # index를 word로 변환함 
    
    # 데이터 프레임에서 100개를 샘플링한다.
    rows = final_pdframe.sample(n = 100)
    labels = list(rows["word"])
    xvals = list(rows["xaxis"])
    yvals = list(rows["yaxis"])
    ```

    > [final_pdframe] > 
    > Out[26]: 
    >          xaxis     yaxis  word_indx    word
    > 0     0.301799  0.000000         25    take
    > 1     0.590210  0.810300        468    pick
    > 2     0.672298  0.000000          1     say
    > 3     0.408792  0.520896          9    know
    > 4     0.387678  0.605502         30    much
    >        ...       ...        ...     ...
    > 2769  1.309759  0.851837         27    mock
    > 2770  0.000000  0.423953        622  master
    > 2771  0.196061  0.299570         83    good
    > 2772  0.000000  0.024289       1516  deserv
    > 2773  0.470771  0.550808        497    plan
    >
    > [2774 rows x 4 columns]

  * 샘플링된 100개 단어를 2차원 공간상에 배치

    * 거리가 가까운 단어들은 서로 관련이 높은 것

    ```python
    plt.figure(figsize=(15, 15))  
    
    for i, label in enumerate(labels):
        x = xvals[i]
        y = yvals[i]
        plt.scatter(x, y)
        plt.annotate(label,xy=(x, y), xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom', fontsize=15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    ```

    > ![image-20200729170642147](markdown-images/image-20200729170642147.png)















## Skip-Gram Negative Sampling(SGNS)

* Skip-Gram의 softmax를 활용했기 때문에 계산량이 많다는 단점을 sigmoid 사용하여 보완함

  * 값이 0~1사이 값이 아니라, 0 아니면 1인 이진 분류로 나옴 
  * 따라서 계산량 감소 

* 방법:

* Skip-Gram Negative Sampling: 

  1.  n-gram으로 선택한 단어 쌍에는 label = 1을 부여하고, 랜덤하게 선택한 단어 쌍에는 label = 0을 부여해서 이진 분류 

     > N-gram으로 선택한 단어 쌍은 서로 연관된 단어로 인식됨 

  2.  2개의 input에 각각의 input, target 값 입력

  3. 각각 vector 값 계산

  4. 두 값 concat(or dot or add) 

  5. sigmoid 계산하여

  6. label(0 or 1) 값이 나오게 

  7. 학습이 완료된 후에는 아래의 왼쪽 네트워크에 특정 단어를 입력하면, 그 단어에 대한 word vector를 얻을 수 있다.

  | Skip-Gram                                                    | Skip-Gram Negative Sampling                                  |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | input 1개<br />input: input data<br /><br />output: target data | input 2개<br />input[1] : input data<br />input[2] : target data<br />output: label |
  | label 無                                                     | label 有: 1 or 0으로 이루어져 있음(이진분류)<br />n-gram으로 선택한 단어 쌍에는 label = 1 <br />랜덤하게 선택한 단어 쌍에는 label = 0 |
  | 출력층: softmax 사용하여 0~1사이의 값                        | 출력층: sigmoid 사용하여 이진분류                            |
  | 거리 연산(cosine 등)을 하지 않는다. <br />latent layer에서 벡터 연산을 통해 나온 x,y 좌표로 그림을 그리던가 해서 <br />맥락 상 비슷한 의미를 가진 단어들을 찾아낼 수 있다. | 거리 연산을 할 수 있다. <br />두 개의 input 값에서 나온 vector 값을 하나로 합칠 때 dot 함수를 쓰면 거리 연산을 하는 것과 같다. 이때 cosine 거리 함수를 쓸 수도 있다. <br />그런데, concate 이나 add 함수를 쓰면 거리 계산을 못한다. |

  ![image-20200729151904052](markdown-images/image-20200729151904052.png)







* Google's trained Word2Vec model:
  * SGNS 방식
  * Pre-trained 방식
  * 문서 → Vector화(수치화) → 일반 DL로 바로 학습 가능