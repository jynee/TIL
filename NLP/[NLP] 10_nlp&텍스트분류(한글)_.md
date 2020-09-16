



# 텍스트 분류



* Skip-Gram
* SGNS
* Hirarchical softmax





## `Hirarchical softmax`

* 연산이 많아진단 단점의 softmax를 개선하여 Binary Tree 사용
* Binary Tree
  * iForest 알고리즘, DB indexing
  * 각 트리마다 나눠서 연산을 함 
* 순서
  * vocab 생성 단계에서 단어들을 sort
  * 출력층을 binary (huffman) tree
  * y와 yHat 사이의 차이를 줄이는 방향으로 역전파 
    * 기존 softmax는 yHat(확률분포 형태)를 구하는 부분에서 계산량이 많다. 이걸 binary tree를 사용하는 것.
    * 그렇게 되면 기존 yHat은 0.01, 0.03 ... 등으로 출력됐는데, 0.6, 0.42 등으로 출력되어 기존의 방법보다 계산량이 작아져 속도도 빨라진다.
    
      

* word2vec
  * **의미 공간이라는 고차원의 공간에 각 단어의 좌표값(벡터)을 부여한다**
  * 기준점이 달라지면 숫자도 달라지는 상대적인 거리로 단어 간의 유사도/차이를 분석할 수 있다.

    



# `PV-DM`

* paragraph vector
* `doc2vec`에 사용됨
  * distributed word representation(분포확률/가설 이론 사용)
    * 한 문장에 같이 쓰인 단어들끼리의 유사성이 높을 것이란 가설 이론
* word vector 가 아닌 paragraph 단위의 vector 표현
  * paragraph : 문장, 문서 단위
* word2vec → (발전 시킴) Doc2vec
* USL
* 유사도 계산에 순서 고려까지 하는 알고리즘 

![image-20200805181033974](markdown-images/image-20200805181033974.png)

> 그림 출처: 아마추어 퀀트, blog.naver.com/chunjein

* concat 후 lstm 넣으면 정보가 조금은 유실된 채로 넣어지는 것이라 우리는 지금까지 embeding layer 거친 것을 바로 lstm으로 넣었음.





* 원리: 
  * 문장 -1 "doc#1" : the cat sat on the table 가 있을 때,

  * 문장마다 unique한 ID(Paragraph id)를 붙여놓고 이것도 하나의 문장으로 취급함.
    * `paragraph vector `
      * 이것들이 추후엔 word가 됨
      * 즉, paragraph token은 또 다른 word라 생각하면 됨 
      * 모든 문장의 paragraph vector를 공유함 

* 따라서

  * concat으로 합쳐져서 y 값이 나오도록 역전파
  * x의 fix value는 3, y는 1라면, window size = 4로 설정

  | context      | x(vocab의 index로 입력) | y(vocab의 index로 출력) | missing value |
  | ------------ | ----------------------- | ----------------------- | ------------- |
  | doc - # 1(7) | the(1), cat(3), sat(4)  | on(13)                  |               |
  | doc - # 1(7) | cat(3), sat(4), on(13)  | the(8)                  | the(1)        |

  이렇게 학습 시킨다.

  > *이렇게 학습: Distributed memory model of paragraph vector(PV-DM): missing된 값을 기억한다

* 학습 후에는 paragraph vector가 1개의 colum vector(word vector)가 됨

* 단점: 학습 단계는 괜찮은데, prediction 단계에서는 새 문장에 paragraph id를 알 수 없어 값을 넣지 못한다는 점

* 해결: inference step(추론 단계) 필요

  * 학습에 사용되지 않은 문장에 대해서는 inference step이 필요하다.
  * GD 방법에 의해 얻을 수 있다.

* 따라서 알고리즘 순서: 

  * 2개의 주 Stage가 있다.
    1. W, D, U 결정(학습단계) 및 고정
    2. **inferece stage**가 필요
       * GD(학습) 통해 알아냄. 
         * 위에서 학습 완료된 W(embedding layer 거친 word vector), U(softmax 가중치), b(softmax의 bias) 고정 시킴
         * 일반적으론 model.fit(x, y)로 학습 한 후, model.predict(x)로 예측했는데 
         * inference 때에는 D(paragraph Matrix) 값을 모르니 D를 random하게 둔다
           * 따라서 model.fit(x, y)로 학습 한 후, model.predict(new x) 수행
           * advantage paragraph vector 

  > PV-DM에서는 다음과 같은 크게 두 가지 단계를 거쳐서 문장의 발화 의 도를 예측한다. 
  > 1) 주어진 데이터로 W(word vectors), U(softmax weights), b(softmax weights), D(paragraph vectors)를 학습한다. 
  > 2) 새로운 문장에 대해서 이미 학습했던 W, U, b는 유지하면서 D에 column을 추가하고 기울기 하강기법(gradient descent)을 적용하여 새로운 문장에 대한 벡터(D’)을 만들게 된다. 
  > 두 번째 단계에서 얻은 D’은 인식기를 통해 미리 정의한 발화 의도 집합 중 하나로 분류된다
  >
  > 출처: 최성호, 김은솔, 장병탁. 2016. Paragraph Vector를 이용한 문장 의도 예측 기법. 2016년 한국컴퓨터종합학술대회 논문집






## `PV-DBOW	`

* PV-DM과 반대 





## CNN & LSTM

![image-20200805175130043](markdown-images/image-20200805175130043.png)

> 그림 출처: 아마추어 퀀트, blog.naver.com/chunjein



* `Word2vec` : LSTM/CNN
  
  * 위 그림 中 1번 그림이다.
  
  * 1개 문장 안에서 단어들의 sequence 분석한다.
    * 즉, 순서/흐름에 주목한 모델이다.
* 부족한 부분은 padding 처리하며 time step을 통일 시켜주었다.
    * 근데 time step의 크기를 굳이 통일시킬 필요가 있나? 어떤 batch는 부족하면 안 되나? 
      
      * 부족해도 된다. 경우에 따라 padding을 써주는데 쓰는 게 일반적이다. 
      
      
  
* `Doc2vec`: 

  * 위 그림 中 2번 그림일 때, batch가 문장 1개라 sequence 분석이 불가능

    * 기계가 입력값을 기억하지 못한다.

    * 인공지능 로봇으로 따지면, 

      나: "지니야 내가 방금 뭐라고 그랬지?"

      기계: "무슨 말씀인지 모르겠어요."

  * 따라서 3번 그림. `Episodic stroy `개념 사용하여 episode가 batch 역할을 한다. 

    * 기계가 입력값을 기억할 수 있다. 

    * 인공지능 로봇으로 따지면, 

      나: "지니야 내가 방금 뭐라고 그랬지?"

      기계: "오늘 날씨가 어떠냐고 물어봤죠?"

    * 현재 개발이 되고는 있지만 쉽지 않은 영역이라 발전은 더딘 편이다.
      * 만약 개발이 된다면, A 작가의 40년치 소설을 모델에 넣어 학습 시켜 가지고 그 작가의 문체적 특징을 파악할 수 있지 않을까*?*

  * 원리: 

    1. 1개 episodic내에서 문장(문단)의 sequence 분석 시행
    2. LSTM layer에 입력

    * chatbot의 경우: 대화 내용의 흐름이 존재. 이걸 학습하기 위해 (내가 챗봇과 대화하는 하나의 세션(대화주제) 개념으로의)episodic story 활용 

    * 이때, time step을 굳이 지정해서 길이를 통일시키지 않아도 된다. 

      > x.shape = (None, 문장 개수, vector_size)
      >
      > xInput = Input(batch_shape = (None, **None**, vector_size))
      >
      > * 다만, 입력된 문장 개수에 따라 recurrent 횟수가 가변적이다.
      > * 추후 학습 시: model.fit(x,y, batch_size=1) 필요.
      > * 단점: 학습 시간이 오래 걸린다.









* 참고: 

  >* 아마추어 퀀트, blog.naver.com/chunjein
  >
  >* 코드 출처: 전창욱, 최태균, 조중현. 2019.02.15. 텐서플로와 머신러닝으로 시작하는 자연어 처리 - 로지스틱 회귀부터 트랜스포머 챗봇까지. 위키북스

