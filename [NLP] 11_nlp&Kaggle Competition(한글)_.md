

# AUC

* Area under the roc curve (AUC)

* Confusion matix(Binaray classification)

* |          | prediction<br />P | <br />N |
  | -------- | ----------------- | ------- |
  | actual P |                   |         |
  | N        |                   |         |

* TPR 
* FPR 

* Thes(임계치)가 작을수록 TPR이 높아진다.
* ROC 면적이 클수록 좋다. => Area Under the ROC(AUC)
  
  * AUC 값을 0~1 사이의 값으로 표현





## Kaggle Competition

* 논문 분석: 
  * Alejandro Peláez외. Sring 2015. Sentiment analysis of IMDb movie reviews. Rutgers University 



### `Negation Handling`

* *새로운 접근* > **Negation Handling**
  * hardly good → **neg**.good



### **`Mutual information(상호 정보량)`**

* (쿡북에서) 사전 생성할 때, 전체 단어 中 6,000개만 썼다.

  * 6,000개: most common() 빈도가 가장 높은 6,000개의 단어를 선택하여 vocab을 만들고, 이걸 가지고 문서 전처리를 했었음.
    * 이때 핵심: **빈도가 가장 높은**
    * 단어 빈도수를 수동 카운트
      * collection.counter()
      * counter.most_common()

  * 하지만 이 분은 Mutual information(상호정보량)이라는 개념을 사용했다.

* *새로운 접근* > **Mutual information**

  * class인 Y와 일정한 관계가 높은 순서를 vocab 생성

  * 순서:

    1. review 전처리

       | review        | y    |
       | ------------- | ---- |
       | I like you    | 1    |
       | I dislike you | 0    |
       | ...           | ...  |

    	

    * 모든 단어의 Mutual information 계산

      | 단어(Unigram) | y    | MI(공식에 의해 연산) |
      | ------------- | ---- | -------------------- |
      | I             | 1    | 0.1                  |
      | I             | 0    |                      |
      | like          | 1    | 0.8                  |
      | dislike       | 0    | 0.7                  |

      > * "I"와 y는 무관 → x, y는 독립
      >
      >   * P(x, y) = P(x|y) * P(y)
      >
      >   'like, dislike'와 y는 연관 → x, y는 종속 
      >
      >   * P(x, y) = P(x) * P(y)
      >
      > * ML에서 Entropy 공부 할 때, KL값으로 Cross Entropy 값을 구했었다.
      >
      >   * KL : 두 분포의 정보량의 차이(두 분포의 유사성)
      >
      > * Mutual Information의 KL
      >
      >   * KL(P(x, y) || P(x) * P(y))
      >
      >   * 예: MI('like') = ?
      >
      >     * P(x, y) = P(x|y) * P(y) = > 
      >
      >     * P(x|y = 0) * P(y = 0)  + P(x|y = 1) * P(y = 1)
      >
      >     * P(like|y = 0) * P(y = 0)
      >
      >       P(like|y = 0) : label(y)가 0(neg)인 리뷰 中 'like' 단어가 등장한 비율
      >
      >       P(y = 0) : 리뷰 25,000개 中 y=0(neg)인 비율
      >
      >     * 따라서 MI('like') = 0.8
      >
      > * 'I'의 경우 긍/부정이 섞여 있어 MI 수치가 적다. 
      >
      > * 'MI 수치가 적다'는 말은, 긍/부정이 명확하지 않다는 뜻이고, 이는 리뷰의 긍/부정을 분류하기 애매하다는 뜻이므로 실질적으로 목적을 이루는 데엔 도움이 되진 않는단 뜻이다.

    2. MI 높은 순서로 Sort

    	| 단어    | MI   |
    | ------- | ---- |
    | like    | 0.8  |
    | dislike | 0.7  |

    4. 상위 N% 선택하여 vocab 생성

       * 상위만 선택한 이유: 분석에 별로 도움되지 않는 단어도 vocab에 포함해 생성해봤자 연산량만 늘어나고 분석에 실질적인 도움은 되지 않으니까 후순위로 밀어낸다.
       
    4. TF - IDF

       * 예: movie → 여러 리뷰에 등장 DF 가 높다 IDF가 낮다

    5. 감정분석

       * 학습 기반(신경망 속 Embedding)이 아닌 사전 기반의 감정분석(VADER 알고리즘) 사용
         * VADER 알고리즘
           * 리뷰 속 단어들을 사전에 등재된 단어에 따라 VADER Score를 매기고, 그 하나의 리뷰의 VADER Score 값을 평균낸 것
       * 각 단어마다 score를 매겨 양수/음수에 따라 positive and negative를 구분한다
         * score 식 > label = 1인 리뷰 中 'like'의 개수 - label = 0인 리뷰 中 'like'의 개수
         * 평균, top k, bottom 값을 matrix로 해서 구해봄

    6. word2vec

       * 단어 vector들을 평균 내서 이걸 문장 vector로 쓰는 것
         * 문제: 정보가 손실되어 별로 좋은 방법은 아니다.

    7. doc2vec

       * 단어를 50~60% 선택할 때 score가 가장 좋았다.
       * 하지만 대체적으로 많은 단어를 선택할 때 score가 좋아지긴 한다.
       * doc2vec + TF-IDF : 의미적관계+구조적 관계를 보기 위해 CONCAT or average 등을 사용하여 두 값을 합침

    8. 감정분석

       * 학습 기반

    9. 결과 ROC : 0.99259

       









# 사전 만들기



* 패키지 다운 경로: C:\Users\jynee\.conda\envs\multi_data\Lib\site-packages\konlpy\java
  * 그 中 사전 파일: open-korean-text-2.1.0.jar
  * open-korean-text-2.1.0.jar ← 여기에 단어 등록하기

* 단어 등록 방법

  * 작업 폴더 생성

    * 폴더 이름: aaa ←  라고 생성함 

  * cmd 창에다가

    * cd C:\Users\jynee\.conda\envs\multi_data\Lib\site-packages\konlpy\java\aaa 입력

    * jar xvf ../open-korean-text-2.1.0.jar 입력

      > * jar xvf  :묶음파일 풀기
      >   ../ : 이전 폴더에 있는
      >   open-korean-text-2.1.0.jar : 이 파일을 풀어라
      > * jar cvf : 묶기
      > * \* : 와일드카드 

  * 사전들이 있는 폴더: C:\Users\jynee\.conda\envs\multi_data\Lib\site-packages\konlpy\java\aaa\org\openkoreantext\processor\util

  * 그 中 명사 사전이 들은 파일: C:\Users\jynee\.conda\envs\multi_data\Lib\site-packages\konlpy\java\aaa\org\openkoreantext\processor\util\noun

    * 이름 정의된 파일: 'names.txt' or 'company_names.txt'
      * 여기에 단어 하나 추가해보기
        * '이자용' 이라고 추가하였음

  * cmd 창에 jar cvf open-korean-text-2.1.0.jar * 입력

  * 다시 'aaa' 폴더로 가기: C:\Users\jynee\.conda\envs\multi_data\Lib\site-packages\konlpy\java\aaa

  * 'aaa' 폴더에 생긴 open-korean-text-2.1.0.jar ← ctrl+c해서, 상위 폴더인 'java'에 붙여넣기(ctrl+v)

  * 적용 완료







* 참고: 

  >* 아마추어 퀀트, blog.naver.com/chunjein
  >

