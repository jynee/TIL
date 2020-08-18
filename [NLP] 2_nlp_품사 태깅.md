---
title: (NLP 기초) 품사 태깅
date: 2020-07-16
update: 2020-08-16
tags:
  - NLP
  - 기초
---





# NLP

* 품사 태깅 원리
* HMM

<br>

<br>

## 품사 태깅

* **`품사 태깅`**: 문장의 N, V, ad, av 판별
  * 문장만 보고 품사를 붙여주는 기계:  **`pos tagger`**
* 문맥 = '문장 내' 주변 단어 = **`Context`**
  * 현재 NLP 상에선 문장 간, 절 간 Context는 불가
* "NLP 분석 시, 몇 개의 Context를 창조할 것인가?"
  * **`n-gram`**:
    * 1개: **`unigram`**
    * 2개: **`Bigram` **
      * ex: (I love) , (love you)
    * 3개: **`Trigram`**
    * 4개: ...
* 분석할 문장의 올바른 품사를 결정하기 위해선(올바른 tagger 기계를 만들기 위해선) 사전에 올바른 품사가 정의된 문서 코퍼스(말뭉치)가 있어야 한다.
  * nltk: 영어용
  * konlpy: 한글용 

<br>

## Tagging 거치는 원리

1. 사람이 학습 문서에 품사를 태깅해 놓았음: Tagged Corpora  `trainX`
2. 학습  `model.fit`
3. 모델 파라미터  `machine`
4. POS Tagger 완성 
5. 추후 input text 입력 시  `test X`
6. POS Tagger 거치면 `model.predict(testX)`
7. Tagging 돼서 출력됨 

<br>

<br>

## HMM

- 히든 마코프 모델(HMM): sequence를 분석

- 1차 Markov Chain:

  : 현재 상태는 직전 상태에만 의존한다

- 2차 Markov Chain:

  : 현재 상태는 전전 상태에만 의존한다<br>

- `Hidden Markov Model(HMM)`

  - 관측데이터(주가, 수익률, 거래량 ,변동성, 등...)에 직접 나타나지 않는 히든 상태(Hidden State)가 있다
  - 이때, **HMM은 관찰 데이터를 가지고 Hidden 상태를 추론하는 것**
  - MLE 개념 사용
  - 용어 정리: 
    - `초기상태` = `초기확률`
    - Hidden State에서 행동 변화가 일어날 확률 `천이확률`
    - 상태 변화가 일어나는 확률: `출력확률`
  - 알고리즘 정리:
    - `Forward 알고리즘`: X가 나올 확률 계산
    - `Viterbi decoding 알고리즘`: Z 추정
      * **`Forward 알고리즘`은 '확률' 계산이고, `Viterbi decoding 알고리즘`은 '시퀀스' 추정임**
    - `Baum Welch 알고리즘`: Z 추정
      - **`Viterbi 알고리즘` 과의 차이점: `Baum Welch 알고리즘`는 사전에 주어진 게 X 밖에 없음**

<br>


#### `Forward 알고리즘`

- Evaluation Question 문제에서 활용

  1. `초기 확률(초기 상태)`, 
  2. `Transition 확률(천이확률)`, 
  3. `Emission 확률(출력확률)`이 주어졌을 때, 
  4. 관측 데이터가 발생할 **확률**을 `Forward 알고리즘`으로 계산(추정)한다.
     * **Forward알고리즘은 '확률' 계산이고, Viterbi decoding 알고리즘은 '시퀀스' 추정임**	


<br>

```python
import numpy as np
from hmmlearn import hmm
```

<br>

- 히든 상태 정의

```python
states = ["Rainy", "Sunny"]
nState = len(states)
```

<br>

- 관측 데이터 정의

```python
observations = ["Walk", "Shop", "Clean"]
# nObervation = len(observations)
```

<br>

- HMM 모델 빌드

```python
model = hmm.MultinomialHMM(n_components=nState) # n_components = 2개 
model.startprob_ = np.array([0.6, 0.4]) # 초기확률(상태)
model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]]) # 천이확률 Transition 
model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]) # 출력확률 Emission
```

>  Multinomial(다항분포): 여러 개의 값을 가질 수 있는 독립 확률변수들에 대한 확률분포

<br>

- `X`: 관측 데이터 시퀀스(Observations Sequence) 
  - ''이렇게 X가 나오도록 Z 값'' 계산하라 中 ''이렇게 X가 나오도록'' 담당 

```python
X = np.array([[0, 2, 1]]).T  # Walk(0) -> Clean(2) -> Shop(1)
```

<br>

- `Forward 알고리즘`: `'.score()'`
  - x가 관측될 likely probability(가능성, 확률) 계산

```python
logL = model.score(X) # Forward 알고리즘. sequnce 값이 커지면 확률값이 굉장히 작아져 '0.00..' 등으로 나오니까 defaulf로 log 함수를 취해줌 
p = np.exp(logL) #log를 exp 씌워주면 일반 확률로 변환됨 
print("\nProbability of [Walk, Clean, Shop] = %.4f%s" % (p*100, '%'))
```

> Probability of [Walk, Clean, Shop] = 3.1038%

<br>

#### `Viterbi 알고리즘`

- Decoding Question 에서 활용
  1. `초기 확률(초기 상태)`, 
  2. `Transition 확률(천이확률)`, 
  3. `Emission 확률(출력확률)`,
  4. `관측 데이터 시퀀스(X)`가 주어졌을 때, 
  5. **`히든 상태의 시퀀스(Z)`** 을    `Viterbi decoding 알고리즘`으로 계산(추정)한다.
     * **Forward알고리즘은 '확률' 계산이고, Viterbi decoding 알고리즘은 '시퀀스' 추정임**

<br>

```python
import numpy as np
from hmmlearn import hmm
```

<br>

- 히든 상태 정의

```python
states = ["Rainy", "Sunny"]
nState = len(states)
```

<br>

- 관측 데이터 정의

```python
observations = ["Walk", "Shop", "Clean"]
# nObervation = len(observations)
```

<br>

- HMM 모델 빌드

```python
model = hmm.MultinomialHMM(n_components=nState) # n_components = 2개 
model.startprob_ = np.array([0.6, 0.4]) # 초기확률(상태)
model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]]) # 천이확률 Transition 
model.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]) # 출력확률 Emission
```

>  Multinomial(다항분포): 여러 개의 값을 가질 수 있는 독립 확률변수들에 대한 확률분포

<br>

- `X`: 관측 데이터 시퀀스(Observations Sequence) 
  - ''이렇게 X가 나오도록 Z 값'' 계산하라 中 `이렇게 X가 나오도록` 담당 

```python
X = np.array([[0, 2, 1, 0]]).T # walk -> clean -> shop -> walk
```

<br>

- `Viterbi 알고리즘`: `'.decode( , algorithm="viterbi")' `
  - Z가 관측될 likely probability(가능성, 확률) 계산

```python
logprob, Z = model.decode(X, algorithm="viterbi") # 여기서 Z는 Z가 될 확률값
```

<br>

- 결과 출력

```python
print("\n  Obervation Sequence :", ", ".join(map(lambda x: observations[int(x)], X)))
print("Hidden State Sequence :", ", ".join(map(lambda x: states[int(x)], Z)))
print("Probability = %.6f" % np.exp(logprob))
```

> Obervation Sequence : Walk, Walk, Shop, Shop, Walk, Walk, Walk, Walk, Walk, Walk, Walk, Clean, Walk, ...
>
> Hidden State Sequence : Sunny, Sunny, Sunny, Sunny, Sunny, Sunny, Sunny, Sunny, Sunny, Sunny, Sunny, Rainy, ...
>
> Probability = 0.000000

<br>

#### `Baum Welch 알고리즘`

- X만 주어진 경우: `Learning Question ` 문제

  1. `초기 확률(초기 상태)`, 

  2. `Transition 확률(천이확률)`, 

  3. `Emission 확률(출력확률)`을 추정,

     1-3 까지 `Baum Welch 알고리즘`

  4. `히든 데이터 시퀀스(Z)`까지 찾아낸다.

     4는 `Viterbi 알고리즘`까지 쓴다면

- 활용: 어떤 사람의 행위를 통해 초기 상태와 천이 확률, 그리고 출력 확률을 먼저 추정한 후 Z를 추정한다 

  - 관찰만으로 전부 추정하는 알고리즘 
  - 아래 code 내에선 정확도도 꽤 괜찮은 편

<br>

```python
import numpy as np
from hmmlearn import hmm
np.set_printoptions(precision=2) # np.set_printoptions: numpy float 출력옵션 변경. 소수점 몇자리까지만 보고 싶은 경우
```

<br>

- 나무랑 w(가중치) 세팅

```python
nState = 2
pStart = [0.6, 0.4]
pTran = [[0.7, 0.3], [0.2, 0.8]]
pEmit = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]
```

> 해당 code는 추후 `Baum Welch`을 통해 나온 결과값과의 정확도를 보기 위한 것으로, Data를 임의로 설정해주는 부분임에 유의

> 걍 가짜로 X data, Z data 만들어내는 부분

<br>

- 주어진 확률 분포대로 관측 데이터 시퀀스를 생성한다. 

```python
# 히든 상태 선택. 확률 = [0.6, 0.4]
s = np.argmax(np.random.multinomial(1, pStart, size=1)) # {1, pStart=[0.6, 0.4]} 3개 중 가장 큰 수(np.argmax) 따라서 s = 1
X = []      # Obervation 시퀀스
Z = []      # 히든 상태 시퀀스
for i in range(5000):
    # Walk, Shop, Clean ?
    a = np.argmax(np.random.multinomial(1, pEmit[s], size=1)) # pEmit[s] = [0.6, 0.3, 0.1]
    X.append(a)
    Z.append(s)
    
    # 히든 상태 천이
    s = np.argmax(np.random.multinomial(1, pTran[s], size=1))

X = np.array(X)
X = np.reshape(X, [len(X), 1])
Z = np.array(Z)
```

> 따라서 현재는 X만 아는 상태 

> Q. Z는 왜 만드는 것?? 바움 알고리즘은 X만 가지고 예측하는 건데??? 
> A: 지금 있는 기본 data x랑 나중에 model 만든 거를 합치면 predict z가 나오는데(yHat) 그거랑 찐 z랑 비교하려고

<br>

- `Forward 알고리즘` -> `Baum Welch 알고리즘` 사용

  - Step 1) `Forward 알고리즘` 활용: Observation 시퀀스만을 이용하여, 초기 확률, Transition, Emmision 확률을 추정한다

  ```python
  zHat = np.zeros(len(Z))
  minprob = 999999999 #3의 큰 수로 줘버림
  for k in range(5):
      model = hmm.MultinomialHMM(n_components=nState, tol=0.0001, n_iter=10000)
      model = model.fit(X) # 가짜 data인 x로 학습(fit)
      predZ = model.predict(X)
      logprob = -model.score(X) # forword 알고리즘 # 원래 값이 음수가 나와서 앞에 '-' 붙여줌으로서 양수로 변환
  ```

  > logprob = -6349.458034174618
  >
  > **EM 알고리즘은 local optimum에 빠질 수 있으므로, 5번 반복하여 로그 우도값이 가장 작은 결과를 채택한다.
  > (그게 가장 큰 결과가 되니까 작은 결과 채택).

  <br>

  * Step 2) `Baum Welch 알고리즘` 활용 : Z를 추정
    * **`Viterbi 알고리즘` 과의 차이점: `Baum Welch 알고리즘`는 사전에 주어진 게 X 밖에 없음**

  ```python
      if logprob < minprob:
          zHat = predZ
          T = model.transmat_
          E = model.emissionprob_
          minprob = logprob
      print("k = %d, logprob = %.2f" % (k, logprob))
  ```

  > k = 4, logprob = -6349.46

  <br>

- `찐 Data 세팅 단계`에서 생성한 `Z`와 위 알고리즘들을 통해 추정한 `zHat`의 **정확도를 측정**한다.

```python
accuracy = (Z == zHat).sum() / len(Z)

if accuracy < 0.5: # 정확도가 0.5보다 작다면 순서를 바꿔주는 부분 
    T = np.fliplr(np.flipud(T)) # np.fliplr: 좌우 순서 변경
    E = np.flipud(E) # np.flipud: 상하 순서 변경
    zHat = 1 - zHat
    print("flipped")
    
accuracy = (Z == zHat).sum() / len(Z)
print("\naccuracy = %.2f %s" % (accuracy * 100, '%'))
```

> accuracy = 76.78 %

<br>

- 추정 결과를 출력한다

```python
print("\nlog prob = %.2f" % minprob)
print("\nstart prob :\n", model.startprob_)
print("\ntrans prob :\n",T)
print("\nemiss prob :\n", E)
print("\niteration = ", model.monitor_.iter) 
```

>- log prob = 5339.32
>
>- start prob :
>   [6.93e-72 1.00e+00]
>
>- trans prob :
>   [[0.74 0.26]
>   [0.15 0.85]]
>
>- emiss prob :
>   [[0.11 0.41 0.48]
>   [0.58 0.3  0.12]]
>
>- iteration =  246

> model.monitor_.iter: for문 몇 번 돌렸단 뜻. 여기서 위에 "model = hmm.MultinomialHMM(n_components=nState, tol=0.0001, **n_iter=10000**)" 라 설정했는데 model.monitor.iter가 10000이라 뜨면 값을 못 찾았다는 거라 20000정도로도 늘려보아야 함 

<br>

<br>

## HMM 참고

* Bigram POS tagging과 시험 데이터를 이용한 평가.
  * `trade-off between accuracy and coverage`
    * Bigram에서는 만약 NNS VBG 시퀀스가 학습 데이터에 없다면,
    * P(VBG|NNS)=0, `*해석: VBG 안에 NNS가 없음 `
    * sparse problem 이므로
    * 그 이후의 모든 시퀀스에 악영향을 미쳐 평가 결과가 낮다.
    * N-gram의 N이 클수록 accuracy는 낮지만 문맥의 coverage는 좋다.  따라서 학습용 데이터를 늘리면 약간 개선되기는 함.

<br>

* N-Gram tagging - Combining Tagger (Backoff Tagger)

  - Bigram tagging을 시도하고 P(tag2 | tag1) = 0 (tag1 tag2 시퀀스가 없으면)이면, Unigram을 적용한다 P(tag2).
  - 만약 이것도 없으면 default tag를 적용한다. 

  ```python
  t0 = nltk.DefaultTagger('NN') 
  t1 = nltk.UnigramTagger(train_sents, backoff = t0) 
  t2 = nltk.BigramTagger(train_sents, backoff = t1) 
  t2.evaluate(test_sents)
  ```

  > 참고 : **`nltk.pos_tag()`**는 PerceptronTagger로 Penn Treebank (Wall Street Journal) **데이터를 사전에 학습**해 놓은 것을 사용한다. 
  >
  > 반면에 UngramTagger나 BigramTagger는 사전에 학습해 놓은 것을 사용하는 것이 아니라 직접 학습해서 사용하는 것이다.

<br>

* 품사 태깅 : N-Gram tagging - Unknown word

  * Tagger가 학습 데이터에서 경험하지 못한 단어를 보면 어떻게 태깅해야 하나?

  ```python
  text = "I go to school in the klaldkf" 
  token = text.split() 
  ```

  ```python
  print(unigram_tagger.tag(token))
  ```

  > [('I', 'PPSS'), ('go', 'VB'), ('to', 'TO'), ('school', 'NN'), ('in', 'IN'), ('the', 'AT'), ('klaldkf', None)]

  ```python
  print(bigram_tagger.tag(token)) 
  ```

  > [('I', 'PPSS'), ('go', 'VB'), ('to', 'TO'), **('school', None), ('in', None), ('the', None), ('klaldkf', None)**]

  ```python
  print(nltk.pos_tag(token))
  ```

  > [('I', 'PRP'), ('go', 'VBP'), ('to', 'TO'), ('school', 'NN'), ('in', 'IN'), ('the', 'DT'), ('klaldkf', 'NN')]

  <br>

  * `Unigram Tagger`는 unknown word에만 영향을 미침. 
  * **`Bigram Tagger`**는 unknown word가 다른 단어에도 영향을 미침.
  * `nltk.pos_tag()`는 unknown word를 명사로 태깅하고 있음.
  * 단, `Unigram과 Bigram`은 충분한 데이터로 학습한 결과가 아니며, `nltk.pos_tag()`은 충분한 데이터로 사전에 학습된 것임.

<br>

* 시험 데이터로 태깅 성능을 측정할 때는 `nltk.ConfusionMatrix`를 이용한다.

``` python
# Confusion Matrix 
test_tags = [tag for sent in brown.sents(categories='editorial') for (word, tag) in t2.tag(sent)] 
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')] 
cm = nltk.ConfusionMatrix(gold_tags, test_tags) 
cm['NN', 'NN']
print(cm.pretty_format(truncate=10, sort_by_count=True))
```

<br>

<br>

<br>

<br>

* 참고: 

  >* 아마추어 퀀트, blog.naver.com/chunjein
  >
  >* 코드 출처: 크리슈나 바브사 외. 2019.01.31. 자연어 처리 쿡북 with 파이썬 [파이썬으로 NLP를 구현하는 60여 가지 레시피]. 에이콘