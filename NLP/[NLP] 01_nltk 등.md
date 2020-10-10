---
title: (NLP 기초) 다양한 Corpus들
date: 2020-07-15
update: 2020-08-16
tags:
  - NLP
  - 기초
---





# NLP

* word: 단어. 언어에서 의미를 갖는 최소 단위 ex: 가위, apple 등
* character: character가 모여 word를 이룸 ex: ㄱ,ㄴ,ㅏ,ㄷ, a,C 등
* 문자 = 음운론 = ㄱㅏㄴㅗㄷㅏ (음운론적 규칙 o), abc(음운론적 규칙x)
* 자연어 학습
  * 인공지능(C, Python)과 달리 사람이 하는 말(자연어) 그대로 배우는 것
  * 이때, NLU+NLG 사용(NLU < NLP(더 포괄적))
  * 현재까진 단어 중심의 NLP 기술 활용 中(단점: 사전을 구축해야 하는 단점. 사전에 없는 신조어를 기계가 이해하지 못함)
  * 근데 영어권은 단어 중심이라 음운론적인 규칙(문자:ㄱ,ㄴ,ㄷ,)이 없음(a다음 b나와야 하고 그다음 c나오는)
  * 하지만 한국어는 이런 음운론적 규칙이 있음(예: 먹었니 먹었어 먹을까?)
    → 따라서, 현재 문자(ㄱ,ㄴ,ㄷ) 중심의 NLP를 연구하기도 함

<br>

<br>

## nltk

* 철자(혹은 단어)  개수가 많은 것들은(ex: I, is, am 등) 분석의 의미가 적어, stopword(불용어 처리)를 한다

* 토큰: space(공백)를 기준으로 분리된 것

  ```python
  # 형태적 기준에 의한 분리 방법
  tokens = nltk.word_tokenize(text)
  
  # 의미적 기준에 의한 분리 방법은 따로 존재함
  의미를 갖는 최소 단위(단어 기반) NLP가 현재 주류를 이룸
  ex: 삼성멀티캠퍼스 -> 삼성, 멀티, 캠퍼스
  ```

<br>

### NLTK Base CODE

* 패키지 불러오기

```python
import nltk
```

<br>

#### 영문 소설 txt data > gutenberg corpus

* 조회하기(불러오기)

```python
textID = nltk.corpus.gutenberg.fileids()

print(textID)
```

> ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', ...]

<br>

* 특정 문서를 word 단위로 읽어온다

```python
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
```

>['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']', ...]

<br>

* 특정 문서를 sentence 단위로 읽어온다

```python
sentence = nltk.corpus.gutenberg.sents('austen-emma.txt')
```

>[['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']'], ['VOLUME', 'I'], ...]

<br>

* longestLen

```python
longestLen = max(len(s) for s in sentence)
```

<br>

* hist

```python
countWord = [len(n) for n in sentence]
n, bins, patches = plt.hist(countWord, bins =50)
plt.show()
```

<br>

* 문장당 평균 단어수 

```python
np.mean(countWord)
```

> 문장당 24개의 단어들이 쓰임 

<br>

<br>

#### 사이트에서 직접 소설 txt 읽어와 분석하기

``` python
from urllib import request
```

```python
url = "http://www.gutenberg.org/files/1342/1342-0.txt"
response = request.urlopen(url)
text = response.read().decode('utf8')
```

<br>

* 원시 문서를 토큰(=space를 기준으로 분리된 것)으로 분리한다.

```python
tokens = nltk.word_tokenize(text)
```

>['\ufeff', 'The', 'Project', 'Gutenberg', 'EBook', 'of', 'Pride', 'and', 'Prejudice', ',']

<br>

```python
nltkText = nltk.Text(tokens)
```

><Text: ﻿ The Project Gutenberg EBook of Pride and...>

<br>

<br>

#### 인터넷의 일반 데이터 webtext(영화 대본, 게시판)

* webtext 코퍼스의 파일 ID 조회(불러오기)

```python
textId = webtext.fileids()
```

>['firefox.txt',
> 'grail.txt',
> 'overheard.txt',
> 'pirates.txt',
> 'singles.txt',
> 'wine.txt'] ...

<br>

* 특정 파일 raw text data 조회(불러오기)

```python
text = webtext.raw('pirates.txt')
```

>PIRATES OF THE CARRIBEAN: DEAD MAN'S CHEST, by Ted Elliott & Terry Rossio
>[view looking straight down at rolling swells, sound of wind and thunder, then a low heartbeat] ... 

<br>

* 문서를 단어(word) 단위로 읽어온다.

```python
word = webtext.words('pirates.txt')
```

>['PIRATES', 'OF', 'THE', 'CARRIBEAN', ':', 'DEAD', ...]

<br>

* 문서를 문장(sentence) 단위로 읽어온다.

```python
sentence = webtext.sents('pirates.txt')
```

>[['PIRATES', 'OF', 'THE', 'CARRIBEAN', ':', 'DEAD', 'MAN', "'", ...

<br>

* word 개수, sentence 개수 확인

```python
print("word 개수 = ", len(word))
print("문장 개수 = ", len(sentence))
```

<br>

<br>

#### 인터넷의 일반 데이터 Chat 데이터

* chat 파일 id 조회

```python
textId = nps_chat.fileids()
```

>['firefox.txt',
> 'grail.txt',
> 'overheard.txt',
> 'pirates.txt',
> 'singles.txt',
> 'wine.txt']

<br>

* 특정 chat session의 raw 텍스트 문서 조회(xml 형식)

```python
text = nps_chat.raw('10-19-20s_706posts.xml')
```

><!-- edited with XMLSpy v2007 sp1 (http://www.altova.com) by Eric Forsyth ...

<br>

* xml의 post 데이터를 읽는다

```python
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
```

>[['now', 'im', 'left', 'with', 'this', 'gay', 'name'], [':P'], ...]
>
>***nps_chat.words('10-19-20s_706posts.xml') 와 결과는 같음***

<br><br>

#### 브라운 코퍼서

- 브라운 코퍼서는 뉴스, 편집기사 등의 카테고리(장르)별로 분류돼 있다

  <br>

* brown 코퍼서의 파일 id 조회

```python
textId = brown.fileids()
```

>['ca01', 'ca02', 'ca03', 'ca04', 'ca05', ...

<br>

* 카테고리 (장르) 목록 조회

```python
cat = brown.categories()
```

>['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

<br>

* 특정 카테고리의 raw data 조회

```python
news = brown.raw(categories = 'news')
```

>The/at Fulton/np-tl County/nn-tl Grand/jj-tl ...

<br>

* 'news' 카테고리의 txt문서를 단어 단위로 조회(품사 제외)

```python
news = brown.words(categories = 'news')
```

>['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]

<br>

* 특정 파일 id의 문서 조회

```python
cg22 = brown.words(fileids = ['cg22'])
```

>['Does', 'our', 'society', 'have', 'a', 'runaway', ',', ...]

<br>

* 장르별 단어의 빈도 분포 확인

```python
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
    )

print(cfd)
```

><ConditionalFreqDist with 15 conditions>

<br>

``` python
cfd.conditions()
```

> ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

<br>

* 특정 카테고리의 빈도 분포 확인

```python
 cfd['adventure']
```

>FreqDist({'.': 4057, ',': 3488, 'the': 3370, 'and': 1622, 'a': 1354, 'of': 1322, 'to': 1309, '``': 998, "''": 995, 'was': 914, ...})

<br>

* 단어의 빈도 분포로 문서의 주제를 파악하는 아이디어 - > 토픽 모델(LD)

```python
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)
```

>cfd.tabulate(conditions=genres, samples=modals)
>                	  can could   may might  must  will 
>           news    93    86    66    38    50   389 
>       religion    82    59    78    12    54    71 
>        hobbies   268    58   131    22    83   264 
>science_fiction    16    49     4    12     8    16 
>        romance    74   193    11    51    45    43 
>          humor    16    30     8     8     9    13 

> " 뉴스 문서에는 will이 가장 많이 등장하고, 로맨스 문서에는 could가 가장 많이 등장한다.
> 단순히 특정 단어의 빈도수만 파악해도 해당 문서의 장르를 추측해볼 수 있다. "

<br>

<br>

#### 로이터 코퍼스

* 10,000개 넘는 뉴스 문서가 90개의 Topic(주제)로 분류돼 있다. 또한, 데이터는 train, test로 분리돼 있다

<br>

* 로이터 코퍼스의 파일 id를 조회

```python
from nltk.corpus import reuters
```

>textId[:10]: 
>
>['test/14826', 'test/14828', 'test/14829', 'test/14832', 'test/14833', 'test/14839', 'test/14840', 'test/14841', 'test/14842', 'test/14843']
>
>
>
>textId[5000:5010]:
>
>['training/13203', 'training/13204', 'training/13205', 'training/13206', 'training/13210', 'training/13211', 'training/13212', 'training/13214', 'training/1322', 'training/13223']

<br>

* 카테고리 목록을 조회

```python
cat = reuters.categories()
```

>['acq', 'alum', 'barley', 'bop', 'carcass', 'castor-oil', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 'copper',  ...

<br>

* 원시 문서를 읽는다

```python
text = reuters.raw('training/9865')
```

>FRENCH FREE MARKET CEREAL EXPORT BIDS DETAILED
>  French operators have requested licences

<br>

* 문서의 주제어를 조회

```python
topic = reuters.categories('training/9865')
```

>['barley', 'corn', 'grain', 'wheat']

<br>

* 해당 주제어를 갖는 문서를 찾는다

```python
textTopic = reuters.fileids('cpu')
```

>['test/21245', 'training/5388', 'training/5460', 'training/5485']

```python
textTopic = reuters.fileids(['cpu', 'naphtha'])
```

>['test/17880', 'test/18480', 'test/19497', 'test/19903', 'test/21245', 'training/5388', 'training/5460', 'training/5485', 'training/6535', 'training/7397']

<br>

* 유사한 주제어를 갖는 문서의 내용을 조회

```python
text = reuters.words('training/5388')
```

>['CANADA', 'MANUFACTURING', 'UTILIZATION', 'RATE', 'RISES', 'Utilization', 'of', 'Canadian', ...

<br>

<br>

#### 대통령 취임 연설문 코퍼스

* id 조회

```python
from nltk.corpus import inaugural
textId = inaugural.fileids()
```

>'test/15749',
> 'test/15751',
> 'test/15753', ...

<br>

* 연도별로 'america'와 'citizen'이란 단어가 사용된 빈도의 변화를 관찰

```python
cfd = nltk.ConditionalFreqDist(
    (target,fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target)
    )

cfd['america']
cfd['citizen']
cfd.plot()
```

<br>

* 영어 단어 목록: word list

```python
wordlist = nltk.corpus.words.words() 
```

>'absence',
> 'absent',
> 'absentation',
> 'absentee',
> 'absenteeism',

<br>

* wordlist에서 'egivrvonl' 단어 검색

```python
puzzleLetters = nltk.FreqDist('egivrvonl')

print(FreqDist)
```

>FreqDist({'v': 2, 'e': 1, 'g': 1, 'i': 1, 'r': 1, 'o': 1, 'n': 1, 'l': 1})
>
>*내림차순 정렬

<br>

- wordlist 中 [1] len(6) 이상, [2] 'r' 이 들어가고, [3] 빈도수가 puzzleLetters보다 작거나 같은 것 

```python
obligatory = 'r'

answer = [w for w in wordlist if len(w) >= 6
          and obligatory in w
          and nltk.FreqDist(w) <= puzzleLetters]
```

<br>

<br>

### 사람 이름(영어) - Names

<br>

* 조회(불러오기)

```python
names = nltk.corpus.names
fileId = names.fileids()
```

<br>

* 이름의 마지막 글자의 분포를 확인한다 
* 눈에 띄는 남자 이름과, 여자 이름의 특징은?

```python
names = nltk.corpus.names
fileId = names.fileids()
```

<br>

* 남자/여자 분리 

```python
f = [w for w in cfd] 
# f : ['female.txt', 'male.txt']
cfd[f[0]] # 남자
cfd[f[1]] # 여자
```

<br>

<br>

### 불용어 stop words

* 영어의 stop words를 확인

```python
stopwords = stopwords.words('english')
```

* 영어 소설에서 stop word를 제거

```python
text = nltk.corpus.gutenberg.words('austen-sense.txt')
removedStopWord = [w for w in text if w.lower() not in stopwords]
```

> ['[', 'Sense', 'Sensibility', 'Jane', 'Austen', '1811', ']', 'CHAPTER', '1', 'family', 'Dashwood', 'long', 'settled', 'Sussex', '.', 'estate', 'large', ',', 'residence', 'Norland']

<br>

### 참고 CODE + 정리

- stop word를 제거한 word의 비중을 확인

```python
"stop word 제거한 word의 비율 = ", len(removedStopWord) / len(text)
```

> stop word 제거한 word의 비율 =  0.5285429733853195

<br>

```python
# raw data 불러오기 
# '.raw'
text = webtext.raw('pirates.txt')

# ID 조회하기(불러오기) 
# '.fileids()'
textID = nltk.corpus.gutenberg.fileids()

# 특정 문서를 word 단위로 읽어오기 
# '.words()'
emma = nltk.corpus.gutenberg.words('austen-emma.txt')

# 특정 문서를 sentence 단위로 읽어오기 
# '.sents()'
sentence = nltk.corpus.gutenberg.sents('austen-emma.txt')

# 원시 문서를 토큰(=space를 기준으로 분리된 것)으로 분리 
# 1. 'word_tokenize()'
# 2. '.Text()'
tokens = nltk.word_tokenize(text)
nltkText = nltk.Text(tokens)

# xml의 post 데이터를 읽는다
# '.posts()'
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
## .posts = .words

# 카테고리 (장르) 목록 조회
# '.categories()'
cat = brown.categories()

# 특정 카테고리의 raw data 조회
# '.raw(categories = '')'
news = brown.raw(categories = 'news')

# 특정 카테고리의 문서를 단어 단위로 조회(품사 제외)
# '.words(categories = '')'
news = brown.words(categories = 'news')

# 장르별 단어의 빈도 분포 확인
# '.ConditionalFreqDist()'
# (genre, word)를 위해 .FreqDist 2개 쓰면 .ConditionalFreqDist

## step 1
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
    )

## step2. 
cfd.conditions()

# 특정 카테고리의 빈도 분포 확인
# cfd['adventure']

# 토픽 모델(LD)
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)

# 문서의 주제어를 조회
topic = reuters.categories('training/9865')

# 연도별로 'america'와 'citizen'이란 단어가 사용된 빈도의 변화를 관찰
cfd = nltk.ConditionalFreqDist(
    (target,fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target)
    )

cfd['america']
cfd['citizen']
cfd.plot()



# 영어 소설에서 stop word를 제거한다
text = nltk.corpus.gutenberg.words('austen-sense.txt')
removedStopWord = [w for w in text if w.lower() not in stopwords]
```

<br>

<br>

### WordNet

* 단순한 사전이 아니라, 단어 사이의 의미를 구분지음으로써 단어들 사이에 상하관계를 가지고 유의어 집단(synset)으로 분류할 수 있음 
  * Stemmer(어간, 어미)











<br>

<br>

<br>

<br>

* 참고: 

  >* 아마추어 퀀트, blog.naver.com/chunjein
  >
  >* 코드 출처: 크리슈나 바브사 외. 2019.01.31. 자연어 처리 쿡북 with 파이썬 [파이썬으로 NLP를 구현하는 60여 가지 레시피]. 에이콘