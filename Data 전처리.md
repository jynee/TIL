# 데이터 전처리

* raw data → 가공 및 변형(feature engineering) → 학습이 가능한 형태의  data → model 학습

  * EDA > Feature Engineering > Modeling > Validation > Application 
* 데이터 전처리: `raw data → 학습이 가능한 형태의 data로 만드는 과정`(전체 과정에 있어 80~90%라고 생각하면 됨) 





## 전처리 방법

* raw data → 학습이 가능한 형태의 data로 만들기 위해서는 

  1. **raw data를 보고 어떻게 가공 및 변형할 것인가에 대해 생각한다**

     * 우선, machine은 data를 **row단위**로 본다는 걸 알고가자
     * (빠른방법) excel의 fivot table로 data를 가공해가며 인사이트를 찾는다.
       * ex: AMT를 예측하는 데에 A feature가 중요할까? 안 중요할까? A+B를 합친 feature가 중요할까?

  2. **null 처리** ← 어느 단계에서 하든지 일단 별로 상관 없는 듯 느껴짐 

  3. **raw data의 특징을 살핀다(EDA).**

     * 특징이라함은 

       * _'해당 raw data가 시간순으로 된 것인지 ?그럼 시계열 분석을 할 것인지? '_

       * 시계열분석: 예) RNN
       
         | 날짜    | 지역   | AGE  | AMT  |
         | ------- | ------ | ---- | ---- |
         | 2019.01 | 서울   | 1    | 200  |
         | 2019.01 | 경기도 | 1    | 300  |
     | 2019.02 | 서울   | 2    | 400  |
         | 2019.03 | 서울   | 3    | 200  |

         * 시간순: 날짜(2019.01~2019.12), 계절(봄~겨울)
       
           * (변형 방법) 시간순으로 분석하기에 data 개수가 적다면_?_ 
         ex: 위 table의 날짜는 2019.01~2020.03로 12개 밖에 없다. 
             
           * 1번 방법: **transepose**
           	
               |      | 2019.01 | 2019.01 | 2019.02 | 2019.03 |
             | ---- | ------- | ------- | ------- | ------- |
             | 지역 | 서울    | 경기도  | 서울    | 서울    |
             | AGE  | 1       | 1       | 2       | 3       |
       | AMT  | 200     | 300     | 3       | 200     |
             
             > 행/열을 바꿔서 각 날짜를 feature로
       
   * '_해당 raw data를 시간순으로 보지 않는다면?'_
     
       * XGBoost, random forest 등 
  
4. **raw data 가공 및 변형**(feature engineering)
  
     * 결과에 영향을 줄 feature들을 적절히 선정하는 과정
   * 방법들
   
   1. 새로운 feature **추가**(외부 data 활용 등)
   
      * 인사이트 필요. 인사이트는 EDA 시 도출된다고 생각
      
      * feature를 추가할 땐, 기존 feature와 추가하려는 feature 사이의 상관이 있어야함. 
      
        ex: 지역별 feature를 보고 지역별 인구수를 추가한다거나
   
   2. 있는 feature **가공**(정규화 등)
   
   3. 필요없는 feature **삭제**
   
        * feature들 사이에 의미가 매우 비슷한 것들도 있을 수 있어, 하나만 보존하고 나머지는 삭제
          * ex: Decision Tree
     * 불필요하거나 영향력 적은 feature들은 메모리 공간을 낭비하고 machine의 계산 시간을 늘림
   
  4. 다른 feature로 **transepose **
     
       | 날짜(기존feature) | 날씨(추가한 feature) | data 기반으로 변환 |
       | ----------------- | -------------------- | ------------------ |
       | 2019.01.02        | 맑음                 | 5                  |
       | 2019.07.22        | 비                   | 1                  |
       | 2020.04.14        | 비                   | 1                  |
       | 2020.02.15        | 안개                 | 2                  |
     
         > 기존 feature(날짜)와 연관있는 외부데이터(날짜)를 추가하기 위해 categorical 변환(str→int) 







------------------





## EDA

* 하는 이유:
  1. 예측 데이터 분석에선, EDA를 통해 각 feature를 다각도로 분석함으로써 feature를 결합/분해해보고, 예측치를 더 정확히 맞출 수 있는 feature를 만들어 활용할 수 있을 것으로 보임
  2. EDA를 통해 앞으로 어떤 분야(산업 등의 feature 인스턴스)의 수치가 클 것이라 예상할 수 있는데, 이때 모델을 통해 나온 예측치에선 그 부분이 유난히 적게 나온다면 모델 튜닝 혹은 그 데이터에 그 모델은 적합하지 않은 것으로 판정할 수 있음

* 방법:
  1. pivot table
  2. 인사이트 → 논문 찾아 기존 실험 data 참고 및 새로운 feature 끌어오기





--------------------



## python pandas pivot table 배우기

* > 피봇테이블(pivot table)이란 데이터 열 중에서 두 개의 열을 각각 행 인덱스, 열 인덱스로 사용하여 데이터를 조회하여 펼쳐놓은 것을 말한다.
  >
  > Pandas는 피봇테이블을 만들기 위한 `pivot` 메서드를 제공한다. 첫번째 인수로는 행 인덱스로 사용할 열 이름, 두번째 인수로는 열 인덱스로 사용할 열 이름, 그리고 마지막으로 데이터로 사용할 열 이름을 넣는다.
  >
  > Pandas는 지정된 두 열을 각각 행 인덱스와 열 인덱스로 바꾼 후 행 인덱스의 라벨 값이 첫번째 키의 값과 같고 열 인덱스의 라벨 값이 두번째 키의 값과 같은 데이터를 찾아서 해당 칸에 넣는다. 만약 주어진 데이터가 존재하지 않으면 해당 칸에 `NaN` 값을 넣는다.
  >
  > * 출처: 데이터사이언스스쿨. 2016.07.08. "피봇테이블과 그룹분석". https://datascienceschool.net/view-notebook/76dcd63bba2c4959af15bec41b197e7c/

* > 행 인덱스와 열 인덱스는 **데이터를 찾는 키(key)**의 역할을 한다. 따라서 키 값으로 데이터가 **단 하나만 찾아져야 한다.** 만약 행 인덱스와 열 인덱스 조건을 만족하는 데이터가 2개 이상인 경우에는 에러가 발생한다. 예를 들어 위 데이터프레임에서 ("지역", "연도")를 키로 하면 ("수도권", "2015")에 해당하는 값이 두 개 이상이므로 다음과 같이 에러가 발생한다.
  >
  > * 예: Error: Index contains duplicate entries, cannot reshape
  >
  >   ```python
  >   #중복된 행의 데이터만 표시하기
  >   check = data[data.columns[:-3]]
  >   display(check[check.duplicated()])
  >   ```

* groupby

  ```python
  columns = ['STD_DD', 'GU_CD', 'DONG_CD', 'MCT_CAT_CD', 'SEX_CD', 'AGE_CD']
  data = data.groupby(columns).sum().reset_index(drop=False)
  ```

* 일자별 대구 수성구/중구 & 서울 노원구/중구의 카드 매출액

  ```python
  data.pivot_table("USE_AMT", index=["STD_DD", "GU_CD"])
  ```

  > |        | USE_AMT |      |
  > | -----: | ------: | ---: |
  > | STD_DD |   GU_CD |      |
  > | 201902 |       1 |    1 |
  > |        |       1 |    1 |
  > |        |       1 |    1 |
  > |        |       1 |    1 |

* 성별 업종별 카드 매출액(사용액)

  ```python
  data.pivot_table("USE_AMT", "SEX_CD", "MCT_CAT_CD", aggfunc="count", margins=True)
  ```

  > | MCT_CAT_CD |   10 |   20 |   21 |   22 |
  > | ---------: | ---: | ---: | ---: | ---: |
  > |     SEX_CD |      |      |      |      |
  > |          F |    1 |    1 |    1 |    1 |
  > |          M |    1 |    1 |    1 |    1 |
  > |        All |    1 |    1 |    1 |    1 |

* 일별 성별 나이대별 업종의 카드거래액

  ```python
  data.pivot_table('USE_AMT', ['STD_DD', 'SEX_CD', 'AGE_CD'], 'MCT_CAT_CD', aggfunc='mean', fill_value=0)
  ```

  > |        |        | MCT_CAT_CD |   10 |   20 |
  > | -----: | -----: | ---------: | ---: | ---: |
  > | STD_DD | SEX_CD |     AGE_CD |      |      |
  > | 201902 |      F |            |    1 |    1 |
  > |        |        |            |    1 |    1 |
  > |        |        |            |    1 |    1 |

* describe()

  * groupby에서만 쓸 수 있음

  ```python
  data.groupby(['STD_DD', "SEX_CD", "MCT_CAT_CD"])[["USE_AMT"]].describe()
  ```

  > |        |        |            | USE_AMT |      |      |
  > | -----: | -----: | ---------: | ------: | ---: | ---: |
  > |        |        |            |   count | mean |  std |
  > | STD_DD | SEX_CD | MCT_CAT_CD |         |      |      |
  > | 201902 |      F |            |       1 |    1 |    1 |
  > |        |        |            |       1 |    1 |    1 |
  > |        |        |            |       1 |    1 |    1 |
  > |        |        |            |       1 |    1 |    1 |
  > |        |        |            |       1 |    1 |    1 |

* describe().T

  * dtype이 int인 것만 자동으로 계산된다.

  ```python
  data.groupby(data['STD_DD']).describe().T
  ```

  > |         | STD_DD | 201902 | 201903 | 201904 | 201905 |
  > | ------: | -----: | -----: | -----: | -----: | -----: |
  > | USE_CNT |  count |      1 |      1 |      1 |      1 |
  > |         |   mean |        |        |        |        |
  > |         |    std |        |        |        |        |
  > |         |    min |        |        |        |        |
  > |         |    25% |        |        |        |        |
  > |         |    50% |        |        |        |        |
  > |         |    75% |        |        |        |        |
  > |         |    max |        |        |        |        |
  > | USE_AMT |  count |      1 |      1 |      1 |      1 |
  > |         |   mean |        |        |        |        |
  > |         |    std |        |        |        |        |
  > |         |    min |        |        |        |        |
  > |         |    25% |        |        |        |        |













-----------------------------















