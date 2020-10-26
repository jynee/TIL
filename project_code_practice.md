



# Pandas

* 이중 list -> 1줄로

  ```python
  df['colum_name'] = [','.join(map(str, _)) for _ in df['list']]
  ```

* 특정 값만 모아서 하나의 변수에 합쳐 넣기

  ```python
  dw
  ```

  > code made by [terry kim](url)

* 여러 row의 string 값들을 하나의 row로 합치기

  ```python
  dw
  ```







# type

* string으로 바꿨는데도 모델 돌릴 때 float 혹은 int가 아닌 str 쓰라고 할 때

  ```python
  [x.replace(x,x) if isinstance(x, str) else x for x in pandas_name["contents"]
  ```

  





# model

* **oom** Error

  * out of memory

    따라서 batch_size값을 줄여본다

    > batch_size = 32 -> batch_size = 16 ...



