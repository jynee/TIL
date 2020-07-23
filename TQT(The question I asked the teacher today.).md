**TQT(The question I asked the teacher today.)**





# LSTM

* 2층: lstm을 2번 써준다
* 양방향: bidirectional + merge_mode = ‘concat’. FNN, BFN 값을 merge_mode 형태로 합쳐서 list형으로 되돌려줌
* many-to-many: return-sequences = True. 의미는 LSTM의 중간 스텝의 출력을 모두 사용
* timedistributed: FFN으로 가기 전 LSTM 마지막 층에서 각 뉴런의 각 지점에서 계산한 오류를 다음 층으로 전파





# Dense

* Dense: fully connected

* ANN(FNN)에서는 여러 Dense를 써도 되지만, LSTM에선 마지막 층에서만 Dense를 써야함.

  * CNN에서는 여러 Dense 써도 됨 

* Dense(1, activation='sigmoid')

* LSTM 마지막 Dense에선 relu 쓰면 안됨

  