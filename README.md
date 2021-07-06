# Deep-Learning
HYU ITE4053
-----------------------
 ## 1. image classification
  이미지 분류 task. CNN classifier 를 사용하고 여러 기법을 더해 성능을 향상 시킨다.
  ### 세부 사항
  > - 10 종류의 이미지를 각각에 맞는 class로 분류하는 것이 목표이다.
  > - class의 종류는 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck이다.
  > - 데이터
  >   - x_trin.npy: 트레이닝 데이터, 이미지
  >     - 용량상의 문제로 압축되어있기 때문에 압축을 푼 뒤 실행해야 정상적으로 실행된다.
  >   - y_train.npy: 트레이닝 데이터, 각 이미지에 대한 라벨
  >   - x_text.npy: 성능 측정을 위한 test 데이터, 이미지
  > - 코드
  >   - image_show.py: 데이터 이미지를 보기 위한 코드
  >   - eval.py: x_test 파일을 분류하여 나온 결과를 성능평가하기 위한 코드
  >   - model4student.py: CNN classifier 및 정확도 평가 코드

  ### CNN classifier
  > - convolutional layer, pooling layer가 각 4개씩 있는 neural network.
  > - batch normalization, dropout, checkpoint를 활용한 2번 학습 을 통해 성능을 높임.
  > - mini batch 사용.
  > - activation 함수로는 ReLU, optimizer는 Adam optimizer 사용
  
  ### 정확도
  > - F1 score: 0.68 -> 0.74
  
   ### 환경
 > - 프로그래밍 언어 : python 3.7.10
 > - tensorflow 1.13.1
 > - OS : windows 10
 ---------------------------
## 2. text classification
  텍스트 감성 분류 task. RNN 계열 모델을 사용하고 여러 기법을 더해 성능을 향상 시킨다.
  ### 세부 사항
  > - 긍정 부정 2종류의 텍스트를 각각에 맞는 class로 분류하는 것이 목표이다.
  > - 데이터
  >   - x_trin.npy: 트레이닝 데이터, 전처리가 끝난 텍스트
  >   - y_train.npy: 트레이닝 데이터, 각 텍스트에 대한 라벨
  >   - x_text.npy: 성능 측정을 위한 test 데이터, 전처리가 끝난 텍스트
  > - 코드
  >   - text_show.py: 주어진 전처리가 끝난 텍스트 데이터를 원래의 데이터로 보기 위한 코드
  >   - eval.py: x_test 파일을 분류하여 나온 결과를 성능평가하기 위한 코드
  >   - model4student.py: RNN 기반 classifier 및 정확도 평가 코드

  ### 딥러닝 layer RNN classifier
  > - GRU cell 사용
  > - attention model 사용
  
  ### 정확도
  > - F1 score: 0.88150
  
   ### 환경
 > - 프로그래밍 언어 : python 3.7.10
 > - tensorflow 1.13.1
 > - OS : windows 10

---------------------------
## 1. assignment1
  MNIST classifier에 dropout과 adam optimizer 적용하기

 ---------------------------
 ## 2. assignment2
  MNIST 데이터를 활용하여 Denoising Auto Encoder 만들기

 ---------------------------
 ## 3. assignment3
  check point 사용하기

 ---------------------------
 ## 4. assignment4
  character sequence RNN. 
  character 단위의 RNN을 실습한다. "if you want you"를 input 값으로 했을 때 "f you want you"가 나오도록 한다.
