![image](https://github.com/user-attachments/assets/1967716c-6899-42d7-a796-549e489d5f75)# TITLE : 100 Sports Image Classification



## Members:
> ##### 김재준, 데이터사이언스학과, apolon0617@gmail.com
> ##### 박찬형, 데이터사이언스학과, chanhyoungpark@naver.com
> ##### 조윤규, 데이터사이언스학과, yg20029@gmail.com


## I.	Proposal (Option 1or 2)–This should be filled bysometime in early Nov.
### -Motivation: Why are you doing this?
이 프로젝트를 통해 100가지 스포츠 이미지를 분류하며, 딥러닝과 이미지 분류의 핵심 기술을 탐구하고 이를 실제 문제에 적용해보는 경험을 쌓고자 한다. 또한, 이 프로젝트는 딥러닝 모델 설계, 전이 학습, 하이퍼파라미터 튜닝 등 딥러닝의 여러 중요한 개념을 실습할 수 있는 기회를 제공한다.
이러한 과정을 통해 딥러닝에 대한 이해도를 높이고, 실제 데이터를 다루며 문제 해결 능력을 키우는 데 도움이 될 것이라 생각한다. 이 데이터셋을 활용해 높은 정확도를 달성함으로써, 향후 이미지 분석 프로젝트에 기여하는 것을 목표로 한다.

### -What do you want to see at the end?
이 프로젝트의 최종 목표는 100개의 다양한 스포츠 이미지를 정확히 분류할 수 있는 딥러닝 모델을 구축하는 것이다. 이를 통해 테스트 데이터에서 90% 이상의 분류 정확도를 달성하고, 전이 학습 등의 여러 모델링 기술들을 활용 및 경험하며 점차적으로 정확도를 높여가는 성취를 기대한다.

## II.	Datasets-Describing your dataset 

이번 프로젝트에서 사용할 데이터셋은 kaggle 사이트에서 가져온 100개의 서로 다른 스포츠 카테고리를 포함하는 이미지 데이터셋이다. 이 데이터셋은 주로 인터넷 검색을 통해 수집되었으며, 다음과 같은 특징을 가지고 있다.
### 1.	데이터셋 구조:
> * #### 훈련 데이터 (Train Set): 총 13,493개의 이미지로, 모델 학습을 위한 주요 데이터이다.
> * #### 검증 데이터 (Validation Set): 총 500개의 이미지로, 모델의 하이퍼파라미터 조정과 성능 검증에 사용된다.
> * #### 테스트 데이터 (Test Set): 총 500개의 이미지로, 모델의 최종 성능을 평가하는 데 사용된다.
### 2.	이미지 정보:
> * #### 모든 이미지는 224x224x3 크기의 RGB 이미지로, .jpg 형식으로 저장되어 있다.
> * #### 데이터는 각 스포츠 카테고리별로 정리되어 있어 ImageFolder와 같은 PyTorch의 데이터 구조로 쉽게 로드할 수 있다.
### 3.	데이터 포맷:
> * #### 이미지 데이터 외에도 CSV 파일이 포함되어 있어, 각 이미지의 상대 경로, 클래스 레이블, 데이터 세트(훈련, 검증, 테스트)를 확인할 수 있다.
> * #### 이러한 구조는 PyTorch의 DataLoader와 같은 데이터 로더를 쉽게 사용할 수 있게 한다.

이 데이터셋을 활용하여 스포츠 이미지를 학습하고, 모델이 다양한 스포츠 활동을 정확히 분류할 수 있도록 하는 것이 목표이다. 이 프로젝트를 통해 이미지 분류 모델의 구축 및 평가, 전이 학습 적용 등 여러 딥러닝 기술을 직접 실험할 수 있다.

데이터 다운로드 링크
https://www.kaggle.com/datasets/gpiosenka/sports-classification?select=train

## III.Methodology -Explaining your choice of algorithms (methods)-Explaining features (if any)

### -Explaining your choice of algorithms (methods)
이번 프로젝트에서 이용할 모델의 기반은 CNN 모델이다.
CNN 모델은 이미지, 영상을 분석하기 위한 패턴을 직접 찾아 학습하고, 찾은 패턴을 이용하여 이미지를 분류한다. CNN 이전의 모델인 Fully Connected layer 를 이용할 시, 지나치게 많은 모델 파라미터로 인하여 높은 overfitting 경향을 보이지만 CNN은 이를 방지 할 수 있는 좋은 선택이다.
CNN은 이미지의 각 픽셀들은 주변의 픽셀과 연관되어 있다는 가정 아래 시작한다. 이미지에서는 비슷한 패턴이 다양한 픽셀 위치에서 생길 수 있으며 머신러닝 알고리즘은 이를 이용해 더 뛰어난 generalization 을 이루고자 한다.

### Convolution:
CNN에서 가중 평균(weighted average) 연산은 이미지나feature map의 모든 위치에서 수행되며, 이를 통해 다양한 패턴이나 특징을 탐지할 수 있다. 어떠한 패턴이 weight를 통과하느냐에 따라 다른 결과를 추출해 낼 수 있다.


### Convoultion Layer:
컨볼루션 레이어는 입력 이미지를 Filter(Kernel)를 이용하여 탐색하면서 이미지의 feature들을 추출하고, 추출한 feature들을 Feature Map으로 생성한다.

<img width="300" alt="Screenshot 2024-11-25 at 22 48 41" src="https://github.com/user-attachments/assets/52923fa9-8ea9-496c-b089-a397a0ae16ce">

Feature Detectors example:

<img width="500" alt="image" src="https://github.com/user-attachments/assets/6d81e728-f120-45da-b9b9-714f8e847754">


### ResNet
ResNet의 핵심 개념은 잔차 학습(Residual Learning)으로, 깊은 네트워크에서 발생하는 gradient 소실 문제를 해결하기 위해Skip Connection을 이용한다. 이 스킵 연결은 입력 데이터를 변환된 출력에 더함으로써, 깊은 네트워크에서 학습이 더 원활하게 이루어지도록 돕는다. ResNet은 Residual Block이라는 구조를 쌓아 올리며, 이를 통해 네트워크가 깊어져도 학습 성능이 유지되거나 개선될 수 있다. ResNet은 이미지 분류, 객체 검출, 이미지 세분화와 같은 컴퓨터 비전 작업에서 탁월한 성능을 보여주며, 특히 전이 학습(Transfer Learning)에서 자주 사용된다.  대표적인 ResNet 모델로는 ResNet34, ResNet50이 있으며, 숫자는 계층의 깊이를 나타낸다.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/b0003f19-18d2-491d-8216-dd4677e49832">

### -Explaining features (if any)

ResNet 모델에서 조절 할 수 있는 하이퍼 파라미터는 learning_rate, batch_size, epochs, weight_decay, optimizer등이 존재한다. learning_rate는 모델이 가중치를 업데이트할 때 사용하는 스텝 크기를 결정한다. batch_size는 한 번의 학습 업데이트에서 처리되는 데이터 샘플의 수를 나타낸다. epochs 전체 데이터셋을 몇 번 학습하는지 나타낸다. weight_decay는 모델의 가중치에 대해 L2 정규화를 적용하여 과적합을 방지하는 데 사용한다. optimizer는 가중치를 업데이트하는 방식이다. SDG와 ADAM의 두 가지를 사용했다. SDG는 기울기(Gradient)를 사용해 손실 함수의 값을 최소화하는 방향으로 모델의 가중치를 업데이트하는 기본적인 최적화 알고리즘이다. ADAM은 학습률을 자동으로 조정하며 SGD의 단점을 개선한 최적화 알고리이다.

참고 문헌은 다음과 같다. 
https://arxiv.org/pdf/1512.03385

## IV. Evaluation & Analysis

우리는 두 가지 ResNet 모델(ResNet18, ResNet34) 로 스포츠 데이터를 학습해 보았다.
또한 각 모델에 대해 여러 하이퍼 파라미터 조합으로 학습을 진행하였다.  사용한 하이퍼 파라미터 조합은 다음과 같다.

![image](https://github.com/user-attachments/assets/b6b699eb-d121-4d72-b625-5a8028af9ad1)

사용한 두 모델, 그리고 위의 하이퍼파라미터 조합 총 324가지이다. 하지만 이 모든 조합으로 모델 트레이닝을 하기에는 언제나 그렇듯이 시간과 컴퓨팅 용량이 충분치 않다. 따라서 우리는 랜덤으로 모델 15가지를 시도해 보았다.

하이퍼파라미터 튜닝을 위해서 우리는 wandb라는 파이썬 라이브러리를 사용했다. Wandb는 머신러닝 및 딥러닝 프로젝트의 추이 추적, 시각화, 모델 관찰 등을 지원하는 도구로 특히 하이퍼파라미터 서칭 이후 각 모델들의 차이를 시각화 하는데 좋다. 

Wandb sweep을 거친 후 나온 파이썬 로그는 다음과 같다.

![image](https://github.com/user-attachments/assets/aa086cc7-8ba8-491b-8e53-1b2359ae7886)

wandb 워크스페이스에서 시각화 한 결과입니다.

![image](https://github.com/user-attachments/assets/d8a3b9c3-9df4-4245-8ce3-9a8afcd3fa73)

그럼 나온 결과들을 하나하나 풀어보자. 
먼저 ResNet18 모델이다.
총 hyperparameter searching 하는데 걸린 시간은 3h 29m 이다.

![image](https://github.com/user-attachments/assets/a13b3a42-a0fe-4a05-b324-a5f4e8791ed5)

전체적인 하이퍼파라미터 조합에 따른 validation accuracy 추이는 다음과 같다.
이제 각 모델별 train epoch에 따른 loss와 validation accuracy 를 보자. 

다음은 Train loss 그래프이다. 
![image](https://github.com/user-attachments/assets/ee9c0fcd-1296-4e89-a78e-da7b2702b009)
모든 모델의 train loss가 감소하는 것을 볼 수 있다.

다음은 Validation loss 그래프이다. 
![image](https://github.com/user-attachments/assets/f3f2316a-46e7-4ae0-b9f3-35a308e23587)

마찬가지로 validation loss 도 감소하는 것을 볼 수 있다. 
하지만 train loss와 다르게 가끔 튀는 현상이 발생한다. 모델은 train data 를 기반으로 학습이 되지만 validation loss는 모델이 처음 보는 데이터, 즉 학습할때 이용하지 않는 데이터에 대한 loss이기에 꼭 꾸준히 감소하지는 않는다.

다음은 Validation accuracy 그래프이다.
![image](https://github.com/user-attachments/assets/eb631da4-ccaa-4d60-8b4a-28dc63b4e8ba)

이 그래프를 통해 확인할 수 있는 것은, 무조건 epoch를 많이 돌린다고 모델의 validation accuracy 가 오르지 않는다는 것이다. 또한 learning rate, optimizer_type 에 따라 모델별로 optimize 되는데 걸리는 epoch 수, 시간이 차이가 난다. 

결론
15가지의 하이퍼 파라미터 조합을 랜덤으로 서칭한 결과 Resnet18의 경우 batch_size = 32, epochs = 5, learning_rate: 0.001, SGD optimizer (momentum = 0.9),  weight_decay = 0.001 의 조합으로 train 한 모델이 validation accuracy 0.912 로 가장 좋은 generalization ability 를 보여주었다.

위 그래프에서 해당 모델의 추이는 serene-sweep-12 에서 확인할 수 있다.

다음으로 ResNet34 모델에 대해서 같은 형식으로 살펴볼 것이다.
총 hyperparameter searching 하는데 걸린 시간은 4h 27m 이다.

![image](https://github.com/user-attachments/assets/b70df130-d294-433b-8e8a-78219ec28058)

다음은 Train loss 그래프이다. 
![image](https://github.com/user-attachments/assets/d1b33546-e022-4626-adc0-9b03b80298ff)

다음은 Validation loss 그래프이다. 
![image](https://github.com/user-attachments/assets/71f98f66-52b6-4678-b18e-bd3fc9cb20ef)

다음은 Validation accuracy 그래프이다.
![image](https://github.com/user-attachments/assets/88aee40e-2200-4e56-9d35-d5eeb49c53e5)


각 그래프가 가지는 의미는 ResNet18 과 동일하다.

ResNet34 의 경우 경우 batch_size = 32, epochs = 10, learning_rate: 0.001, SGD optimizer (momentum = 0.9),  weight_decay = 0.001 의 조합으로 train 한 모델이 validation accuracy 0.952 로 가장 좋은 generalization ability 를 보여주었다. 해당 모델의 추이는 restful-sweep-10 으로 확인 할 수 있다.

우리가 직접 수집한 몇가지 test data로 모델의 분류 여부를 확인해보려 한다.













