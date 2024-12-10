# AI-X_DeepLearning
AI-X 딥러닝 : 최종 프로젝트를 위한 블로그

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

100가지 스포츠 class 당 하나씩 그림을 출력해 보았다.

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/17f13efb-1fb1-478e-b22f-175ab982d0a4">


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

ResNet의 참고 문헌은 다음과 같다. 
https://arxiv.org/pdf/1512.03385

ResNet 모델에서 조절 할 수 있는 하이퍼 파라미터는 learning_rate, batch_size, epochs, weight_decay, optimizer등이 존재한다. 

learning_rate는 모델이 가중치를 업데이트할 때 사용하는 스텝 크기를 조절하는데, 이를 통해서 학습 속도와 학습 정확도를 조절 할 수 있다. 

batch_size는 한 번의 학습 업데이트에서 처리되는 데이터 샘플의 수로 사이즈가 클수록 학습이 안정적이고 속도가 증가하지만, 많은 메모리를 사용한다. 어떤 GPU를 사용하느냐에 따라 선택지가 바뀔 수 있다. 

epochs는 전체 데이터셋을 몇 번 학습하는지 나타내는데, 대부분의 경우 높을수록 로스가 감소하게 되지만 너무 높은 경우 데이터의 overfitting이 일어날 수 있다. Train loss 와 Validation loss의 추이를 보면서 overfitting, underfitting 문제를 잘 확인해야 한다. 

optimizer는 가중치를 업데이트하는 방식이다. SDG와 ADAM의 두 가지를 사용했다. SDG는 기울기(Gradient)를 사용해 손실 함수의 값을 최소화하는 방향으로 모델의 가중치를 업데이트하는 기본적인 최적화 알고리즘이다 우리가 사용한 ResNet에서는 기본 SGD optimizer에 0.9의 momentum을 건 optimizer를 사용했다. 일반 SGD optimizer에 모멘텀을 적용하면 이전 기울기도 최적화에 이용해 느린 최적화와 같은 일반 SGD의 문제점을 해결한다. ADAM은 학습률을 자동으로 조정하며 이전 단계의 기울기의 평균과 분산을 통해 SGD의 단점을 개선한 최적화 알고리즘이다.

weight_decay는 모델의 가중치에 대해 L2 정규화를 적용하여 과적합을 방지하는 데 사용한다. weight_decay 값은 optimizer 내에서 선언하면 된다. 

### Data preprocessing

머신러닝에서 모델만큼이나 중요한것이 data preprocessing 이다. 

우리가 가지고 있는 train 데이터는 총 13493개다. 하지만 총 class 수는 100 개로 데이터셋의 크기가 많다고 볼 수 는 없다. 따라서 overfitting을 줄이고 generalization ability 를 높이기 위해 적당한 augmentation 을 줄 필요가 있다. 
우리가 적용한 첫번째 augmentation 은 가장 기초적인 것 중 하나로서 사진을 0.5(default) 확률로 좌 우로 뒤집는 것이다. 이를 위해 transforms.RandomHorizontalFlip() 을 이용했으며, 다음 그림은 이 augmentation을 적용한 예시이다.

![image](https://github.com/user-attachments/assets/7b47fd47-db25-4ef6-a0aa-ca433fdb6eee)


다음으로 이미지를 랜덤한 비율로 자른 후 이를 지정 픽셀(224*224)로 확대하여 데이터로 활용하는 transforms.RandomResizedCrop(224) 도 활용하였다. 예시는 다음과 같다.

![image](https://github.com/user-attachments/assets/982c00c1-2daa-4662-bd64-7e473289934a)


또한 우리는 IMAGENET1K_V1 데이터셋에 최적화된 model parameter을 초기 파라미터로 두고 train을 하였기에  IMAGENET1K_V1의 각 채널(R, G, B)의 평균과 표준편차를 가지고 데이터들에 normalization 을 진행하였다.
또한 모든 이미지 데이터는 224*224 모양으로 변환하여 사용하였다.

다음은 사용한 DataPreprocessing 코드이다

![image](https://github.com/user-attachments/assets/4a43671d-2a37-4339-ba87-8f073e78224d)

DataPreprocessing 과정을 거치고 나면, 컴퓨터가 인식하는 이미지는 다음과 같다. 

![image](https://github.com/user-attachments/assets/fee1674d-99a5-4f41-9ecf-1f30e382304c)


## IV. Evaluation & Analysis

### Validation

우리는 두 가지 ResNet 모델(ResNet18, ResNet34) 로 스포츠 데이터를 학습해 보았다.
또한 각 모델에 대해 여러 하이퍼 파라미터 조합으로 학습을 진행하였다.  사용한 하이퍼 파라미터 조합은 다음과 같다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/b6b699eb-d121-4d72-b625-5a8028af9ad1">

사용한 두 모델, 그리고 위의 하이퍼파라미터 조합 총 324가지이다. 하지만 이 모든 조합으로 모델 트레이닝을 하기에는 언제나 그렇듯이 시간과 컴퓨팅 용량이 충분치 않다. 따라서 우리는 랜덤으로 모델 15가지를 시도해 보았다.

하이퍼파라미터 튜닝을 위해서 우리는 wandb라는 파이썬 라이브러리를 사용했다. Wandb는 머신러닝 및 딥러닝 프로젝트의 추이 추적, 시각화, 모델 관찰 등을 지원하는 도구로 특히 하이퍼파라미터 서칭 이후 각 모델들의 차이를 시각화 하는데 좋다. 

Wandb sweep 중 한 hyperparameter 조합으로 model train을 한 후 나온 파이썬 로그는 다음과 같다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/aa086cc7-8ba8-491b-8e53-1b2359ae7886">


wandb 워크스페이스에서 다음 그림처럼 한 sweep 별 시각화 자료들을 볼 수 있다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/d8a3b9c3-9df4-4245-8ce3-9a8afcd3fa73">

그럼 나온 결과들을 하나하나 풀어보자. 
#### ResNet18 
총 hyperparameter searching 하는데 걸린 시간은 3h 29m 이다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/a13b3a42-a0fe-4a05-b324-a5f4e8791ed5">


전체적인 하이퍼파라미터 조합에 따른 validation accuracy 추이는 다음과 같다.
이제 각 모델별 train epoch에 따른 loss와 validation accuracy 를 보자. 

Train loss 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/ee9c0fcd-1296-4e89-a78e-da7b2702b009">

모든 모델의 train loss가 감소하는 것을 볼 수 있다.

Validation loss  

<img width="600" alt="image" src="https://github.com/user-attachments/assets/f3f2316a-46e7-4ae0-b9f3-35a308e23587">


마찬가지로 validation loss 도 감소하는 것을 볼 수 있다. 
하지만 train loss와 다르게 가끔 튀는 현상이 발생한다. 모델은 train data 를 기반으로 학습이 되지만 validation loss는 모델이 처음 보는 데이터, 즉 학습할때 이용하지 않는 데이터에 대한 loss이기에 꼭 꾸준히 감소하지는 않는다.

Validation accuracy 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/eb631da4-ccaa-4d60-8b4a-28dc63b4e8ba">

이 그래프를 통해 확인할 수 있는 것은, 무조건 epoch를 많이 돌린다고 모델의 validation accuracy 가 오르지 않는다는 것이다. 또한 learning rate, optimizer_type 에 따라 모델별로 optimize 되는데 걸리는 epoch 수, 시간이 차이가 난다. 

결론
15가지의 하이퍼 파라미터 조합을 랜덤으로 서칭한 결과 Resnet18의 경우 batch_size = 32, epochs = 5, learning_rate: 0.001, SGD optimizer (momentum = 0.9),  weight_decay = 0.001 의 조합으로 train 한 모델이 validation accuracy 0.912 로 가장 좋은 generalization ability 를 보여주었다.

위 그래프에서 해당 모델의 추이는 serene-sweep-12 에서 확인할 수 있다.

#### ResNet34
총 hyperparameter searching 하는데 걸린 시간은 4h 27m 이다.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/b70df130-d294-433b-8e8a-78219ec28058">

Train loss 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/d1b33546-e022-4626-adc0-9b03b80298ff">


Validation loss

<img width="600" alt="image" src="https://github.com/user-attachments/assets/71f98f66-52b6-4678-b18e-bd3fc9cb20ef">


Validation accuracy 

<img width="600" alt="image" src="https://github.com/user-attachments/assets/88aee40e-2200-4e56-9d35-d5eeb49c53e5">



각 그래프가 가지는 의미는 ResNet18 과 동일하다.

ResNet34 의 경우 경우 batch_size = 32, epochs = 10, learning_rate: 0.001, SGD optimizer (momentum = 0.9),  weight_decay = 0.001 의 조합으로 train 한 모델이 validation accuracy 0.952 로 가장 좋은 generalization ability 를 보여주었다. 

해당 모델의 추이는 restful-sweep-10 으로 확인 할 수 있다.

### Test
하이퍼파라미터 서칭을 통해 최적의 모델을 찾은 후 이를 이용해 제공된 500개의 test set을 이용해 final test accuracy를 도출해 보았다.

물론 모델을 처음 훈련할 때 사용한 validation transform 을 먼저 test set 이미지에 적용해야 한다. 여기에 IMAGENET1K_V1 의 평균과 표준편차로 normalization 을 진행하는 것과 224*224 로 픽셀 사이즈를 맞추는 것이 있다. 간단한 변형을 거친 후 모델이 보는 이미지는 다음과 같다.

![image](https://github.com/user-attachments/assets/e3fd2624-8abe-4d2a-8387-f951bffaeeae)

이미지 위에 있는 class가 모델이 예측한 class이다.

Test set 에 대한 최종 정확도는 96% 가 나왔다.

![image](https://github.com/user-attachments/assets/bd275b0b-ca38-402f-98b9-d89f71bc05f8)

이외에도 우리가 이전에 직접 찍었던 사진들을 모델에 넣어 보았다.

![image](https://github.com/user-attachments/assets/add30114-f87c-40d9-bb3e-4b3c7aaf7dad) ![image](https://github.com/user-attachments/assets/8deca5ac-b555-4128-92ab-d113ad85d47a)

![image](https://github.com/user-attachments/assets/1375cbaa-799e-4b3e-b7eb-96b5e5751244) ![image](https://github.com/user-attachments/assets/6d8d1463-a2b9-49b7-b80b-5ed3ae40818c)

축구 사진이 ampute football로 분류 된 이유는 100개의 클래스에 축구가 없었기 때문이다. 하지만 장애인 축구인 ampute football이 class로 있었기에 축구와 가장 가까운 종목인 ampute football로 분류된 것으로 보인다.

결론적으로 우리는 모델이 본적 없는 새로운 데이터에 대한 검증을 통해 스포츠 카테고리 분류에 모델이 우수한 성능을 보임을 확인할 수 있었다.




## V. Related Work (e.g., existing studies)

### 1. 라이브러리 및 프레임워크

PyTorch: PyTorch는 모델 개발을 위한 주요 딥러닝 프레임워크로 사용되었다. 텐서 등의 다양한 수학 함수가 포함되어 있으며, numpy와 유사한 구조를 가진다.

Torchvision: 사전 학습된 모델(예: ResNet-34) 및 데이터 증강, 데이터셋 로드와 같은 필수 유틸리티에 사용되었다.

wandb: wandb 사이트와 연결하여 모델의 하이퍼 파라미터 서칭을 하는 과정에서 사용되었다. 

### 2. 도구 및 플랫폼

Google Colab: GPU 지원을 통해 효율적인 모델 학습을 제공한 주요 개발 환경으로 사용되었다.

Kaggle: 고품질의 라벨링된 데이터를 제공하여 프로젝트에서 사용된 100가지 스포츠 데이터셋의 출처가 되었다.

wandb : 딥러닝 실험 과정을 손쉽게 Tracking하고, 시각화해서 하이퍼파라미터 최적화, 실험 추적 및 결과 시각화를 지원하는 Tool이다 

### 3. 기존 연구 및 문헌

ResNet: Deep Residual Learning for Image Recognition (He et al., 2015): ResNet 모델에 대한 기초적인 논문으로, ResNet의 구성을 이해하고, 하이퍼 파라미터의 역할을 이해하는 과정에 도움을 주었다.

## VI. Conclusion: Discussion

이번 프로젝트를 통해 ResNet를 활용한 스포츠 이미지 분류 시스템의 구축과 최적화에 성공하였다. 특히 사전 학습된 ResNet모델은 적은 epoch로도 높은 정확도를 달성하며, 전이 학습의 효과를 확실히 입증하였다.
하이퍼파라미터 탐색 과정에서 랜덤 하이퍼파라미터 서칭(Random Search)을 활용하여 학습률, 배치 크기, 옵티마이저 등의 다양한 조합을 효율적으로 탐색할 수 있었다. 이 과정은 전체 파라미터 공간을 탐색하지 않으면서도 최적의 조합을 빠르게 찾는 데 큰 도움을 주었으며, 하이퍼파라미터가 모델 성능에 미치는 영향을 체계적으로 분석할 수 있었다. 
또한, 제한된 데이터셋과 적은 학습 시간에도 불구하고 ResNet모델의 강력한 Feature extraction능력을 통해 뛰어난 성능을 달성하였다. 이는 데이터가 부족한 환경에서도 사전 학습된 모델을 활용하면 효과적인 성능 향상이 가능함을 보여준다.
Weights & Biases를 사용한 실험의 실시간 메트릭 추적과 시각화는 실험 과정을 효율적이고 투명하게 관리할 수 있도록 도왔다. 하이퍼파라미터 변경에 따른 성능 변화를 직관적으로 확인할 수 있었으며, 이는 실험의 생산성과 신뢰성을 크게 높였으며 시각화가 프로젝트 진행에 미치는 영향을 체감할 수 있었다.
이번 프로젝트는 딥러닝 모델 개발 및 최적화 과정에서 얻은 중요한 통찰과 함께 전이 학습과 하이퍼파라미터 서칭의 실질적인 가치를 확인할 수 있는 기회였다. 향후 연구에서는 데이터 증강 기법의 다양화를 통해 데이터셋 불균형 문제를 개선하고, 보다 복잡한 모델 아키텍처를 도입하여 추가적인 성능 향상을 목표할 예정이다.












