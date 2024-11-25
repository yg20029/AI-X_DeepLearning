# TITLE : 100 Sports Image Classification



## Members:
> ##### 김재준, 데이터사이언스학과, apolon0617@gmail.com
> ##### 박찬형, 데이터사이언스학과, chanhyoungpark@naver.com
> ##### 조윤규, 데이터사이언스학과, yg20029@gmail.com


## I.	Proposal (Option 1or 2)–This should be filled bysometime in early Nov.
### -Motivation: Why are you doing this?
이 프로젝트를 통해 100가지 스포츠 이미지를 분류하며, 딥러닝과 이미지 분류의 핵심 기술을 탐구하고 이를 실제 문제에 적용해보는 경험을 쌓고자 합니다. 또한, 이 프로젝트는 딥러닝 모델 설계, 전이 학습, 하이퍼파라미터 튜닝 등 딥러닝의 여러 중요한 개념을 실습할 수 있는 기회를 제공합니다.
이러한 과정을 통해 딥러닝에 대한 이해도를 높이고, 실제 데이터를 다루며 문제 해결 능력을 키우는 데 도움이 될 것이라 생각합니다. 이 데이터셋을 활용해 높은 정확도를 달성함으로써, 향후 이미지 분석 프로젝트에 기여하는 것을 목표로 하고 있습니다.

### -What do you want to see at the end?
이 프로젝트의 최종 목표는 100개의 다양한 스포츠 이미지를 정확히 분류할 수 있는 딥러닝 모델을 구축하는 것입니다. 이를 통해 테스트 데이터에서 90% 이상의 분류 정확도를 달성하고, 전이 학습 등의 어러 모델링 기술들을 활용 및 경험해가며 점차적으로 정확도를 높여가는 성취를 기대하고 있습니다. 

## II.	Datasets-Describing your dataset 

이번 프로젝트에서 사용할 데이터셋은 100개의 서로 다른 스포츠 카테고리를 포함하는 이미지 데이터셋입니다. 이 데이터셋은 주로 인터넷 검색을 통해 수집되었으며, 다음과 같은 특징을 가지고 있습니다:
### 1.	데이터셋 구조:
> * #### 훈련 데이터 (Train Set): 총 13,493개의 이미지로, 모델 학습을 위한 주요 데이터입니다.
> * #### 검증 데이터 (Validation Set): 총 500개의 이미지로, 모델의 하이퍼파라미터 조정과 성능 검증에 사용됩니다.
> * #### 테스트 데이터 (Test Set): 총 500개의 이미지로, 모델의 최종 성능을 평가하는 데 사용됩니다.
### 2.	이미지 정보:
> * #### 모든 이미지는 224x224x3 크기의 RGB 이미지로, .jpg 형식으로 저장되어 있습니다.
> * #### 데이터는 각 스포츠 카테고리별로 정리되어 있어 ImageFolder와 같은 PyTorch의 데이터 구조로 쉽게 로드할 수 있습니다.
### 3.	데이터 포맷:
> * #### 이미지 데이터 외에도 CSV 파일이 포함되어 있어, 각 이미지의 상대 경로, 클래스 레이블, 데이터 세트(훈련, 검증, 테스트)를 확인할 수 있습니다.
> * #### 이러한 구조는 PyTorch의 DataLoader와 같은 데이터 로더를 쉽게 사용할 수 있게 합니다.

이 데이터셋을 활용하여 스포츠 이미지를 학습하고, 모델이 다양한 스포츠 활동을 정확히 분류할 수 있도록 하는 것이 목표입니다. 이 프로젝트를 통해 이미지 분류 모델의 구축 및 평가, 전이 학습 적용 등 여러 딥러닝 기술을 직접 실험할 수 있습니다


## III.	Methodology -Explaining your choice of algorithms (methods)-Explaining features (if any)

이번 프로젝트에서 이용할 모델의 기반은 CNN 모델입니다.
CNN 모델은 이미지, 영상을 분석하기 위한 패턴을 직접 찾아 학습하고, 찾은 패턴을 이용하여 이미지를 분류합니다. CNN 이전의 모델인 Fully Connected layer 를 이용할 시, 지나치게 많은 모델 파라미터로 인하여 높은 overfitting 경향을 보이지만 CNN은 이를 방지 할 수 있습니다.
CNN은 이미지의 각 픽셀들은 주변의 픽셀과 연관되어 있다는 가정 아래 시작합니다. 이미지에서는 비슷한 패턴이 다양한 픽셀 위치에서 생길 수 있으며 머신러닝 알고리즘은 이를 이용해 더 뛰어난 generalization 을 이루고자 합니다.

### Convolution:
CNN에서 가중 평균(weighted average) 연산은 이미지나feature map의 모든 위치에서 수행되며, 이를 통해 다양한 패턴이나 특징을 탐지할 수 있습니다. 어떠한 패턴이 weight를 통과하느냐에 따라 다른 결과를 추출해 낼 수 있습니다.

### Convoultion Layer:
컨볼루션 레이어는 입력 이미지를 Filter(Kernel)를 이용하여 탐색하면서 이미지의 feature들을 추출하고, 추출한 feature들을 Feature Map으로 생성합니다.
<img width="565" alt="Screenshot 2024-11-25 at 22 48 41" src="https://github.com/user-attachments/assets/52923fa9-8ea9-496c-b089-a397a0ae16ce">

Feature Detectors example:
<img width="450" alt="image" src="https://github.com/user-attachments/assets/6d81e728-f120-45da-b9b9-714f8e847754">






