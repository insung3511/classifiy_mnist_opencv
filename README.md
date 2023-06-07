# Classification MNIST Datasets using OpenCV, Numpy

In this project, I will classification MNIST Datasets using OpenCV Convolution, Mean Squared Error and HoG (Histogram Of Oriented Gradient). This commonly worked as handcraft machine learning, becuase all of feature extraction phase.

# Whole phase

All phase will be 2 sections, train phase and test phase.

## Train Phase

학습 과정은 데이터의 특성을 추출하고 테스트 과정에서 입력 이미지와 실제 이미지를 비교하기 위한 용도로 쓰입니다. 해당 과정은 train.py 에서 이루어지며 Feature Extraction은 feature_extraction.py 에서 진행이 됩니다. 학습된 데이터를 우리는 __Train Threshold__ 라고 정의합니다. Train Threshold는 데이터는 Train Phase에서 Feature Extraction 거친 이미지에서 각각의 한 레이블에 대한 평균값을 구해져 있습니다.

## Test Phase

Test Phase 혹은 Prediction 에서는 입력 이미지와 Train Phase에서 거친 Feature Extraction을 동일하게 거치고 MSE로 비교를 합니다. 여기서 MSE가 가장 낮은 레이블이 가장 높은 확률로 동일한 이미지로 구별합니다.