# Deepfake detection

## 0. 프로젝트 소개
There has been recent rise in sales and abusive usages of deepfake videos, fake news, political and social issues along with the deepfake videos.    Phishing crimes using deepfake are assumable.     

By detecting modified and fake images online, we are pursuing to seek below,
* Building safety guidelines on reckless false video production with AI technology
* Prevent fake videos being uploaded our most public video content platform – Youtube
* Prevent dissemination of false information with unguaranteed images/ videos     

You can clone this repository into your favorite directory:
```
$ git clone https://github.com/Fake-Is-Now-Detected/deepfake_detection.git
```

## 1. Requirement (환경설정)
* Ubuntu 18.04
* Tensorflow 2.4.1
* Keras
* scikit-learn
* Matplotlib
* Numpy
* Gradio

## 2. Dataset (데이터셋)
종류|사진/영상|갯수|size|unseen|링크
---|:-------:|---:|:----:|:------:|:----:
Face Forensics++|영상|5000|8.42GB|X|[LINK](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)
Celeb-DF|영상|6229|10GB|X|[LINK](https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform)
Deepfake Detection Challenge(DFDC)|영상|100,000|90GB|X|[LINK](https://ai.facebook.com/datasets/dfdc)
Dacon 딥페이크 탐지 경진대회|영상|1300,000|127.6GB|X|[LINK](https://dacon.io/competitions/official/235655/data)
UADFV|영상|98|-|X|-
real data|영상|-|-|O|-

Download Link for the dataset

## 3. Model Explanation (모델 설명)
### XceptionNet
![Xcept](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcURENc%2FbtqGdQ4oEj2%2F7kbxgeNBccVQSZMbYZn2Kk%2Fimg.png)
* Currently ensembles of EfficientNet & XceptionNet are listed as the latest models developed for deepfake face detection sector. We picked to use XceptionNet as our backbone model to extract features. 

By using XceptionNet’s depth wise separable convolution we reduced the computation capability by 8 times.  We adopted XceptionNet since it has accelerating performance in distinguishing between real and fake images. 


## 4. Preprocessing (전처리 기술)
### MTCNN

### data augmentation


## 5. Results (결과)
정확도, confusion matrix
## Reference

