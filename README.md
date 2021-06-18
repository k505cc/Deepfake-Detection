# FIND (Fake Is Now Detected)
# Deepfake Image Detection 

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
Dataset|Images/Videos|Numbers of Dataset|Size|Unseen|Link to Download 
---|:-------:|---:|:----:|:------:|:----:
Face Forensics++|Vidoes|5000|8.42GB|X|[LINK](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)
Celeb-DF|Videos|6229|10GB|X|[LINK](https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform)
Deepfake Detection Challenge(DFDC)|Videos|100,000|90GB|X|[LINK](https://ai.facebook.com/datasets/dfdc)
Dacon Deepfake Detection Championship|Videos|1300,000|127.6GB|X|[LINK](https://dacon.io/competitions/official/235655/data)
UADFV|Videos|98|-|X|-
real data|Videos|-|-|O|-

Download Link for the dataset

## 3. Model Explanation (모델 설명)
### XceptionNet
![Xcept](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcURENc%2FbtqGdQ4oEj2%2F7kbxgeNBccVQSZMbYZn2Kk%2Fimg.png)

![SOTA Deepfake detection](https://user-images.githubusercontent.com/76925087/122522090-873fd280-d050-11eb-9f81-1fde7a3b1714.png)

Currently ensembles of EfficientNet & XceptionNet are listed as the latest models developed for deepfake face detection sector. We picked to use XceptionNet as our backbone model to extract features. 

By using XceptionNet’s depth wise separable convolution we reduced the computation capability by 8 times.  We adopted XceptionNet since it has accelerating performance in distinguishing between real and fake images. 


## 4. Data Preprocessing & Model Ensemble (적용된 전처리와 모델 적용)

![그림2](https://user-images.githubusercontent.com/76925087/122522214-aa6a8200-d050-11eb-8145-3e97bdf00010.png)

* Face Cropping & Masking
* Image Compression, Saturation, Flipping 
* Noise Data Removal & Loss Function Conversion
* Model Ensemble of 2 Best models of XceptionNet


## 5. Final Accuracy & Recall 

* Accuracy : 0.783
* Recall : 0.82
* Confusion Matrix

![그림3](https://user-images.githubusercontent.com/76925087/122522877-63c95780-d051-11eb-83a9-5b65fa700f09.png)


## Reference

