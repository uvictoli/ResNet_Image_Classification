# 2022 - 2 HUFS 기계학습 기말 프로젝트
- 학과 : 컴퓨터전자시스템공학부
- 이름 : 유승리  

<br>
이미지 데이터와 모델 파일은 용량 제한으로 업로드할 수 없어 구글 드라이브 링크 첨부하였습니다.  

[구글 드라이브](https://drive.google.com/drive/folders/15cdZvbcXZ5NYE_a4bkA2VejoP4Zlcc7W?usp=sharing)  
HUFS 계정으로 접속하시면 액세스 권한 요청 없이 드라이브 열람 가능합니다!   
  
#### [프로젝트 설명]
스타벅스 음료 메뉴 중 일부를 사용하여 해당 메뉴들에 대한 이미지 분류 모델을 만들었습니다.   

메뉴의 이미지는 네이버 이미지, 구글 이미지, 인스타그램 크롤링으로 수집하였고,   
카페라떼, 카푸치노, 카라멜 마끼아또 등 커피 메뉴는 비슷한 외관 특성 상 분류 성능이 좋지 않아(약 0.5)  
외관 상 차이가 뚜렷한 메뉴들을 위주로 프로젝트를 진행하였습니다.

<br>  

## 이미지분류모델 실행 파일 구성 안내
#### [데이터 폴더] 	/model_image

- 인스타그램, 네이버 이미지, 구글이미지 크롤링을 통하여
 수집한 메뉴별 평균 287장의 사진파일 폴더
- 메뉴 목록 :   
	/Cold_Brew_or_Iced_Caffe_Americano_or_Iced_Coffee : 301장  
		- 아이스 콜드브루, 아이스 아메리카노, 아이스 커피  
	/Iced_Caffe_Latte_or_Dolce_Cold_Brew_or_Iced_Starbucks_Dolce_Latte : 302장  
		- 아이스 카페라떼, 돌체 콜드브루, 아이스 스타벅스 돌체라떼  
	/Iced_Caramel_Macchiato : 299장  
		- 아이스 카라멜 마키아또  
	/Iced_Grapefruit_Honey_Black_Tea : 300장  
		- 아이스 자몽허니블랙티  
	/Iced_Malcha_Latte_from_Jeju_Organic_Farm : 200장  
		- 아이스 제주 유기농 말차로 만든 라떼  
	/Iced_Mango_Passion_Fruit_Blended : 300장  
		- 망고 패션 프루트 블렌디드  
	/Iced_Mint_Chocolate_Chip_Blended : 278장  
		- 민트 초콜릿 칩 블렌디드  
	/Iced_Strawberry_Delight_Yogurt_Blended : 293장  
		- 딸기 딜라이트 요거트 블렌디드  
	/Java_Chip_Frappuccino : 300장  
		- 자바칩 프라푸치노  
	/Vanila_Cream_Cold_Brew : 300장  
		- 바닐라 크림 콜드브루  
- 폴더 구성 :   
	/model_image  
		/image_file  
			- 스타벅스 top 10 메뉴 사진 평균 287장 (위 메뉴 목록에 각 메뉴별 이미지 파일 개수 함께 기재)  
		/split_image  
			- 스타벅스 top 10 메뉴를 train, validation, test split에 따라  
			  나눠진 폴더 (4:3:3)  
			/train  
			/val  
			/test  
		/predict_image  
			- 각 메뉴별 음료 이미지 한 장을 넣었을 때 예측을 확인하기 위한 10장의 이미지  

***

#### [코드 파일] 	/model_transfer_only.ipynb

- ResNet50 finetuning 파일  
- 코드 실행 시, "모듈 불러오기 및 설치"에서 에러 없이 실행되도록 없는 모듈 install  
- 코드 실행 시, "파일 경로 설정"에서 DIR에 /model_submit 파일의 경로 설정  
- 주석 처리한 셀은 실행시킬 필요 없음  
- 코드 실행 위한 모듈 목록 :  
```
import os  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from imutils import paths
import easydict
from time import sleep
from IPython.display import clear_output
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.layers  import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.metrics import AUC
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from keras.models import load_model
```  
- 모듈 버전 정보:  
```
python		3.9.12   
matplotlib	3.5.1  
seaborn		0.11.2  
numpy		1.21.5  
easydict		1.9  
keras		2.9.0  
tensorflow	2.9.1  
scikit-learn	1.0.2  
```
***

#### [모델 파일] 	/ResNet50V2_model_final_cate  

- 저장된 모델  
- test acc 0.76, auc 0.86 성능 모델  
