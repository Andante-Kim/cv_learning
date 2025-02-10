import cv2
import numpy as np
import matplotlib.pyplot as plt # matplotlib 설치
#matplotlib : 파이썬에서 데이타를 차트나 플롯(Plot)으로 그려주는 라이브러리 패키지로서 가장 많이 사용되는 데이타 시각화(Data Visualization) 패키지

gray = cv2.imread("CT-brain-image.jpg", cv2.IMREAD_GRAYSCALE)
print(gray.shape)
h_gray=cv2.calcHist([gray],[0],None,[256],[0,256]) # 1바이트 gray의 0번 채널에서 히스토그램 구함
plt.plot(h_gray,'k', linewidth=3)   # linestyle='solid'

print(h_gray)

gray2 = cv2.subtract(255,gray)  # 반전 이미지
h_gray2=cv2.calcHist([gray2],[0],None,[256],[0,256]) # 1바이트 gray의 0번 채널에서 히스토그램 구함
plt.plot(h_gray2,'m', linestyle='dashed', linewidth=3)

img_gray=np.hstack((gray,gray2))
cv2.imshow('Gray image',img_gray)

plt.show()

