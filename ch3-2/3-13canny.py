import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)

# Sobel 에지 : Canny 1~2 단계
blur=cv2.blur(gray,(3,3))	# Canny 1단계 : (전처리과정) 노이즈 제거
sobel_grad_x = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)	# Canny 2단계 : 소벨 에지 검출
sobel_grad_y = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_grad_x)
sobel_y = cv2.convertScaleAbs(sobel_grad_y)
sobel_edge = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)

_, th = cv2.threshold(sobel_edge, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # sobel edge에서 나온 값을 임계값으로 만들기(임계값 1개)

# Canny : 다양한 임계값
canny1=cv2.Canny(gray,50,150)	# Tlow=50, Thigh=150으로 설정
canny2=cv2.Canny(gray,100,200)	# Tlow=100, Thigh=200으로 설정

canny=np.hstack((sobel_edge, th, canny1, canny2))
cv2.imshow('Canny',canny)

cv2.waitKey()
cv2.destroyAllWindows()