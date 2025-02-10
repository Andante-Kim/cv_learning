import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)
gray=cv2.imread('coins.png',cv2.IMREAD_GRAYSCALE)
#gray=cv2.imread('check.png',cv2.IMREAD_GRAYSCALE)

blur=cv2.blur(gray,(3,3))	# 에지 검출의 전처리과정 : 스무딩, 잡음 제거

#1차 미분 : Prewitt
prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) # 수직 필터
prewitt_filter_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) # 수평 필터
prewitt_grad_x = cv2.filter2D(blur, -1, prewitt_filter_x) # 수직 에지
prewitt_grad_y = cv2.filter2D(blur, -1, prewitt_filter_y) # 수평 에지
prewitt_x = cv2.convertScaleAbs(prewitt_grad_x)   # 절대값을 취해 양수 영상으로 변환, 수직 에지
prewitt_y = cv2.convertScaleAbs(prewitt_grad_y)   # 수평 에지
prewitt_edge = cv2.addWeighted(prewitt_x,0.5, prewitt_y,0.5,0)  # 에지 강도 계산, 수평 + 수직
#print('[100,100]의 그래디언트는 ',prewitt_x[100,100],',',prewitt_y[100,100])
#print(prewitt_edge[100,100])
prewitt=np.hstack((blur,prewitt_x,prewitt_y,prewitt_edge))
cv2.imshow('Prewitt',prewitt)

#1차 미분 : Sobel
sobel_grad_x = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)	# 소벨 연산자 적용 - prewitt과 달리 함수 존재 # 수직 에지
sobel_grad_y = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3) # 수평 에지
sobel_x = cv2.convertScaleAbs(sobel_grad_x)
sobel_y = cv2.convertScaleAbs(sobel_grad_y)
sobel_edge = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)	# 수평 + 수직
sobel=np.hstack((sobel_x,sobel_y,sobel_edge))
cv2.imshow('Sobel',sobel)

cv2.waitKey()
cv2.destroyAllWindows()