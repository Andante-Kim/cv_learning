import cv2
import numpy as np

gray=cv2.imread('lenna256.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original',gray)

# 1 blur
# 평균값 마스크 직접 선언 -> 마스크가 커지면 불편함(다 입력해야 하므로..)
faverage=np.array([[1.0/9.0, 1.0/9.0, 1.0/9.0],
                   [ 1.0/9.0, 1.0/9.0, 1.0/9.0],
                   [ 1.0/9.0, 1.0/9.0, 1.0/9.0]])
average1 = cv2.filter2D(gray, -1, faverage)
average2 = cv2.blur(gray,(3,3))
cv2.imshow('Average - filter2D',average1)
cv2.imshow('Average - blur',average2)

#2 다양한 크기의 스무딩 필터
smooth=np.hstack((gray,cv2.blur(gray,(3,3),0.0),cv2.blur(gray,(7,7),0.0),cv2.blur(gray,(11,11),0.0),cv2.blur(gray,(15,15),0.0)))
cv2.imshow('Smooth',smooth)

#3 Gaussian
blur5 = cv2.blur(gray, (5,5), 0.0)
gaussian = cv2.GaussianBlur(gray,(5,5),1.0)
gaussian1=np.hstack((gray,blur5,gaussian))
cv2.imshow('Smooth - gaussian1',gaussian1)

#4 Gaussian - 다양한 크기의 필터
gaussian2=np.hstack((gray,cv2.GaussianBlur(gray,(3,3),1.0),cv2.GaussianBlur(gray,(7,7),1.0),cv2.GaussianBlur(gray,(11,11),1.0),cv2.GaussianBlur(gray,(15,15),1.0)))
cv2.imshow('Smooth - Gaussian2',gaussian2)

#5 Gaussian - sigma는 보통 0, 값이 클수록 블러링 효과 커짐
gaussian3=np.hstack((gray,cv2.GaussianBlur(gray,(5,5),1.0),cv2.GaussianBlur(gray,(5,5),3.0),cv2.GaussianBlur(gray,(5,5),7.0),cv2.GaussianBlur(gray,(5,5),11.0)))
cv2.imshow('Smooth - Gaussian3',gaussian3)

cv2.waitKey()
cv2.destroyAllWindows()