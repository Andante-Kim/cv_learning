import cv2
import numpy as np

gray=cv2.imread('original-thresholding.png', cv2.IMREAD_GRAYSCALE) # BGR 컬러 영상을 명암 영상으로 변환하여 저장

# 1. 전역 threshold 값 적용 : 1개의 threshold 값을 모든 픽셀에 적용
ret, img_binary50 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
cv2.putText(img_binary50,"BINARY-50",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0, 2)
ret, img_binary100 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
cv2.putText(img_binary100,"BINARY-100",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0, 2)
ret, img_binary200 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.putText(img_binary200,"BINARY-200",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 255, 2)

img_binary=np.hstack((gray,img_binary50, img_binary100, img_binary200))
cv2.imshow('threshold',img_binary)

# 2. Adaptive threshold : 각 픽셀마다 다른 threshold 값을 계산하여 적용
img_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 7)
img_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
cv2.imshow('adaptive threshold',img_adaptive)

cv2.waitKey()
cv2.destroyAllWindows()