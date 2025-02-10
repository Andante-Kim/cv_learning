import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('original', gray)

median = cv2.medianBlur(gray, 5)
cv2.imshow('Smooth - Median',median)

#1 다양한 크기의 필터
median1=np.hstack((gray,cv2.medianBlur(gray, 3),cv2.medianBlur(gray, 7),cv2.medianBlur(gray, 11)))
#cv2.imshow('Smooth - Median1',median1)

cv2.waitKey()
cv2.destroyAllWindows()