import cv2
import sys

img=cv2.imread('soccer.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv2.imshow('Original Image', img)

# 1 그레이 이미지
#grayImg = cv2.imread('soccer.jpg',  cv2.IMREAD_GRAYSCALE)
#grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('GrayImage', grayImg)

# 2 컬러 모델 변환
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ycrcbImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('HSVImage', hsvImg)
cv2.imshow('YCrCbImage', ycrcbImg)
cv2.imshow('RGBImage', rgbImg)

cv2.waitKey()
cv2.destroyAllWindows()
