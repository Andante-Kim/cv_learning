import cv2     # opencv 모듈 import
import sys

img=cv2.imread('soccer.jpg')	# 이미지 읽기
grayImg = cv2.imread('soccer.jpg',  cv2.IMREAD_GRAYSCALE)   # 그레이로 이미지 읽기

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
print(type(img))
print(img.shape)

print(type(grayImg))
print(grayImg.shape)

print(img[0,0])

print(grayImg[0,0])

print(img[0,0,0], img[0,0,1], img[0,0,2])

cv2.imshow('Image Display',img)	            # 윈도우에 영상 표시
cv2.imshow('Gray Image Display',grayImg)	# 윈도우에 영상 표시

cv2.waitKey()               # 키보드 입력 대기 
cv2.destroyAllWindows()     # 모든 윈도우 제거 후 프로그램 종료