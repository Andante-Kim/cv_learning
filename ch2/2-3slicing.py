import cv2
import sys
import numpy as np

img=cv2.imread('soccer.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

print(type(img))
print(img.shape)

# 1 numpy의 슬라이싱slicing
cv2.imshow('Upper left half', img[0:img.shape[0]//2, 0:img.shape[1]//2,:])
cv2.imshow('Central half', img[img.shape[0]//4:3*img.shape[0]//4,img.shape[1]//4:3*img.shape[1]//4,:])

cv2.imshow('Red channel', img[:,:,2])   # OpenCV는 BGR로 색상값 저장
#cv2.imshow('Green channel', img[:,:,1])
#cv2.imshow('Blue channel', img[:,:,0])

# 2 OpenCV의 split
b,g,r = cv2.split(img)
black = np.zeros((img.shape[0],img.shape[1]), np.uint8)
img_B = cv2.merge((b, black, black))
img_R = cv2.merge((black,black, r))
cv2.imshow('Blue in Color', img_B)
cv2.imshow('Red in Color', img_R)

cv2.waitKey()
cv2.destroyAllWindows()