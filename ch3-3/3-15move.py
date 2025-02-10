import cv2
import numpy as np

def contain(p, shape):                              # 좌표(y,x)가 범위내 인지 검사
    return 0<= p[0] <shape[0] and 0<= p[1] <shape[1]

img=cv2.imread('flower.jpg')
dst = np.zeros(img.shape, img.dtype)  # 입력 이미지(img)와 같은 크기의 출력 이미지(검정)

pt = (50,100)   # 입력 이미지를 pt만큼 이동
for i in range(dst.shape[0]):  # 출력 이미지의 각 픽셀값을 계산 – 역방향 사상
    for j in range(dst.shape[1]):
        x, y = np.subtract((j, i), pt)  # 역변환 : 출력 이미지를 (-j,-i) 이동한 위치 결정
        if contain((y, x), img.shape):
            dst[i, j] = img[y, x]  # 출력 이미지 픽셀 결정

move=np.hstack((img,dst))
cv2.imshow('Move', move)

cv2.waitKey()
cv2.destroyAllWindows()