import cv2
import numpy as np

img=cv2.imread('lenna256.png')
img=cv2.imread('rose_small.png')
cv2.imshow('Original', img)

(h, w) = img.shape[:2]      # 이미지의 크기
(cX, cY) = (w // 2, h // 2) # 이미지의 중심

# 회전
M45 = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)   # 이미지의 중심을 중심으로 이미지를 회전합시키는 행렬
rotated_45 = cv2.warpAffine(img, M45, (w, h))  # 회전 행렬을 이미지에 적용하여 이미지 회전시킴

img_rotate=np.hstack((img,rotated_45))
cv2.imshow('Rotation', img_rotate)

# 회전 affine
#90 rotate
# M90 = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)   # 이미지의 중심을 중심으로 이미지를 회전합시키는 행렬
src_points = np.float32([[0,0], [0,h-1], [w-1,0]])      # 입력 이미지의 위치
dst_points = np.float32([[0,w-1], [h-1,w-1], [0,0]])    # 출력 이미지의 위치
affineM90 = cv2.getAffineTransform(src_points, dst_points) # 3개의 좌표점 쌍으로 변환시키는 행렬

rotated_90 = cv2.warpAffine(img, affineM90, (w, h)) # affine 변환 행렬을 이미지에 적용하여 이미지 회전시킴, (h,w)
cv2.imshow('Rotate - 90', rotated_90)

cv2.waitKey()
cv2.destroyAllWindows()