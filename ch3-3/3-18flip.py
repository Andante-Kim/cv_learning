import cv2
import numpy as np

img=cv2.imread('rose_small.png')

(h, w) = img.shape[:2]      # 이미지의 크기

# 대칭
#Horizontal
src_points1 = np.float32([ [0,0], [0,h-1], [w-1,0] ])
dst_points1 = np.float32([ [w-1,0], [w-1,h-1], [0,0] ])

#Vertical
src_points2 = np.float32([ [0,0], [w-1,0], [0,h-1] ])
dst_points2 = np.float32([ [0,h-1], [w-1,h-1], [0,0] ])

#Origin
src_points3 = np.float32([ [0,0], [0,h-1], [w-1,0] ])
dst_points3 = np.float32([ [w-1,h-1], [w-1,0], [0, h-1] ])

affine_matrix1 = cv2.getAffineTransform(src_points1, dst_points1)
affine_matrix2 = cv2.getAffineTransform(src_points2, dst_points2)
affine_matrix3 = cv2.getAffineTransform(src_points3, dst_points3)

img_symmetry1 = cv2.warpAffine(img, affine_matrix1, (w,h)) # 3번째 결과영상크기(가로,세로)
img_symmetry2 = cv2.warpAffine(img, affine_matrix2, (w,h))
img_symmetry3 = cv2.warpAffine(img, affine_matrix3, (w,h))

img_symmetry=np.vstack((img,img_symmetry1,img_symmetry2,img_symmetry3))
cv2.imshow('Symmetry', img_symmetry)

# openCV 함수 : flip
img_symmetry2=np.vstack((img,cv2.flip(img,1),cv2.flip(img,0),cv2.flip(img,-1)))
#cv2.imshow('Symmetry2', img_symmetry2)

cv2.waitKey()
cv2.destroyAllWindows()