import cv2
import numpy as np

#gray=cv2.imread('morph.jpg',cv2.IMREAD_GRAYSCALE)
gray=cv2.imread('morph_j.png', cv2.IMREAD_GRAYSCALE)
t,b=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

se1=np.uint8([[0,0,1,0,0],			# 구조 요소
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]])
se2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
k=2 # 반복 횟수


b_dilation=cv2.dilate(b,se2,iterations=k)	# 팽창
b_erosion=cv2.erode(b,se2,iterations=k)	# 침식
b_opening=cv2.dilate(cv2.erode(b,se2,iterations=k),se2,iterations=k)	# 열기
#b_opening=cv2.morphologyEx(b,cv2.MORPH_OPEN, se2)
b_closing=cv2.erode(cv2.dilate(b,se2,iterations=k),se2,iterations=k)	# 닫기
# b_closing=cv2.morphologyEx(b,cv2.MORPH_CLOSE, se2)

morphology=np.vstack((b,b_dilation,b_erosion,b_opening,b_closing))
cv2.imshow('Morphology',morphology)

cv2.waitKey()
cv2.destroyAllWindows()