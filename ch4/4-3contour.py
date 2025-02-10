import cv2
import sys

#img=cv2.imread('soccer.jpg')	 # 영상 읽기
img=cv2.imread('shapes2.png')
img=cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

img2 = img.copy()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny=cv2.Canny(gray,100,200)    # 에지 영상
cv2.imshow('Canny edges',canny)

contours,hierarchy=cv2.findContours(canny,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_NONE)
#contours,hierarchy=cv2.findContours(canny,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)

lcontours=[]
for i in range(len(contours)):
    if contours[i].shape[0]>100:	# 길이가 100보다 크면
        lcontours.append(contours[i])

cv2.drawContours(img, contours, -1, (255, 255, 0), 2)  # 모든 contour(길이 100이하 포함)
cv2.imshow('Contours - all',img)

cv2.drawContours(img2, lcontours, -1,(255, 0, 255),2)   # 길이 100이상 contour만
cv2.imshow('Contours - long',img2)

cv2.waitKey()
cv2.destroyAllWindows()