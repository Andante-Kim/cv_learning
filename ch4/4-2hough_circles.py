import cv2
import sys

img=cv2.imread('apples.jpg')
#img=cv2.imread('coins.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.blur(gray,(3,3))

circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,200,param1=150,param2=20,minRadius=50,maxRadius=120)    # 허프 원 검출 ---②
#circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,80,param1=150,param2=20,minRadius=20,maxRadius=40)
# method : 검출방법, HOUGH_GRADIENT
# dp : 이미지해상도 : accumulator해상도, 1이면 두 해상도 같음
# dist : 검출된 원 중심 사이의 최소 거리
# param1 : canny의 높은 threshold
# param2 : 누적 threshold
# minRadius, maxRadius : 검출할 원 반지름 범위

if circles is not None:
    for i in circles[0]:
        cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)	# 검출된 원 그리기 ---③

cv2.imshow('Hough circles',img)

cv2.waitKey()
cv2.destroyAllWindows()