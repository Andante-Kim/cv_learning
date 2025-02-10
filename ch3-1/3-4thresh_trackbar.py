import cv2
import numpy as np

def onChange(value):               # 트랙바 콜백 함수, value는 트랙바의 현재 위치(값)
    global gray, bar_name, title   # 전역 변수 참조

    th = cv2.getTrackbarPos(bar_name, title)
    ret, img_thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)   # Trackbar의 현재 위치 값으로 이진화
    cv2.imshow(title, img_thresh)

gray=cv2.imread('lenna512.png', cv2.IMREAD_GRAYSCALE) # BGR 컬러 영상을 명암 영상으로 변환하여 저장

title = 'Trackbar Event'    # 이미지 창 이름 : 트랙바가 추가될 창
cv2.imshow(title, gray) # 트랙바 생성 전에 이미지 창 생성

bar_name = "Threshold"      # 트랙바 이름
cv2.createTrackbar(bar_name, title, gray[0][0], 255, onChange)   # 트랙바 콜백 함수 등록

cv2.waitKey(0)
cv2.destroyAllWindows()
