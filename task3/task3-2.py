# [3-2] 지동차 번호판 후보 검출하기2

import cv2
import numpy as np
import sys

#① 자동차 이미지 번호를 입력 받는다.
num = str(input('00-05 중 숫자 하나를 입력하세요: '))
img1 = cv2.imread('{}.jpg'.format(num))
img2 = cv2.imread('{}.jpg'.format(num))
img_grayscale = cv2.imread('{}.jpg'.format(num),cv2.IMREAD_GRAYSCALE)

if (img1 is None) or (img2 is None):
    sys.exit('파일을 찾을 수 없습니다.')

#② HW#2-2를 이용하여 전처리를 한다.
def car_number(img_grayscale,num):
    # 전처리(잡음 제거)
    smooth = cv2.GaussianBlur(img_grayscale, (5,5), 7.0)
    # 세로 에지 검출
    edge = cv2.Sobel(smooth, cv2.CV_8U, 1, 0, ksize=3) # 수직 에지
    # 임계값을 이용한 검은 배경과 흰 에지 분리
    _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
    # 닫힘(close)을 통해 흰 숫자 에지를 팽창 -> 침식
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (65,3)) # 가로로 긴 구조 요소 생성
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, structuringElement)

    # 결과 return_closing 이미지 return
    cv2.imshow('closing_{}'.format(num),closing)
    return closing

#③ 윤곽선(contours)을 찾아 최소면적 사각형을 찾는다.
def find_contours(img1, img2, closing,num):
    # canny 검출
    canny = cv2.Canny(closing, 100, 200)

    # contours 검출
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_len = len(contours) # contours 개수 확인
    print('전체 contours 개수:',contours_len)

    for i in range(contours_len):
        contour = contours[i]

        # 최소 면적 사각형 계산
        rect = cv2.minAreaRect(contour)  # 중심, (가로, 세로), 각도 반환

        box = cv2.boxPoints(rect)  # 사각형 꼭짓점 좌표 계산
        box = np.int32(box)  # 정수형 변환

        # 사각형을 이미지에 그림
        cv2.drawContours(img1, [box], 0, (0, 255, 0), 2)

        #④ 최소면적 사각형 중 가로 세로의 비율 등의 조건을 이용하여 자동차 번호판 후보를 찾는다.
        size = rect[1]
        check_size = verify_aspect_size(size)
        if check_size != False:
            cv2.drawContours(img2,[box],0,(0,0,255),2)

    # contours_사각형 표시한 이미지 출력
    cv2.imshow('contours_{}'.format(num), img1)
    cv2.imshow('find_car_num_{}'.format(num), img2)
    return 0 #contours_img

#가로 세로의 비율 등의 조건 함수
def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False
    aspect = h/ w if h > w else w/ h    # 종횡비 계산
    chk1 = 3000 < (h * w) < 12000       # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5           # 번호판 종횡비 조건
    return (chk1 and chk2)

# 최종 결과 출력
closing = car_number(img_grayscale,num)
find_contours(img1, img2, closing,num)

cv2.waitKey()
cv2.destroyAllWindows()