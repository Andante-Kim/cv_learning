# [2-2] 지동차 번호판 후보 검출하기

import cv2
import numpy as np

# 이미지 불러오기
img_00=cv2.imread('00.jpg', cv2.IMREAD_GRAYSCALE)
img_01=cv2.imread('01.jpg', cv2.IMREAD_GRAYSCALE)
img_02=cv2.imread('02.jpg', cv2.IMREAD_GRAYSCALE)
img_03=cv2.imread('03.jpg', cv2.IMREAD_GRAYSCALE)
img_04=cv2.imread('04.jpg', cv2.IMREAD_GRAYSCALE)
img_05=cv2.imread('05.jpg', cv2.IMREAD_GRAYSCALE)

def car_number(img, name):

    # 전처리(잡음 제거)
    smooth = cv2.GaussianBlur(img, (5,5), 7.0)
    # 세로 에지 검출
    edge = cv2.Sobel(smooth, cv2.CV_8U, 1, 0, ksize=3) # 수직 에지
    # 임계값을 이용한 검은 배경과 흰 에지 분리
    _, binary = cv2.threshold(edge, 130, 255, cv2.THRESH_BINARY)
    # 닫힘(close)을 통해 흰 숫자 에지를 팽창 -> 침식
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (70,3)) # 가로로 긴 구조 요소 생성
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, structuringElement)

    # 결과 출력
    img_set = np.hstack((smooth, edge))
    img_set2 = np.hstack((binary, closing))
    cv2.imshow('{}: smooth-edge'.format(name), img_set)
    cv2.imshow('binary, closing', img_set2)
    return img_set

# 최종 결과 출력
car_number(img_00, 'img_00')
cv2.waitKey()
cv2.destroyAllWindows()

car_number(img_01, 'img_01')
cv2.waitKey()
cv2.destroyAllWindows()

car_number(img_02, 'img_02')
cv2.waitKey()
cv2.destroyAllWindows()

car_number(img_03, 'img_03')
cv2.waitKey()
cv2.destroyAllWindows()

car_number(img_04, 'img_04')
cv2.waitKey()
cv2.destroyAllWindows()

car_number(img_05, 'img_05')
cv2.waitKey()
cv2.destroyAllWindows()