'''
miniProject: 자동차 번호판 인식하기
지금까지 배운 알고리즘을 이용하여 자동차 번호판의 숫자와 글자를 인식한다.
    1. 자동차 이미지 번호를 입력받는다.
    2. HW#2-2를 이용하여 전처리를 한다.
    3. HW#3-2를 이용하여 번호판 후보를 검출한다.
        * 비스듬한 번호판을 warp 변환을 이용하여 직사각형으로 변환한다.
    4. 후보 번호판에서 숫자와 글자를 분리한다.
    5. 분리된 숫자와 글자를 인식한다. : 사전 학습된 모델(pytesseract, easy OCR 등)
평가 방법
    10개의 테스트 이미지에 대한 번호판 숫자와 글자 인식률로 평가함
제출물
    - 코드 파일
    - 10개의 테스트 이미지에 대한 결과를 포함한 보고서
'''

import cv2
import sys
import numpy as np
import pytesseract
import os

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tesseract 언어 데이터 경로 설정
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

def preprocessing(image):
    # 이미지 전처리 함수
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    blur = cv2.blur(gray, (5, 5))  # 흑백 블러링
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)   # Sobel 필터 적용
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)         # 이진화

    kernel = np.ones((5, 17), np.uint8)   # 모폴로지 커널
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)       # 모폴로지 연산

    return morph

def find_candidates(image):
    # 번호판 후보 검출
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽선의 최소 외접 직사각형

    # 후보 필터링
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates

def verify_aspect_size(size):
    # 번호판의 종횡비 빛 크기 조건 검증
    w, h = size
    if h == 0 or w == 0:
        return False

    aspect = h/ w if h > w else w/ h       # 종횡비 계산
                                                                                                                  
    chk1 = 3000 < (h * w) < 12000          # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5       # 번호판 종횡비 조건

    return (chk1 and chk2)

def clean_plate_image(crop_img):
    # 번호판 이미지에서 글자/숫자를 강조하고 배경을 제거하는 함수

    # 히스토그램 평활화
    h = cv2.equalizeHist(crop_img)
    
    # 이진화
    _, binary = cv2.threshold(h, 100, 255, cv2.THRESH_OTSU)

    # 모폴로지 연산자로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1) # 모폴로지 열기 연산

    # 가장자리와 연결된 흰색 픽셀을 검정색으로 변경
    h, w = cleaned.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    for x in range(w):
        if cleaned[0, x] == 255:
            cv2.floodFill(cleaned, mask, (x, 0), 0)
        if cleaned[h-1, x] == 255:
            cv2.floodFill(cleaned, mask, (x, h-1), 0)

    for y in range(h):
        if cleaned[y, 0] == 255:
            cv2.floodFill(cleaned, mask, (0, y), 0)
        if cleaned[y, w - 1] == 255:
            cv2.floodFill(cleaned, mask, (w - 1, y), 0)
    return cleaned

def rotate_plate(image, rect):
    # 번호판 후보 영역을 회전 및 크기 조정
    center, (w, h), angle = rect  # rect는 중심점, 크기, 회전 각도로 표시
    w = w + 15        # 여유 공간 추가
    h = h + 15

    if w < h :  # 세로가 긴 영역이면
        w, h = h, w    # 가로와 세로 맞바꿈
        angle -= 90   # 회전 각도 조정

    size = image.shape[1::-1]   # 행태와 크기는 역순
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)   # 회전 행렬 계산
    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 적용

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)  # 후보영역 가져오기 (영역 자르기)
    crop_img = 255 - cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)       # 흑백 반전

    # 추가 전처리로 이미지 정리
    cleaned_img = clean_plate_image(crop_img)
    
    return cv2.resize(cleaned_img, (255, 56))   # pytesseract로 인식하기 적합한 크기로 조절


def recognize_plate(image, candidates):
    # 번호판 후보에서 문자 인식
    plates = []
    for i, candidate in enumerate(candidates):
        # 후보 영역 회전 및 추출
        rotated_plate = rotate_plate(image, candidate)
        
        # 문자 인식
        text = pytesseract.image_to_string(
            rotated_plate,
            lang='kor+eng',  # 한글과 영어(숫자) 인식
            config='--psm 7')    # 단일 텍스트 줄 인식

        if len(text) > 7:
            # 텍스트와 이미지 매칭
            plates.append((rotated_plate, text.strip()))  # 번호판 이미지와 인식된 테스트 저장
    return plates


car_no = str(input("자동차 영상 번호 (00~09): "))
img = cv2.imread('cars/'+car_no+'.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
#cv2.imshow('original',img)

# 1 전처리 단계 (hw2-2)
preprocessed = preprocessing(img)
#cv2.imshow('plate candidate 0(Preprocessed)',preprocessed)

# 2 번호판 후보 영역 검출 (hw3-2)
candidates = find_candidates(preprocessed)

img2 = img.copy()
for candidate in candidates:  # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))   # 외곽선 꼭짓점 좌표
    cv2.polylines(img2, [pts], True, (0, 225, 255), 3)

#cv2.imshow('plate candidate 1', img2)

# 3. 번호판 문자 인식
plates = recognize_plate(img, candidates)
for i, (plate_img, text) in enumerate(plates):
    print('{}'.format(text))
    cv2.imshow('plate {}'.format(car_no), plate_img)

cv2.waitKey()
cv2.destroyAllWindows()