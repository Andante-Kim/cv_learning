import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Face 분류기 로드
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')      # Eye 분류기 로드

img = cv2.imread('face_w1.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴 검출

for (x,y,w,h) in faces:
    if h > 0 and w > 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형으로 표시

        roi_gray = gray[y:y+h, x:x+w]   # 검출된 얼굴에 대한 사각형 영역을 관심 영역(ROI)로 설정
        eyes = eye_cascade.detectMultiScale(roi_gray)   # 눈 검출

        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(img, (x+x_eye, y+y_eye), (x+x_eye + w_eye, y+y_eye + h_eye), (0, 0, 255), 2)  # 사각형으로 표시

        cv2.imshow('Eye detection with haarcascade',img)

cv2.waitKey()
cv2.destroyAllWindows()
