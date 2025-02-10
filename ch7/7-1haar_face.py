import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Face 분류기 로드

#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap=cv2.VideoCapture('face2.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret,frame=cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴 검출
    # face_cascade.detectMultiScale(이미지 변수, 스케일 요소, 얼굴 신뢰도)
    # 신뢰도 : 얼굴에 최소한 5개의 후보 경계 박스가 있어야 해당 얼굴을 검출

    ksize = 31  # 모자익 3-19 참조
    for (x, y, w, h) in faces:  # 검출된 모든 얼굴에 대해
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # 사각형으로 표시

        roi = frame[y : y + h, x : x + w]

        # 1 스무딩을 이용한 모자익
        # roi = cv2.GaussianBlur(roi, (ksize, ksize), 0.0) # 블러링 : 모자이크1-1
        # roi = cv2.blur(roi, (ksize, kize), 0.0) # 블러링 : 모자이크1-2
        # roi = cv2.medianBlur(roi, ksize)

        # 2 크기 변환을 이용한 모자익
        roi = cv2.resize(roi,(w//ksize,h//ksize))
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_NEAREST)

        frame[y:y+h, x:x+w] = roi   # 원본 이미지에 적용

    cv2.imshow('Face detection with haarcascade',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()		# 카메라와 연결을 끊음
cv2.destroyAllWindows()