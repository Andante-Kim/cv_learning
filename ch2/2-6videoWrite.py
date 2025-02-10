import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('slow_traffic_small.mp4')    # 비디오 파일
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)    # 웹캠 비디오

if not cap.isOpened():
    sys.exit('동영상 연결 실패')

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   # 비디오 크기 지정
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('frame_size =', frame_size)

fourcc = cv2.VideoWriter_fourcc(*'XVID')        # 비디오 저장 방식 지정 
outV = cv2.VideoWriter('./record.mp4', fourcc, 20.0, frame_size) # 비디오 저장 객체 생성 

while True:
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    outV.write(frame)  # 비디오로 프레임 저장
    cv2.imshow('Video display', frame)

    key = cv2.waitKey(1)  # 1밀리초 동안 키보드 입력 기다림
    if key == ord('q'):  # 'q' 키가 들어오면 루프를 빠져나감
        break

cap.release()  # 카메라와 연결을 끊음
cv2.destroyAllWindows()