import cv2
import sys

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        return None     #[]


cap = cv2.VideoCapture('slow_traffic_small.mp4')    # 비디오 파일
if not cap.isOpened():
    sys.exit('동영상 연결 실패')

prev_frame = get_frame(cap)
cur_frame = get_frame(cap)
next_frame = get_frame(cap)

while True:

    diff = cv2.absdiff(next_frame, cur_frame)
    cv2.imshow('Difference in Video', diff)

    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = get_frame(cap)

    if next_frame is None : 
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    key = cv2.waitKey(1)  # 1밀리초 동안 키보드 입력 기다림(정수)
    if key == ord('q'):  # 'q' 키가 들어오면 루프를 빠져나감, ord()는 문자를 아스키 값으로 변환하는 함수
        break

cap.release()  # 카메라와 연결을 끊음
cv2.destroyAllWindows()
