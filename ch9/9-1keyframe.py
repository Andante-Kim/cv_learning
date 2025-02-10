import cv2
import sys
import numpy as np

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        return None     


cap = cv2.VideoCapture('face2.mp4') #slow_traffic_small.mp4')    # 비디오 파일
#cap = cv2.VideoCapture('KBS-Special-2011-10-09.mp4')
if not cap.isOpened():
    sys.exit('동영상 연결 실패')

prev_frame = get_frame(cap)
cur_frame = get_frame(cap)
next_frame = get_frame(cap)

images=[]   # key frame을 저장할 공간
images.append(cv2.resize(next_frame, dsize=(0, 0), fx=0.2, fy=0.2))  # 동영상 크기 resize
cv2.imshow("keyframes", images[0])

while True:

    diff = cv2.absdiff(next_frame, cur_frame)
    cv2.imshow('Difference in Video', diff)

    sum_diff = np.sum(diff)   # 한 동영상에서 차이의 합

    if sum_diff > 20000000:   # 숫자는 임의로 조정 가능. 차이를 구별 잘 하게(KBS에서는 100000)
        images.append(cv2.resize(next_frame, dsize=(0, 0), fx=0.2, fy=0.2))
        print(sum_diff)
        concat_image = np.hstack((images))  # 가로로 붙여 key frame 출력
        cv2.imshow("keyframes", concat_image)

    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = get_frame(cap)

    if next_frame is None :    
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    key = cv2.waitKey(1)
    if key == ord('q'):  # 'q' 키가 들어오면 루프를 빠져나감, ord()는 문자를 아스키 값으로 변환하는 함수
        break

cv2.waitKey()
cap.release()  # 카메라와 연결을 끊음
cv2.destroyAllWindows()
