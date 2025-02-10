import cv2
import sys

# cap = cv2.VideoCapture(0, cv.CAP_DSHOW)  # 동영상을 가져오는 클래스

cap = cv2.VideoCapture('face2.mp4')

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:  # 무한루프로
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득(frame)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 1 피부색 마스크
    skin_mask = cv2.inRange(hsv_img, (0, 30, 0), (20, 180, 255))  # skin color 범위
    # 2 피부색 그대로
    img_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    cv2.imshow('skin color detection', skin_mask)
    #cv2.imshow('skin color detection', img_skin)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
