# [3-1] 손가락 개수를 이용한 가위바위보 놀이

# module
import cv2
import numpy as np
import math
import copy
import math

# 손가락 계산 함수
def calcultate_fingers(res, drawing):
    try:
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            detects = cv2.convexityDefects(res, hull)
            if detects is not None:
                cnt = 0
                for i in range(detects.shape[0]):
                    s, e, f, d = detects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # 코사인
                    if angle <= math.pi / 2:  # 각도가 90보다 작으면, 손가락으로 간주
                        cnt += 1
                        cv2.circle(drawing, far, 8, [255, 0, 0], -1)
                if cnt > 0:
                    return True, cnt + 1
                else:
                    return True, 0
    except cv2.error as e:
        print(e)
        return False, 0

# Open Video
cap = cv2.VideoCapture('hand_final.mp4')


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./hand_result.mp4', fourcc, fps, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # 전처리
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skin_mask = cv2.inRange(hsv_img, lower, upper)

    cv2.imshow('canny', skin_mask)

    # contours
    contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area = -1

    # contour
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > max_area:
                max_area = area
                ci = i
                res = contours[ci]
        hull = cv2.convexHull(res)
        cv2.drawContours(frame, [res], 0,  (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)

        isFinishCal, cnt = calcultate_fingers(res, frame)
        print('Fingers:', cnt)
        if cnt < 2 :
            cv2.putText(frame, "Rock", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        elif cnt < 5:
            cv2.putText(frame, "Scissors", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Paper", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # 최종 출력
        out.write(frame)
        cv2.imshow('original', frame)

    # 종료
    key = cv2.waitKey(30)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()