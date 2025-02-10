# [2-1] 비디오에서 특수효과(special effects) 적용하기

import cv2
import sys

video = cv2.VideoCapture('face2.mp4')

if not video.isOpened():
    sys.exit('카메라 연결 실패')

# 현재 적용 중인 효과를 저장할 변수
current_effect = 'Original'

while True:  # 무한루프
    ret, frame = video.read()  # 비디오를 구성하는 프레임 획득(frame)
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    # 키 입력 확인
    key = cv2.waitKey(1)

    # 키 입력에 따라 현재 효과 변경
    if key == ord('n'):
        current_effect = 'Original'
    elif key == ord('b'):
        current_effect = 'Bilateral'
    elif key == ord('s'):
        current_effect = 'Stylization'
    elif key == ord('g'):
        current_effect = 'Gray Pencil Sketch'
    elif key == ord('c'):
        current_effect = 'Color Pencil Sketch'
    elif key == ord('o'):
        current_effect = 'Oil Painting'
    elif key == ord('q'):
        break

    # 각 효과에 따른 처리
    if current_effect == 'Original':
        result = frame
        cv2.putText(result, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
    elif current_effect == 'Bilateral':
        result = cv2.bilateralFilter(frame, -1, 10, 5)
        cv2.putText(result, 'Bilateral', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
    elif current_effect == 'Stylization':
        result = cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
        cv2.putText(result, 'Stylization', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
    elif current_effect == 'Gray Pencil Sketch':
        result, _ = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
        cv2.putText(result, 'Gray Pencil Sketch', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
    elif current_effect == 'Color Pencil Sketch':
        _, result = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
        cv2.putText(result, 'Color Pencil Sketch', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
    elif current_effect == 'Oil Painting':
        result = cv2.xphoto.oilPainting(frame, 7, 1)
        cv2.putText(result, 'Oil Painting', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)

    # 결과 출력
    cv2.imshow('video', result)

# 비디오 및 윈도우 해제
video.release()
cv2.destroyAllWindows()
