import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('slow_traffic_small.mp4')    # 비디오 파일
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)    # 웹캠 비디오

if not cap.isOpened():
    sys.exit('동영상 연결 실패')

# 1 비디오에서 프레임 수집
frames=[]       # 수집한 프레임을 저장할 배열 변수 초기화
while True:
    ret,frame=cap.read()			# 비디오를 구성하는 프레임 획득
    
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
        
    cv2.imshow('Video display',frame)
    
    key=cv2.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림
    if key==ord('c'):	# 'c' 키가 들어오면 프레임을 리스트에 추가
        frames.append(frame)    # 프레임 저장(수집)
    elif key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv2.destroyAllWindows() # 비디오 창 닫기

# 2 수집된 프레임을 스택(stack)으로 보여주기
if len(frames)>0:		# 수집된 이미지가 있으면
    imgs_h=frames[0]
    for i in range(1,min(3,len(frames))):	# 최대 3개까지 이어 붙임
        imgs_h = np.hstack((imgs_h, frames[i]))     # 가로로 연결

    cv2.imshow('collected images - Hstack', imgs_h) # 연결된 이미지의 크기 출력
#    cv2.imshow('collected images - Vstack', imgs_v) # 연결된 이미지의 크기 출력

    print(imgs_h.shape)
#    print(imgs_v.shape)

    cv2.waitKey()
    cv2.destroyAllWindows()