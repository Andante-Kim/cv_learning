import cv2
import numpy as np
import math

# 600*900 크기의 컬러 이미지를 만들어 흰색으로 칠하기
img = np.ones((600,900,3), np.uint8) * 255

# 마우스 콜백 함수
def draw(event, x, y, flags, param):
    global ix, iy

    #Alt 키와 마우스 왼쪽 버튼의 다운/업을 이용하여 직사각형 그리기
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
    elif event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_ALTKEY:
        cv2.rectangle(img,(ix,iy),(x,y),(0,0,255),2)

    #Alt 키와 마우스 오른쪽 버튼의 다운/업을 이용하여 내부가 칠해진 직사각형 그리기
    elif event == cv2.EVENT_RBUTTONDOWN:
        ix,iy = x,y
    elif event == cv2.EVENT_RBUTTONUP and flags == cv2.EVENT_FLAG_ALTKEY:
        cv2.rectangle(img,(ix,iy),(x,y),(0,0,150),-1)

    # Ctrl 키와 마우스 왼쪽 버튼의 다운/업을 이용하여 원을 그리기
    elif event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
    elif event == cv2.EVENT_LBUTTONUP and flags == cv2.EVENT_FLAG_CTRLKEY:
        cv2.circle(img,(ix,iy),int(math.sqrt((x-ix)**2 + (y-iy)**2)),(0,255,0),2)

    # Ctrl 키와 마우스 오른쪽 버튼의 다운/업을 이용하여 내부가 칠해진 원을 그리기
    elif event == cv2.EVENT_RBUTTONDOWN:
        ix,iy = x,y
    elif event == cv2.EVENT_RBUTTONUP and flags == cv2.EVENT_FLAG_CTRLKEY:
        cv2.circle(img,(ix,iy),int(math.sqrt((x-ix)**2 + (y-iy)**2)),(0,150,0),-1)

    # 마우스 움직일 때
    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 왼쪽 버튼을 누르면서 움직이면 파란색 원(반지름5)이 따라 그려진다.
        if flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(img, (x, y), 5, (255,0,0), -1)
        # 마우스 오른쪽 버튼을 누르면서 움직이면 빨간색 원(반지름 5)이 따라 그려진다.
        elif flags == cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        # 마우스 왼쪽 버튼과 shift키를 누르면서 움직이면 초록색 원(반지름 5)이 따라 그려진다.
        elif flags == (cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        # 마우스 오른쪽 버튼과 shift키를 누르면서 움직이면 노란색 원(반지름 5)이 따라 그려진다.
        elif flags == (cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_RBUTTON):
            cv2.circle(img, (x,y),5,(0,255,255),-1)

    # 수정된 이미지를 다시 그림
    cv2.imshow('Drawing',img)

# 창을 만들고 현재 캔버스 표시
cv2.namedWindow('Drawing')
cv2.imshow('Drawing',img)

# 마우스 콜백 설정
cv2.setMouseCallback('Drawing',draw)

while(True):
    # 's' 키를 누르면 현재 이미지 저장
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('drawing.png', img)

    # 'q' 키를 누르면 프로그램 종료
    elif cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        break