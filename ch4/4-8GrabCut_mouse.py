import cv2
import numpy as np

img=cv2.imread('fish1.jpg')	# 영상 읽기
#img=cv2.imread('fish2.jpeg')	# 영상 읽기

img_show=np.copy(img)

mask=np.zeros((img.shape[0],img.shape[1]),np.uint8) # 모든 화소를 0(cv.2GC_BGD) 배경으로 초기화
#mask=np.zeros(img.shape[:2], np.uint8)
mask[:,:]=cv2.GC_PR_BGD     # cv2.GC_INIT_WITH_MASK인 경우 반드시

BrushSiz=9				# 붓의 크기
LColor,RColor=(255,0,0),(0,0,255)	# 파란색(물체)과 빨간색(배경)

def painting(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_show,(x,y),BrushSiz,LColor,-1)	# 왼쪽 버튼 클릭하면 파란색
        cv2.circle(mask,(x,y),BrushSiz,cv2.GC_FGD,-1)
    elif event==cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img_show,(x,y),BrushSiz,RColor,-1)	# 오른쪽 버튼 클릭하면 빨간색
        cv2.circle(mask,(x,y),BrushSiz,cv2.GC_BGD,-1)
    elif event==cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img_show,(x,y),BrushSiz,LColor,-1)# 왼쪽 버튼 클릭하고 이동하면 파란색
        cv2.circle(mask,(x,y),BrushSiz,cv2.GC_FGD,-1)
    elif event==cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_RBUTTON:
        cv2.circle(img_show,(x,y),BrushSiz,RColor,-1)	# 오른쪽 버튼 클릭하고 이동하면 빨간색
        cv2.circle(mask,(x,y),BrushSiz,cv2.GC_BGD,-1)

    cv2.imshow('Painting',img_show)
    
cv2.namedWindow('Painting')
cv2.setMouseCallback('Painting',painting)

while(True):				# 붓 칠을 끝내려면 'q' 키를 누름
    if cv2.waitKey(1)==ord('q'):
        break

background=np.zeros((1,65),np.float64)	# 배경 히스토그램 0으로 초기화
foreground=np.zeros((1,65),np.float64)	# 물체 히스토그램 0으로 초기화
cv2.grabCut(img,mask,None,background,foreground,5,cv2.GC_INIT_WITH_MASK)
#4,5 : 배경, 전경에 대한 모델 지정. cv2.GC_INIT_WITH_MASK 모드에서만 사용됨.
#6 : 반복 횟수
#7 mode: GrabCut 적용 방법

mask2=np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')
# if (mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD)이면  mask2=0, 아니면 1
grab=img*mask2[:,:,np.newaxis]  # if mask2가 0이면 0, 아니면(1이면) 자기 자신의 색상 그대로
# np.newaxis : 차원을 높여줌 2차원 -> 3차원

img_grab=np.vstack((img,grab))
cv2.imshow('Grab cut image',img_grab)

cv2.waitKey()
cv2.destroyAllWindows()