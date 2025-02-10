import cv2
import numpy as np

img=cv2.imread('fish1.jpg')	# 영상 읽기
#img=cv2.imread('fish2.jpeg')	# 영상 읽기

mask=np.zeros(img.shape[:2], np.uint8)    # 모든 화소를 0(cv.2GC_BGD) 배경으로 초기화
#mask[:,:]=cv2.GC_PR_BGD

rect = cv2.selectROI(img)
cv2.grabCut(img,mask,rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
#3 rect : 범위 지정. cv2.GC_INIT_WITH_RECT 모드에서만 사용됨.
#6 : 반복 횟수
#7 mode: GrabCut 적용 방법

mask2=np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')
# if (mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD)이면  mask2=0, 아니면 1
grab=img*mask2[:,:,np.newaxis]  # if mask2가 0이면 0, 아니면(1이면) 자기 자신의 색상 그대로
# np.newaxis : 차원을 높여줌 2차원 -> 3차원

cv2.imshow('Grab cut image',grab)

cv2.waitKey()
cv2.destroyAllWindows()