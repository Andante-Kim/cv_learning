import cv2
import sys
import cvlib as cv  # 설치

img = cv2.imread('face2.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

faces, confidences = cv.detect_face(img)	 # 얼굴 검출

for (x, y, x2, y2), conf in zip(faces, confidences):    # 검출된 모든 얼굴에 대해
    cv2.putText(img, str(conf), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # 확률 출력하기
    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2) # 사각형으로 표시

cv2.imshow('CVLIB - face detection',img)

cv2.waitKey()
cv2.destroyAllWindows()