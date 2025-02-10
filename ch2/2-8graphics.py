import cv2
import numpy as np

img = np.ones((600,300,3), np.uint8) * 255 	# 600*300*3*8bits 행렬, 흰색으로 초기화

cv2.line(img, (50,50), (150,150), (255,0,0), 3)

cv2.rectangle(img, (50,50), (150,150), (0,255,0), 2)
cv2.rectangle(img, (50,200), (150,300), (0,255,0), cv2.FILLED)

cv2.circle(img, (220,100), 50, (0,0,255), 4)
cv2.circle(img, (220,250), 30, (0,255,255), -1)

cv2.ellipse(img, (100, 400), (75, 50), 0, 0, 360, (0,255,0), 3)

pts = np.array([[220,350], [180,500], [260,500]], dtype=np.int32)
cv2.polylines(img, [pts], False, (255,0,0), 10)

cv2.putText(img, "text1", (50,500), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 128, 0), 2)
cv2.putText(img, "text2", (50,570), cv2.FONT_HERSHEY_TRIPLEX, 2, (221, 160, 221), 4)

cv2.imshow('OpenGL Graphics',img)

cv2.waitKey()
cv2.destroyAllWindows()