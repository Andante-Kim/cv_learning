import cv2
import numpy as np

gray=cv2.imread('grayscale.png', cv2.IMREAD_GRAYSCALE) # BGR 컬러 영상을 명암 영상으로 변환하여 저장

ret, img_binaryB = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
cv2.putText(img_binaryB,"BINARY",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 255, 2)
ret, img_binaryBINV = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
cv2.putText(img_binaryBINV,"BINARY_INV",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0, 2)
ret, img_binaryT = cv2.threshold(gray, 120, 255, cv2.THRESH_TRUNC)
cv2.putText(img_binaryT,"TRUNC",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0, 2)
ret, img_binaryT0 = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
cv2.putText(img_binaryT0,"TOZERO",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 255, 2)
ret, img_binaryT0INV = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO_INV)
cv2.putText(img_binaryT0INV,"TOZERO_INV",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0, 2)

img_binary=np.vstack((gray,img_binaryB, img_binaryBINV, img_binaryT, img_binaryT0, img_binaryT0INV))
cv2.imshow('threshold',img_binary)

cv2.waitKey()
cv2.destroyAllWindows()