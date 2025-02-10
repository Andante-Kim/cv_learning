import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)

# Embossing
gray16=np.int16(gray)              # int8로 음수를 표현하는 경우 -128~127까지만 표현 가능
# int16은 -256~255까지 표현 가능

femboss1=np.array([[-1.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 1.0]])
emboss1a = np.uint8(cv2.filter2D(gray16,-1,femboss1))
emboss1b = np.uint8(cv2.filter2D(gray16,-1,femboss1)+128)
emboss1=np.uint8(np.clip(cv2.filter2D(gray16,-1,femboss1)+128,0,255))  # 0보다 작으면 0, 255보다 크면 255

femboss2=np.array([[-1.0, -1.0, 0.0],
                 [ -1.0, 0.0, 1.0],
                  [ 0.0, 1.0, 1.0]])
emboss2=np.uint8(np.clip(cv2.filter2D(gray16,-1,femboss2)+128,0,255))  # 0보다 작으면 0, 255보다 크면 255

emboss=np.hstack((emboss1,emboss2))
cv2.imshow('Emboss',emboss)

cv2.waitKey()
cv2.destroyAllWindows()