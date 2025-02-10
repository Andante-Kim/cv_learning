import cv2
import sys
import numpy as np

#1
img=cv2.imread('lenna256.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환하여 저장
cv2.imshow('original image - gray',gray)

#2 사칙연산 -> 단순한 수식은 결과가 제대로 나오지 않음
img_plus = gray + 50       # y = x + 50
cv2.putText(img_plus,"+50",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_minus = gray - 50      # y = x - 50
cv2.putText(img_minus,"-50",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_multi = gray * 2       # y = 2 * x
cv2.putText(img_multi,"*2",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_div = gray / 2         # y = x / 2
cv2.putText(img_div,"/2",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_reverse = 255 - gray    # y = 255 - x
cv2.putText(img_reverse,"255-",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
pp=np.hstack((gray, img_plus, img_minus, img_multi, img_div, img_reverse)) # hstack은 높이(세로)가 같아야 함
#cv2.imshow('point processing',pp)

#3 opencv 함수
img_plus2 = cv2.add(gray, 50)         # y = x + 50
cv2.putText(img_plus2,"add(50)",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_minus2 = cv2.subtract(gray, 50)   # y = x - 50
cv2.putText(img_minus2,"subtract(50)",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_multi2 = cv2.multiply(gray, 2)    # y = 2 * x
cv2.putText(img_multi2,"multiply(2)",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_div2 = cv2.divide(gray, 2)        # y = x / 2
cv2.putText(img_div2,"divide(2)",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
img_reverse2 = cv2.subtract(255,gray)    # y = 255 - x
cv2.putText(img_reverse2,"subtract(255)",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, 0,2)
pp2=np.hstack((gray, img_plus2, img_minus2, img_multi2, img_div2, img_reverse2)) # hstack은 높이(세로)가 같아야 함
#cv2.imshow('point processing - opencv',pp2)

#4 크기가 같은 두개의 이미지
img2=cv2.imread('opencv_logo256.png')
img_plus3 = cv2.add(img, img2)         # y = x1 + x2
img_minus3 = cv2.subtract(img, img2)   # y = x1 - x2
img_multi3 = cv2.multiply(img, img2)    # y = x1 * x2
img_div3 = cv2.divide(img, img2)        # y = x1 / x2
img_addW = cv2.addWeighted(img, 0.5, img2, 0.5, 0) # 이미지 블렌딩(Image Blending)
img_addW1 = cv2.addWeighted(img, 0.8, img2, 0.2, 0)
img_addW2 = cv2.addWeighted(img, 0.2, img2, 0.8, 0)
pp4=np.hstack((img_plus3, img_minus3, img_multi3, img_div3,img_addW, img_addW1, img_addW2))
#cv2.imshow('point processing - two images',pp4)

#5 비선형 연산 : 감마연산
f = gray/255
#f = img/255     # 픽셀값을 0~1사이 실수로 변환
img_gamma05 = np.uint8(255*(f**0.5)) # 감마 연산 :
cv2.putText(img_gamma05,"gamma=0.5",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, (255,0,255), 2)
img_gamma075 = np.uint8(255*(f**0.75))
cv2.putText(img_gamma075,"gamma=0.75",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, (255,0,255), 2)
img_gamma10 = np.uint8(255*(f**1.0))
cv2.putText(img_gamma10,"gamma=1.0",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, (255,0,255), 2)
img_gamma20 = np.uint8(255*(f**2.0))
cv2.putText(img_gamma20,"gamma=2.0",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, (255,0,255), 2)
img_gamma30 = np.uint8(255*(f**3.0))
cv2.putText(img_gamma30,"gamma=3.0",(20,30),cv2.FONT_HERSHEY_PLAIN,1.5, (255,0,255), 2)
pp3=np.hstack((img_gamma05,img_gamma075,img_gamma10,img_gamma20,img_gamma30)) # hstack은 높이(세로)가 같아야 함
cv2.imshow('point processing - gamma',pp3)

cv2.waitKey()
cv2.destroyAllWindows()