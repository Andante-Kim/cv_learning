import cv2

#1 skin color
src1 = cv2.imread('hand.jpg')
hsv1 = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)
lowerb1 = (0, 30, 0)
upperb1 = (20, 180, 255)
dst1 = cv2.inRange(hsv1, lowerb1, upperb1)
cv2.imshow('hands', dst1)

# 흰색 부분은 색깔 그대로 반영
dst_skin = cv2.bitwise_and(src1,src1,mask=dst1)
cv2.imshow('hands_skin', dst_skin)

#2 flower
src2 = cv2.imread('flower.jpg')
hsv2 = cv2.cvtColor(src2,cv2.COLOR_BGR2HSV)
lowerb2 = (150, 100, 100)
upperb2 = (180, 255, 255)
dst2 = cv2.inRange(hsv2, lowerb2, upperb2)
cv2.imshow('flower', dst2)

cv2.waitKey()
cv2.destroyAllWindows()