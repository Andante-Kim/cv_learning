import cv2
import numpy as np

gray=cv2.imread('morph_j.png', cv2.IMREAD_GRAYSCALE)
t,b=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# opencv skeletonize
size = np.size(b)
skeleton = np.zeros(b.shape,np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
while(not done):
    eroded = cv2.erode(b,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(b,temp)
    skeleton = cv2.bitwise_or(skeleton,temp)
    cv2.imshow('skeletonize', skeleton)
    # cv2.imshow('b',b)
    cv2.waitKey()
    b = eroded.copy()

    zeros = size - cv2.countNonZero(b)
    if zeros==size:
        done = True

cv2.imshow('skeletonize', skeleton)

cv2.waitKey()
cv2.destroyAllWindows()