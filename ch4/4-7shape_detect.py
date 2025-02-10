import cv2
import sys
import numpy as np

def shape_detect(c):
    shape = "undefined"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    #print(len(approx))

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 10:
        shape = "star"
    else:
        roundness = (4.0 * np.pi * cv2.contourArea(c)) / (peri * peri)
        if roundness >= 0.85 and roundness <=1.15 :
            shape = "circle"

    return shape

img = cv2.imread('shapes1.png')
#img = cv2.imread('shapes2.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

h,w = img.shape[:2]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(gray,100,200)
contours,hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for contour in contours:
    shape = shape_detect(contour)

    m = cv2.moments(contour)
    area = m['m00']  # Contour 면적, cv.contourArea(contour)
    cx, cy = int(m['m10'] / area), int(m['m01'] / area)
    #print(cx,cy,area)

    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    cv2.putText(img, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.imshow("Image", img)

    k=cv2.waitKey()
    if k==ord('q'):
        cv2.destroyAllWindows()
        break