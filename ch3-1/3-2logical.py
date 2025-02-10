import cv2
import sys
import numpy as np

src1=cv2.imread('lenna512.png')
src2=cv2.imread('opencv_logo256.png')

if src1 is None or src2 is None:
    sys.exit('파일을 찾을 수 없습니다.')

mask = cv2.imread('opencv_logo256_mask.png',cv2.IMREAD_GRAYSCALE)
mask_inv = cv2.imread('opencv_logo256_mask_inv.png',cv2.IMREAD_GRAYSCALE)

sy, sx = 0,0
rows,cols,channels = src2.shape
roi = src1[sy:sy+rows, sx:sx+cols]
cv2.imshow('roi', roi)

src1_bg = cv2.bitwise_and(roi, roi, mask=mask) # mask의 흰색(1)에 해당하는 roi는 그대로, 검정색(0)은 검정색으로

src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv) # mask_inv의 흰색(1)에 해당하는 src2는 그대로, 검정색(0)은 검정색으로

dst = cv2.bitwise_or(src1_bg, src2_fg)
cv2.imshow('dst', dst)

src1[sy:sy+rows, sx:sx+cols] = dst

pp=np.hstack((src1_bg,src2_fg, dst))
cv2.imshow('point processing - logical',pp)
cv2.imshow('combine', src1)

cv2.waitKey()
cv2.destroyAllWindows()