import cv2
import sys
import numpy as np
import seam_carving

img=cv2.imread('beach.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

h,w,c = img.shape

girl_mask = cv2.imread("beach_girl.png", cv2.IMREAD_GRAYSCALE)
girl_removed = seam_carving.resize(img,  drop_mask=girl_mask)   # size=(w,h),
#bird_mask = cv2.imread("beach_bird.png", cv2.IMREAD_GRAYSCALE)
#bird_removed = seam_carving.resize(img, drop_mask=bird_mask)    # size=(w,h),

padding = np.full((h, 8, c), 255, dtype=np.uint8)
dst = np.hstack((img, padding, girl_removed)) #, padding, bird_removed))

cv2.imshow('Seam Carving - object delete', dst)

cv2.waitKey()
cv2.destroyAllWindows()
