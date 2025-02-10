import cv2
import sys
import numpy as np
import seam_carving

img=cv2.imread('beach.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

h,w,c = img.shape

dst = seam_carving.resize(
    img,  # input image (rgb or gray)
    size=(w+200, h),  # target size
    energy_mode="backward",  # choose from {backward, forward}
    order="width-first",  # choose from {width-first, height-first}
    keep_mask=None,  # object mask to protect from removal
)

padding = np.full((h, 8, c), 255, dtype=np.uint8)
img_carved=np.hstack((img,padding, cv2.resize(img,dsize=(w+200,h)),padding, dst))
cv2.imshow('Seam Carving - resizing', img_carved)

cv2.waitKey()
cv2.destroyAllWindows()
