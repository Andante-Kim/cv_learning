import cv2
import os

stitcher = cv2.Stitcher.create()
images = []

# 1-1 stitch 이미지 불러오기
images.append(cv2.imread('.\stitch_images\stitch_img20.png'))
images.append(cv2.imread('.\stitch_images\stitch_img40.png'))
images.append(cv2.imread('.\stitch_images\stitch_img60.png'))
images.append(cv2.imread('.\stitch_images\stitch_img80.png'))

# 1-2 저장된 이미지를 불러오는 경우
#path = '.\\stitch_images\\'
#for root, directories, files in os.walk(path):
#    for file in files:
#        if '.png' in file:
#            src = cv2.imread(os.path.join(root,file))
#            images.append(src)

# 2-1 차례대로 stitch 수행
status, dst2 = stitcher.stitch((images[0],images[1]))
status, dst3 = stitcher.stitch((dst2, images[2]))
status, dst4 = stitcher.stitch((dst3, images[3]))
#print(status)

#cv2.imshow('dst2',  dst2)
#cv2.imshow('dst3',  dst3)
#cv2.imshow('dst4',  dst4)

# 2-2 모든 이미지를 한까번에 stitch 수행
status, dst = stitcher.stitch((images[0],images[1],images[2],images[3]))
#status, dst = stitcher.stitch(images)

if status == cv2.STITCHER_OK:
    # cv2.imwrite('.\stitch_images\stitch_out.jpg', dst)
    cv2.imshow('stitching',  dst)

cv2.waitKey()
cv2.destroyAllWindows()