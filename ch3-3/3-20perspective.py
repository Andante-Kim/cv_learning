import cv2
import numpy as np

img = cv2.imread('perspective3.jpg')

pts1 = np.float32([(75, 11),  (74, 357), (420, 95), (450, 317)] )  # 입력 이미지의 위치
pts2 = np.float32([(0, 0),  (0, 383), (511, 0), (511, 383)])    # 출력 이미지의 위치

cv2.circle(img,(75,11),8,(0,0,255),-1)
cv2.circle(img,(74,357),8,(255,0,0),-1)
cv2.circle(img,(420,95),8,(255,0,255),-1)
cv2.circle(img,(450,317),8,(0,255,0),-1)

perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, perspect_mat, img.shape[1::-1], cv2.INTER_CUBIC) # w=512, h=384

img_perspective=np.vstack((img,dst))
cv2.imshow('Perspective', img_perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()
