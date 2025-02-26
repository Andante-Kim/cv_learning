import cv2
import numpy as np

num = str(input('1~4 중 입력'))
img1 = cv2.imread('book{}.jpg'.format(num))
img1 = cv2.resize(img1, dsize=(0, 0), fx=0.3, fy=0.3)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)			     # 장면 영상
img2 = cv2.imread('books.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)  # 최근접 2개

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# good_match를 찾음
# ==========================================================
# good_match 특징점의 위치
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])

# homography를 계산
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

h1, w1 = img1.shape[0], img1.shape[1]  # 첫 번째 영상의 크기(검색 이미지)
h2, w2 = img2.shape[0], img2.shape[1]  # 두 번째 영상의 크기

# homography가 적용된 위치 계산
box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
box2 = cv2.perspectiveTransform(box1, H)

# 다각형으로 그림
img2 = cv2.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8)

img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
cv2.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_match = cv2.resize(img_match, dsize=(0, 0), fx=0.3, fy=0.3)
cv2.imshow('Matches and Homography', img_match)

cv2.waitKey()
cv2.destroyAllWindows()