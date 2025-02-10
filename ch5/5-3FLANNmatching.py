import cv2
import numpy as np
import time

img1=cv2.imread('mot_color70.jpg')[190:350,440:560] # 버스를 크롭하여 모델 영상으로 사용
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2=cv2.imread('mot_color83.jpg')			     # 장면 영상
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift=cv2.SIFT_create()
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)
print('특징점 개수:',len(kp1),len(kp2)) 

# 1 전수조사 + 모든 특징점
bf_matcher=cv2.BFMatcher()
bf_matches = bf_matcher.match(des1,des2)		# 모든 특징점을 받음
print(len(bf_matches))
print(bf_matches[0].queryIdx, bf_matches[0].trainIdx, bf_matches[0].distance)

# 매칭 전략을 만족하는 매칭쌍(good_match1)을 찾음
T1=200
good_match1=[]
for nearest in bf_matches:
    if nearest.distance<T1: # 1) 고정 임계값
        good_match1.append(nearest)
print('bf_matches:', len(good_match1))

# 2 전수조사 + k개의 유사한 특징점
bf_knn_matches = bf_matcher.knnMatch(des1,des2, 2) 	# 가장 유사한 특징점 k(=2)개를 받음
print(len(bf_knn_matches))
print(bf_knn_matches[0][0].distance,bf_knn_matches[0][1].distance) # bf_matches와 차이 주목

# 매칭 전략을 만족하는 매칭쌍(good_match2)을 찾음
T2=0.7
good_match2=[]
for nearest1,nearest2 in bf_knn_matches:
    #if nearest1.distance < T2: # 1) 고정 임계값
    if (nearest1.distance/nearest2.distance)<T2: # 3) 최근접 이웃 거리 비율 전략
        good_match2.append(nearest1)
print('bf_knn_matches:', len(good_match2))

# 3 빠른 매칭 + + k개의 유사한 특징점
flann_matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
flann_knn_matches=flann_matcher.knnMatch(des1,des2,2)   # 가장 유사한 특징점 k(=2)개를 받음
print(len(flann_knn_matches))

# 매칭 전략을 만족하는 매칭쌍(good_match3)을 찾음
T3=0.7
good_match3=[]
for nearest1,nearest2 in flann_knn_matches:
    if (nearest1.distance/nearest2.distance)<T3: # 최근접 이웃 거리 비율 전략
        good_match3.append(nearest1)
        print(nearest1.distance,nearest2.distance)
print('flann_knn_matches:',len(good_match3))
print(good_match3[0].queryIdx,' -- ', good_match3[0].trainIdx, ' : ', good_match3[0].distance)


img_match=np.empty((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3),dtype=np.uint8)
#cv2.drawMatches(img1,kp1,img2,kp2,good_match3,img_match,flags=cv2.DrawMatchesFlags_DEFAULT)  # good_match3만 그림
cv2.drawMatches(img1,kp1,img2,kp2,good_match3,img_match,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # good_match3만 그림
# cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, cv2.DrawMatchesFlags_DEFAULT, cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG, cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
cv2.imshow('Good Matches', img_match)

cv2.waitKey()
cv2.destroyAllWindows()