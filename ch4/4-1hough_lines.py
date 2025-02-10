import cv2
import sys
import numpy as np

#img = cv2.imread('road2.jpg')
#img = cv2.imread('sudoku.png')
img = cv2.imread('check.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

h,w = img.shape[:2]
imgP = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)   # 이미지마다 임계값을 달리 설정

# 1
lines = cv2.HoughLines(edges, 1, np.pi/180, 100) # 허프 선 검출 ---②
# 4번째 : 직선으로 판단할 threshold, 같은 직선에 있는 최소 픽셀 수
for line in lines: 				# 검출된 모든 선 순회
   r, theta = line[0] 				# 거리와 각도
   tx, ty = np.cos(theta), np.sin(theta) 		# x, y축에 대한 삼각비
   x0, y0 = tx*r, ty*r 			#x, y 기준(절편) 좌표
   cv2.circle(img, (int(x0), int(y0)), 3, (0,255,0), -1) # 기준 좌표에 빨강색 점 그리기
   x1, y1 = int(x0 + w*(-ty)), int(y0 + h * tx) 	# 직선 방정식으로 그리기 위한 시작점 계산
   x2, y2 = int(x0 - w*(-ty)), int(y0 - h * tx) 	# 끝점 계산
   cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 2) 	# 선그리기

# 2
linesp = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, minLineLength=50, maxLineGap=5) 	# 허프 선 검출 ---②
# 4번째 : 직선으로 판단할 threshold, 같은 직선에 있는 최소 픽셀 수
# 5번째 : 최소 직선 길이
# 6번째 : 이웃하는 픽셀 간 최대 허용 갭
for line in linesp: 				# 검출된 모든 선 순회
   x1, y1, x2, y2 = line[0] 			# 시작점과 끝점
   cv2.line(imgP, (x1,y1), (x2, y2), (0,0,255), 2) 	# 검출된 선 그리기 ---③

img_houghline=np.hstack((img,imgP))
cv2.imshow("Hough lines", img_houghline)

cv2.waitKey()
cv2.destroyAllWindows()