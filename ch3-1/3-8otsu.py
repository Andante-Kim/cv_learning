import cv2
import matplotlib.pyplot as plt

gray=cv2.imread('soccer.jpg',cv2.IMREAD_GRAYSCALE) # gray 이미지
h_gray=cv2.calcHist([gray],[0],None,[256],[0,256]) # 1바이트 gray의 0번 채널에서 히스토그램 구함
plt.plot(h_gray,'k.', linewidth=3)
plt.show()

t,bin_gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적 임계값=',t)

cv2.imshow('Gray',gray)			# gray 이미지
cv2.imshow('Gray binarization',bin_gray)	# gray 이미지 이진화 영상

cv2.waitKey()
cv2.destroyAllWindows()