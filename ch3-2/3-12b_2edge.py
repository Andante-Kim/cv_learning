import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(gray, -1)
cv2.imshow('Laplacian',laplacian)

# 샤프닝 필터와 비교
fsharpen=np.array([[0.0, 1.0, 0.0],
                  [1.0, -4.0, 1.0],
                  [ 0.0, 1.0, 0.0]])
sharpen = cv2.filter2D(gray, -1, fsharpen)

print(sharpen[100,100])
print(laplacian[100,100])
grad=np.hstack((gray, sharpen, laplacian))
cv2.imshow('Second derivatives',grad)

cv2.waitKey()
cv2.destroyAllWindows()