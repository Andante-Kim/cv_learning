import cv2
import numpy as np

gray=cv2.imread('lenna256.png', cv2.IMREAD_GRAYSCALE)

# Sharpening
fsharpen1=np.array([[0.0, -1.0, 0.0],
                  [-1.0, 4.0, -1.0],
                  [ 0.0, -1.0, 0.0]])
sharpen1 = cv2.filter2D(gray, -1, fsharpen1)
cv2.putText(sharpen1,'sharpen 4',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),2)

fsharpen2=np.array([[0.0, -1.0, 0.0],
                  [-1.0, 5.0, -1.0],
                  [ 0.0, -1.0, 0.0]])
sharpen2 = cv2.filter2D(gray, -1, fsharpen2)
cv2.putText(sharpen2,'sharpen 5',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,100),2)

fsharpen3=np.array([[-1.0, -1.0, -1.0],
                  [-1.0, 8.0, -1.0],
                  [ -1.0, -1.0, -1.0]])
sharpen3 = cv2.filter2D(gray, -1, fsharpen3)
cv2.putText(sharpen3,'sharpen 8',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),2)

fsharpen4=np.array([[-1.0, -1.0, -1.0],
                  [-1.0, 9.0, -1.0],
                  [-1.0, -1.0, -1.0]])
sharpen4 = cv2.filter2D(gray, -1, fsharpen4)
cv2.putText(sharpen4,'sharpen 9',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,100,100),2)

sharpen=np.hstack((sharpen1,sharpen2,sharpen3,sharpen4))
cv2.imshow('Sharpening',sharpen)

cv2.waitKey()
cv2.destroyAllWindows()