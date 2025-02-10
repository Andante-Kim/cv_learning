import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Face 분류기 로드

#img = cv2.imread('face_w1.jpg')
#img = cv2.imread('goyunjeong.jpg')
#img = cv2.imread('hani.jpg')
#img = cv2.imread('herin.jpg')
#img = cv2.imread('jisu.jpg')
img = cv2.imread('minji.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴 검출

#face_mask = cv2.imread('mask_hannibal2.png')   # 가면 이미지
face_mask=cv2.imread('mask_fire.png')
h_mask, w_mask = face_mask.shape[:2]

for (x,y,w,h) in faces:
    if h > 0 and w > 0:
        roi = img[y:y+h, x:x+w]   # 검출된 얼굴에 대한 사각형 영역을 관심 영역(ROI)로 설정
        #cv2.imshow('0',roi)
        #cv2.moveWindow('0', 1200, 50)

        face_mask_small=cv2.resize(face_mask,(w,h),interpolation=cv2.INTER_AREA) # 가면 이미지의 크기를 검출된 얼굴의 크기와 같도록 resize
        #cv2.imshow('11', face_mask_small)

        gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY) # ① 가면 마스크에서 검지 않은 부분만 통과(검은색 부분은 투명)
        ret, mask = cv2.threshold(gray_mask, 220, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('12', mask)
        masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

        mask_inv = cv2.bitwise_not(mask)  # ② 검출된 얼굴에서 가면 마스크의 검은 부분만 통과(검지 않은 부분은 투명)
        #cv2.imshow('13', mask_inv)
        #cv2.moveWindow('13', 600, 450)
        masked_img = cv2.bitwise_and(roi, roi, mask=mask_inv)

        img[y:y+h, x:x+w] = cv2.add(masked_face, masked_img) # 흰색 마스크와 마스크된 얼굴 합성 (① + ②)

        #cv2.imshow('1',masked_face)
        #cv2.moveWindow('1', 600, 50)
        #cv2.imshow('2', masked_img)
        #cv2.moveWindow('2', 900, 50)  # Move it to(x,y)
        cv2.imshow('Play with face',img)

cv2.waitKey()
cv2.destroyAllWindows()