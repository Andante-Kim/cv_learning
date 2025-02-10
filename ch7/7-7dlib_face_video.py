import cv2
import sys
import dlib

detector = dlib.get_frontal_face_detector()	# 얼굴 검출기
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 랜드마크 검출기

#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap=cv2.VideoCapture('Snowman.mp4')
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret,frame=cap.read()
    if not ret:
        sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 그레이 이미지로 변환
    faces = detector(gray)		# 얼굴 검출기로 얼굴 영역 검출

    for rect in faces:
        #x,y = rect.left(), rect.top()	 # 얼굴 영역(rect)을 좌표로 변환
        #w,h = rect.right()-x, rect.bottom()-y
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # 얼굴 영역 좌표로 사각형 표시

        shape = predictor(gray, rect)	# 랜드마크 검출기로 얼굴 랜드마크 검출
        for i in range(48,68):   	# 각 랜드마크 좌표 추출 및 표시
            part = shape.part(i)
            cv2.circle(frame, (part.x, part.y), 2, (255, 0, 255), -1)

    cv2.imshow('DLIB - face landmarks - lip',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()		# 카메라와 연결을 끊음
cv2.destroyAllWindows()