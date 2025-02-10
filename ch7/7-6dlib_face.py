import cv2
import sys
import dlib # cmake, wheel 패키지 설치 후 dlib 패키지 설치

detector = dlib.get_frontal_face_detector()	# 얼굴 검출기
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 랜드마크 검출기

img = cv2.imread("kimjiwon.jpg")
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 그레이 이미지로 변환
faces = detector(gray)		# 얼굴 검출기로 얼굴 영역 검출

for rect in faces:
    x,y = rect.left(), rect.top()	 # 얼굴 영역(rect)을 좌표로 변환
    w,h = rect.right()-x, rect.bottom()-y
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1) # 얼굴 영역 좌표로 사각형 표시

    shape = predictor(gray, rect)	# 랜드마크 검출기로 얼굴 랜드마크 검출
    for i in range(68):   	# 각 랜드마크 좌표 추출 및 표시
        part = shape.part(i)
        cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
        cv2.putText(img, str(i), (part.x, part.y), cv2.FONT_HERSHEY_PLAIN, 0.5,(255,255,255), 1, cv2.LINE_AA)

cv2.imshow('DLIB - face landmarks',img)

cv2.waitKey()
cv2.destroyAllWindows()