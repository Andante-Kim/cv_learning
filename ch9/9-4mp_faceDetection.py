import cv2 
import mediapipe as mp  # mediapipe 설치

img=cv2.imread('BSDS_376001.jpg')

mp_face_detection=mp.solutions.face_detection  # mediapipe 솔루션
mp_drawing=mp.solutions.drawing_utils          # drawing

face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)
# 모델 인덱스는 0 또는 1입니다.
# 0을 사용하면 카메라 2m 이내의 부분적 모델 촬영에 적합하고,
# 1은 5m 이내에서 전신 모델을 촬영하는데 적합합니다.
# 지정하지 않을 경우의 기본값은 0입
res=face_detection.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    print(len(res.detections))
    for detection in res.detections:
        #print(detection)
        mp_drawing.draw_detection(img,detection)
    cv2.imshow('Face detection by MediaPipe',img)

cv2.waitKey()
cv2.destroyAllWindows()