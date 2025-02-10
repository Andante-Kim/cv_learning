import cv2
import numpy as np
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic # Initializing mediapipe class.
mp_drawing = mp.solutions.drawing_utils     # Import drawing_utils and drawing_styles.
mp_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(static_image_mode=False, enable_segmentation=True, #model_complexity=2,
                                refine_face_landmarks=True)

#cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('aerobics.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(frame, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              landmark_drawing_spec=None,
                              connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                              connection_drawing_spec=mp_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                              connection_drawing_spec=mp_styles.get_default_hand_connections_style())

    cv2.imshow('MediaPipe holistic', cv2.flip(frame, 1))
    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()