import cv2
import numpy as np

def embossing(img):
    femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray16 = np.int16(gray)
    embossing = np.uint8(np.clip(cv2.filter2D(gray16, -1, femboss) + 128, 0, 255))

    return embossing


# 얼굴 검출
def haar_face(img, dir):
    face_cascade = cv2.CascadeClassifier(dir+'haarcascade_frontalface_default.xml')  # Face 분류기 로드
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이 이미지로 변환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 검출

    for (x, y, w, h) in faces:  # 검출된 모든 얼굴에 대해
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형으로 표시

    return img

# YOLO
def construct_yolo_v3(dir):
    f = open(dir+'coco_names.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    model = cv2.dnn.readNet(dir+'yolov3.weights', dir+'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names


def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    test_img = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)

    yolo_model.setInput(test_img)  # 테스트 이미지를 YOLO 학습모델의 새 입력 값으로 설정
    output3 = yolo_model.forward(out_layers)  # YOLO 학습모델의 out_layers로 출력을 계산하기 위해 정방향으로 전달

    box, conf, id = [], [], []  # 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]  # p1,p2,⋯,p80, 80개 부류의 인식률
            class_id = np.argmax(scores)  # 가장 큰 인식률의 부류
            confidence = scores[class_id]  # 가장 큰 인식률
            if confidence > 0.5:  # 신뢰도가 50% 이상인 경우만 취함
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)  # [0~1]표현을 이미지 내 실제 객체 중심위치로 변환
                w, h = int(vec85[2] * width), int(vec85[3] * height)  # [0~1]표현을 이미지 내 실제 객체 크기로 변환
                x, y = int(centerx - w / 2), int(centery - h / 2)  # 객체의 좌측 상단 위치
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)  # 비최대억제(NonMaximum Suppression) 알고리즘을 적용  # <3> 0.4 -> 1.0
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects

def yolo_v3(img, dir):
    model, out_layers, class_names = construct_yolo_v3(dir)  # YOLO 모델 생성
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔 다르게

    res = yolo_detect(img, model, out_layers)  # YOLO 모델로 객체 검출

    for i in range(len(res)):  # 검출된 물체를 영상에 표시
        x1, y1, x2, y2, confidence, id = res[i]
        text = str(class_names[id]) + '%.3f' % confidence
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
        cv2.putText(img, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)

    return img

# Segmentation
def mask_rcnn(img, div):
    height, width = img.shape[0], img.shape[1]

    f = open(div+'object_detection_classes_coco.txt', 'r')
    class_names = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔 다르게

    # Loading Mask RCNN
    net = cv2.dnn.readNetFromTensorflow(div+"frozen_inference_graph.pb", div+"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

    # Detect objects
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])

    for i in range(boxes.shape[2]):
        box = boxes[0, 0, i]
        class_id = int(box[1])
        confidence = box[2]
        if confidence < 0.5:
            continue
        # Get box Coordinates
        x = int(box[3] * width)  # 주어진 이미지에서 실제 위치 계산
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        # 1 detection : boxes
        text = str(class_names[class_id]) + '%.3f' % confidence
        #cv2.rectangle(img, (x, y), (x2, y2), colors[class_id], 2)
        cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[class_id], 2)

        # 2 segmentation : masks
        roi = img[y: y2, x: x2]  # 객체 영역 roi
        roi_height, roi_width, _ = roi.shape
        # Get the mask
        mask = masks[i, class_id] # 15*15
        mask = cv2.resize(mask, (roi_width, roi_height)) # 마스크 크기를 실제와 같게 변형

        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  # mask로 객체 영역 획득
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # 객체 영역의 윤곽선 획득
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], colors[class_id])  # 윤곽선 내부 확인 -> segmentation, fillPoly: 윤곽선 내부 색칠
            img[y: y2, x: x2] = roi
        cv2.waitKey()

    return img