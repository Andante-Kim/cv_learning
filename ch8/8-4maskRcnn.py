import cv2
import sys
import numpy as np

img=cv2.imread('london_street.png')
if img is None:
    sys.exit('파일이 없습니다.')
height,width=img.shape[0],img.shape[1]

f = open('object_detection_classes_coco.txt', 'r')
class_names = [line.strip() for line in f.readlines()]

colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔 다르게

# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Detect objects
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)

boxes, masks = net.forward(["detection_out_final", "detection_masks"])

for i in range(boxes.shape[2]):
    box = boxes[0, 0, i]
    print(box)
    class_id = int(box[1])    # 객체 부류
    confidence = box[2]       # 객체 신뢰도
    if confidence < 0.5:
        continue
    # Get box Coordinates
    x = int(box[3] * width)  # 주어진 이미지에서 실제 위치 계산
    y = int(box[4] * height)
    x2 = int(box[5] * width)   # 객체 실제 크기
    y2 = int(box[6] * height)

    # 1 detection : boxes
    text = str(class_names[class_id]) + '%.3f' % confidence
    cv2.rectangle(img, (x, y), (x2, y2), colors[class_id], 2)
    cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[class_id], 2)

    # 2 segmentation : masks
    roi = img[y: y2, x: x2]     # 객체 영역 roi
    roi_height, roi_width, _ = roi.shape
    # Get the mask
    mask = masks[i, class_id]        # 객체 마스크mask
    #cv2.imshow('mask', mask) # 원래 마스크 크기는 항상 15x15
    #cv2.moveWindow('mask',100,50)

    # 마스크 크기를 실제와 같게 변형
    mask = cv2.resize(mask, (roi_width, roi_height))
    #cv2.imshow('mask-resize', mask)
    #cv2.moveWindow('mask-resize', 100, 200)

    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)  # mask로 객체 영역 획득
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 객체 영역의 윤곽선 획득
    for cnt in contours:
        cv2.fillPoly(roi, [cnt], colors[class_id])  # 윤곽선 내부 확인 -> segmentation, fillPoly: 윤곽선 내부 색칠
        img[y: y2, x: x2] = roi
    cv2.waitKey()

cv2.imshow("Object detection by mask RCNN",img)

cv2.waitKey()
cv2.destroyAllWindows()