import cv2
import numpy as np
import copy
import math
import sys

def calculateFingers(res, drawing):
    #  convexity defect
    try :
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                if cnt > 0:
                    return True, cnt + 1
                else:
                    return True, 0
    except cv2.error as e:
        print(e)
    return False, 0


# Open Camera
camera = cv2.VideoCapture('hand_final.mp4')
camera.set(10, 200)   # while True:
while camera.isOpened():
    # Main Camera
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing
    cv2.imshow('original', frame)
    frame = cv2.flip(frame, 1)  # Horizontal Flip
    #cv2.imshow('original', frame)
    # Background Removal
    bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    fgmask = bgModel.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    img = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Skin detect and thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('Threshold Hands', skinMask)  # Getting the contours and convex hull
    skinMask1 = copy.deepcopy(skinMask)
    contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
                res = contours[ci]
        hull = cv2.convexHull(res)
        #drawing = np.zeros(img.shape, np.uint8)
        drawing = frame
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        isFinishCal, cnt = calculateFingers(res, drawing)
        print("Fingers", res, cnt)
        if cnt < 2 :
            cv2.putText(drawing, "Rock", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        elif cnt < 5:
            cv2.putText(drawing, "Scissors", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(drawing, "Paper", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('output', drawing)
        k = cv2.waitKey(10)

        if k == ord('q'):
            break
    if k == 27:  # press ESC to exit
        break