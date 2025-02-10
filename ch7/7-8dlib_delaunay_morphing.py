import cv2
import sys
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()	# 얼굴 검출기
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 랜드마크 검출기

# Check if a point is inside a rectangle
def rect_contains(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Locate Facial Landmarks
def face_delaunay(theImage1, theImage2):
    corresp = np.zeros((68,2))

    imgList = [theImage1[:,:],theImage2[:,:]]
    list1 = []
    list2 = []
    j = 1

    for img in imgList:

        size = (img.shape[0],img.shape[1])
        if(j == 1):
            currList = list1
        else:
            currList = list2

        dets = detector(img, 1) # Detect the points of face.

        j=j+1
        for k, rect in enumerate(dets):
            shape = predictor(img, rect)    # Get the landmarks/parts for the face in rect.

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y

            # Add back the background
            currList.append((1,1))
            currList.append((size[1]-1,1))                 # 배경에 대한 랜드마크 추가
            currList.append(((size[1]-1)//2,1))
            currList.append((1,size[0]-1))
            currList.append((1,(size[0]-1)//2))
            currList.append(((size[1]-1)//2,size[0]-1))
            currList.append((size[1]-1,size[0]-1))
            currList.append(((size[1]-1),(size[0]-1)//2))

    # Add back the background
    narray = corresp/2
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)

    f_w = size[1]
    f_h = size[0]
    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary.
    theList = narray .tolist()
    points = [(int(x[0]), int(x[1])) for x in theList]
    dictionary = {x[0]: x[1] for x in list(zip(points, range(76)))}

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            list4.append((dictionary[pt1], dictionary[pt2], dictionary[pt3]))

    dictionary = {}

    return [size,imgList[0],imgList[1],list1,list2,list4]

# Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

img1 = cv2.imread('morph_images/Clinton.jpg')
img2 = cv2.imread('morph_images/Bush.jpg')
cv2.imshow('morphing face1',img1)
cv2.imshow('morphing face2',img2)

[size, img1, img2, points1, points2, tri_list] = face_delaunay(img1, img2)

num_images = 12
for j in range(0, num_images):

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Read array of corresponding points
    points = []
    alpha = j/(num_images-1)

    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))

    morphed_frame = np.zeros(img1.shape, dtype=img1.dtype)    # 출력 이미지를 위한 변수, 공간 선언
    for i in range(len(tri_list)):
        x = int(tri_list[i][0])
        y = int(tri_list[i][1])
        z = int(tri_list[i][2])

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
        pts = np.array([[int(t[0][0]), int(t[0][1])], [int(t[1][0]), int(t[1][1])], [int(t[2][0]), int(t[2][1])]], dtype=np.int32)
        #cv2.polylines(morphed_frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('morphed face', np.uint8(morphed_frame) )
    cv2.waitKey()

cv2.waitKey()
cv2.destroyAllWindows()