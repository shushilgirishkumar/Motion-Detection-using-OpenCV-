import cv2
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


kernel = np.ones((5, 5), np.uint8)
frame_width = 200
frame_height = 500
cap = cv2.VideoCapture(0)
if not (cap.isOpened()): print("not opened")
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 100)

while True:
    success, img = cap.read()
    print(success)
    img_GRay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(img_GRay, (7, 7), 3)
    imgCanny = cv2.Canny(img, 200, 200)
    imDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imErode = cv2.erode(imDial, kernel, iterations=1)
    imgstack = stackImages(0.8, ([img, img_GRay, imDial], [imgBlur, imgCanny, imErode]))
    cv2.imshow("shape", imgstack)
    # cv2.imshow("Video1", img_GRay)

    # cv2.imshow("Video1", img)
    if cv2.waitKey(1) and 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"""
--------------
import cv2

frame_width = 640
frame_height = 480

camera_port = 0
camera = cv2.VideoCapture(camera_port)

camera.set(3, frame_width)
camera.set(4, frame_height)
camera.set(10, 100)
while True:

    return_value, image = camera.read()
    cv2.imwrite("image.png", image)

camera.release()  # Error is here
#-----------

import cv2
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
