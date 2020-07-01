import cv2
import numpy as np
frame_width = 400
frame_height = 300
kernel = np.zeros((512,512,3),np.uint8)

face_Cascade = cv2.CascadeClassifier("E:\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3,frame_width)
cap.set(4,frame_height)
cap.set(10,150)
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def getcontours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: continue
        print(area)

        #cv2.imwrite(imgray,kernel)



while True:
    success,img = cap.read()
    # print(success)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imBlur = cv2.GaussianBlur(imgray,(7,7),2)
    imCanny = cv2.Canny(imBlur,150,200,edges=5,apertureSize=3)
    imDial = cv2.dilate(imCanny,None,iterations=2)
    imErode = cv2.erode(imDial,None,iterations=1)
    faces = face_Cascade.detectMultiScale(imgray,1.1,4)
    getcontours(imErode)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
    #cv2.imshow("Result",img)
    imgH = stackImages(0.6,([img,imgray],[imBlur,imErode]))
    cv2.imshow("Image",imgH)
    if cv2.waitKey(1) and 0xff == ord('q'): break
cap.release()