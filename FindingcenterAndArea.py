
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt

def nothing(x):
    pass

cap = cv.VideoCapture(0)

cv.namedWindow("HSV", cv.WINDOW_NORMAL)

cv.createTrackbar("LH", "HSV", 0, 180, nothing)
cv.createTrackbar("UH", "HSV", 180, 180, nothing)
cv.createTrackbar("LS", "HSV", 0, 255, nothing)
cv.createTrackbar("US", "HSV", 255, 255, nothing)
cv.createTrackbar("LV", "HSV", 0, 255, nothing)
cv.createTrackbar("UV", "HSV", 255, 255, nothing)

kernel = np.ones((5,5), "uint8")
while True:
 
    Success, frame = cap.read()

    lh = cv.getTrackbarPos("LH", "HSV")
    uh = cv.getTrackbarPos("UH", "HSV")
    ls = cv.getTrackbarPos("LS", "HSV")
    us = cv.getTrackbarPos("US", "HSV")
    lv = cv.getTrackbarPos("LV", "HSV")
    uv = cv.getTrackbarPos("UV", "HSV")
        
    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV) 

    mask = cv.inRange(hsv, lower, upper)
    
    resultantFrame = cv.bitwise_and(frame, frame, mask=mask)
    grayFrame = cv.cvtColor(resultantFrame, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(grayFrame, 100, 200)
    dilaation = cv.dilate(canny, kernel, iterations=1)
    contours, h = cv.findContours(dilaation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    frame = cv.drawContours(frame, contours, -1, (0, 255, 0), 5)


    cv.imshow("frame", frame)
    cv.imshow("gray", grayFrame)
    cv.imshow("canny", canny)
    cv.imshow("resultant", resultantFrame)
   
    k = cv.waitKey(1)

    if k == ord('q'):
        break
    
    
cap.release()
cv.destroyAllWindows()