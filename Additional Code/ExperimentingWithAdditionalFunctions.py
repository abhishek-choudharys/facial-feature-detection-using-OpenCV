import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # hsv hue sat value
    
    lower_red = np.array([0,0,20])
    upper_red = np.array([175,80,210])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    
    #kernel = np.ones((15,15), np.float32)/255
    #smoothed = cv2.filter2D(res, -1, kernel)
    
    #blur = cv2.GaussianBlur(res, (15,15), 0)
    #median = cv2.medianBlur(res, 15)
    
    #bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(mask, kernel, iterations = 2)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    
    #tophat = 
    #blackhat = 
    
    
    #cv2.imshow('tophat', tophat)
    #cv2.imshow('blackhat', blackhat)
    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #cv2.imshow('smoothed', smoothed)
    #cv2.imshow('blur', blur)
    #cv2.imshow('medianblur', median)
    #cv2.imshow('bil', bilateral)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output7.avi', fourcc, 20.0, (640, 480))
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cam.release()