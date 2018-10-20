import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5 )
    edges = cv2.Canny(frame, 60, 200)
    
    
    
    cv2.imshow('frame', frame)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('edges', edges)
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cam.release()