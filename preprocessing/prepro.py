import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2a_010.JPG',0)
ret,thresh1 = cv2.threshold(img,149,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY']
images = [img, thresh1]

cv2.imshow('Original Image',img)
cv2.imshow('After Thresholding',thresh1)


import numpy as np 
kernel = np.ones((4,4), np.uint8) 

img = thresh1
img_dilation = cv2.dilate(img, kernel, iterations=2) 
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)



#cv2.imshow('Input', img) 
cv2.imshow('Erosion', img_erosion) 
cv2.imshow('Dilation', img_dilation) 
  
cv2.waitKey(0) 
