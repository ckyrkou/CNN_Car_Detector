#!/usr/bin/python

import numpy as np 
import cv2 

from utils import *

#Define windows for output
cv2.namedWindow('Output',0)
cv2.namedWindow('controls',0)

# create trackbars for color change
cv2.createTrackbar('R/H L','controls',0,255,nothing)
cv2.createTrackbar('G/S L','controls',0,255,nothing)
cv2.createTrackbar('B/V L','controls',0,255,nothing)

cv2.createTrackbar('R/H U','controls',255,255,nothing)
cv2.createTrackbar('G/S U','controls',255,255,nothing)
cv2.createTrackbar('B/V U','controls',255,255,nothing)

# Set default lower and upper bounds
lower = np.array([0,0,0], dtype = "uint8")
upper = np.array([255,255,255], dtype = "uint8")


############################
#### EDIT ONLY THIS BLOCK
# Read Image
# Convert Image to HSV from RGB/BGR
# use cv2.cvtColor()
# input is img_init
# output is img_hsv

img_init = cv2.imread('test_col.png')
img_hsv= cv2.cvtColor(img_init, cv2.COLOR_BGR2HSV)

############################

while(True):

    # get current positions of four trackbars
    rh_l = cv2.getTrackbarPos('R/H L','controls')
    gs_l = cv2.getTrackbarPos('G/S L','controls')
    bv_l = cv2.getTrackbarPos('B/V L','controls')
    
    rh_h = cv2.getTrackbarPos('R/H H','controls')
    gs_h = cv2.getTrackbarPos('G/S H','controls')
    bv_h = cv2.getTrackbarPos('B/V H','controls')

    # set lower and upper bounds
    lower = np.array([rh_l, gs_l, bv_l], dtype="uint8")
    upper = np.array([rh_h, gs_h, bv_h], dtype="uint8")

    img_out = img_hsv.copy()

    # Find the pixels that correspond to road
    img_out2 = cv2.inRange(img_out,lower, upper)

    # Clean from noisy pixels and keep only the largest connected segment
    img_out = post_process(img_out2)

    # Display the road mask and overlay on the original image
    display(img_init, img_hsv, img_out2, img_out)

    k=cv2.waitKey(33)
    
    if(k & 0xFF == ord('q')):
        cv2.destroyWindow("Video")
        break

cv2.destroyAllWindows()
