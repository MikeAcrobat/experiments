import numpy as np
import cv2
import os

image_path = os.path.join("images", "stereo")

imgL = cv2.imread(os.path.join(image_path, 'tsukuba_l.png'), 0)
imgR = cv2.imread(os.path.join(image_path, 'tsukuba_r.png'), 0)

disp = 16
size = 1

def compute():
    #stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, disp, size)
    stereo = cv2.StereoSGBM(0, disp, size, 
        P1=8*1*size*size, 
        P2=32*1*size*size, 
        disp12MaxDiff=1, 
        preFilterCap=63, 
        uniquenessRatio=10, 
        speckleWindowSize=100, 
        speckleRange=32, 
        fullDP=True)
    
    disparity = stereo.compute(imgL, imgR)
    image = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('depth', image)

def on_disp(value):
    global disp
    if value > 0:
        disp = value * 16
        compute()

def on_size(value):
    global size
    if value % 2 == 1 and value >= 1:
        size = value
        compute()

compute()

cv2.createTrackbar('disparity', 'depth', 0, 10, on_disp)
cv2.createTrackbar('size', 'depth', 1, 100, on_size)

cv2.waitKey(0)
