import numpy as np
import cv2
import os
import utils

image_path = os.path.join("images", "stereo")

cameraL = cv2.VideoCapture(0)
cameraR = cv2.VideoCapture(2)

disp = 32
size = 1
P1 = 8*3
P2 = 32*3

stereo = None

def createBM():
    global stereo
    global P1
    global P2
    global size
    global disp 

    P1 = P1 * size * size
    P2 = P2 * size * size

    #stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, disp, size)
    stereo = cv2.StereoSGBM(0, disp, size, 
        P1=P1,#*size*size, 
        P2=P2,#*size*size, 
        disp12MaxDiff=15, 
        preFilterCap=67, 
        uniquenessRatio=15, 
        speckleWindowSize=100, 
        speckleRange=10, 
        fullDP=True)

createBM()

def compute(imgL, imgR):
    
    disparity = stereo.compute(imgL, imgR)
    image = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return image


def on_disp(value):
    global disp
    if value > 0:
        disp = value * 16
        createBM()

def on_size(value):
    global size
    if value % 2 == 1 and value >= 1:
        size = value
        createBM()

def on_P1(value):
    global P1
    P1 = value
    createBM()

def on_P2(value):
    global P2
    P2 = value
    createBM()

cv2.namedWindow('2')
cv2.createTrackbar('disp', '2', disp / 16, 10, on_disp)
cv2.createTrackbar('size', '2', size, 15, on_size)
cv2.createTrackbar('P1', '2', P1, 100, on_P1)
cv2.createTrackbar('P2', '2', P2, 100, on_P2)

while True:
    (grabbedL, imgL) = cameraL.read()
    (grabbedR, imgR) = cameraR.read()

    imgL = utils.resize(imgL, width=200)
    imgR = utils.resize(imgR, width=200)

    #cv2.imshow('left', imgL)
    #cv2.imshow('right', imgR)

    if not grabbedL or not grabbedR:
        break

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    depth = compute(imgL, imgR)

    utils.show_images([imgL, imgR, depth])

    if cv2.waitKey(14) == 27:
        break