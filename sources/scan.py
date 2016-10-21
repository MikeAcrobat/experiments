import cv2
import numpy as np
import os
import utils
from skimage.filters import threshold_adaptive

image_path = os.path.join("images", "receipts", "receipt3.jpg")

image = cv2.imread(image_path)

resized = utils.resize(image, height = 500)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(blurred, 15, 150)

(contours, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

def customKey(curve):
    return cv2.arcLength(curve, False)
    #return cv2.contourArea(curve)

for i, contour in enumerate(contours):
    length = cv2.arcLength(contour, True)
    contours[i] = cv2.approxPolyDP(contour, length * 0.05, True)

contours = sorted(contours, key=customKey, reverse=True)

print(contours[0])

cv2.drawContours(resized, contours[:2], -1, (0, 255, 0), 2)

'''
temp = edged.copy()
lines = cv2.HoughLinesP(temp, 10, 180 * 3.14 / 180.0, 1)
if lines is not None:
    for line in lines[0]:
        print line
        cv2.line(resized, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
'''

#gray = threshold_adaptive(gray, 11, "gaussian", 0, "nearest").astype("uint8") * 255

utils.show_images([resized, edged, blurred])

utils.wait_esc_key()

cv2.destroyAllWindows()