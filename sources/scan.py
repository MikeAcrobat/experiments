import cv2
import numpy as np
import os
import utils
from skimage.filters import threshold_adaptive

image_path = os.path.join("images", "receipts", "receipt1.jpg")

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

warped_image = None

contour = np.append(contours[0], contours[1], axis=0)

if len(contour) >= 4:
    contour = contour.reshape(len(contour), 2)

    rect = np.zeros((4, 2), dtype = "float32")
    '''
    [0] . [1]
     .     .
    [3] . [2]
    '''
    sum = contour.sum(axis = 1)
    rect[0] = contour[np.argmin(sum)]
    rect[2] = contour[np.argmax(sum)]

    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bot = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    
    height_left  = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    max_width = max(int(width_bot), int(width_bot))
    max_height = max(int(height_left), int(height_right))

    fixed_rect = np.array([
        [0, 0], 
        [max_width - 1, 0], 
        [max_width - 1, max_height - 1], 
        [0, max_height - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, fixed_rect)
    warped_image = cv2.warpPerspective(resized, matrix, (max_width, max_height))
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

cv2.drawContours(resized, contours, -1, (0, 255, 0), 2)

utils.show_images([resized, edged, warped_image])

radius = 1
offset = 0

def apply():
    copy_image = threshold_adaptive(warped_image, radius, "gaussian", offset, "nearest")
    copy_image = copy_image.astype("uint8") * 255
    cv2.imshow("2", copy_image)

def callback_radius(value):
    if value % 2 == 1:
        global radius
        radius = value
        apply()

def callback_offset(value):
    global offset
    offset = value
    apply()

cv2.createTrackbar("radius", "2", 1, 251, callback_radius)
cv2.createTrackbar("offset", "2", 0, 100, callback_offset)

utils.wait_esc_key()

cv2.destroyAllWindows()