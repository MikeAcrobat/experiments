import cv2
import numpy as np

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    dim = None
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized_image = cv2.resize(image, dim, interpolation=inter)

    return resized_image

def show_images(images):
    for i, image in zip(range(len(images)), images):
        if image is None:
            continue
        cv2.imshow(str(i), image)
        cv2.moveWindow(str(i), i * 350, 0)

def wait_esc_key():
    while True:
        key = cv2.waitKey(100)
        if key == 27:
            break