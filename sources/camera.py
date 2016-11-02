import cv2
import numpy as np

camera = cv2.VideoCapture(2)

cv2.namedWindow('frame')

orb = cv2.Feature2D_create("ORB")

print(dir(orb))

while True:

    (grabbed, frame) = camera.read()

    if not grabbed:
        break

    cv2.imshow('frame', frame)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, descriptors = orb.detectAndCompute(grey, None)
    grey = cv2.drawKeypoints(grey, kp, np.zeros((15, 15)))

    print(descriptors)

    cv2.imshow('grey', grey)
    
    if cv2.waitKey(1000/60) == 27:
        break

camera.release()
cv2.destroyAllWindows()
print("done")