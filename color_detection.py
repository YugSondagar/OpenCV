import cv2
import numpy as np
from PIL import Image

orange = [0, 165, 255]
cap = cv2.VideoCapture(0)

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerlimit = hsvC[0][0][0] - 10,100,100
    upperlimit = hsvC[0][0][0] + 10,255,255

    lowerlimit = np.array(lowerlimit,dtype = np.uint8)
    upperlimit = np.array(upperlimit,dtype = np.uint8)
    return lowerlimit, upperlimit

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerlimit, upperlimit = get_limits(orange)

    mask = cv2.inRange(hsvImage, lowerlimit, upperlimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1,y1,x2,y2 = bbox

        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5)


    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()









#OpenCV uses NumPy arrays as the primary data structure for images, colors, masks, etc.