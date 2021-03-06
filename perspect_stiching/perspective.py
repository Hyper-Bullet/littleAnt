import cv2
import argparse
import numpy as np
from cam import *

camback_src = np.float32([[328, 363], [433, 359], [447, 482], [314, 488]])
camback_dst = np.float32([[814, 1359], [947, 1359], [947, 1488], [814, 1488]])

camleft_src = np.float32([[340, 426], [446, 424], [470, 554], [333, 554]])
camleft_dst = np.float32([[833, 1426], [970, 1426], [970, 1554], [833, 1554]])

camright_src = np.float32([[226, 428], [338, 430], [332, 572], [193, 568]])
camright_dst = np.float32([[693, 1430], [838, 1430], [838, 1572], [693, 1572]])

def warpImage(image, src, dst):
    image_size = (int(image.shape[1]*3), int(image.shape[0]*3))
    M =    cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(image, M,image_size, flags=cv2.INTER_LINEAR)
    return warped_image, M, Minv

def mouse(event, x, y, flags, param):
    global frame0
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness = 1)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)


img = cv2.imread('./capture/right.jpg')
cv2.namedWindow("img",2)
cv2.namedWindow("pe_img",2)
cv2.setMouseCallback("img", mouse)
cv2.imshow("img", img)
while (True):
    key = cv2.waitKey(1)
    if key  == ord('q'):
        break

pe_frame,_,_ = warpImage(img,camright_src,camright_dst)
cv2.imshow('pe_img', pe_frame)
cv2.imshow('img', img)
key = cv2.waitKey(1)
while (True):
    key = cv2.waitKey(1)
    if key  == ord('q'):
        cv2.destroyAllWindows()
        break
