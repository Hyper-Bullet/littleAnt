import cv2
import argparse
import numpy as np
from cam import *

cam0_src = np.float32([[326, 411], [434, 414], [450, 546], [317, 543]])
cam0_dst = np.float32([[1017, 1511], [1150, 1511], [1150, 1646], [1026, 1646]])


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


img = cv2.imread('./capture/2_0.jpg')
cv2.namedWindow("img",2)
cv2.namedWindow("pe_img",2)
cv2.setMouseCallback("img", mouse)
cv2.imshow("img", img)
while (True):
    key = cv2.waitKey(1)
    if key  == ord('q'):
        break

pe_frame,_,_ = warpImage(img,cam0_src,cam0_dst)
cv2.imshow('pe_img', pe_frame)
cv2.imshow('img', img)
key = cv2.waitKey(1)
while (True):
    key = cv2.waitKey(1)
    if key  == ord('q'):
        cv2.destroyAllWindows()
        break
