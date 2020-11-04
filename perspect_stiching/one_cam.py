import cv2
import argparse
import numpy as np
from cam import *

cam0_mtx = np.array(cam0_mtx)
cam1_mtx = np.array(cam1_mtx)
cam2_mtx = np.array(cam2_mtx)
cam3_mtx = np.array(cam3_mtx)

cam0_dist = np.array(cam0_dist)
cam1_dist = np.array(cam1_dist)
cam2_dist = np.array(cam2_dist)
cam3_dist = np.array(cam3_dist)

ap = argparse.ArgumentParser()
ap.add_argument("--i", type=int, default=0)
args = vars(ap.parse_args())

video0 = cv2.VideoCapture(args["i"])
width = (int(video0.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = (int(video0.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frame0 = 0
src = np.float32([[319, 401], [415, 389], [440, 451], [306, 470]])
dst = np.float32([[306, 403], [440, 403], [440, 470], [306, 470]])

def mouse(event, x, y, flags, param):
    global frame0
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(frame0, (x, y), 1, (0, 0, 255), thickness = -1)
        cv2.putText(frame0, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness = 1)
        cv2.imshow("source", frame0)

def warpImage(image, src, dst):
    image_size = (image.shape[1], image.shape[0])
    # rows = img.shape[0] 720
    # cols = img.shape[1] 1280
    M =    cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(image, M,image_size, flags=cv2.INTER_LINEAR)
    
    return warped_image, M, Minv

while (video0.isOpened()):
    ret0, frame0 = video0.read()
    # frame0 = cv2.resize(frame0, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('source', frame0)
    cv2.setMouseCallback("source", mouse)
    
    un_frame = cv2.undistort(frame0, cam3_mtx,cam3_dist,None,cam3_mtx)
    cv2.imshow('undist', un_frame)
    pe_frame,_,_ = warpImage(un_frame,src,dst)
    cv2.imshow('perspective', pe_frame)
    key = cv2.waitKey(1)
    if key  == ord('q'):
        break
video0.release()

