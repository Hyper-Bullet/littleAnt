import cv2
import time
import sys
import os
import multiprocessing as mp
import numpy as np
from cam import *

count = 0

def adjust_saturation(image):
    image = image.astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image[:, :, 2] = 0.5 * image[:, :, 2]
    image[:, :, 2][image[:, :, 2] > 1] = 1
    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    image = (image*255).astype(np.uint8)
    return image
    
def image_put(q, i):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print('ok for ',i)
    while True:
        img = cap.read()[1]
        if i == 0:
            img = cv2.flip(img,1)   # 1 for upside_down
            img = cv2.undistort(img, cam0_mtx,cam0_dist,None,cam0_mtx)
            img = adjust_saturation(img)
        if i == 1:
            img = cv2.flip(img,0)   # 0 for left_right flip
            img = cv2.undistort(img, cam1_mtx,cam1_dist,None,cam1_mtx)
        if i == 2:
            img = cv2.flip(img,0)
            img = cv2.undistort(img, cam2_mtx,cam2_dist,None,cam2_mtx)
        if i == 3:
            img = cv2.undistort(img, cam1_mtx,cam1_dist,None,cam1_mtx)
        q.put(img)
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, i):
    global count
    cv2.namedWindow(str(i), 2)
    while True:
        frame = q.get()
        cv2.imshow(str(i), frame)
        key = cv2.waitKey(1)
        if key  == ord('q'):
            break
        if key  == ord('s'):
            cv2.imwrite('./capture/'+str(i)+'_'+str(count)+'.jpg', frame)
            count = count + 1


def run_multi_camera():
    camera_indexs = [0,1,2]
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_indexs]
    processes = []
    for queue, camera_index in zip(queues, camera_indexs):
        processes.append(mp.Process(target=image_put, args=(queue, camera_index)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_index)))
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    # run_single_camera()
    run_multi_camera()
    pass
