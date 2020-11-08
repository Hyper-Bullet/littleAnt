import cv2
import numpy as np

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

b_img = cv2.imread('./capture/back.jpg')
b_img = cv2.flip(b_img,1)

l_img = cv2.imread('./capture/left.jpg')

r_img = cv2.imread('./capture/right.jpg')


cv2.namedWindow("b",2)
cv2.namedWindow("l",2)
cv2.namedWindow("r",2)

pl_frame,_,_ = warpImage(l_img,camleft_src,camleft_dst)
pb_frame,_,_ = warpImage(b_img,camback_src,camback_dst)
pr_frame,_,_ = warpImage(r_img,camright_src,camright_dst)
pr_frame = np.rot90(pr_frame)


cv2.imshow('b', pb_frame)
cv2.imshow('l', pl_frame)
cv2.imshow('r', pr_frame)
key = cv2.waitKey(1)
while (True):
    key = cv2.waitKey(1)
    if key  == ord('q'):
        cv2.destroyAllWindows()
        break
