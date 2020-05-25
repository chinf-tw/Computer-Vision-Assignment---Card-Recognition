import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
kpImg = cv2.imread('./Pokers/3S.png',0)
# kpImg = 255 - kpImg
# img = cv2.imread('./test.jpg',0)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(kpImg,None)
# compute the descriptors with ORB
kp, des = orb.compute(kpImg, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(kpImg, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()