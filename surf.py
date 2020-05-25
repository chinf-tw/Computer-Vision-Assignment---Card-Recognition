import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

img = cv2.imread('3S.png',0)
# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)
# surf = cv2.SURF(400)
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)
len(kp)