import numpy as np
import cv2
from cv2 import cv2
import time

def ORB(warp,kpImg):
    # ORB
    warpGray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(kpImg,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(warpGray, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(warp, kp, None, color=(0,255,0), flags=0)
    return img2
def mathc_img(image, Target, value):
    if len(image.shape) > 2 :
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image.copy()
    w, h = Target.shape[::-1]
    res = cv2.matchTemplate(img_gray, Target, cv2.TM_CCOEFF_NORMED)
    threshold = value
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 1)


def getDiagonalPoints(c):
    # 畫出矩形框架,返回值x，y是矩陣左上點的座標，w，h是矩陣的寬和高
    (x, y, w, h) = cv2.boundingRect(c)
    pointOne = (x-10, y-10)
    pointTwo = (x+w+10, y+h+10)
    return pointOne, pointTwo


def getCanny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(Gaussian, 100, 250)
    return canny


def DrawLine(canny, img):
    draw = img.copy()
    rhoAccuracy = 1
    thetaAccuracy = np.pi/180.0
    minVotes = 30
    lines = cv2.HoughLines(canny, rhoAccuracy, thetaAccuracy, minVotes)
    # count = 0
    # while len(lines) != 2 and count < 100:
    #     count += 1
    #     if len(lines) > 2:
    #         minVotes += 1
    #     else:
    #         minVotes -= 1
    #     lines = cv2.HoughLines(canny, rhoAccuracy, thetaAccuracy, minVotes)

    # print("minVotes ",minVotes)
    # print("count ",count)
    # print(lines)
    if (lines is not None):
        for line in lines:
            # print(len(line))
            for rho, theta in line:
                # print(theta)
                a = np.cos(theta)
                b = np.sin(theta)
                print("a = ", a, " b = ", b)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return draw


def DrawLineP(canny, img):
    draw = img.copy()
    rhoAccuracy = 1
    thetaAccuracy = np.pi/45.0
    minVotes = 30
    minLineLength = 1000
    maxLineGap = 3

    lines = cv2.HoughLinesP(canny, rhoAccuracy, thetaAccuracy,
                            minVotes, minLineLength, maxLineGap)
    if (lines is not None):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return draw


def drawFourPointContours(img, c):
    box = []
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    for i in range(len(box)):
        cv2.circle(img, tuple(box[i]), 8, (i*85, 0, 0), -1)
        pass


def PT(img, c):
    pass


def getSubImg(img, pointOne, pointTwo):
    x0, y0 = pointOne
    x1, y1 = pointTwo
    return img[y0:y1, x0:x1, :]

# angle 為角度非逕度


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


def saftyRotate(image, angle, center=None, scale=1.0):
    # Create a safty rotation matrix
    maxValue = max(image.shape)
    rotated = np.zeros((maxValue, maxValue, image.shape[2]))
    print(rotated.shape)
    # Put the original image into the rotated matrix center
    new_tuple_list = sorted(image.shape[:2])
    a = new_tuple_list[0]//2
    rotated[maxValue//2-a:maxValue//2+a, :, :] = image

    # rotate
    rotated = rotate(rotated, angle)
    return rotated


def GetRect(screenCnt):
    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # multiply the rectangle by the original ratio
    # rect *= ratio
    return rect


def GetDst(rect):
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    return dst, maxWidth, maxHeight
