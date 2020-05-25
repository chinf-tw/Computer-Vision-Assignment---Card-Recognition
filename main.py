import numpy as np
import time
import func
import math
from cv2 import cv2


def runSimpleFindContours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(Gaussian, 100, 250)
    contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (contours, hierarchy)


def showImgWithPlt(img, Gaussian, canny):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Original Image')
    ax1.imshow(img, interpolation='bicubic')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Gaussian')
    ax2.imshow(Gaussian, cmap='gray')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('canny')
    ax3.imshow(canny, cmap='gray')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Original + Line')
    draw = func.DrawLine(canny, img)
    ax4.imshow(draw)


def drawApproxPolyDP(img, c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)


def drawRotatedRectangle(img, c):
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print(box.shape)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
    # for i in range(len(box)):
    #     for j in range(len(box)):
    #         result = twoPointEuclideanDistance(box[i],box[j])
    #         print("point {i} to point {j} = {result}".format(i=i,j=j,result=result))
    #     pass
    for i in range(len(box)):
        x, y = box[i, :]
        cv2.putText(img, str(i), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        pass
    pass


def getMinAreaRectPoints(c):
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def getPerspectiveSrc(c):
    result = []
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    for i in range(len(box)):
        for j in range(len(box)):
            distance = int(twoPointEuclideanDistance(box[i], box[j]))
            if distance not in result and distance > 0:
                result.append(distance)
            if len(result) == 3:
                return result
        pass
    return None


def srcToDstConvert(src):
    src.sort()
    a = src[0]
    b = src[1]
    if a > b:
        raise ValueError
    rect = np.array([(0, 0), (a, 0), (a, b), (0, b)])
    return rect


def Perspective(img, c):
    src = getPerspectiveSrc(c)
    src.sort()
    maxWidth, maxHeight = src[0], src[1]
    rect = getMinAreaRectPoints(c)
    dst = srcToDstConvert(src)
    dst, maxWidth, maxHeight = func.GetDst(rect)
    rect = rect.astype(np.float32)
    dst = dst.astype(np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # 畫點的數字
    # for i in range(len(dst)):
    #     x = int(dst[i,0])
    #     y = int(dst[i,1])
    #     if x <= 0 :
    #         x += 10
    #     else:
    #         x -= 10
    #     if y <= 0 :
    #         y += 20
    #     else:
    #         y -= 20
    #     cv2.putText(warp,str(i) , (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return warp


def twoPointEuclideanDistance(pointOne, pointTwo):
    if len(pointOne) != len(pointTwo):
        raise ValueError
    result = 0
    for i in range(len(pointOne)):
        result += math.pow(pointOne[i]-pointTwo[i], 2)
    result = math.sqrt(result)
    return result


def runAllContours(img, contours, func):
    for c in contours:
        # 設置敏感度
        # contourArea計算輪廓面積
        if cv2.contourArea(c) < 1000:
            continue
        else:
            func(c)


def drawAllContours(img, contours):
    count = 0

    for c in contours:
        # 設置敏感度
        # contourArea計算輪廓面積
        if cv2.contourArea(c) < 1000:
            continue
        else:
            pointOne, pointTwo = func.getDiagonalPoints(c)
            # rectangle(原圖,(x,y)是矩陣的左上點座標,(x+w,y+h)是矩陣的右下點座標,(0,255,0)是畫線對應的rgb顏色,2是所畫的線的寬度)
            cv2.rectangle(img, pointOne, pointTwo, (0, 255, 0), 2)
            # fitEllipse 為透過contour來得知該contour的角度跟中心點，以及長軸短軸
            center, (MA, ma), angle = cv2.fitEllipse(c)
            # putText 顯示文字套件
            # cv2.putText(img,str(angle) , (x, y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            func.drawFourPointContours(img, c)
            subimg = func.getSubImg(img, pointOne, pointTwo)
            rotated = func.rotate(subimg, angle)
            cv2.imshow("Sub Image - "+str(count), rotated)
            count += 1


if __name__ == "__main__":
    resizeLen = 1
    img = cv2.imread('./FA35C405-BE5B-4D49-B4F2-4FE4994E940A_1_105_c.jpeg')
    h, w, _ = img.shape
    if resizeLen > 1:
        img = cv2.resize(img, (w//resizeLen, h//resizeLen))
    # canny = func.getCanny(img)
    contours, hierarchy = runSimpleFindContours(img)

    window_name = 'Image'

    # c = contours[1]
    # pointOne,pointTwo = func.getDiagonalPoints(c)
    # subimg = getSubImg(img,pointOne,pointTwo)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # cv2.imshow(window_name, canny)

    # the perspective to grab the screen
    # rect = func.GetRect(contours[0])
    # dst,maxWidth,maxHeight = func.GetDst(rect)
    # M = cv2.getPerspectiveTransform(rect, dst)
    # warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # img = func.DrawLine(canny, img)

    # drawRotatedRectangle(img, contours[0])

    # kpImg = cv2.imread('./Pokers/3S.png',0)
    # kpImg = 255 - kpImg
    Target = cv2.imread('./3S_.png',0)
    resizeLen = 1
    h, w, _ = img.shape
    if resizeLen > 1:
        Target = cv2.resize(Target, (w//resizeLen, h//resizeLen))
    print(Target.shape)
    cv2.imshow("Target", Target)
    count = 0
    # contours = [contours[1]]
    for c in contours:
        # 設置敏感度
        # contourArea計算輪廓面積
        if cv2.contourArea(c) < 1000:
            continue
        else:
            warp = Perspective(img, c)
            # 倒下轉正問題
            if warp.shape[0] < warp.shape[1] :
                print("T")
                warp = func.rotate(warp, 90)

            # print("warp",warp.shape)
            # print("Target",Target.shape)
            # Target = cv2.resize(Target,(Target.shape[1]*warp.shape[0]//Target.shape[0],warp.shape[0]))
            # print(Target.shape)
            warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            ret2,th2 = cv2.threshold(warp_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            func.mathc_img(th2,Target,0.9)
            # img2 = func.ORB(th2,Target)
            print("Poker - " + str(count)," ",th2.shape)
            cv2.imshow("Poker - " + str(count), th2)
            # if count == 0:
            #     cv2.imwrite("3S_.png",func.getSubImg(warp,(3,13),(19,62)))
            count += 1

    # cv2.imshow("before", img)
    # cv2.imshow("after", warp)
    while True:
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    cv2.destroyAllWindows()
    # plt.show()
