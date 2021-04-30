"""
项目实战2：OCR文字识别
"""

import cv2
import numpy as np
import argparse

from imutils.perspective import four_point_transform

from day03 import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image to be scanned")
args = vars(ap.parse_args())


def cv_show(name, age):
    cv2.imshow(name, age)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取数据
image = cv2.imread(args["image"])
# 预处理 灰度图---去除噪音点---边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.medianBlur(gray, 5)
gray = cv2.GaussianBlur(gray, (5, 5), 1)
edged = cv2.Canny(gray, 75, 200)

# cv_show("gray", gray)
#
# cv_show("edge", edged)

# canny 边缘检测后的图像为二值图  找轮廓，排序，留最大的边缘
cnts, hierachy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]

# 遍历轮廓
for c in cnts:
    # 轮廓周长、True表示轮廓是闭合的
    peri = cv2.arcLength(c, True)
    # 轮廓近似
    """
    cv.approxPolyDP() :
    参数1是源图像的某个轮廓；
    参数2(epsilon)是一个距离值，表示多边形的轮廓接近实际轮廓的程度，值越小，越精确；
    参数3表示是否闭合

    返回的是轮廓点的坐标值
    """
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # print(len(approx))
    # 如果是4个点，就是矩形了，就是我想要的轮廓
    if len(approx) == 4:
        screenCnt = approx
        # print(screenCnt)
        break

# 第二个参数是轮廓本身，在Python中是一个list。
#
"""screencnt不是函数内部的局部变量嘛，为什么能用？？"""
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv_show("outline", image)
#
# 坐标也会相同变化(图片上坐标的比例变化)
ratio = image.shape[0] / 500.0
orig = image.copy()

image = myutils.resize(orig, height=500)
#  透视变换：类似于扫描后的效果，规整图片
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# cv_show("warp", warped)
# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ret = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv_show("ret", ret)

