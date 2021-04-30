import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("04_3.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值图
# show_img("thresh", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw_img = img.copy()
# 绘制图像、轮廓、轮廓索引(内层+外层)、颜色模式(BGR)、线条厚度
res = cv2.drawContours(draw_img, contours, 1, (0, 0, 125), 2)


show_img("res", res)


