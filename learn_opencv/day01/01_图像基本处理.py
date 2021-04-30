import cv2   # opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('noise.png')

# 均值滤波
# 用3*3的核对图片进行卷积操作，核上的参数都是1/9，达到均值的效果
blur = cv2.blur(img, (3, 3))
# 方框滤波（归一化）=均值滤波
box1 = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# 方框滤波（不归一化）
box2 = cv2.boxFilter(img, -1, (3, 3), normalize=False)
# 高斯滤波
# 用5*5的核进行卷积操作，但核上离中心像素近的参数大。
guassian = cv2.GaussianBlur(img, (5, 5), 1)
# 中值滤波
# 将某像素点周围5*5的像素点提取出来，排序，取中值写入此像素点。
mean = cv2.medianBlur(img, 5)

res=np.hstack((blur,box1,guassian,mean))
# vstack是竖着展示，hstack是横着展示
# print(res)

cv2.imshow("noise_low",res)
cv2.waitKey(0)
cv2.destroyAllWindows()