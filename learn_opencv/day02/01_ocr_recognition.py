"""
项目实战 1：信用卡数字识别

整体思路：
1. 使用模板匹配
  整体图像、模板图像
2. 整体图像中的数字 与 模板图像中数字的轮廓 去匹配

3. 模板与图像大小不同，轮廓大小也可能不同，因此我们找到轮廓后，进行resize，再进行外界矩形来匹配

4. 匹配之前的预处理：
 灰度图 --- 二值图 --- 筛选需要的轮廓 --- 依次匹配

https://blog.csdn.net/weixin_41874898/article/details/99624454
"""

import cv2
import numpy as np
import argparse
from day02 import myutils
from imutils import contours   # 排序是用到的安装包是一样的
import matplotlib.pyplot as plt
"""
argparse 是 Python 内置的一个用于命令项选项与参数解析的模块
主要有三个步骤：
- 创建 ArgumentParser() 对象
- 调用 add_argument() 方法添加参数
- 使用 parse_args() 解析添加的参数
"""
# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# 绘图显示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 展示模板图像
img = cv2.imread(args['template'])
cv_show("img", img)

# 灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show("img_gray",img_gray)

# 二值化 (threshold返回值是两个参数，我们取第二个)
ref = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 绘制轮廓
# ref.copy():会改变原图,所以需要复制; cv2.RETR_EXTERNAL：只绘制外部轮廓;
refcnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, refcnts, -1, (0, 0, 255), 2)
cv_show("img", img)

# dtype=object 有的版本不会出现警告，不需要
print(np.array(refcnts, dtype=object).shape)  # 一共有10个轮廓

# 对轮廓进行排序
refcnts = contours.sort_contours(refcnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refcnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # 每个图像对应一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取图像，预处理
image = cv2.imread(args["image"])
cv_show("image",image)

image = cv2.resize(image, (300, 186))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show("image", image_gray)

# 去除一些不需要的信息：有多种操作，这里选择礼帽操作
# 礼帽操作: 礼帽=原始输入-开运算结果
tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show("tophat", tophat)

# 算子计算边缘
# gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # ksize=-1 相当于3*3
#
# gradX = np.absolute(gradX)
# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
# gradX = gradX.astype("uint8")

canny = cv2.Canny(tophat, 50, 100)  # 边缘检测效果比较好
cv_show("canny",canny)
# print(canny.shape)

# 边缘有了，下一步是把需要找的区域过滤出来
# 1、可以膨胀出来,再腐蚀
kernel = np.ones((3, 3))  # 卷积核大小
dige_dilate = cv2.dilate(canny, kernel, iterations=4)
erosion = cv2.erode(dige_dilate, kernel, iterations=2)
cv_show("kernel", erosion)
# # 2、直接进行闭操作：效果不好
# closing=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
# cv_show("kernel",closing)

# res=np.hstack((gradX,canny))
# cv_show("res",res)

# 图像二值化后，绘制轮廓
# 这里0，255，的阈值是适合双峰图像的，也就是两色图像，可以自动处理阈值
chresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show("res",chresh)
#
# res=np.hstack((erosion,chresh))
# cv_show("res",res)
cnts, hierarchy = cv2.findContours(chresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt_img = image.copy()
cnt_img = cv2.drawContours(cnt_img, cnts, -1, (0, 255, 0), 1)
cv_show("img", cnt_img)
# cv2.imwrite("cnt_img.jpg",cnt_img)

# 55/21  54/19
# 找出想要的区域
locs = []
for (i, c) in enumerate(cnts):
    # 计算轮廓的外界矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 选择想保留的区域
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 56) and (h > 10 and h < 24):
            locs.append((x, y, w, h))
print(locs)

# 将符合的轮廓进行排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 将大轮廓中的每一个数据遍历，来进行匹配
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    # 找到了4个组，每组都是一个大块
    group = image_gray[gy - 6:gy + gh + 6, gx - 6:gx + gw + 6]
    cv_show('group', group)
    # 截出来每一个块之后，去找小块
    # 1、二值化
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show("group",group)
    # 2、找轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 3、排序
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算小轮廓
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # cv_show("roi", roi)

        # 进行匹配得分
        scores = []

        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))

    # 结果
    cv2.rectangle(image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    output.extend(groupOutput)


# 打印结果
print("Credit Card Type:{}".format(FIRST_NUMBER[output[0]]))  # 这里其实需要循环的
# 如果换图片之后出现keyError format的问题，就是之前筛选区域的范围出问题了
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
