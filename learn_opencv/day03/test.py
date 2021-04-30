"""
中文字体库识别：
    https://www.cnblogs.com/mrlayfolk/p/12617077.html


该测试模块有遗留问题：
1、没有安装tesseract.exe (pip install py..过了)
 - 安装了之后，去找.py文件，在默认路径前添加一个r，不然 / 的符号, windows系统无法识别
 - r：保持字符原始值的意思
2、该模板的中文版本可能不好用，可以直接使用百度OCR去进行文字识别
3、目前还没有需要识别文字，可以去尝试识别英文即可
"""

from PIL import Image
import pytesseract
import cv2
import os

preprocess = "blur"

image = cv2.imread("./images/4.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

if preprocess == "thresh":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.show("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
