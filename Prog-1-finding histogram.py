# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')

histb = cv2.calcHist([img],[0],None,[256],[0,256])
histg = cv2.calcHist([img],[1],None,[256],[0,256])
histr = cv2.calcHist([img],[2],None,[256],[0,256])

plt.plot(histb), plt.plot(histg), plt.plot(histr)
plt.xlim([0,256])
plt.legend("BGR")
plt.show()