
# coding: utf-8

# In[5]:

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg',0)

# create a mask1
mask = np.zeros(img.shape[:2], np.uint8)
mask[0:113, 0:136] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

# create a mask2
mask2 = np.zeros(img.shape[:2], np.uint8)
mask2[0:113, 136:272] = 255
masked_img2 = cv2.bitwise_and(img,img,mask = mask2)
hist_mask2 = cv2.calcHist([img],[0],mask2,[256],[0,256])

# create a mask3
mask3 = np.zeros(img.shape[:2], np.uint8)
mask3[0:113, 272:410] = 255
masked_img3 = cv2.bitwise_and(img,img,mask = mask3)
hist_mask3 = cv2.calcHist([img],[0],mask3,[256],[0,256])

# create a mask4
mask4 = np.zeros(img.shape[:2], np.uint8)
mask4[113:226, 0:136] = 255
masked_img4 = cv2.bitwise_and(img,img,mask = mask4)
hist_mask4 = cv2.calcHist([img],[0],mask4,[256],[0,256])

# create a mask5
mask5 = np.zeros(img.shape[:2], np.uint8)
mask5[113:226, 136:272] = 255
masked_img5 = cv2.bitwise_and(img,img,mask = mask5)
hist_mask5 = cv2.calcHist([img],[0],mask5,[256],[0,256])

# create a mask6
mask6 = np.zeros(img.shape[:2], np.uint8)
mask6[113:226, 272:410] = 255
masked_img6 = cv2.bitwise_and(img,img,mask = mask6)
hist_mask6 = cv2.calcHist([img],[0],mask6,[256],[0,256])

# create a mask7
mask7 = np.zeros(img.shape[:2], np.uint8)
mask7[226:341, 0:136] = 255
masked_img7 = cv2.bitwise_and(img,img,mask = mask7)
hist_mask7 = cv2.calcHist([img],[0],mask7,[256],[0,256])

# create a mask8
mask8 = np.zeros(img.shape[:2], np.uint8)
mask8[226:341, 136:272] = 255
masked_img8 = cv2.bitwise_and(img,img,mask = mask8)
hist_mask8 = cv2.calcHist([img],[0],mask8,[256],[0,256])

# create a mask9
mask9 = np.zeros(img.shape[:2], np.uint8)
mask9[226:339, 272:408] = 255
masked_img9 = cv2.bitwise_and(img,img,mask = mask9)
hist_mask9 = cv2.calcHist([img],[0],mask9,[256],[0,256])

plt.figure(1)
plt.subplot(121), plt.imshow(img, 'gray')
plt.subplot(122), plt.plot(hist_full,label="Full image"), plt.plot(hist_mask,label="mask1"),plt.plot(hist_mask2,label="mask2"),plt.plot(hist_mask3,label="mask3"),plt.plot(hist_mask4,label="mask4"),plt.plot(hist_mask5,label="mask5"),plt.plot(hist_mask6,label="mask6"),plt.plot(hist_mask7,label="mask7"),plt.plot(hist_mask8, label="mask8"),plt.plot(hist_mask9,label="mask9")
plt.legend()

plt.figure(2)
plt.subplot(331), plt.imshow(masked_img, 'gray')
plt.subplot(332), plt.imshow(masked_img2, 'gray')
plt.subplot(333), plt.imshow(masked_img3, 'gray')
plt.subplot(334), plt.imshow(masked_img4, 'gray')
plt.subplot(335), plt.imshow(masked_img5, 'gray')
plt.subplot(336), plt.imshow(masked_img6, 'gray')
plt.subplot(337), plt.imshow(masked_img7, 'gray')
plt.subplot(338), plt.imshow(masked_img8, 'gray')
plt.subplot(339), plt.imshow(masked_img9, 'gray')

plt.show()
