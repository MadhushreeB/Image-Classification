
# coding: utf-8

# In[8]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv

img=plt.imread('img.jpg')

print(img.shape[:2])

hist_maxr = np.zeros([9])
index_maxr = np.zeros([9])
hist_maxg = np.zeros([9])
index_maxg = np.zeros([9])
hist_maxb = np.zeros([9])
index_maxb = np.zeros([9])

# create a mask1
mask = np.zeros(img.shape[:2], np.uint8)
mask[0:85, 0:85] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])

hist_maskr = cv2.calcHist([img],[0],mask,[256],[0,256])
hist_maskg = cv2.calcHist([img],[1],mask,[256],[0,256])
hist_maskb = cv2.calcHist([img],[2],mask,[256],[0,256])

hist_maxr[0] = max(hist_maskr)
index_maxr[0] = np.argmax(hist_maskr)

hist_maxg[0] = max(hist_maskg)
index_maxg[0] = np.argmax(hist_maskg)

hist_maxb[0] = max(hist_maskb)
index_maxb[0] = np.argmax(hist_maskb)

# create a mask2
mask2 = np.zeros(img.shape[:2], np.uint8)
mask2[0:85, 85:170] = 255
masked_img2 = cv2.bitwise_and(img,img,mask = mask2)
hist_mask2r = cv2.calcHist([img],[0],mask2,[256],[0,256])
hist_mask2g = cv2.calcHist([img],[1],mask2,[256],[0,256])
hist_mask2b = cv2.calcHist([img],[2],mask2,[256],[0,256])

hist_maxr[1] = max(hist_maskr)
index_maxr[1] = np.argmax(hist_maskr)

hist_maxg[1] = max(hist_maskg)
index_maxg[1] = np.argmax(hist_maskg)

hist_maxb[1] = max(hist_maskb)
index_maxb[1] = np.argmax(hist_maskb)


# create a mask3
mask3 = np.zeros(img.shape[:2], np.uint8)
mask3[0:85, 170:256] = 255
masked_img3 = cv2.bitwise_and(img,img,mask = mask3)
hist_mask3r = cv2.calcHist([img],[0],mask3,[256],[0,256])
hist_mask3g = cv2.calcHist([img],[1],mask3,[256],[0,256])
hist_mask3b = cv2.calcHist([img],[2],mask3,[256],[0,256])

hist_maxr[2] = max(hist_mask3r)
index_maxr[2] = np.argmax(hist_mask3r)

hist_maxg[1] = max(hist_mask3g)
index_maxg[1] = np.argmax(hist_mask3g)

hist_maxb[1] = max(hist_mask3b)
index_maxb[1] = np.argmax(hist_mask3b)

# create a mask4
mask4 = np.zeros(img.shape[:2], np.uint8)
mask4[85:170, 0:85] = 255
masked_img4 = cv2.bitwise_and(img,img,mask = mask4)
hist_mask4r = cv2.calcHist([img],[0],mask4,[256],[0,256])
hist_mask4g = cv2.calcHist([img],[1],mask4,[256],[0,256])
hist_mask4b = cv2.calcHist([img],[2],mask4,[256],[0,256])

hist_maxr[3] = max(hist_mask4r)
index_maxr[3] = np.argmax(hist_mask4r)

hist_maxg[3] = max(hist_mask4g)
index_maxg[3] = np.argmax(hist_mask4g)

hist_maxb[3] = max(hist_mask4b)
index_maxb[3] = np.argmax(hist_mask4b)


# create a mask5
mask5 = np.zeros(img.shape[:2], np.uint8)
mask5[85:170, 85:170] = 255
masked_img5 = cv2.bitwise_and(img,img,mask = mask5)
hist_mask5r = cv2.calcHist([img],[0],mask5,[256],[0,256])
hist_mask5g = cv2.calcHist([img],[1],mask5,[256],[0,256])
hist_mask5b = cv2.calcHist([img],[2],mask5,[256],[0,256])

hist_maxr[4] = max(hist_mask5r)
index_maxr[4] = np.argmax(hist_mask5r)

hist_maxg[4] = max(hist_mask5g)
index_maxg[4] = np.argmax(hist_mask5g)

hist_maxb[4] = max(hist_mask5b)
index_maxb[4] = np.argmax(hist_mask5b)


# create a mask6
mask6 = np.zeros(img.shape[:2], np.uint8)
mask6[85:170, 170:256] = 255
masked_img6 = cv2.bitwise_and(img,img,mask = mask6)
hist_mask6r = cv2.calcHist([img],[0],mask6,[256],[0,256])
hist_mask6g = cv2.calcHist([img],[1],mask6,[256],[0,256])
hist_mask6b = cv2.calcHist([img],[2],mask6,[256],[0,256])

hist_maxr[5] = max(hist_mask6r)
index_maxr[5] = np.argmax(hist_mask6r)

hist_maxg[5] = max(hist_mask6g)
index_maxg[5] = np.argmax(hist_mask6g)

hist_maxb[5] = max(hist_mask6b)
index_maxb[5] = np.argmax(hist_mask6b)


# create a mask7
mask7 = np.zeros(img.shape[:2], np.uint8)
mask7[170:256, 0:85] = 255
masked_img7 = cv2.bitwise_and(img,img,mask = mask7)
hist_mask7r = cv2.calcHist([img],[0],mask7,[256],[0,256])
hist_mask7g = cv2.calcHist([img],[1],mask7,[256],[0,256])
hist_mask7b = cv2.calcHist([img],[2],mask7,[256],[0,256])

hist_maxr[6] = max(hist_mask7r)
index_maxr[6] = np.argmax(hist_mask7r)

hist_maxg[6] = max(hist_mask7g)
index_maxg[6] = np.argmax(hist_mask7g)

hist_maxb[6] = max(hist_mask7b)
index_maxb[6] = np.argmax(hist_mask7b)


# create a mask8
mask8 = np.zeros(img.shape[:2], np.uint8)
mask8[170:256, 85:170] = 255
masked_img8 = cv2.bitwise_and(img,img,mask = mask8)
hist_mask8r = cv2.calcHist([img],[0],mask8,[256],[0,256])
hist_mask8g = cv2.calcHist([img],[1],mask8,[256],[0,256])
hist_mask8b = cv2.calcHist([img],[2],mask8,[256],[0,256])

hist_maxr[7] = max(hist_mask8r)
index_maxr[7] = np.argmax(hist_mask8r)

hist_maxg[7] = max(hist_mask8g)
index_maxg[7] = np.argmax(hist_mask8g)

hist_maxb[7] = max(hist_mask8b)
index_maxb[7] = np.argmax(hist_mask8b)


# create a mask9
mask9 = np.zeros(img.shape[:2], np.uint8)
mask9[170:256, 170:256] = 255
masked_img9 = cv2.bitwise_and(img,img,mask = mask9)
hist_mask9r = cv2.calcHist([img],[0],mask9,[256],[0,256])
hist_mask9g = cv2.calcHist([img],[1],mask9,[256],[0,256])
hist_mask9b = cv2.calcHist([img],[2],mask9,[256],[0,256])

hist_maxr[8] = max(hist_mask9r)
index_maxr[8] = np.argmax(hist_mask9r)

hist_maxg[8] = max(hist_mask9g)
index_maxg[8] = np.argmax(hist_mask9g)

hist_maxb[8] = max(hist_mask9b)
index_maxb[8] = np.argmax(hist_mask9b)

print(index_maxr)
print(index_maxg)
print(index_maxb)

index_max = []
index_max.extend(index_maxr)
index_max.extend(index_maxg)
index_max.extend(index_maxb)
print(index_max)

with open('features.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(index_max)

plt.figure(1)
plt.imshow(img)

plt.figure(2)
plt.subplot(331), plt.imshow(masked_img)
plt.subplot(332), plt.imshow(masked_img2)
plt.subplot(333), plt.imshow(masked_img3)
plt.subplot(334), plt.imshow(masked_img4)
plt.subplot(335), plt.imshow(masked_img5)
plt.subplot(336), plt.imshow(masked_img6)
plt.subplot(337), plt.imshow(masked_img7)
plt.subplot(338), plt.imshow(masked_img8)
plt.subplot(339), plt.imshow(masked_img9)

plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



