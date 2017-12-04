# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

images = []
path = "C:/Users/acer/tentestclasses/"
for image in os.listdir(path):
    images.append(image)
plt.figure(1)

pos = np.arange(1,100)
for (image,b) in zip(images, pos):
    img = cv2.imread("%s%s"%(path, image))    # Load the image 
    channels = cv2.split(img)       # Set the image channels
    colors = ("b", "g", "r")        # Initialize tuple      
    plt.subplot(10,10,b)
    for (i, col) in zip(channels, colors):       # Loop over the image channels
        hist = cv2.calcHist([i], [0], None, [256], [0, 256])   # Create a histogram for current channel
        plt.plot(hist, color = col)      # Plot the histogram  
        plt.xlim([0, 256])
        plt.xticks(hist," ")
        plt.yticks(hist," ")
    
plt.show()