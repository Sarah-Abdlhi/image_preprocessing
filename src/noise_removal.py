# importing libraries 
import numpy as np 
import cv2
from matplotlib import pyplot as plt 

# load image/Reading image from folder where it is stored 
img = cv2.imread('/home/sarah/Documents/img_preprocessing/src/balloons_noisy.png')
#show image
#cv2.imshow('gol', img)
#cv2.waitKey(3000)


# denoising of image saving it into dst image 
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) 

# Plotting of source and destination image 
plt.subplot(121), plt.imshow(img) 
plt.subplot(122), plt.imshow(dst) 

plt.show() 

