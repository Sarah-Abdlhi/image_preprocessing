import numpy as np 
import cv2 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode

# Reading image from folder where it is stored 
img = cv2.imread('/home/sarah/Documents/img_preprocessing/src/balloons.png') 

# denoising of image saving it into dst image 
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 

# Plotting of source and destination image 
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title('Denoised Image')
plt.axis('off')

plt.show()

# Wait for user input before exiting
input("Press Enter to close the plot...")
