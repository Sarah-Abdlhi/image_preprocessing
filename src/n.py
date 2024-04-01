# fastNlMeansDenoisingColored
# Wait for result, takes time to respond
import cv2
from tkinter import filedialog
from tkinter import *

root = Tk()
# Do not show graphics window
root.withdraw()

# Load the original color image
origColorImage = cv2.imread(filedialog.askopenfilename(image.jpeg),1)

# Image must have 3 channels
print("Shape of image ", origColorImage.shape)

dest = cv2.fastNlMeansDenoisingColored(origColorImage,None,10,10,7,21)

cv2.imshow('Original image',origColorImage)
cv2.imshow('fastNlMeansDenoisingColored',dest)




import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('/home/sarah/Documents/img_preprocessing/src/balloons_noisy.png')

# Check if image was loaded successfully
if img is None:
    print("Error: Image not found.")
    exit()

# Denoise image
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# Convert BGR to RGB for proper display in matplotlib
dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# Plot source and denoised images
plt.subplot(121), plt.imshow(img)
plt.title('Original Image')

plt.subplot(122), plt.imshow(dst_rgb)
plt.title('Denoised Image')

plt.show()
