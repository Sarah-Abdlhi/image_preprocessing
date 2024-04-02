import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

# Load the image 
image = cv2.imread('/home/sarah/Documents/img_preprocessing/src/flower_blur.png') 

# Plot the original image 
plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 

# Create the sharpening kernel 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

# Split the image into color channels 
b, g, r = cv2.split(image)

# Sharpen each color channel 
sharpened_b = cv2.filter2D(b, -1, kernel) 
sharpened_g = cv2.filter2D(g, -1, kernel) 
sharpened_r = cv2.filter2D(r, -1, kernel) 

# Merge the sharpened color channels 
sharpened_image = cv2.merge((sharpened_b, sharpened_g, sharpened_r))

# Save the image 
cv2.imwrite('sharpened_image.jpg', sharpened_image) 

# Plot the sharpened image 
plt.subplot(1, 2, 2) 
plt.title("Sharpening") 
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)) 
plt.show()
