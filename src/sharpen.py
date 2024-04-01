import cv2
import numpy as np
from matplotlib import pyplot as plt


original= cv2.imread('balloons_noisy.png', cv2.IMREAD_UNCHANGED)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Remove axis labels
plt.show()
print("Blur Image")

# create a sharpening kernel
sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])
# applying kernels to the input image to get the sharpened image

sharp_image=cv2.filter2D(original,-1,sharpen_filter)
plt.imshow(cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Remove axis labels
plt.show()
print("Sharpened Image")




import cv2

# Load the image in its original format
original = cv2.imread('balloons_noisy.png', cv2.IMREAD_UNCHANGED)

# Define the sharpening kernel
sharpen_filter = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Apply the sharpening filter
sharp_image = cv2.filter2D(original, -1, sharpen_filter)

# Display the original and sharpened images using OpenCV
cv2.imshow("Original Image", original)
cv2.waitKey(0)  # Wait for a key press to close the window

cv2.imshow("Sharpened Image", sharp_image)
cv2.waitKey(0)  # Wait for a key press to close the window

cv2.destroyAllWindows()  # Close all OpenCV windows

