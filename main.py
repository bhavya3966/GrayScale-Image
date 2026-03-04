import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Convert to grayscale using NumPy
gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])

#Blur using simple averaging filter
kernel = np.ones((5,5)) / 25

blur = cv2.filter2D(gray, -1, kernel)

#Edge detection (Sobel filter)
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

edges = cv2.filter2D(gray, -1, sobel_x)

#Thresholding
threshold = gray > 120

# Plot results
plt.figure(figsize=(10,8))

plt.subplot(221)
plt.title("Original")
plt.imshow(img)

plt.subplot(222)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")

plt.subplot(223)
plt.title("Blur")
plt.imshow(blur, cmap="gray")

plt.subplot(224)
plt.title("Edges")
plt.imshow(edges, cmap="gray")

plt.show()