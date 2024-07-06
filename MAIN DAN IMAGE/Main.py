import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('image/animal.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

pixel_vals = image.reshape((-1, 3))

pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 3

retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

segmented_data = centers[labels.flatten()]

segmented_image = segmented_data.reshape((image.shape))

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Gambar Tersegmentasi (K=3)')
plt.axis('off')

plt.tight_layout()
plt.show()
