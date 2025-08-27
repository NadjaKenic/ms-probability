import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('slika4.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

_, binar = cv2.threshold(opening, 10, 255, cv2.THRESH_BINARY)

radius = 20
y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x*x + y*y <= radius*radius
kernel_circle = mask.astype(np.uint8)


closing = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel_circle)

plt.figure()
plt.imshow(closing, cmap='gray')

inv = cv2.bitwise_not(closing)

contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)
num_to_mask = 1
largest_contours = contours[:num_to_mask]

block_mask = np.ones_like(closing, dtype=np.uint8) * 255
cv2.drawContours(block_mask, largest_contours, -1, 0, thickness=cv2.FILLED)

masked_result = cv2.bitwise_and(closing, block_mask)

plt.figure()
plt.imshow(block_mask, cmap='gray')

plt.figure()
plt.imshow(masked_result, cmap='gray')

plt.show()
