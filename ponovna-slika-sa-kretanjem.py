import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Maskiranje najveće konture ---
img = cv2.imread('slika4.png', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
_, binar = cv2.threshold(opening, 10, 255, cv2.THRESH_BINARY)

# kružni kernel
radius = 18
y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = (x*x + y*y) <= radius*radius
kernel_circle = mask.astype(np.uint8)

closing = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel_circle)
inv = cv2.bitwise_not(closing)

contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest_contours = contours[:1]

block_mask = np.ones_like(closing, dtype=np.uint8) * 255
cv2.drawContours(block_mask, largest_contours, -1, 0, thickness=cv2.FILLED)

# --- Braunovo kretanje ---
def simulate_2d_bm(nsteps=100000, t=0.5, mask=None):
    h, w = mask.shape
    x, y = [w // 2], [h // 2]  # start iz centra slike

    for _ in range(nsteps):
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        new_x, new_y = x[-1] + dx, y[-1] + dy

        if 0 <= new_x < w and 0 <= new_y < h and mask[new_y, new_x] != 0:
            x.append(new_x)
            y.append(new_y)

    return np.array(x), np.array(y)

x, y = simulate_2d_bm(mask=block_mask)

# --- Prikaz ---
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray', origin='upper')
plt.plot(x, y, lw=0.5, c='red')
plt.show()
