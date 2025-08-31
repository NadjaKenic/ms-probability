import numpy as np
import matplotlib.pyplot as plt

def get_ring_pixels(cx, cy, radius, w, h):
    """Vrati samo piksele na prstenu udaljenosti 'radius' od centra (cx, cy)."""
    pixels = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                # Udaljenost od centra u Manhattan metrici
                if abs(dx) + abs(dy) == radius:
                    pixels.append((nx, ny))
    return pixels

def simulate_2d_bm(nsteps=500, mask=None, radius=1):
    h, w = mask.shape
    x, y = [w // 2], [h // 2]
    circle_pixels_all = []

    for _ in range(nsteps):
        dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
        new_x, new_y = x[-1] + dx, y[-1] + dy

        if 0 <= new_x < w and 0 <= new_y < h and mask[new_y, new_x] != 0:
            x.append(new_x)
            y.append(new_y)

            # dodaj piksele tačno na zadatom radijusu
            circle_pixels = get_ring_pixels(new_x, new_y, radius, w, h)
            circle_pixels_all.extend(circle_pixels)

    return np.array(x), np.array(y), circle_pixels_all


# Primer maske
mask = np.ones((50, 50), dtype=np.uint8)

# Pokreni sa radijusom 1 i 2
x1, y1, pixels_r1 = simulate_2d_bm(mask=mask, radius=2)

# Vizuelizacija


plt.plot(x1, y1, lw=0.5, color="blue")
px, py = zip(*pixels_r1)
plt.scatter(px, py, color="orange", s=4, alpha=0.7)

plt.show()