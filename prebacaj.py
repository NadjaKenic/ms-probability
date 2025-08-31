import numpy as np # type: ignore
import tifffile # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path

# Učitaj .npz fajl
npz_data = np.load("tissuenet_v1.1_test.npz")

# Ako ima više nizova u fajlu, možemo ih ispisati da vidimo imena
print("Ključevi u fajlu:", list(npz_data.keys()))

data = npz_data["X"]

# Sačuvaj kao .tif
tifffile.imwrite("izlazne_slike.tif", data)

print("Fajl je uspešno sačuvan kao izlazne_slike.tif")

# --- PODEŠAVANJA ---
in_tif = "izlazne_slike.tif"       # .tif koji si već generisao iz .npz
out_base = Path("izlazne_slike")   # bazni naziv za izlazne fajlove (bez ekstenzije)
num_to_show = 5                    # koliko primera prikazati

# --- UČITAVANJE ---
images = tifffile.imread(in_tif)
print("Dimenzije:", images.shape)   # npr. (N, H, W, 2)
print("Tip podataka:", images.dtype)

# --- POMOĆNE FUNKCIJE ---
def to_uint16(arr: np.ndarray) -> np.ndarray:
    """Normalizuje na [0, 65535] i vraća uint16 (za kompaktniji TIFF i bolju kompatibilnost)."""
    a = arr.astype(np.float64)
    a = a - np.nanmin(a)
    mx = np.nanmax(a)
    if mx > 0:
        a = a / mx
    a = (a * 65535.0).round()
    return a.astype(np.uint16)

# --- PRIKAZ ---
N = images.shape[0]
num_to_show = min(num_to_show, N)

if images.ndim == 4 and images.shape[-1] == 2:
    # Prikaz kanala odvojeno: gornji red kanal 0, donji red kanal 1
    fig, axes = plt.subplots(2, num_to_show, figsize=(4*num_to_show, 8))
    if num_to_show == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # ujednači indeksiranje kada je 1 kolona

    for i in range(num_to_show):
        ch0 = images[i, :, :, 0]
        ch1 = images[i, :, :, 1]

        axes[0, i].imshow(ch0, cmap="gray")
        axes[0, i].set_title(f"Slika {i} – kanal 0")
        axes[0, i].axis("off")

        axes[1, i].imshow(ch1, cmap="gray")
        axes[1, i].set_title(f"Slika {i} – kanal 1")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()
else:
    # Standardni slučajevi (grayscale ili RGB)
    cols = num_to_show
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
    if cols == 1:
        axes = [axes]
    for i in range(cols):
        if images.ndim == 3:  # (N, H, W)
            axes[i].imshow(images[i], cmap="gray")
        elif images.ndim == 4 and images.shape[-1] in (3, 4):  # (N, H, W, 3/4)
            axes[i].imshow(images[i])
        else:
            raise ValueError(f"Nepodržan oblik za prikaz: {images.shape}")
        axes[i].set_title(f"Slika {i}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# --- ČUVANJE U .TIF ---
# Preporuka: sačuvati svaki kanal posebno radi kompatibilnosti sa pregledima/slikarskim softverom
out_ch0 = f"{out_base}_ch0.tif"
out_ch1 = f"{out_base}_ch1.tif"

if images.ndim == 4 and images.shape[-1] == 2:
    ch0 = images[..., 0]
    ch1 = images[..., 1]

    # Ako želiš zadržati float, može i direktno: tifffile.imwrite(out_ch0, ch0.astype(np.float32))
    # Ali za bolju kompatibilnost koristimo uint16:
    tifffile.imwrite(out_ch0, to_uint16(ch0), photometric="minisblack")
    tifffile.imwrite(out_ch1, to_uint16(ch1), photometric="minisblack")
    print(f"Sačuvano:\n- {out_ch0}\n- {out_ch1}")
else:
    # Jednokanalan ili RGB: može direktno
    # (po potrebi prevedi u uint16 ili uint8 – zavisi od tvog pipeline-a)
    out_single = f"{out_base}.tif"
    data = images
    if data.dtype == np.float64:
        data = data.astype(np.float32)  # TIFF lepo podržava float32
    tifffile.imwrite(out_single, data)
    print(f"Sačuvano: {out_single}")

# --- OPCIONO: sastavi pseudo-RGB iz 2 kanala da bi se videlo u standardnim viewer-ima ---
# (kanal 0 -> R, kanal 1 -> G, B = 0)
# Napomena: ovo je samo vizuelizacija, ne naučno spajanje kanala.
if images.ndim == 4 and images.shape[-1] == 2:
    ch0_u16 = to_uint16(images[..., 0])
    ch1_u16 = to_uint16(images[..., 1])
    rgb = np.stack([ch0_u16, ch1_u16, np.zeros_like(ch0_u16)], axis=-1)
    out_rgb = f"{out_base}_pseudoRGB.tif"
    tifffile.imwrite(out_rgb, rgb, photometric="rgb")
    print(f"Sačuvan i pseudo-RGB prikaz: {out_rgb}")