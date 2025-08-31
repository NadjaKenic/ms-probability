import numpy as np
import tifffile

data = np.load('e:\\petnica-slike-tissuenet\\tissuenet_v1.1_test.npz')

# Uzmi prvu sliku
image_array = data['X'][0]          # (256, 256, 2)

# Preuredi u (kanali, visina, širina)
image_array = np.moveaxis(image_array, -1, 0)  # (2, 256, 256)

# Sačuvaj kao višekanalni TIFF
tifffile.imwrite('output_multichannel.tif', image_array)

print("Sačuvano: output_multichannel.tif, shape:", image_array.shape)




