import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Paths
images_dir = "./data/train_data/images/"
densitymaps_dir = "./data/train_data/densitymaps/"

# Pick one file to test
filename = "20250829_202417.jpg"   # change this to any file you want
image_path = os.path.join(images_dir, filename)
npy_path   = os.path.join(densitymaps_dir, filename.replace(".jpg", ".npy"))

# Load image + density map
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
density = np.load(npy_path)

print("Image shape:", image.shape)
print("Density map shape:", density.shape)
print("Density sum (should equal #points):", density.sum())
print("Max density value:", density.max())

# Normalize density for visualization
density_vis = density / density.max() if density.max() > 0 else density

# Plot side by side
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(density_vis, cmap="jet")
plt.title(f"Density Map (sum={density.sum():.2f})")
plt.colorbar()

plt.show()
