import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from model import CSRNet
import os

# Config
BEST_EPOCH = 93
CHECKPOINT_PATH = f'./checkpoints/{BEST_EPOCH}.pth'
IMAGE_PATH = './data/train_data/images/20250829_202417.jpg'  # can be new/unseen image
# IMAGE_PATH = './data/train_data/images/20250829_202417.jpg'  # can be new/unseen image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (normalize with ImageNet-style stats)
transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5],
              std=[0.225, 0.225, 0.225])
])

# Load image
img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Load model
model = CSRNet().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Forward pass
with torch.no_grad():
    output = model(img_tensor)
    count = output.sum().item()
    density_map = output.squeeze().cpu().numpy()

print(f'Predicted count: {count:.2f}')

# Try to load GT density map if available
DENSITY_PATH = IMAGE_PATH.replace("images", "densitymaps").replace(".jpg", ".npy").replace(".jpeg", ".npy")
if os.path.exists(DENSITY_PATH):
    gt_densitymap = np.load(DENSITY_PATH)
    print("GT count:", gt_densitymap.sum())
    print("ET count:", density_map.sum())
else:
    print("⚠️ No ground truth density map found (new/unseen image). Skipping GT comparison.")

# Show image and density map
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img)
axs[0].set_title(f'Input Image\nPredicted count: {count:.2f}')
axs[0].axis('off')

axs[1].imshow(density_map, cmap='jet')
axs[1].set_title('Predicted Density Map')
axs[1].axis('off')

plt.show()
