import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import CSRNet
from dataset import get_transform   # <-- we'll create this helper
import numpy as np

# Config
BEST_EPOCH = 93
CHECKPOINT_PATH = f'./checkpoints/{BEST_EPOCH}.pth'
IMAGE_PATH = './data/train_data/images/20250829_202417.jpg'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Use the same transform as training ---
transform = get_transform()  # Reuse transform from dataset.py

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

# Show image and density map
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img)
axs[0].set_title(f'Input Image\nPredicted count: {count:.2f}')
axs[0].axis('off')

axs[1].imshow(density_map, cmap='jet')
axs[1].set_title('Predicted Density Map')
axs[1].axis('off')

plt.show()
