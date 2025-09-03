import torch
from model import CSRNet
from dataset import create_test_dataloader
import numpy as np

# Set your best epoch checkpoint here
BEST_EPOCH = 97 # Replace with your best epoch number, e.g. 7
CHECKPOINT_PATH = f'./checkpoints/{BEST_EPOCH}.pth'

# Load model
model = CSRNet()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()

# Load test dataloader
TEST_ROOT = './data'
test_loader = create_test_dataloader(TEST_ROOT)

mae = 0.0
with torch.no_grad():
    for data in test_loader:
        image = data['image']
        gt_densitymap = data['densitymap']
        et_densitymap = model(image)
        gt_count = gt_densitymap.sum().item()
        et_count = et_densitymap.sum().item()
        mae += abs(et_count - gt_count)
mae /= len(test_loader)
print(f'Test MAE: {mae:.2f}')
