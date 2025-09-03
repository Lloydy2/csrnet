# import torch
# import matplotlib.pyplot as plt
# import matplotlib.cm as CM
# from tqdm import tqdm
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# # from csrnet import CSRNet
# from dataset import CrowdDataset
# from model import CSRNet


# def cal_mae(model_param_path):
#     '''
#     Calculate the MAE of the test data.
#     model_param_path: the path of specific CSRNet parameters.
#     '''
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CSRNet()
#     model.load_state_dict(torch.load(model_param_path))
#     model.to(device)
#     img_trans = Compose([Resize((512, 512)), ToTensor(), Normalize(mean=[0.5,0.5,0.5], std=[0.225,0.225,0.225])])
#     dmap_trans = Compose([Resize((512, 512)), ToTensor()])
#     dataset = CrowdDataset(root='./data', phase='test', img_transform=img_trans, dmap_transform=dmap_trans)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#     model.eval()
#     mae = 0
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(dataloader)):
#             img = data['image'].to(device)
#             gt_dmap = data['densitymap'].to(device)
#             et_dmap = model(img)
#             mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
#             del img, gt_dmap, et_dmap
#     print("model_param_path:" + model_param_path + " mae:" + str(mae / len(dataloader)))

# def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
#     '''
#     Show one estimated density-map.
#     img_root: the root of test image data.
#     gt_dmap_root: the root of test ground truth density-map data.
#     model_param_path: the path of specific mcnn parameters.
#     index: the order of the test image in test dataset.
#     '''
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model=CSRNet().to(device)
#     model.load_state_dict(torch.load(model_param_path))
#     dataset=CrowdDataset(img_root,gt_dmap_root,8)
#     dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
#     model.eval()
#     for i,(img,gt_dmap) in enumerate(dataloader):
#         if i==index:
#             img=img.to(device)
#             gt_dmap=gt_dmap.to(device)
#             # forward propagation
#             et_dmap=model(img).detach()
#             et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
#             print(et_dmap.shape)
#             plt.imshow(et_dmap,cmap=CM.jet)
#             break


# if __name__=="__main__":
#     torch.backends.cudnn.enabled=False
#     model_param_path = './checkpoints/78.pth'  # Update to your best checkpoint
#     cal_mae(model_param_path)
#     # estimate_density_map(img_root,gt_dmap_root,model_param_path,3)
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from model import CSRNet

# Set your best epoch checkpoint and image path here
BEST_EPOCH = 78  # Use your best epoch
CHECKPOINT_PATH = f'./checkpoints/{BEST_EPOCH}.pth'

# Paths
IMAGE_PATH = './data/train_data/images/20250829_205259.jpg'   # test image
DENSITY_PATH = './data/train_data/densitymaps/20250829_205259.npy'  # matching .npy file

# Preprocessing
transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
])

# Load image
img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Load ground truth density map (from .npy)
gt_densitymap = np.load(DENSITY_PATH)

# Load model
model = CSRNet()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    et_densitymap = output.squeeze().cpu().numpy()
    pred_count = et_densitymap.sum()
    gt_count = gt_densitymap.sum()

print(f'Predicted count: {pred_count:.2f}, Ground Truth count: {gt_count:.2f}')

# Display image, GT density, and Predicted density
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(img)
axs[0].set_title(f'Input Image\nGT: {gt_count:.2f}, Pred: {pred_count:.2f}')
axs[0].axis('off')

axs[1].imshow(gt_densitymap, cmap='jet')
axs[1].set_title('Ground Truth Density')
axs[1].axis('off')

axs[2].imshow(et_densitymap, cmap='jet')
axs[2].set_title('Predicted Density')
axs[2].axis('off')

plt.show()
