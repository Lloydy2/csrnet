'''
This script is for generating the ground truth density map 
for ShanghaiTech PartB. 
'''
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import json


# def generate_fixed_kernel_densitymap(image,points,sigma=3):
#     '''
#     Use fixed size kernel to construct the ground truth density map 
#     for ShanghaiTech PartB. 
#     image: the image with type numpy.ndarray and [height,width,channel]. 
#     points: the points corresponding to heads with order [col,row]. 
#     sigma: the sigma of gaussian_kernel to simulate a head. 
#     '''
#     # the height and width of the image
#     image_h = image.shape[0]
#     image_w = image.shape[1]

#     # coordinate of heads in the image
#     points_coordinate = points
#     # quantity of heads in the image
#     points_quantity = len(points_coordinate)

#     # generate ground truth density map
#     densitymap = np.zeros((image_h, image_w))
#     for point in points_coordinate:
#         c = min(int(round(point[0])),image_w-1)
#         r = min(int(round(point[1])),image_h-1)
#         densitymap[r,c] = 1
#     densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

#     total = densitymap.sum()
#     if total > 0:
#         densitymap = densitymap / total * points_quantity
#     else:
#         print('Warning: densitymap.sum() == 0 for image with', points_quantity, 'points')
#         # densitymap remains as is, or you can skip normalization
#     return densitymap    

def generate_fixed_kernel_densitymap(image, points, sigma=4, resize=None):
    """
    Generate ground truth density map for CSRNet.
    image: numpy.ndarray (H, W, C)
    points: list of [x, y] coordinates
    sigma: Gaussian sigma
    resize: (W, H) if you want to resize both image and densitymap
    """
    if resize is None:
        h, w = image.shape[0], image.shape[1]
        scale_x, scale_y = 1.0, 1.0
    else:
        h, w = resize[1], resize[0]
        scale_x = w / image.shape[1]
        scale_y = h / image.shape[0]

    densitymap = np.zeros((h, w))

    # Rescale annotation points
    points_resized = [(round(p[0] * scale_x), round(p[1] * scale_y)) for p in points]

    for x, y in points_resized:
        if 0 <= x < w and 0 <= y < h:
            densitymap[y, x] = 1

    # Apply Gaussian
    densitymap = gaussian_filter(densitymap, sigma=sigma, mode='constant')

    # Normalize to match object count
    total = densitymap.sum()
    if total > 0:
        densitymap = densitymap / total * len(points_resized)

    return densitymap

def load_via_points(json_path, image_filename):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Find the entry for the image
    for key, value in data.items():
        if value['filename'] == image_filename:
            regions = value['regions']
            points = []
            # Handle both dict and list formats
            if isinstance(regions, dict):
                region_iter = regions.values()
            elif isinstance(regions, list):
                region_iter = regions
            else:
                region_iter = []
            for region in region_iter:
                shape = region.get('shape_attributes', {})
                if 'cx' in shape and 'cy' in shape:
                    cx = shape['cx']
                    cy = shape['cy']
                    points.append([cx, cy])
            return points
    return []

if __name__ == '__main__':
    for phase in ['train', 'test']:
        images_dir = f'../data/{phase}_data/images/'
        densitymaps_dir = f'../data/{phase}_data/densitymaps/'
        json_path = '../data/via_export_json.json'
        if not os.path.exists(densitymaps_dir):
            os.makedirs(densitymaps_dir)
        if not os.path.exists(images_dir):
            print(f'Warning: {images_dir} does not exist. Skipping.')
            continue
        image_file_list = os.listdir(images_dir)
        for image_file in tqdm(image_file_list):
            image_path = os.path.join(images_dir, image_file)
            image = plt.imread(image_path)
            points = load_via_points(json_path, image_file)
            densitymap = generate_fixed_kernel_densitymap(image, points, sigma=4)
            np.save(os.path.join(densitymaps_dir, image_file.replace('.jpg', '.npy').replace('.jpeg', '.npy')), densitymap)
        print(f'Density maps have been generated for {phase}_data.')