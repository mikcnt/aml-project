# For everything
import numpy as np
import torch
# For dataset class
from torchvision import datasets, transforms
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
from utils import compute_smoothed


class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img_original = self.transform(img)
        img_original = np.asarray(img_original)
        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = compute_smoothed(img_ab)
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original).unsqueeze(0).float()

        return img_original, img_ab
