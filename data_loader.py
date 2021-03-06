from torchvision import datasets
from skimage.color import rgb2lab
from utils import *


class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        transposed_dim = (2, 0, 1)

        # crop image
        img_original = self.transform(img)
        img_original = np.asarray(img_original)

        # transform to lab
        img_lab = rgb2lab(img_original)

        # gray channel
        img_gray = img_lab[:, :, 0] / 100

        # ab channel
        img_ab = img_lab[:, :, 1:3]
        img_smooth = compute_smoothed(img_ab)

        # numpy to torch
        img_ab = img_ab.transpose(transposed_dim)
        img_ab = torch.from_numpy(img_ab).float()
        img_ab = (img_ab + 128) / 255
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_ab, img_smooth
