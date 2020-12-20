import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from scipy.spatial import cKDTree

# equations numbers in the following refer to Colorful Image Colorization, Zhang et al.
# https://arxiv.org/abs/1603.08511

pts_hull = torch.from_numpy(np.load("data/pts_in_hull.npy"))
p_tilde_tensor = torch.from_numpy(np.load('data/prior_factor.npy'))
tree = cKDTree(pts_hull)

# parameters for smoothed soft-encoding
n_points = 1000
sigma = 5
lambda_ = 0.5
eps = 1e-5
q = 313
h, w = 224, 224


def multicrossentropy_loss(z_pred, z_true):
    batch_size = z_pred.shape[0]
    z_pred = z_pred
    z_true = z_true
    z_true = z_true.reshape(batch_size, -1, q)
    q_star = z_true.argmax(axis=2)

    use_gpu = z_pred.device.type == 'cuda'

    # rebalancing weighting term
    weight = p_tilde_tensor[q_star]

    if use_gpu:
        weight = weight.cuda()

    min_values = z_pred.min(axis=1).values.unsqueeze(1)
    z_pred_shifted = z_pred - min_values
    z_pred_shifted = z_pred_shifted + eps
    z_pred_shifted = z_pred_shifted / z_pred_shifted.sum(axis=1).unsqueeze(1)
    z_pred_shifted = z_pred_shifted.reshape(batch_size, -1, q)
    loss = (z_true * torch.log(z_pred_shifted)).sum(axis=2)
    loss = - loss * weight

    return loss.sum(axis=1).sum()


def gray_ab_tensor2rgb(img_gray, img_ab):
    """

    Parameters
    ----------
    img_gray : tensor of dim (1, 224, 224)
    img_ab : tensor of dim (2, 224, 224)

    Returns
    -------
    numpy image in rgb of dim (224, 224, 3)
    """
    img_gray = img_gray.detach().cpu().numpy().transpose((1, 2, 0))
    img_ab = img_ab.detach().cpu().numpy().transpose((1, 2, 0))

    out_lab = np.zeros((h, w, 3))
    out_lab[:, :, 0] = img_gray.reshape((h, w)) * 100
    out_lab[:, :, 1:] = img_ab * 255 - 128

    return lab2rgb(out_lab)


def compute_smoothed_tensor(img_ab):
    """

        Parameters
        ----------
        img_ab : torch of dim (224, 224, 2)

        Returns
        -------

        """
    if img_ab.dtype != torch.float32:
        img_ab = torch.from_numpy(img_ab)

    img_ab_flat = img_ab.reshape(-1, 2)

    # finds closest 5 bins to each pixel
    distance_matrix = torch.sqrt((img_ab_flat.unsqueeze(1) - pts_hull).pow(2).sum(axis=2))
    torch.save(distance_matrix, "distance.pt")
    dists, indices = torch.topk(distance_matrix, 5, largest=False)

    wts = torch.exp(- dists ** 2 / (2 * sigma ** 2))
    wts = wts / wts.sum(axis=1)[:, None]

    z_soft_encoding = torch.zeros((img_ab_flat.shape[0], q))
    idx_pts = np.arange(img_ab_flat.shape[0])[:, np.newaxis]
    z_soft_encoding[idx_pts, indices] = wts

    return z_soft_encoding.reshape((h, w, q)).permute(2, 0, 1)


def compute_smoothed(img_ab):
    """

    Parameters
    ----------
    img_ab : numpy of dim (224, 224, 2)

    Returns
    -------

    """
    img_ab = img_ab.reshape(-1, 2)

    # distances and indices of the 5 nearest neighbour of the true image ab channel
    dists, ind = tree.query(img_ab, k=5)

    wts = np.exp(- dists ** 2 / (2 * sigma ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    z_soft_encoding = np.zeros((img_ab.shape[0], q))

    idx_pts = np.arange(img_ab.shape[0])[:, np.newaxis]
    z_soft_encoding[idx_pts, ind] = wts

    return z_soft_encoding.reshape((h, w, q)).transpose((2, 0, 1))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gray_smooth_tensor2lab(img_gray, img_smooth, temperature=0.38):
    """

        Parameters
        ----------
        img_gray : tensor of dim (1, 224, 224)
        img_smooth : tensor of dim (313, 224, 224)

        Returns
        -------

        """
    z_exp = torch.exp(torch.log(img_smooth + eps) / temperature)
    z_mean = (z_exp / z_exp.sum(axis=0)).reshape((q, h * w))

    q_a = pts_hull[:, 0].reshape(1, -1)
    q_b = pts_hull[:, 1].reshape(1, -1)

    x_a = (z_mean.T * q_a).sum(axis=1).reshape((h, w))
    x_b = (z_mean.T * q_b).sum(axis=1).reshape((h, w))

    x_np = img_gray.reshape(h, w)

    out_lab = np.zeros((h, w, 3))
    out_lab[:, :, 0] = x_np * 100
    out_lab[:, :, 1] = x_a.detach().cpu().numpy()
    out_lab[:, :, 2] = x_b.detach().cpu().numpy()

    return out_lab


def gray_smooth_tensor2rgb(img_gray, img_smooth, temperature=0.38, save_path=None, save_name=None):
    """

    Parameters
    ----------
    img_gray : tensor of dim (1, 224, 224)
    img_smooth : tensor of dim (313, 224, 224)
    save_path : string
    save_name : string

    Returns
    -------

    """
    z_exp = torch.exp(torch.log(img_smooth + eps) / temperature)
    z_mean = (z_exp / z_exp.sum(axis=0)).reshape((q, h * w))

    q_a = pts_hull[:, 0].reshape(1, -1)
    q_b = pts_hull[:, 1].reshape(1, -1)

    x_a = (z_mean.T * q_a).sum(axis=1).reshape((h, w))
    x_b = (z_mean.T * q_b).sum(axis=1).reshape((h, w))

    x_np = img_gray.reshape(h, w)

    out_lab = np.zeros((h, w, 3))
    out_lab[:, :, 0] = x_np * 100
    out_lab[:, :, 1] = x_a.cpu().numpy()
    out_lab[:, :, 2] = x_b.cpu().numpy()

    img_rgb = lab2rgb(out_lab)

    if save_path is not None and save_name is not None:
        plt.imsave(arr=x_np, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=img_rgb, fname='{}{}'.format(save_path['colorized'], save_name))

    return img_rgb


def plot_comparison(img_gray, img_ab, img_smooth):
    """ Plot the original image and the predicted colorization side to side

    Parameters
    ----------
    img_gray : tensor of dim (1, 224, 224)
    img_ab : tensor of dim (2, 224, 224)
    img_smooth : tensor of dim (313, 224, 224)

    """
    n = len(img_gray)
    fig, axs = plt.subplots(n, 2, figsize=(10, 10))
    axs[0, 0].set_title("Original Image")
    axs[0, 1].set_title("Model Colorization")

    for i in range(n):
        img_true = gray_ab_tensor2rgb(img_gray[i], img_ab[i])
        prediction_rgb = gray_smooth_tensor2rgb(img_gray[i], img_smooth[i])
        axs[i, 0].imshow(img_true)
        axs[i, 1].imshow(prediction_rgb)


