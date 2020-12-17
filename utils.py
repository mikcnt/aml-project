import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import sklearn.neighbors as nn

# equations numbers in the following refer to Colorful Image Colorization, Zhang et al.
# https://arxiv.org/abs/1603.08511

pts_hull = np.load("data/pts_in_hull.npy")
p_tilde = np.load('data/prior_factor.npy')
nearest = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nearest.fit(pts_hull)

# parameters for smoothed soft-encoding
n_points = 1000
sigma = 5
lambda_ = 0.5
eps = 1e-5
q = 313
temperature = 0.38
h, w = 224, 224


def multicrossentropy_loss(z_pred, z_true):
    batch_size = z_pred.shape[0]
    z_pred = z_pred.cuda()
    z_true = z_true.cuda()
    z_true = z_true.reshape(batch_size, -1, q)
    q_star = z_true.argmax(axis=2)

    # rebalancing weighting term
    p_tilde_tensor = torch.from_numpy(p_tilde).cuda()
    weight = p_tilde_tensor[q_star]
    min_values = z_pred.min(axis=1).values.unsqueeze(1)
    z_pred_shifted = z_pred - min_values
    z_pred_shifted = z_pred_shifted + eps
    z_pred_shifted = z_pred_shifted / z_pred_shifted.sum(axis=1).unsqueeze(1)
    z_pred_shifted = z_pred_shifted.reshape(batch_size, -1, q)
    loss = (z_true * torch.log(z_pred_shifted)).sum(axis=2)
    loss = - loss * weight

    return loss.sum(axis=1).mean()


def compute_prior_prob(image_ab):
    """
    image_ab: numpy of shape (2, 224, 224)
    """
    # flattened array of shape (224*224, 2)
    image_a = image_ab[:, :, 0].ravel()
    image_b = image_ab[:, :, 1].ravel()
    image_ab = np.vstack((image_a, image_b)).T

    # distances and indices of the 5 nearest neighbour of the true image ab channel
    dists, ind = nearest.kneighbors(image_ab)

    ind = np.ravel(ind)
    counts = np.bincount(ind)
    idxs = np.nonzero(counts)[0]

    # create vector of prior probabilities from non zero occurring pixels
    prior_prob = np.zeros(pts_hull.shape[0])
    prior_prob[idxs] = counts[idxs]
    prior_prob = prior_prob / (1.0 * counts.sum())

    return prior_prob


def compute_smoothed(z_true):
    """

    Parameters
    ----------
    z_true : torch of dim (224, 224, 2)

    Returns
    -------

    """
    image_a = z_true[:, :, 0].ravel()
    image_b = z_true[:, :, 1].ravel()
    image_ab = np.vstack((image_a, image_b)).T

    # distances and indices of the 5 nearest neighbour of the true image ab channel
    dists, ind = nearest.kneighbors(image_ab)

    wts = np.exp(- dists ** 2 / (2 * sigma ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    z_soft_encoding = np.zeros((image_ab.shape[0], q))

    idx_pts = np.arange(image_ab.shape[0])[:, np.newaxis]
    z_soft_encoding[idx_pts, ind] = wts

    return z_soft_encoding.reshape((h, w, q))


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


def to_rgb(img_gray, img_smooth, save_path=None, save_name=None):
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
    out_lab[:, :, 1] = x_a
    out_lab[:, :, 2] = x_b
    img_rgb = lab2rgb(out_lab)

    if save_path is not None and save_name is not None:
        plt.imsave(arr=x_np, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=img_rgb, fname='{}{}'.format(save_path['colorized'], save_name))

    return img_rgb
