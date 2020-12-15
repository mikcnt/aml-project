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


def multinomial_cross_entropy(z_pred, z_true):
    smoothed_normalized = compute_smoothed(z_true)
    q_star = smoothed_normalized.argmax(axis=1)

    # rebalancing weighting term
    weight = p_tilde[q_star]
    z_pred_shifted = z_pred - z_pred.min(axis=0)
    z_pred_shifted = z_pred_shifted + eps
    z_pred_shifted = z_pred_shifted / z_pred_shifted.sum(axis=0)
    z_pred_shifted = z_pred_shifted.reshape(-1, q)
    loss = (smoothed_normalized * np.log(z_pred_shifted)).sum(axis=1)
    loss = - loss * weight

    return loss.sum()


def compute_prior_prob(image_ab):
    """
    image_ab: numpy of shape (2, 224, 224)
    """
    # flattened array of shape (224*224, 2)
    image_a = image_ab[0, :, :].ravel()
    image_b = image_ab[1, :, :].ravel()
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
    # flattened array of shape (224*224, 2)
    h, w, _ = z_true.shape
    image_a = z_true[:, :, 0].ravel()
    image_b = z_true[:, :, 1].ravel()
    image_ab = np.vstack((image_a, image_b)).T

    # distances and indices of the 5 nearest neighbour of the true image ab channel
    dists, ind = nearest.kneighbors(image_ab)

    wts = np.exp(- dists ** 2) / (2 * sigma ** 2)
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


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
         Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf()  # clear matplotlib
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
