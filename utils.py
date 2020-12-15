import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import sklearn.neighbors as nn
from scipy.signal import gaussian, convolve

# equations numbers in the following refer to Colorful Image Colorization, Zhang et al.
# https://arxiv.org/abs/1603.08511

# Load the color prior factor that encourages rare colors
prior_factor = np.load("data/prior_factor.npy").astype(np.float32)
pts_hull = np.load("data/pts_in_hull.npy")
q = 313

# parameters for smoothed soft-encoding
n_points = 1000
sigma = 5

def multinomial_cross_entropy(z_pred, z_true):
    pass


def compute_prior_prob(image_ab):
    """
    image_ab: numpy of shape (2, 224, 224)
    """
    nearest = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    nearest.fit(pts_hull)

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


def compute_smoothed(image_ab):
    """
    image_ab: numpy of shape (2, 224, 224)
    """
    # noinspection DuplicatedCode
    nearest = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    nearest.fit(pts_hull)

    # flattened array of shape (224*224, 2)
    image_a = image_ab[0, :, :].ravel()
    image_b = image_ab[1, :, :].ravel()
    image_ab = np.vstack((image_a, image_b)).T

    # distances and indices of the 5 nearest neighbour of the true image ab channel
    dists, ind = nearest.kneighbors(image_ab)

    n = dists.shape[0]
    z_soft_enc = np.zeros((n, q))

    # create soft encoding vector with 0s everywhere but the 5-nearest pixels
    for i in range(n):
        z_soft_enc[i][ind[i]] = dists[i]

    # gaussian smoothing
    window = gaussian(n_points, sigma)
    smoothed = np.array([convolve(x, window, mode='same') for x in z_soft_enc])
    smoothed_normalized = smoothed / smoothed.sum(axis=1).reshape(-1, 1)

    return smoothed_normalized


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
    plt.clf() # clear matplotlib 
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0))    # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128     
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None: 
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))