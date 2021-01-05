import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
from skimage.transform import resize
from PIL import Image, ImageTk
from pathlib import Path

# constants used to make the model flexible for both regression and classification
CLASSIFICATION = "classification"
REGRESSION = "regression"

# equations numbers in the following refer to Colorful Image Colorization, Zhang et al.
# https://arxiv.org/abs/1603.08511

pts_hull = torch.from_numpy(np.load("objects/pts_in_hull.npy"))
prior_factor = torch.from_numpy(np.load('objects/prior_factor.npy'))  # vector w of Eq.4
prior_prob_smoothed = torch.from_numpy(np.load('objects/prior_prob_smoothed.npy'))  # p tilde of Eq.4
tree = cKDTree(pts_hull)

# parameters for smoothed soft-encoding
n_points = 1000
sigma = 5
eps = 1e-5
q = 313
h, w = 176, 176  # celeb dataset


class CustomLoss(nn.Module):
    def __init__(self, type='classification', alpha=.5):
        super(CustomLoss, self).__init__()
        self.prior_factor = prior_factor
        self.type = type

        if type == 'classification':
            self.loss = self.multicrossentropy_loss
            if alpha != .5:
                self.prior_factor = compute_prior_factor(alpha)
        elif type == 'regression':
            self.loss = self.l2_loss

    def multicrossentropy_loss(self, z_pred, z_true):
        """

        Parameters
        ----------
        z_pred : torch tensor of dim (batch_size, h, w, q)
            image colorization probability distribution output by the model
        z_true : torch tensor of dim (batch_size, h, w, q)
            original colorization smoothed with gaussian filter
        Returns
        -------

        """
        batch_size = z_pred.shape[0]
        z_true = z_true.reshape(batch_size, -1, q)
        q_star = z_true.argmax(axis=2)

        use_gpu = z_pred.device.type == 'cuda'

        # rebalancing weighting term
        weight = self.prior_factor[q_star]

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

    def forward(self, z_pred, z_true):
        return self.loss(z_pred, z_true)

    def l2_loss(self, output_batch_ab, input_batch_ab):
        """

        Parameters
        ----------
        output_batch_ab : torch tensor of dim (batch_size, 2, h, w)
            image colorization probability distribution output by the model
        input_batch_ab : torch tensor of dim (batch_size, h, w, q)
            original colorization smoothed with gaussian filter

        Returns
        -------
            L2 loss according to Eq.1 at page 4 of Zhang et al.
        """
        return torch.sqrt(((output_batch_ab - input_batch_ab) ** 2).sum(axis=1)).sum()


def compute_prior_factor(alpha=.5):
    """ Compute prior factor according to Eq.4 (alpha = lambda) """
    weight = ((1 - alpha) * prior_prob_smoothed + alpha / q) ** (-1)
    return weight / (prior_prob_smoothed * weight).sum()


def gray_ab_tensor2lab(img_gray, img_ab):
    """

    Parameters
    ----------
    img_gray : tensor of dim (1, 224, 224)
    img_ab : tensor of dim (2, 224, 224)

    Returns
    -------
    numpy image in lab of dim (224, 224, 3)
    """
    img_gray = img_gray.detach().cpu().numpy().transpose((1, 2, 0))
    img_ab = img_ab.detach().cpu().numpy().transpose((1, 2, 0))

    out_lab = np.zeros((h, w, 3))
    out_lab[:, :, 0] = img_gray.reshape((h, w)) * 100
    out_lab[:, :, 1:] = img_ab * 255 - 128

    return out_lab


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
    out_lab = gray_ab_tensor2lab(img_gray, img_ab)

    return lab2rgb(out_lab)


def compute_smoothed_tensor(img_ab):
    """

        Parameters
        ----------
        img_ab : torch of dim (h, w, 2)

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


def gray_smooth_to_ab_tensor(img_gray, img_smooth):
    """

    Parameters
    ----------
    img_gray
    img_smooth

    Returns
    -------
        tensor
    """
    img_lab = gray_smooth_tensor2lab_npy(img_gray, img_smooth)
    img_ab = img_lab[..., 1:]
    return torch.tensor(img_ab.transpose((2, 0, 1)))


def ab_tensor_from_graysmooth(batch_gray, batch_smooth, temperature=.38):
    """

    Parameters
    ----------
    batch_gray : tensor batch of dim (batch_size, 1, h, w)
    batch_smooth : tensor batch of dim (batch_size, 331, h, w)
    temperature : float

    Returns
    -------
        a tensor batch of dim (batch_size, 2, h, w) representing
        the smoothed point estimate in the ab channel
    """
    use_gpu = batch_gray.device.type == 'cuda'

    batch_size = batch_gray.shape[0]
    z_exp = torch.exp(torch.log(batch_smooth + eps) / temperature)
    z_mean = torch.div(z_exp, z_exp.sum(axis=1).unsqueeze(1))
    z_mean = z_mean.reshape((batch_size, q, h * w))
    q_a = pts_hull[:, 0].reshape(1, -1)
    q_b = pts_hull[:, 1].reshape(1, -1)

    if use_gpu:
        q_a = q_a.cuda()
        q_b = q_b.cuda()

    x_a = (z_mean.T * q_a.unsqueeze(2)).sum(axis=1).reshape((batch_size, h, w))
    x_b = (z_mean.T * q_b.unsqueeze(2)).sum(axis=1).reshape((batch_size, h, w))

    batch_ab = torch.zeros((batch_size, 2, h, w))
    batch_ab[:, 0, ...] = x_a
    batch_ab[:, 1, ...] = x_b

    return batch_ab


def gray_smooth_tensor2lab_npy(img_gray, img_smooth, temperature=0.38):
    """

        Parameters
        ----------
        temperature
        img_gray : tensor of dim (1, h, w)
        img_smooth : tensor of dim (313, h, w)

        Returns
        -------

        """
    use_gpu = img_gray.device.type == 'cuda'

    z_exp = torch.exp(torch.log(img_smooth + eps) / temperature)
    z_mean = (z_exp / z_exp.sum(axis=0)).reshape((q, h * w))

    q_a = pts_hull[:, 0].reshape(1, -1)
    q_b = pts_hull[:, 1].reshape(1, -1)

    if use_gpu:
        q_a = q_a.cuda()
        q_b = q_b.cuda()

    x_a = (z_mean.T * q_a).sum(axis=1).reshape((h, w))
    x_b = (z_mean.T * q_b).sum(axis=1).reshape((h, w))

    x_np = img_gray.reshape(h, w)

    out_lab = np.zeros((h, w, 3))
    out_lab[:, :, 0] = x_np.detach().cpu().numpy() * 100
    out_lab[:, :, 1] = x_a.detach().cpu().numpy()
    out_lab[:, :, 2] = x_b.detach().cpu().numpy()

    return out_lab


def gray_smooth_tensor2rgb(img_gray, img_smooth, temperature=0.38):
    """

    Parameters
    ----------
    temperature
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

    img_rgb = lab2rgb(out_lab)

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


# ----------- FUNCTIONS FOR GUI VISUALIZATION ----------- #


def get_img(filename, ):
    """ Generate png image from jpg """
    img = Image.open(filename).resize((350, 350))
    return ImageTk.PhotoImage(img)


def load_img_np(pathname):
    return np.array(Image.open(pathname).convert('RGB'))


def get_img_prediction(model, pathname, color_space='rgb'):
    img = load_img_np(pathname)
    img_original_size = img.shape[:2]
    img_lab = rgb2lab(img)
    img_gray = img_lab[:, :, 0] / 100
    img_gray_small = resize(img_gray, (h, w))
    img_gray_tensor = torch.from_numpy(img_gray_small).unsqueeze(0).float()
    img_gray_batch = img_gray_tensor.unsqueeze(0)
    img_smooth = model(img_gray_batch)[0]
    if model.loss_type == 'classification':
        img_lab = gray_smooth_tensor2lab_npy(img_gray_tensor, img_smooth)
    else:
        img_lab = gray_ab_tensor2lab(img_gray_tensor, img_smooth)
    img_lab_resized = resize(img_lab, img_original_size)
    img_gray = img_gray * 100
    img_lab_resized[:, :, 0] = img_gray

    if color_space == 'rgb':
        img_rgb = lab2rgb(img_lab_resized)
        img_from_array = (img_rgb * 255).astype(np.uint8)
        return img_from_array
    elif color_space == 'lab':
        return img_lab_resized


def get_img_prediction_as_tk(model, pathname, img_size):
    """
    Parameters
    ----------
    model : pyTorch pretrained model
    pathname : target img pathname
    img_size : target img size

    Returns
    -------
        an image object that can be visualized in PySimpleGUI
    """
    img_pred = get_img_prediction(model, pathname)
    img_pred = Image.fromarray(img_pred).resize(img_size)
    return ImageTk.PhotoImage(image=img_pred)


def lab_image_from_file(filename):
    """ Returns the ab channels from the image in filename """
    true_img = load_img_np(filename)
    true_img_lab = rgb2lab(true_img)

    return true_img_lab


def image_list_from_dir(pathname):
    """ Returns the list of images in the target directory
        (search recursively in subdirectories)

    Parameters
    ----------
    pathname : string

    Returns
    -------
    image_list : list of strings
    """
    return list(map(str, Path(pathname).rglob('*.jpg')))


def img_ab_from_list(pathname_list):
    """ Returns the list of images ab channels from the pathaname list

    Parameters
    ----------
    pathname_list : list of strings

    Returns
    -------
    image_ab_list : list of numpy of dim (h, w, 2)
    """
    return [lab_image_from_file(x)[..., 1:] for x in pathname_list]


def img_ab_prediction_from_list(model, pathname_list):
    """ Returns the list of ab channels of predicted images provided by pathnames

    Parameters
    ----------
    model : ColorizationNet object
    pathname_list : list of strings

    Returns
    -------
    img_ab_list : list of numpy of dim (h, w, 2)
    """
    img_ab_list = [get_img_prediction(model, pathname,
                                      color_space='lab')[..., 1:]
                   for pathname in pathname_list[:10]]
    return img_ab_list
