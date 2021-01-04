import random
from utils import *


def get_random_ab_channel(pathname_list):
    """
    Returns the ab channel of a random image from the target set
    Parameters
    ----------
    pathname_list : list
        list of pathname from which to sample a random image
    Returns
    -------
    image_ab : numpy
        vector of dim (h, w, 2)
    """
    train_set_dir = 'datasets/celeba/train/human/'
    idx = random.randrange(0, len(pathname_list))
    path = pathname_list[idx]
    filename = path.split('\\')[-1]
    path = train_set_dir + filename
    img_lab = lab_image_from_file(path)
    return img_lab[..., 1:]


class ImageTest(object):
    def __init__(self, img_ab):
        """ Creates an object to run accuracy tests against the image

        Parameters
        ----------
        img_ab : numpy of dim (h, w, 2)
            the image against which all tests will be run
        """
        self.img_ab = img_ab

    def raw_accuracy(self, test_img_ab):
        """ Take two numpy representing the AB channel of the ground truth
            and the predicted colorization, returns the raw accuracy as
            defined in https://arxiv.org/abs/1603.08511 at page 11 """
        n_pixels = self.img_ab.shape[0] * self.img_ab.shape[1]
        thresholds = np.arange(150)
        l2_dist = np.sqrt(((self.img_ab - test_img_ab) ** 2).sum(axis=2))
        accs = ((l2_dist[..., None] <= thresholds).sum(axis=2) / n_pixels).sum()
        return accs / 150

    def rmse(self, test_img_ab):
        np.sqrt((self.img_ab - test_img_ab) ** 2)