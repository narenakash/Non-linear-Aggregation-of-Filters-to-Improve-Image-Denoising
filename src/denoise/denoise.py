from copy import deepcopy
from random import randint, random
import os

import numpy as np
import cv2
import skimage.restoration

# from .errors import ImageNotFoundError, InvalidImageError

def denoiseMethods():
    dm = ['median', 'gaussian', 'bilateral', 'NLmeans', 'TVchambolle',\
            'richardsonLucy', 'inpaint']

    return dm


class Denoise:
    """
    To add noise to an image image
    """

    def __init__(self, img):
        self.img = img
        self.ext = ".jpg"

    @staticmethod
    def imread(img_path):
        """
        Raises exception if file doesn't exist or is invalid
        Returns the image if valid
        """
        if not os.path.exists(img_path):
            raise ImageNotFoundError(f"Image {img_path} could'nt be located")

        img = cv2.imread(img_path)

        if img is None:
            raise InvalidImageError(f"Image {img_path} could'nt be loaded")

        return img

    def denoiseNAME(self, img, method):
        getattr(self, method, 'img', img)

        return self.img
        

    def median(self, k=5):
        self.img = cv2.medianBlur(self.img, k)

        return self

    def gaussian(self, k=(5,5)):
        self.img = cv2.GaussianBlur(self.img, k, 0) 

        return self

    def bilateral(self, k = 9, s1 = 75, s2 = 75):
        self.img = cv2.bilateralFilter(self.img, k, s1, s2)

        return self

    def NLmeans(self, h = 10, hc = 10, tws = 7, sws = 21):
        self.img = cv2.fastNlMeansDenoisingColored(self.img, None, h, hc, tws ,sws) 

        return self

    def TVchambolle(self):
        self.img = skimage.restoration.denoise_tv_chambolle(self.img, multichannel=True)

        return self

    def richardson_lucy(self, point_spread_rl = 5):
        psf = np.ones((point_spread_rl, point_spread_rl)) / point_spread_rl**2
        result = np.zeros(self.img.shape)
        result[:,:,0] =  skimage.restoration.richardson_lucy(self.img[:,:,0], psf, point_spread_rl)
        result[:,:,1] =  skimage.restoration.richardson_lucy(self.img[:,:,1], psf, point_spread_rl)
        result[:,:,2] =  skimage.restoration.richardson_lucy(self.img[:,:,2], psf, point_spread_rl)
        self.img = result

        return self

    def inpaint(self):
        mask = (self.img.mean(axis=2) == 1)
        self.img = skimage.restoration.inpaint.inpaint_biharmonic(self.img, mask, multichannel=True)

        return self

    def copy(self):
        """
        Returns a copy of this
        """
        return deepcopy(self)

    def returnImage(self):
        return self.img

    def write(self, name=None, ext=None, directory=None):
        """
        Writes the image to disk
        """
        delim = "_"
        if directory is not None:
            name = os.path.join(directory, name)
        if ext is None:
            ext = self.ext

        return cv2.imwrite(name + ext, self.img)


if __name__ == "__main__":
    img = cv2.imread('./nin_speckle:0.1_.png')
    Denoise(img).median().write(name="median")
    Denoise(img).richardson_lucy().write(name="rl")
    Denoise(img).TVchambolle().write(name="tv")
    Denoise(img).NLmeans().write(name="NLm")
