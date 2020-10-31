from copy import deepcopy
from random import randint, random
import os

import numpy as np
import cv2

from .errors import ImageNotFoundError, InvalidImageError


class Noise:
    """
    To add noise to an image image
    """

    def __init__(self, img_path):
        self.img_path = img_path

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        self.edits = [name]
        self.ext = ext

        self.img = self.imread(img_path)

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

   
    def salt(self, prob=0.08):
        """
        Adds salt noise to the image
        """
        h, w, c = self.img.shape
        for i in range(h):
            for j in range(w):
                if random() < prob:
                    self.img[i, j] = 255

        self.edits.append(f"salt:{prob}")
        return self

    def pepper(self, prob=0.08):
        """
        Adds pepper noise to the image
        """
        h, w, c = self.img.shape
        for i in range(h):
            for j in range(w):
                if random() < prob:
                    self.img[i, j] = 0

        self.edits.append(f"pepper:{prob}")
        return self

    def gaussian(self, var = 0.1, mean = 0):
        """
        Adds gaussian noise to the image
        """
        h, w, c = self.img.shape
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,self.img.shape)
        gauss = gauss.reshape(self.img.shape)
        noisy = self.img + gauss
        self.img = noisy
        return self

    def speckle(self, var=0.1, mean=0.0):
        """
        Adds speckle noise to the image
        """
        noise = np.random.normal(mean, var**0.5, self.img.shape)
        noisy = self.img + self.img * noise
        self.img = noisy
        return self
      
    def poisson(self):
        """
        Adds poisson noise to the image
        """
        # determine unique values in image and calculate the next power of two
        vals = len(np.unique(self.img))
        vals = 2 ** np.ceil(np.log2(vals))

        # ensure image is exclusively positive
        if self.img.min() < 0:
            old_max = self.img.max()
            self.img = (self.img + 1.0) / (old_max + 1.0)

        # generating noise for each unique value in the image
        noisy = np.random.poisson(self.img * vals) / float(vals)

        # return image to original range if input was signed
        if self.img.min() < 0:
            noisy = noisy * (old_max + 1.0) - 1

        self.img = noisy
        return self

    def patchSupression(self):
        return self


    def copy(self):
        """
        Returns a copy of this
        """
        return deepcopy(self)

    def write(self, name=None, ext=None, directory=None):
        """
        Writes the image to disk
        """
        delim = "_"
        if name is None:
            name = delim.join(self.edits) + delim
        if directory is not None:
            name = os.path.join(directory, name)
        if ext is None:
            ext = self.ext

        return cv2.imwrite(name + ext, self.img)


if __name__ == "__main__":
    Noise("./inp.jpg").salt().pepper().write()
