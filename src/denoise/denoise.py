from copy import deepcopy
from random import randint, random
import os

import numpy as np
import cv2

from .errors import ImageNotFoundError, InvalidImageError


class Denoise:
    """
    To add noise to an image image
    """

    def __init__(self, img):
        self.img = img
        self.ext = "jpg"

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
        self.img = cv.fastNlMeansDenoisingColored(self.img, None, h, hc, tws ,sws) 

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
        if directory is not None:
            name = os.path.join(directory, name)
        if ext is None:
            ext = self.ext

        return cv2.imwrite(name + ext, self.img)


if __name__ == "__main__":
    Denoise(img).median().write(name="file1")
