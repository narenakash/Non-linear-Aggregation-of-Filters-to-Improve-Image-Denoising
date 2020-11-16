from copy import deepcopy
from random import randint, random
import os

import numpy as np
import cv2

# from .errors import ImageNotFoundError, InvalidImageError


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
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,self.img.shape)
        gauss = gauss.reshape(self.img.shape).astype(np.uint8)
        self.img = cv2.add(self.img, gauss)

        self.edits.append(f"gaussian:{var}")
        return self

    def speckle(self, var=0.1, mean=0.0):
        """
        Adds speckle noise to the image
        """
        noise = np.random.normal(mean, var**0.5, self.img.shape)
        noisy = self.img + self.img * noise
        self.img = noisy
        self.edits.append(f"speckle:{var}")
        return self
      
    def poisson(self, maxv = 0.2):
        """
        Adds poisson noise to the image
        """
        noisy = np.random.poisson(self.img / 255.0 * maxv) / maxv * 255  

        self.img = noisy.astype(np.uint8)
        self.edits.append(f"poisson")
        return self

    def patchSupression(self, patch_nb = 5, patch_size = 20):
        """
        Suppress random patch from the original image
        """
        noisy = np.copy(self.img)
        for i in range(patch_nb):
            x = np.random.randint(0, self.img.shape[0] - patch_size)
            y = np.random.randint(0, self.img.shape[1] - patch_size)
            noisy[x:x + patch_size,y:y + patch_size] = 1

        self.img = noisy
        self.edits.append(f"patchsup")
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
    # Noise("./nin.png").salt().pepper().write()
    # Noise("./nin.png").patchSupression().write()
    # Noise("./nin.png").gaussian().write()
    # Noise("./nin.png").speckle().write()
    Noise("./Aaron_Eckhart_0001.jpg").poisson().write()
