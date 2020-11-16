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

        self.originalImg = self.imread(img_path)
        self.saltImg = np.empty(self.originalImg.shape)
        self.pepperImg = np.empty(self.originalImg.shape)
        self.gaussianImg = np.empty(self.originalImg.shape)
        self.speckleImg = np.empty(self.originalImg.shape)
        self.poissonImg = np.empty(self.originalImg.shape)
        self.patchSupressionImg = np.empty(self.originalImg.shape)
        self.multiNoiseImg = np.empty(self.originalImg.shape)
        self.allNoises = []

    @staticmethod
    def imread(img_path):
        """
        Raises exception if file doesn't exist or is invalid
        Returns the image if valid
        """
        if not os.path.exists(img_path):
            raise ImageNotFoundError(f"Image {img_path} could'nt be located")

        img = cv2.imread(img_path, 0)

        if img is None:
            raise InvalidImageError(f"Image {img_path} could'nt be loaded")

        return img

    def noiseMethods(self):
      dm = ['salt', 'pepper', 'gaussian', 'speckle', 'poisson',
            'patchSupression', 'multiNoise']

      return dm

    def salt(self, prob=0.08):
        """
        Adds salt noise to the image
        """
        h, w = self.originalImg.shape
        self.saltImg = self.originalImg.copy()
        for i in range(h):
            for j in range(w):
                if random() < prob:
                    self.saltImg[i, j] = 255

        self.edits.append(f"salt:{prob}")
        return self.saltImg

    def pepper(self, prob=0.08):
        """
        Adds pepper noise to the image
        """
        h, w = self.originalImg.shape
        self.pepperImg = self.originalImg.copy()
        for i in range(h):
            for j in range(w):
                if random() < prob:
                    self.pepperImg[i, j] = 0

        self.edits.append(f"pepper:{prob}")
        return self.pepperImg

    def gaussian(self, var = 0.1, mean = 0):
        """
        Adds gaussian noise to the image
        """
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,self.originalImg.shape)
        gauss = gauss.reshape(self.originalImg.shape).astype(np.uint8)
        self.gaussianImg = self.originalImg.copy()
        self.gaussianImg = cv2.add(self.gaussianImg, gauss)

        self.edits.append(f"gaussian:{var}")
        return self.gaussianImg

    def speckle(self, var=0.1, mean=0.0):
        """
        Adds speckle noise to the image
        """
        noise = np.random.normal(mean, var**0.5, self.originalImg.shape)
        noisy = self.originalImg + self.originalImg * noise
        self.speckleImg = noisy
        self.edits.append(f"speckle:{var}")
        return self.speckleImg
      
    def poisson(self, maxv = 0.2):
        """
        Adds poisson noise to the image
        """
        # noisy = np.random.poisson(self.img / 255.0 * maxv) / maxv * 255  

        # self.img = noisy.astype(np.uint8)
        rng = np.random.default_rng(None)
        self.poissonImg = rng.poisson(self.img)
        # noisy = (np.random.poisson(self.originalImg / 255.0 * maxv) / maxv * 255).astype('uint8')
        # self.poissonImg = noisy.astype(np.uint8)
        self.edits.append(f"poisson")
        return self.poissonImg

    def patchSupression(self, patch_nb = 5, patch_size = 2):
        """
        Suppress random patch from the original image
        """
        self.patchSupressionImg = self.originalImg.copy()
        noisy = np.copy(self.patchSupressionImg)
        for i in range(patch_nb):
            x = np.random.randint(0, self.patchSupressionImg.shape[0] - patch_size)
            y = np.random.randint(0, self.patchSupressionImg.shape[1] - patch_size)
            noisy[x:x + patch_size,y:y + patch_size] = 1

        self.patchSupressionImg = noisy
        self.edits.append(f"patchsup")
        return self.patchSupressionImg
    
    def addPatch(self, I):
      tmp = self.originalImg.copy()
      self.originalImg = I.copy()
      self.patchSupression()
      self.originalImg = tmp.copy()
      return self.patchSupressionImg
    
    def getAllNoises(self):
        self.allNoises = []
        self.allNoises.append(self.salt())
        self.allNoises.append(self.pepper())
        self.allNoises.append(self.gaussian())
        self.allNoises.append(self.speckle())
        self.allNoises.append(self.poisson())
        self.allNoises.append(self.patchSupression())
        self.allNoises.append(self.multiNoise())
        return self.allNoises

    def multiNoise(self):
        self.gaussian()
        self.salt()
        self.pepper()
        self.poisson()
        self.speckle()
        self.patchSupression()
        self.multiNoiseImg = np.zeros(self.originalImg.shape)
        self.multiNoiseImg[0:self.originalImg.shape[0]//2, 0:self.originalImg.shape[1]//2] = self.gaussianImg[0:self.originalImg.shape[0]//2, 0:self.originalImg.shape[1]//2]
        self.multiNoiseImg[0:self.originalImg.shape[0]//2, self.originalImg.shape[1]//2:self.originalImg.shape[1]] = self.saltImg[0:self.originalImg.shape[0]//2, self.originalImg.shape[1]//2:self.originalImg.shape[1]]
        self.multiNoiseImg[self.originalImg.shape[0]//2:self.originalImg.shape[0], 0:self.originalImg.shape[1]//2] = self.poissonImg[self.originalImg.shape[0]//2:self.originalImg.shape[0], 0:self.originalImg.shape[1]//2]
        self.multiNoiseImg[self.originalImg.shape[0]//2:self.originalImg.shape[0], self.originalImg.shape[1]//2:self.originalImg.shape[1]] = self.speckleImg[self.originalImg.shape[0]//2:self.originalImg.shape[0], self.originalImg.shape[1]//2:self.originalImg.shape[1]]
        self.multiNoiseImg = self.addPatch(self.multiNoiseImg)
        return self.multiNoiseImg

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
