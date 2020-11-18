from copy import deepcopy
from random import randint, random
import os
from skimage import img_as_float
import scipy.ndimage
import numpy as np
import cv2
import skimage.restoration

# from .errors import ImageNotFoundError, InvalidImageError

def denoiseMethods():
    dm = ['median', 'gaussian', 'bilateral', 'NLmeans', 'TVchambolle', 
          'richardson_lucy', 'inpaint']

    return dm

class Denoise:
    """
    To add noise to an image image
    """

    def __init__(self, img):
        self.img = img.astype('uint8')
        self.ext = ".jpg"
        self.medianImg = np.empty(self.img.shape)
        self.gaussianImg = np.empty(self.img.shape)
        self.bilateralImg = np.empty(self.img.shape)
        self.NLmeansImg = np.empty(self.img.shape)
        self.TVchambolleImg = np.empty(self.img.shape)
        self.richardsonLucyImg = np.empty(self.img.shape)
        self.inpaintImg = np.empty(self.img.shape)
        self.allNoises = []

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

    def denoiseNAME(self, method):
        ret = getattr(self, method)
        return ret()

    def denoiseMethods(self):
        dm = ['median', 'gaussian', 'bilateral', 'NLmeans', 'TVchambolle', 
              'richardson_lucy', 'inpaint']

        return dm        

    def median(self, k=5):
        self.medianImg = cv2.medianBlur(self.img, k)

        return self.medianImg

    def gaussian(self, k=(5,5)):
        self.gaussianImg = cv2.GaussianBlur(self.img, k, 0) 

        return self.gaussianImg

    def bilateral(self, k = 9, s1 = 75, s2 = 75):
        self.bilateralImg = cv2.bilateralFilter(self.img, k, s1, s2)

        return self.bilateralImg

    def NLmeans(self, h = 10, tws = 7, sws = 21):
        self.NLmeansImg = cv2.fastNlMeansDenoising(self.img, None, h, tws ,sws) 

        return self.NLmeansImg

    def TVchambolle(self):
        self.TVchambolleImg = skimage.restoration.denoise_tv_chambolle(self.img, multichannel=True)

        return self.TVchambolleImg

    def richardson_lucy(self, point_spread_rl = 5):
        psf = np.ones((point_spread_rl, point_spread_rl)) / point_spread_rl ** 2
        self.richardson_lucyImg = skimage.restoration.richardson_lucy(img_as_float(self.img), psf, point_spread_rl)

        return self.richardson_lucyImg

    def inpaint(self):
        mask = (self.img == 1)
        self.inpaintImg = skimage.restoration.inpaint.inpaint_biharmonic(self.img, mask, multichannel=False)

        return self.inpaintImg

    def getAllNoises(self):
        self.allNoises = []
        self.allNoises.append(self.median())
        self.allNoises.append(self.gaussian())
        self.allNoises.append(self.bilateral())
        self.allNoises.append(self.NLmeans())
        self.allNoises.append(self.TVchambolle())
        self.allNoises.append(self.richardson_lucy())
        self.allNoises.append(self.inpaint())
        return self.allNoises

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
    img = cv2.imread('../../dataset/test/Aaron_Eckhart_0001.jpg')
    x = Denoise(img)
    ret = x.denoiseNAME('bilateral')
    print(ret.shape)
    # Denoise(img).median().write(name="median")
    # Denoise(img).richardson_lucy().write(name="rl")
    # Denoise(img).TVchambolle().write(name="tv")
    # Denoise(img).NLmeans().write(name="NLm")
