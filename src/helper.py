import numpy as np

class denoiseEvaluation :
    def __init__(self, img1, img2):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        
        self.PSNR = 0
        self.euclidianDist = 0
        self.RMSE = 0

def getNeighbours(I, x, y, k):
    """
    load and reshape the training data 
    """
    assert(0 <= x - k and x + k < I.shape[0] and 0 <= y - k and y + k < I.shape[1])
    return I[x - k:x + k + 1, y - k:y + k + 1].flatten()