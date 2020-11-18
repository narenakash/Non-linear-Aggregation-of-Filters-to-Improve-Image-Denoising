import numpy as np
from noise import *

class denoiseEvaluation :
    def __init__(self, img1, img2):
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        
        self.PSNR = 0
        self.euclidianDist = 0
        self.RMSE = 0

    def getPSNR(self, peak=255):
        """
        Computes PSNR
        """
        x = (np.array(self.img1).squeeze() - np.array(self.img2).squeeze()).flatten()
        self.PSNR = np.log10(peak**2 / np.mean(x**2)) * 10
        return self.PSNR

    def euclidianDistance(self):
        """
        Compute euclidian distance
        """
        self.euclidianDist = np.linalg.norm(self.img1 - self.img2)
        return self.euclidianDist

    def getRMSE(self):
        """
        Computes RMSE
        """
        self.RMSE = np.sqrt(((self.img1 - self.img2) ** 2).mean())
        return self.RMSE

    def evaluateAll(self):
        self.euclidianDistance()
        self.getPSNR()
        self.getRMSE()
        print("Euclidian distance = ", self.euclidianDist)
        print("PSNR = ", self.PSNR)
        print("RMSE = ", self.RMSE)
        print('---------------------------------------------')
        return

def getNeighbours(I, x, y, k):
    """
    load and reshape the training data 
    """
    assert(0 <= x - k and x + k < I.shape[0] and 0 <= y - k and y + k < I.shape[1])
    return I[x - k:x + k + 1, y - k:y + k + 1].flatten()

def loadTrainingData(train_names, totNoises, k=0):
    trainingData  = []
    trainingData1 = []
    trainingData2 = []
    testingData  = []
    for f in train_names :
        noiseClass = Noise(f)
        noisyImgs = noiseClass.getAllNoises()
        for i in totNoises:
            sigma = 0.1
            epsilon = np.random.normal(0, sigma, noiseClass.originalImg.shape)

            testingData += [[noiseClass.originalImg[i1, i2]] for i1 in range(k, noiseClass.originalImg.shape[0] - k) for i2 in range(k, noiseClass.originalImg.shape[1] - k)]
            noisyImg = noisyImgs[i]

            v1 = noisyImg + sigma * epsilon
            v1 = (v1.min() - v1) / (v1.min() - v1.max())
            v2 = noisyImg - sigma * epsilon
            v2 = (v2.min() - v2) / (v2.min() - v2.max())
            
            for x in range(k,noiseClass.originalImg.shape[0]-k):
              for y in range(k,noiseClass.originalImg.shape[1]-k):
                trainingData1 += [getNeighbours(v1, x, y, k)]
                trainingData2 += [getNeighbours(v2, x, y, k)]
                trainingData += [getNeighbours(noisyImg, x, y, k)]

    return (trainingData, trainingData1, trainingData2, testingData)