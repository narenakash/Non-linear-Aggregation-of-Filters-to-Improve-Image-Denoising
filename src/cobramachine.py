# WIP 
from denoise import denoiseMethods, Denoise

class CobraMachine:
    def __init__(self, denoise, size):
        self.size = size
        self.denoise = denoise
        self.methods = denoiseMethods()

    def predict(self, img):
        self.denoisedIMG = []

        dn = Denoise(img)
        for i in range(img.shape[0]):
            ret = dn.denoiseNAME(self.denoise)
            
            if 0 < self.size < min(ret.shape[0], ret.shape[1]):
                self.denoisedIMG.append([ret[self.size,self.size]])
            else:
                self.denoisedIMG.append([ret])

        return self.denoisedIMG


