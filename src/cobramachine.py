# WIP 
from .denoise import denoiseMethods, Denoise

class CobraMachine:
    def __init__(self, denoise, size):
        self.size = size
        self.denoise = denoise
        self.methods = denoiseMethods()

    def predict(self, img):
        self.denoisedIMG = []

        for i in range(img.shape[0]):
            dn = Denoise(img)
            ret = dn.denoiseNAME(self.denoise)
            
            if size:
                self.denoisedIMG.append([ret[self.size,self.size]])
            else:
                self.denoisedIMG.append([ret])

        return self.denoisedIMG


