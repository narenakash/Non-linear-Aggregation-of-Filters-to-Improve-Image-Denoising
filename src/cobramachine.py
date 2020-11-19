# WIP 
from denoise import denoiseMethods, Denoise

class CobraMachine:
    """
    cobra machine definition
    """
    def __init__(self, denoise, size):
        """
        denoise : denoise method name
        size: patch size
        """
        self.size = size
        self.denoise = denoise
        self.methods = denoiseMethods()

    def predict(self, img):
        self.denoisedIMG = []
        count = 1 if len(img.shape)==1 else img.shape[0]
        for i in range(count):
            if len(img.shape)==1:
                img2 = img.reshape((2*self.size+1,2*self.size+1))
            else:
                img2 = img[i].reshape((2*self.size+1, 2*self.size+1))
            dn = Denoise(img2)
            ret = dn.denoiseNAME(self.denoise)
            
            if 0 < self.size :
                self.denoisedIMG.append([ret[self.size,self.size]])
            else:
                self.denoisedIMG.append([ret])

        return self.denoisedIMG

