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

        if len(img.shape)==1:
            img2 = img.reshape((2*self.size+1,2*self.size+1))
            dn = Denoise(img2)
            ret = dn.denoiseNAME(self.denoise)

            if 0 < self.size :
                self.denoisedIMG.append([ret[self.size,self.size]])
            else:
                self.denoisedIMG.append([ret])

            return self.denoisedIMG

        for i in range(img.shape[0]):

            img2 = img[i].reshape((2*self.size+1, 2*self.size+1))
            dn = Denoise(img2)
            ret = dn.denoiseNAME(self.denoise)
            
            if 0 < self.size :
                self.denoisedIMG.append([ret[self.size,self.size]])
            else:
                self.denoisedIMG.append([ret])

        return self.denoisedIMG

