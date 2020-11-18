import numpy as np
from helper import *
from cobra import *
from cobramachine import *
from noise import *
from denoise import *
import subprocess
import os, pickle, matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

class saveModel:
    def __init__(self, model, alpha, eps):
        self.model = model
        self.alpha = alpha
        self.eps = eps

if __name__ == "__main__":

    train_path = '../dataset/train/'
    train_names=[train_path+i for i in os.listdir(train_path)]
    print("Number of train images = " + str(len(train_names)))

    file_path = '../dataset/test/'
    file_names=[file_path+i for i in os.listdir(file_path)]
    file_name=file_names[0]

    noise_class = Noise(file_name)
    noise_class.multiNoise()
    noise_class.getAllNoises()

    noisyImgs = [noise_class.originalImg] + noise_class.allNoises
    noisyLabels = ['Original'] + noise_class.noiseMethods()
    cnt = 0
    f, axarr = plt.subplots(2, 4, figsize=(20,10))
    for i in range(2):
      for j in range(4):
        axarr[i][j].imshow(noisyImgs[cnt], cmap='gray')
        axarr[i][j].title.set_text(noisyLabels[cnt])
        cnt += 1

    plt.savefig("./results/1:Input.png")
    plt.cla()
    plt.clf()

    denoise_class = Denoise(noise_class.multiNoiseImg)
    denoise_class.getAllNoises()

    denoisedImgs = [denoise_class.img] + denoise_class.allNoises
    denoisedLabels = ['Original'] + denoise_class.denoiseMethods()
    cnt = 0
    f, axarr = plt.subplots(2, 4, figsize=(20,10))
    for i in range(2):
      for j in range(4):
        axarr[i][j].imshow(denoisedImgs[cnt], cmap='gray')
        axarr[i][j].title.set_text(denoisedLabels[cnt])
        cnt += 1

    plt.savefig("./results/2:MultiNoise-Output.png")
    plt.cla()
    plt.clf()

    training_noise_kind = [ i for i in range(len(noisyImgs) - 2) ]
    patch = 1
    noisy = noisyImgs[-1]
    loadModel = False
    model  = None
    alpha = 0
    eps = 0
    if loadModel:
        with open('best.pkl', 'rb') as input:
            model = pickle.load(input)
            alpha = model.alpha
            eps = model.eps
            model = model.model
    else:
        model, alpha, eps = cobraModelInit(train_names, training_noise_kind, noisy.shape, patchSize=patch, best=False)
    tosave = saveModel(model, alpha, eps)
    with open('best.pkl', 'wb') as output:
        pickle.dump(tosave, output, pickle.HIGHEST_PROTOCOL)

    Y = cobraDenoise(noisy, model,noise_class, alpha, p_size=patch)
    im_denoise = np.array(Y).reshape(noisy.shape)

    print('Display of the cobra denoising result')
    plt.imshow(im_denoise, cmap = plt.get_cmap('gray'))
    plt.savefig("./results/3:Cobra-Output.png")
    plt.cla()
    plt.clf()

    print("Evaluation...")
    evaluate = denoiseEvaluation(im_denoise, noise_class.originalImg)
    evaluate.evaluateAll()
    print("Saving the difference between denoised image and original one...")
    plt.imshow(evaluate.Idiff, cmap = plt.get_cmap('gray'))
    plt.savefig("./results/4:Difference.png")
    plt.cla()
    plt.clf()
