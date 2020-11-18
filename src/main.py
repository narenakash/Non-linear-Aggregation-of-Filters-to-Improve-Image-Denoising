import numpy as np
from helper import *
from cobra import *
from cobramachine import *
from noise import *
from denoise import *
import subprocess
import matplotlib.pyplot as plt

if __name__ == "__main__":
    command = "ls ../dataset/train"
    train_names = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    train_names.pop()
    train_names = [ '../dataset/train/' + name for name in train_names ]

    command = "ls ../dataset/test"
    file_name = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    file_name.pop()
    file_name = '../dataset/test/' + file_name[0]

    print("Number of train images = " + str(len(train_names)))

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

    plt.savefig("1.png")
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

    plt.savefig("2.png")
    plt.cla()
    plt.clf()

    training_noise_kind = [ i for i in range(len(noisyImgs) - 2) ]
    patch = 31
    noisy = noisyImgs[-1]
    model, alpha, eps = cobraModelInit(train_names, training_noise_kind, noisy.shape, patchSize=patch, best=False)
    Y = cobraDenoise(noisy, model,noise_class, alpha, p_size=patch)
    im_denoise = np.array(Y).reshape(noisy.shape)
    

    print('Display of the cobra denoising result')
    plt.imshow(im_denoise, cmap = plt.get_cmap('gray'))
    plt.savefig("3.png")
    plt.cla()
    plt.clf()

    print("Evaluation...")
    evaluate = denoiseEvaluation(im_denoise, noise_class.originalImg)
    evaluate.evaluateAll()
    print("Saving the difference between denoised image and original one...")
    plt.imshow(evaluate.Idiff, cmap = plt.get_cmap('gray'))
    plt.savefig("4.png")
    plt.cla()
    plt.clf()
