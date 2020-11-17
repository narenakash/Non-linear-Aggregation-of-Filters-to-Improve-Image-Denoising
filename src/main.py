import numpy as np
from .helper import *
from cobra import *
from cobramachine import *
from .noise import *
from .denoise import *

if __name__ == "__main__":
    command = "ls train"
    train_names = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    train_names.pop()
    train_names = [ 'train/' + name for name in train_names ]

    command = "ls test"
    file_name = subprocess.check_output(command, shell=True).decode('utf-8').split('\n')
    file_name.pop()
    file_name = 'test/' + file_name[0]

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