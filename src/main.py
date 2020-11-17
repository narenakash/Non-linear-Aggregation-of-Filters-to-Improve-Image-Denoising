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
