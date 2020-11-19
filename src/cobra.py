import numpy as np
from denoise.denoise import denoiseMethods
from helper import *
from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation
from cobramachine import *

def cobraModelInit(trainNames, noiseType, imShape, patchSize=1, best=True):
    """
    Initalise and train cobra mode
    
    """
    print("Making training data ready")
    trainingData, trainingData1, trainingData2, testingData = loadTrainingData(
        trainNames, noiseType, patchSize)
    denoisemethods = denoiseMethods()
    epsilon = 0.2
    machines = 3
    cobra = Cobra(epsilon=epsilon, machines=machines)
    print("Training model")
    cobra.fit(trainingData, testingData)

    for i, denoise in enumerate(denoisemethods):
        cobra.load_machine(denoise, CobraMachine(denoise, patchSize))

    cobra.load_machine_predictions()
    # print("Predictions:", cobra.machine_predictions_)

    if best:

        print("Running Diagnostics")
        cobra_diagnostics = Diagnostics(cobra, trainingData, testingData, load_MSE=False)
        print("epsilon")
        epsilon, _ = cobra_diagnostics.optimal_epsilon(
            trainingData, testingData, line_points=100, info=False)
        print("machines")
        machines, _ = cobra_diagnostics.optimal_alpha(
            trainingData, testingData, epsilon=epsilon, info=False)

        cobra = Cobra(epsilon=epsilon, machines=machines)
        print("fit")
        cobra.fit(
            trainingData,
            testingData,
            default=False,
            X_k=trainingData1,
            X_l=trainingData2,
            y_k=testingData,
            y_l=testingData)
        for i, denoise in enumerate(denoisemethods):
            cobra.load_machine(denoise, CobraMachine(denoise, patchSize))
        cobra.load_machine_predictions()
        # print("Predictions:", cobra.machine_predictions_)

    return cobra, machines, epsilon

def cobraDenoise(noisy, model,noise_class, machines, patchSize=1):
    """
    denoise an image based on cobra model
    noisy: noisy image
    model: trained cobra model
    noise_class: Noise class of orginal image
    machines: minimum no of machines that should satisfy:
    patchSize: patch size of image to consider
    """
    print("Image denoising...")
    testX = []
    for x in range(patchSize, noise_class.originalImg.shape[0]-patchSize):
      for y in range(patchSize,noise_class.originalImg.shape[1]-patchSize):
        testX.append(getNeighbours(noisy, x, y, patchSize))
    print("Dolty")
    Y = model.predict(testX, machines)
    print("Dolty2")
    # Padding image
    if patchSize:
      Ytmp = noisy.copy()
      for x in range(patchSize, noise_class.originalImg.shape[0]-patchSize) :
        for y in range(patchSize,noise_class.originalImg.shape[1]-patchSize) : 
          Ytmp[x,y] = Y[(x-patchSize)*(noise_class.originalImg.shape[1]-2*patchSize)+(y-patchSize)]

      Y = Ytmp.reshape(-1)
    return Y
