import numpy as np
from denoise.denoise import denoiseMethods
from helper import *
from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation
from cobramachine import *

def cobraModelInit(trainNames, noiseType, imShape, patchSize=1, best=True):

    trainingData, trainingData1, trainingData2, testingData = loadTrainingData(
        trainNames, noiseType, patchSize)
    denoisemethods = denoiseMethods()
    epsilon = 0.2
    machines = 4
    cobra = Cobra(epsilon=epsilon, machines=machines)

    cobra.fit(trainingData, testingData)

    for i, denoise in enumerate(denoisemethods):
        cobra.load_machine(denoise, CobraMachine(denoise, patchSize))

    cobra.load_machine_predictions()
    # print("Predictions:", cobra.machine_predictions_)

    if best:

        cobra_diagnostics = Diagnostics(cobra, trainingData, testingData)
        epsilon, _ = cobra_diagnostics.optimal_epsilon(
            trainingData, testingData, line_points=100, info=False)
        machines, _ = cobra_diagnostics.optimal_alpha(
            trainingData, testingData, epsilon=Epsilon_opt, info=False)

        cobra = Cobra(epsilon=epsilon, machines=machines)
        cobra.fit(
            trainingData,
            testingData,
            default=False,
            X_k=trainingData1,
            X_l=trainingData2,
            y_k=testingData,
            y_l=testingData)
        for i, denoise in enumerate(denoisemethods):
            cobra.load_machine(denoise, CobraMachine(denoise, i, patchSize))
        cobra.load_machine_predictions()
        # print("Predictions:", cobra.machine_predictions_)

    return cobra, machines, epsilon

def cobraDenoise(noisy, model,noise_class, n_of_machines, p_size=1):
    print("Image denoising...")
    testX = []
    for x in range(p_size, noise_class.originalImg.shape[0]-p_size):
      for y in range(p_size,noise_class.originalImg.shape[1]-p_size):
        testX.append(getNeighbours(noisy, x, y, p_size))
    print("Dolty")
    Y = model.predict(testX, n_of_machines)
    print("Dolty2")
    # Padding image
    if p_size:
      Ytmp = noisy.copy()
      for x in range(p_size, noise_class.originalImg.shape[0]-p_size) :
        for y in range(p_size,noise_class.originalImg.shape[1]-p_size) : 
          Ytmp[x, y] = (noise_class.originalImg.shape[1]-2*p_size) * Y[(x - p_size)+(y - p_size)]

      Y = Ytmp.reshape(-1)
    return Y